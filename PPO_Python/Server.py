import socket
import threading
import time
import sys
import numpy as np
import CasualtiesEnv
import Train
import CriticTraining
import ReinforceTraining
import Inference
import Validation
import Types

TCP_HOST = "127.0.0.1"
OBS_PORT = 45701
ACTION_PORT = 45702
running = True
reset_requested = threading.Event()
action_write_lock = threading.Lock()
observation_lock = threading.Lock()
shutdown_lock = threading.Lock()
shutdown_started = False
simulation_paused = threading.Event()
pause_applied = threading.Event()
resume_applied = threading.Event()
reset_ready_condition = threading.Condition()
reset_ready_results = {}

def CreateTcpListener(port):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((TCP_HOST, port))
    listener.listen(1)
    return listener


print(f"Creating observation TCP listener on {TCP_HOST}:{OBS_PORT}...")
obs_listener = CreateTcpListener(OBS_PORT)

print(f"Creating action TCP listener on {TCP_HOST}:{ACTION_PORT}...")
action_listener = CreateTcpListener(ACTION_PORT)

obs_pipe = None
action_pipe = None


def RecvExact(connection, size):
    chunks = []
    remaining = size

    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Unity observation connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def BuildActionMessage():
    return f"{move},{jump},{vertMove},{crouch},{lookdX},{lookdY},{attack},{interact},{targetSlotIndex},{selectedSlotIndex},{dropItem},{moveItem},{selectedBagIndex},{useItem},{useItemWorld},{selectedLimb},{useItemMedical},{selectedRecipe},{favoriteItem},{switchMainHand},{trySleep},{ragdoll},{exercise},{bark},{throw},{liquidAmount},{drainLiquid},{pullLiquidFromWorld},{action_sequence}\n".encode("utf-8")


def ControlAckLoop():
    """Receive pause-state acknowledgements from the Unity bridge."""
    buffer = b""
    while running:
        try:
            data = action_pipe.recv(256)
            if not data:
                return
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line == b"PAUSED":
                    pause_applied.set()
                elif line == b"RESUMED":
                    resume_applied.set()
                elif line.startswith(b"RESET_READY "):
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            reset_token = int(parts[1])
                            first_observation_id = int(parts[2])
                            world_seed = int(parts[3])
                        except ValueError:
                            print(f"Malformed Unity reset acknowledgement: {line!r}")
                        else:
                            with reset_ready_condition:
                                reset_ready_results[reset_token] = (
                                    first_observation_id,
                                    world_seed,
                                )
                                reset_ready_condition.notify_all()
                    else:
                        print(f"Malformed Unity reset acknowledgement: {line!r}")
        except OSError:
            return


def WaitForResetReady(reset_token, expected_world_seed, timeout=30.0):
    """Wait for Unity to acknowledge the seeded reset and first observation."""
    deadline = time.monotonic() + timeout
    with reset_ready_condition:
        while reset_token not in reset_ready_results:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Unity did not acknowledge reset token {reset_token} "
                    f"within {timeout:.1f}s"
                )
            reset_ready_condition.wait(timeout=remaining)

        first_observation_id, applied_world_seed = reset_ready_results.pop(
            reset_token
        )
        if applied_world_seed != expected_world_seed:
            raise RuntimeError(
                "Unity acknowledged the wrong world seed for reset "
                f"{reset_token}: expected {expected_world_seed}, "
                f"applied {applied_world_seed}"
            )
        return first_observation_id


def PauseSimulation():
    """Freeze Unity before PPO's optimizer touches the collected rollout."""
    if simulation_paused.is_set():
        return

    pause_applied.clear()
    simulation_paused.set()
    with action_write_lock:
        action_pipe.sendall(b"PAUSE\n")

    if not pause_applied.wait(timeout=5.0):
        raise RuntimeError("Unity did not acknowledge the PPO pause")


def ResumeSimulation():
    """Resume Unity with the newly decoded post-update action already queued."""
    if not simulation_paused.is_set():
        return

    resume_applied.clear()
    with action_write_lock:
        if reset_requested.is_set():
            raise RuntimeError("Cannot resume Unity while a reset is in progress")
        action_pipe.sendall(b"RESUME\n")
        action_pipe.sendall(BuildActionMessage())
    simulation_paused.clear()

    if not resume_applied.wait(timeout=5.0):
        raise RuntimeError("Unity did not acknowledge the PPO resume")

def SendDecodedAction():
    """Send exactly one packet for the already-decoded CB1 decision."""
    if reset_requested.is_set():
        raise RuntimeError("Cannot send a policy action while a reset is in progress")

    if simulation_paused.is_set():
        # ResumeSimulation atomically writes RESUME and this fresh action.
        ResumeSimulation()
        return

    with action_write_lock:
        if reset_requested.is_set():
            raise RuntimeError("Cannot send a policy action while a reset is in progress")
        action_pipe.sendall(BuildActionMessage())

def Shutdown():
    global running, shutdown_started

    with shutdown_lock:
        if shutdown_started:
            return

        shutdown_started = True
        running = False
        reset_requested.set()

        try:
            with action_write_lock:
                if action_pipe is not None:
                    action_pipe.sendall(b"SHUTDOWN\n")
        except OSError:
            pass

        for connection in (obs_pipe, action_pipe, obs_listener, action_listener):
            try:
                if connection is not None:
                    connection.close()
            except OSError:
                pass

if len(sys.argv) < 2 or sys.argv[1] not in (
    "train", "critic-train", "reinforce-train", "inference", "validate"
):
    raise SystemExit(
        "Usage: python Server.py train [checkpoint-directory] | "
        "critic-train <checkpoint-directory> [shadow|active] | "
        "reinforce-train <checkpoint-directory> [shadow|active] | "
        "inference <checkpoint.zip> [world-seed] | "
        "validate [steps] [world-seed]"
    )

if (
    (sys.argv[1] == "train" and len(sys.argv) not in (2, 3))
    or (sys.argv[1] == "critic-train" and len(sys.argv) not in (3, 4))
    or (sys.argv[1] == "reinforce-train" and len(sys.argv) not in (3, 4))
    or (sys.argv[1] == "inference" and len(sys.argv) not in (3, 4))
    or (sys.argv[1] == "validate" and len(sys.argv) not in (2, 3, 4))
):
    raise SystemExit(
        "Usage: python Server.py train [checkpoint-directory] | "
        "critic-train <checkpoint-directory> [shadow|active] | "
        "reinforce-train <checkpoint-directory> [shadow|active] | "
        "inference <checkpoint.zip> [world-seed] | "
        "validate [steps] [world-seed]"
    )

inference_world_seed = None
validation_steps = 1_500
validation_world_seed = None
if sys.argv[1] == "inference" and len(sys.argv) == 4:
    inference_world_seed = CasualtiesEnv.NormalizeWorldSeed(int(sys.argv[3]))
elif sys.argv[1] == "validate":
    if len(sys.argv) >= 3:
        validation_steps = int(sys.argv[2])
    if len(sys.argv) == 4:
        validation_world_seed = CasualtiesEnv.NormalizeWorldSeed(
            int(sys.argv[3])
        )

critic_mode = "shadow"
if sys.argv[1] == "critic-train" and len(sys.argv) == 4:
    critic_mode = sys.argv[3]
    if critic_mode not in ("shadow", "active"):
        raise SystemExit("critic-train mode must be 'shadow' or 'active'")

reinforce_mode = "shadow"
if sys.argv[1] == "reinforce-train" and len(sys.argv) == 4:
    reinforce_mode = sys.argv[3]
    if reinforce_mode not in ("shadow", "active"):
        raise SystemExit("reinforce-train mode must be 'shadow' or 'active'")

print("Waiting for Unity observation connection...")
obs_pipe, _ = obs_listener.accept()

print("Waiting for Unity action connection...")
action_pipe, _ = action_listener.accept()

print("Unity connected to both TCP streams!")

move = 0
jump = 0
vertMove = 0
crouch = 0
lookdX = 0
lookdY = 0
attack = 0
interact = 0
targetSlotIndex = 0 # 0 = primaryhand, 1 = secondaryhand, 2 = mouth, 3 = upperback, 4 = middleback, 5 = lowerback, 6-24 are all wearable slots you can find them in Plugin.cs
selectedSlotIndex = 0 # same index rules as target but -1 = no slot
dropItem = 0
moveItem = 0
selectedBagIndex = -1 # -1 means the bag, anything above is anything inside
useItem = 0
useItemWorld = 0
selectedLimb = 0 # 0-14 are your 15 limbs in game
# here lies medamount... what a baby.
useItemMedical = 0
# tryMedProcedure also died
selectedRecipe = -1 # -1 none, 0-131 are recipe indices
favoriteItem = 0
switchMainHand = 0
trySleep = 0
ragdoll = 0
exercise = -1 # -1 is nothing, 0 = pushups, 1 = squats, 2 = plank
bark = 0
throw = 0
liquidAmount = 0 # mL to transfer, currently between 0 and 1000
drainLiquid = 0 # simple hold input to drain liquid
pullLiquidFromWorld = 0 # simple button to pull liquids from watercontaineritems in the world like minibarrels
action_sequence = 0  # CB1 monotonically identifies each policy decision.

# AUXILIARY VARIABLES
mode = "none"
chosenRecipe = 0

aux = {
    "Mode": mode,
    "SelectedSlot": selectedSlotIndex,
    "SelectedBagIndex": selectedBagIndex,
    "ChosenRecipe": chosenRecipe,
    "SelectedLimb": selectedLimb,
    "TargetSlot": targetSlotIndex,
    "LiquidAmount": liquidAmount
}

CasualtiesEnv.Start(sys.modules[__name__])

threading.Thread(target=ControlAckLoop, daemon=True).start()

OBSERVATION_DATA_SIZE = Types.OBSERVATION_DTYPE.itemsize
OBSERVATION_ID_SIZE = 8
OBSERVATION_MESSAGE_SIZE = OBSERVATION_DATA_SIZE + OBSERVATION_ID_SIZE
EXPECTED_OBSERVATION_DATA_SIZE = (
    1031536 if Types.INCLUDE_UNUSED_OBSERVATIONS
    else 42658 if Types.INCLUDE_LIMB_OBSERVATIONS
    else 41998
)
EXPECTED_OBSERVATION_MESSAGE_SIZE = EXPECTED_OBSERVATION_DATA_SIZE + OBSERVATION_ID_SIZE

if OBSERVATION_DATA_SIZE != EXPECTED_OBSERVATION_DATA_SIZE:
    raise RuntimeError(
        f"Observation dtype is {OBSERVATION_DATA_SIZE} bytes; "
        f"expected {EXPECTED_OBSERVATION_DATA_SIZE} bytes before metadata"
    )

if OBSERVATION_MESSAGE_SIZE != EXPECTED_OBSERVATION_MESSAGE_SIZE:
    raise RuntimeError(
        f"Observation message is {OBSERVATION_MESSAGE_SIZE} bytes; "
        f"expected {EXPECTED_OBSERVATION_MESSAGE_SIZE} bytes"
    )

env = CasualtiesEnv.Env()

if sys.argv[1] == "train":
    resume_dir = sys.argv[2] if len(sys.argv) == 3 else None
    threading.Thread(
        target=Train.Begin_Training,
        args=(env, PauseSimulation, resume_dir),
        daemon=True,
    ).start()
elif sys.argv[1] == "critic-train":
    threading.Thread(
        target=CriticTraining.Begin_Critic_Training,
        args=(env, PauseSimulation, sys.argv[2], critic_mode),
        daemon=True,
    ).start()
elif sys.argv[1] == "reinforce-train":
    threading.Thread(
        target=ReinforceTraining.Begin_Reinforce_Training,
        args=(env, PauseSimulation, sys.argv[2], reinforce_mode),
        daemon=True,
    ).start()
elif sys.argv[1] == "inference":
    threading.Thread(
        target=Inference.Infer,
        args=(env, sys.argv[2], inference_world_seed),
        daemon=True,
    ).start()
elif sys.argv[1] == "validate":
    threading.Thread(
        target=Validation.Run,
        args=(env, validation_steps, 0, validation_world_seed),
        daemon=True,
    ).start()

while running:
    try:
        data = RecvExact(obs_pipe, OBSERVATION_MESSAGE_SIZE)

        if len(data) != OBSERVATION_MESSAGE_SIZE:
            raise RuntimeError(
                f"Observation message was {len(data)} bytes; "
                f"expected {OBSERVATION_MESSAGE_SIZE}"
            )

        observation_id = int.from_bytes(
            data[OBSERVATION_DATA_SIZE:], byteorder="little", signed=False
        )
        obs = np.frombuffer(
            data[:OBSERVATION_DATA_SIZE], dtype=Types.OBSERVATION_DTYPE, count=1
        )[0]
        # If the event is already set, the previous observation has not yet
        # been consumed by Env.step(); this arrival will replace latest_obs.
        with observation_lock:
            env.latest_obs = obs
            env.latest_observation_id = observation_id
            env.obs_ready.set()

        # Update Auxiliary
        aux["Mode"] = mode
        aux["SelectedSlot"] = selectedSlotIndex
        aux["SelectedBagIndex"] = selectedBagIndex
        aux["ChosenRecipe"] = chosenRecipe
        aux["SelectedLimb"] = selectedLimb
        aux["TargetSlot"] = targetSlotIndex
        aux["LiquidAmount"] = liquidAmount

        # Visualization.Update(obs, aux)

    except Exception as e:
        print(e)
        running = False
        break
