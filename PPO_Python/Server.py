import socket
import os
import csv
import Human_Input
import Visualization
import json
import threading
import time
import sys
import numpy as np
import CasualtiesEnv
import gymnasium as gym
import Train
import Inference
import Types
from pathlib import Path
from datetime import datetime

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

PIPELINE_PROFILE_ENABLED = os.environ.get("PPO_PIPELINE_PROFILE", "").lower() in (
    "1", "true", "yes", "on"
)
PIPELINE_PROFILE_DIR = Path(
    os.environ.get("PPO_PIPELINE_PROFILE_DIR", "profiles")
)
pipeline_profile_lock = threading.Lock()
pipeline_profile_file = None
pipeline_profile_writer = None
pipeline_profile_rows = 0


def ProfileEvent(phase, duration_ms=0.0, size_bytes=0, observation_id=0, interval_ticks=0):
    global pipeline_profile_rows
    if not PIPELINE_PROFILE_ENABLED or pipeline_profile_writer is None:
        return
    with pipeline_profile_lock:
        pipeline_profile_writer.writerow((
            time.time_ns(),
            time.perf_counter_ns(),
            threading.current_thread().name,
            phase,
            f"{duration_ms:.6f}",
            size_bytes,
            observation_id,
            interval_ticks,
        ))
        pipeline_profile_rows += 1
        if pipeline_profile_rows >= 100:
            pipeline_profile_file.flush()
            pipeline_profile_rows = 0


if PIPELINE_PROFILE_ENABLED:
    PIPELINE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PIPELINE_PROFILE_DIR / (
        f"pipeline_python_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    pipeline_profile_file = profile_path.open("w", newline="", encoding="utf-8")
    pipeline_profile_writer = csv.writer(pipeline_profile_file)
    pipeline_profile_writer.writerow((
        "wall_time_ns", "monotonic_ns", "thread", "phase", "duration_ms",
        "size_bytes", "observation_id", "interval_ticks",
    ))
    pipeline_profile_file.flush()
    print(f"Pipeline Python profile: {profile_path}")

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
    started = time.perf_counter_ns()
    chunks = []
    remaining = size

    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Unity observation connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)

    result = b"".join(chunks)
    ProfileEvent(
        "observation_socket_recv",
        (time.perf_counter_ns() - started) / 1_000_000,
        len(result),
    )
    return result


def BuildActionMessage():
    started = time.perf_counter_ns()
    message = f"{move},{jump},{vertMove},{crouch},{lookdX},{lookdY},{attack},{interact},{targetSlotIndex},{selectedSlotIndex},{dropItem},{moveItem},{selectedBagIndex},{useItem},{useItemWorld},{selectedLimb},{useItemMedical},{selectedRecipe},{favoriteItem},{switchMainHand},{trySleep},{ragdoll},{exercise},{bark},{throw},{liquidAmount},{drainLiquid},{pullLiquidFromWorld}\n".encode("utf-8")
    ProfileEvent("action_build", (time.perf_counter_ns() - started) / 1_000_000, len(message))
    return message


def ConfigureTimeScaleMultiplier(multiplier):
    started = time.perf_counter_ns()
    commands = []
    if PIPELINE_PROFILE_ENABLED:
        commands.append("PPO_PROFILE 1\n")
    commands.append(f"PPO_SCALE {multiplier:g}\n")
    message = "".join(commands).encode("ascii")
    with action_write_lock:
        action_pipe.sendall(message)
    ProfileEvent("scale_send", (time.perf_counter_ns() - started) / 1_000_000, len(message))


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
        except OSError:
            return


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
        action_pipe.sendall(b"RESUME\n")
        action_pipe.sendall(BuildActionMessage())
    simulation_paused.clear()

    if not resume_applied.wait(timeout=5.0):
        raise RuntimeError("Unity did not acknowledge the PPO resume")

def ActionLoop():
    global move, jump, vertMove, crouch, running
    running = True
    while running:
        try:
            if reset_requested.is_set() or simulation_paused.is_set():
                time.sleep(0.005)
                continue

            msg = BuildActionMessage()

            with action_write_lock:
                if not reset_requested.is_set():
                    send_started = time.perf_counter_ns()
                    action_pipe.sendall(msg)
                    ProfileEvent("action_socket_send", (time.perf_counter_ns() - send_started) / 1_000_000, len(msg))
            time.sleep(0.005)
        except OSError as e:
            print(f"Action TCP connection closed: {e}")
            running = False
            break

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

if (
    len(sys.argv) < 2
    or len(sys.argv) > 3
    or sys.argv[1] not in ("train", "inference")
    or (sys.argv[1] == "inference" and len(sys.argv) != 3)
):
    raise SystemExit(
        "Usage: python Server.py train [checkpoint-directory] | "
        "inference <checkpoint.zip>"
    )

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

# Human_Input.Start(sys.modules[__name__])

CasualtiesEnv.Start(sys.modules[__name__])

ConfigureTimeScaleMultiplier(4.0 if sys.argv[1] == "train" else 1.0)

threading.Thread(target=ActionLoop, daemon=True).start()
threading.Thread(target=ControlAckLoop, daemon=True).start()

OBSERVATION_DATA_SIZE = Types.OBSERVATION_DTYPE.itemsize
OBSERVATION_ID_SIZE = 8
OBSERVATION_INTERVAL_SIZE = 2
OBSERVATION_METADATA_SIZE = OBSERVATION_ID_SIZE + OBSERVATION_INTERVAL_SIZE
OBSERVATION_MESSAGE_SIZE = OBSERVATION_DATA_SIZE + OBSERVATION_METADATA_SIZE
EXPECTED_OBSERVATION_DATA_SIZE = (
    1031515 if Types.INCLUDE_UNUSED_OBSERVATIONS else 42637
)
EXPECTED_OBSERVATION_MESSAGE_SIZE = EXPECTED_OBSERVATION_DATA_SIZE + OBSERVATION_METADATA_SIZE

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
elif sys.argv[1] == "inference":
    if len(sys.argv) == 3:
        threading.Thread(target=Inference.Infer, args=(env, sys.argv[2]), daemon=True).start()
    else:
        raise SystemExit("Usage: python Server.py inference <checkpoint.zip>")

while running:
    try:
        data = RecvExact(obs_pipe, OBSERVATION_MESSAGE_SIZE)

        if len(data) != OBSERVATION_MESSAGE_SIZE:
            raise RuntimeError(
                f"Observation message was {len(data)} bytes; "
                f"expected {OBSERVATION_MESSAGE_SIZE}"
            )

        parse_started = time.perf_counter_ns()
        observation_id_start = OBSERVATION_DATA_SIZE
        observation_id_end = observation_id_start + OBSERVATION_ID_SIZE
        observation_id = int.from_bytes(
            data[observation_id_start:observation_id_end],
            byteorder="little",
            signed=False,
        )
        decision_interval_ticks = int.from_bytes(
            data[observation_id_end:observation_id_end + OBSERVATION_INTERVAL_SIZE],
            byteorder="little",
            signed=False,
        )
        obs = np.frombuffer(
            data[:OBSERVATION_DATA_SIZE], dtype=Types.OBSERVATION_DTYPE, count=1
        )[0]
        ProfileEvent(
            "observation_parse",
            (time.perf_counter_ns() - parse_started) / 1_000_000,
            len(data),
            observation_id,
            decision_interval_ticks,
        )
        # If the event is already set, the previous observation has not yet
        # been consumed by Env.step(); this arrival will replace latest_obs.
        with observation_lock:
            env.latest_obs = obs
            env.latest_observation_id = observation_id
            env.latest_decision_interval_ticks = decision_interval_ticks
            env.obs_ready.set()
        ProfileEvent(
            "observation_publish",
            observation_id=observation_id,
            interval_ticks=decision_interval_ticks,
        )

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
