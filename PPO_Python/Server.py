import socket
import os
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
    return f"{move},{jump},{vertMove},{crouch},{lookdX},{lookdY},{attack},{interact},{targetSlotIndex},{selectedSlotIndex},{dropItem},{moveItem},{selectedBagIndex},{useItem},{useItemWorld},{selectedLimb},{useItemMedical},{selectedRecipe},{favoriteItem},{switchMainHand},{trySleep},{ragdoll},{exercise},{bark},{throw},{liquidAmount},{drainLiquid},{pullLiquidFromWorld}\n".encode("utf-8")


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
                    action_pipe.sendall(msg)
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

threading.Thread(target=ActionLoop, daemon=True).start()
threading.Thread(target=ControlAckLoop, daemon=True).start()

# Fixed binary observation protocol.
MAX_NEARBY_BUILDINGS = 16
MAX_NEARBY_ITEMS = 16 # 16
SIGHT_RANGE_X = 42
SIGHT_RANGE_Y = 24
MAX_BAG_ITEMS = 32; # 32
MAX_INGREDIENTS = 16
MAX_LIQUID_COMPONENTS = 16 # 16
MAX_QUALITIES = 8 # 8
MAX_SOUNDS_HEARD = 16 # 16

# Packed wire dtypes. Keep align=False: the C# writer emits fields back-to-back
# with no native padding.
BOOL_DTYPE = np.dtype("u1")
BYTE_DTYPE = np.dtype("u1")
SBYTE_DTYPE = np.dtype("i1")
SHORT_DTYPE = np.dtype("<i2")
USHORT_DTYPE = np.dtype("<u2")
INT_DTYPE = np.dtype("<i4")
FLOAT_DTYPE = np.dtype("<f4")

VECTOR2_INT_DTYPE = np.dtype([
    ("X", SHORT_DTYPE),
    ("Y", SHORT_DTYPE),
], align=False)

VECTOR2_DTYPE = np.dtype([
    ("X", FLOAT_DTYPE),
    ("Y", FLOAT_DTYPE),
], align=False)

QUALITY_DTYPE = np.dtype([
    ("ID", SBYTE_DTYPE),
    ("Amount", USHORT_DTYPE),
], align=False)

LIQUID_QUALITY_DTYPE = np.dtype([
    ("ID", SBYTE_DTYPE),
    ("Amount", INT_DTYPE),
], align=False)

LIQUID_DTYPE = np.dtype([
    ("ID", SBYTE_DTYPE),
    ("Amount", USHORT_DTYPE),
    ("Qualities", LIQUID_QUALITY_DTYPE, (MAX_QUALITIES,)),
], align=False)

# Contents are serialized exactly one level deep. Child items do not contain
# another Contents payload, while root/world/inventory items do.
CHILD_ITEM_DTYPE = np.dtype([
    ("ID", SHORT_DTYPE),
    ("Condition", FLOAT_DTYPE),
    ("Liquids", LIQUID_DTYPE, (MAX_LIQUID_COMPONENTS,)),
    ("Qualities", QUALITY_DTYPE, (MAX_QUALITIES,)),
], align=False)

ITEM_DTYPE = np.dtype([
    ("ID", SHORT_DTYPE),
    ("Condition", FLOAT_DTYPE),
    ("Contents", CHILD_ITEM_DTYPE, (MAX_BAG_ITEMS,)),
    ("Liquids", LIQUID_DTYPE, (MAX_LIQUID_COMPONENTS,)),
    ("Qualities", QUALITY_DTYPE, (MAX_QUALITIES,)),
], align=False)

BLOCK_DTYPE = np.dtype([
    ("Health", FLOAT_DTYPE),
    ("Toxicity", FLOAT_DTYPE),
    ("SleepQuality", SBYTE_DTYPE),
    ("ItemPool", SHORT_DTYPE, (3,)),
], align=False)

FLUID_TILE_DTYPE = np.dtype([
    ("Type", BYTE_DTYPE),
], align=False)

BUILDING_DTYPE = np.dtype([
    ("Exists", BOOL_DTYPE),
    ("RelativePosition", VECTOR2_INT_DTYPE),
    ("Health", FLOAT_DTYPE),
    ("DropPool", SHORT_DTYPE, (5,)),
], align=False)

ITEM_REQUIREMENT_DTYPE = np.dtype({
    "names": ["ID", "MinimumCondition", "Condition"],
    "formats": [SHORT_DTYPE, FLOAT_DTYPE, FLOAT_DTYPE],
    "offsets": [0, 2, 2],
    "itemsize": 6,
})

RECIPE_RESULT_DTYPE = np.dtype([
    ("ID", SHORT_DTYPE),
    ("Condition", FLOAT_DTYPE),
    ("Amount", BYTE_DTYPE),
], align=False)

WORLD_ITEM_DTYPE = np.dtype([
    ("RelativePosition", VECTOR2_INT_DTYPE),
    ("Item", ITEM_DTYPE),
], align=False)

RECIPE_DTYPE = np.dtype({
    "names": [
        "IsCraftable",
        "ItemRequirements",
        "QualityRequirements",
        "Output",
        # Compatibility alias for Visualization.py's previous schema.
        "OutputQuantity",
    ],
    "formats": [
        BOOL_DTYPE,
        (ITEM_REQUIREMENT_DTYPE, (MAX_INGREDIENTS,)),
        (QUALITY_DTYPE, (MAX_QUALITIES,)),
        RECIPE_RESULT_DTYPE,
        BYTE_DTYPE,
    ],
    "offsets": [0, 1, 97, 121, 127],
    "itemsize": 128,
})

LIMB_DTYPE = np.dtype([
    ("SkinHealth", FLOAT_DTYPE),
    ("MuscleHealth", FLOAT_DTYPE),
    ("Pain", FLOAT_DTYPE),
    ("InfectionAmount", FLOAT_DTYPE),
    ("DisinfectionTime", FLOAT_DTYPE),
    ("Dislocated", BOOL_DTYPE),
    ("Broken", BOOL_DTYPE),
    ("Splinted", BOOL_DTYPE),
    ("Infected", BOOL_DTYPE),
    ("DislocationTimer", FLOAT_DTYPE),
    ("BoneHealTimer", FLOAT_DTYPE),
    ("IsVital", BOOL_DTYPE),
    ("IsHead", BOOL_DTYPE),
    ("IsAbdomen", BOOL_DTYPE),
    ("IsLegLimb", BOOL_DTYPE),
    ("IsArm", BOOL_DTYPE),
    ("DistanceToHeart", BYTE_DTYPE),
    ("Shrapnel", BYTE_DTYPE),
    ("Dismembered", BOOL_DTYPE),
    ("TotalBleedAmount", FLOAT_DTYPE),
], align=False)

SOUND_DTYPE = np.dtype([
    ("ID", SHORT_DTYPE),
    ("RelativeTilePosition", VECTOR2_INT_DTYPE),
    ("Volume", FLOAT_DTYPE),
], align=False)

OBSERVATION_DTYPE = np.dtype([
    ("RelativeBlockMap", BLOCK_DTYPE, (SIGHT_RANGE_X * 2 + 1, SIGHT_RANGE_Y * 2 + 1)),
    ("VisibleBuildings", BUILDING_DTYPE, (MAX_NEARBY_BUILDINGS,)),
    ("VisibleItems", WORLD_ITEM_DTYPE, (MAX_NEARBY_ITEMS,)),
    ("RelativeFluidMap", FLUID_TILE_DTYPE, (SIGHT_RANGE_X * 2 + 1, SIGHT_RANGE_Y * 2 + 1)),
    ("Velocity", VECTOR2_DTYPE),
    ("IsRight", BOOL_DTYPE),
    ("MaxSpeed", FLOAT_DTYPE),
    ("RelativeLookPos", VECTOR2_INT_DTYPE),
    ("JumpCooldown", FLOAT_DTYPE),
    ("Grounded", BOOL_DTYPE),
    ("TimeSinceGrounded", FLOAT_DTYPE),
    ("StandingOn", BLOCK_DTYPE),
    ("TimeRagdolled", FLOAT_DTYPE),
    ("CrawlTime", FLOAT_DTYPE),
    ("InWater", BOOL_DTYPE),
    ("LiquidSlipTime", FLOAT_DTYPE),
    ("LiquidRagdollBar", FLOAT_DTYPE),
    ("LiquidDrinkTime", FLOAT_DTYPE),
    ("CanWalljumpLeft", BOOL_DTYPE),
    ("CanWalljumpRight", BOOL_DTYPE),
    ("AttackCooldown", FLOAT_DTYPE),
    ("CrouchAmount", FLOAT_DTYPE),
    ("Crouching", BOOL_DTYPE),
    ("IsClimbing", BOOL_DTYPE),
    ("ClimbableProgress", FLOAT_DTYPE),
    ("ClimbVelocity", FLOAT_DTYPE),
    ("HeartRate", FLOAT_DTYPE),
    ("FibrillationProgress", FLOAT_DTYPE),
    ("FibrillationForced", BOOL_DTYPE),
    ("FibrillationRising", BOOL_DTYPE),
    ("HasPulmonaryEmbolism", BOOL_DTYPE),
    ("BloodOxygen", FLOAT_DTYPE),
    ("BloodVolume", FLOAT_DTYPE),
    ("BloodPressure", FLOAT_DTYPE),
    ("BloodVesselSize", FLOAT_DTYPE),
    ("BloodViscosity", FLOAT_DTYPE),
    ("TotalBleedSpeed", FLOAT_DTYPE),
    ("InternalBleeding", FLOAT_DTYPE),
    ("Hemothorax", FLOAT_DTYPE),
    ("VenomTotal", FLOAT_DTYPE),
    ("VenomCurrent", FLOAT_DTYPE),
    ("RespiratoryRate", FLOAT_DTYPE),
    ("Breathing", BOOL_DTYPE),
    ("Adrenaline", FLOAT_DTYPE),
    ("CurAdrenaline", FLOAT_DTYPE),
    ("StimulantMultiplier", FLOAT_DTYPE),
    ("OnHardStimulants", BOOL_DTYPE),
    ("OpiateHappiness", FLOAT_DTYPE),
    ("AntidepressantHappiness", FLOAT_DTYPE),
    ("BrainGrowSickness", FLOAT_DTYPE),
    ("UsedNeuralBooster", BOOL_DTYPE),
    ("MindWiped", BOOL_DTYPE),
    ("Caffeinated", FLOAT_DTYPE),
    ("OverdoseIndex", BYTE_DTYPE),
    ("WeightOffset", FLOAT_DTYPE),
    ("Hunger", FLOAT_DTYPE),
    ("Thirst", FLOAT_DTYPE),
    ("Stamina", FLOAT_DTYPE),
    ("Energy", FLOAT_DTYPE),
    ("Immunity", FLOAT_DTYPE),
    ("TotalHappiness", FLOAT_DTYPE),
    ("Dirtyness", FLOAT_DTYPE),
    ("ClawHealth", FLOAT_DTYPE),
    ("BrainHealth", FLOAT_DTYPE),
    ("Consciousness", FLOAT_DTYPE),
    ("Shock", FLOAT_DTYPE),
    ("ReversedControls", BOOL_DTYPE),
    ("BrainDying", BOOL_DTYPE),
    ("PlayerDead", BOOL_DTYPE),
    ("StrokeAmount", FLOAT_DTYPE),
    ("Temperature", FLOAT_DTYPE),
    ("ClothingTemperature", FLOAT_DTYPE),
    ("AveragePain", FLOAT_DTYPE),
    ("PainShock", FLOAT_DTYPE),
    ("HearingLoss", FLOAT_DTYPE),
    ("BothHandsUnusable", BOOL_DTYPE),
    ("SicknessAmount", FLOAT_DTYPE),
    ("SepticShock", FLOAT_DTYPE),
    ("RadiationSickness", FLOAT_DTYPE),
    ("CorpsesSeen", USHORT_DTYPE),
    ("TraumaAmount", FLOAT_DTYPE),
    ("HorrifiedLevel", FLOAT_DTYPE),
    ("FocusedLevel", FLOAT_DTYPE),
    ("Disfigured", BOOL_DTYPE),
    ("EyeGone", BOOL_DTYPE),
    ("BothEyesGone", BOOL_DTYPE),
    ("TotalEncumberance", FLOAT_DTYPE),
    ("OverEncumberance", FLOAT_DTYPE),
    ("MaxEncumberance", FLOAT_DTYPE),
    ("Sleeping", BOOL_DTYPE),
    ("CurSleep", SBYTE_DTYPE),
    ("BadSleepAmount", FLOAT_DTYPE),
    ("GoodSleepTime", FLOAT_DTYPE),
    ("ForcedSleepQuality", SBYTE_DTYPE),
    ("UsingSleepingBag", BOOL_DTYPE),
    ("CanTakeNap", BOOL_DTYPE),
    ("TriedRollingLastStand", BOOL_DTYPE),
    ("LastStandTime", FLOAT_DTYPE),
    ("STR", BYTE_DTYPE),
    ("RES", BYTE_DTYPE),
    ("INT", BYTE_DTYPE),
    ("STRProgress", FLOAT_DTYPE),
    ("RESProgress", FLOAT_DTYPE),
    ("INTProgress", FLOAT_DTYPE),
    ("Inventory", ITEM_DTYPE, (25,)),
    ("Recipes", RECIPE_DTYPE, (132,)),
    ("Limbs", LIMB_DTYPE, (15,)),
    ("LayerProgress", FLOAT_DTYPE),
    ("CurrentLayer", BYTE_DTYPE),
    ("BestLayerDepth", SHORT_DTYPE),
    ("LayerTimeRemaining", INT_DTYPE),
    ("RadLineDisplacement", SHORT_DTYPE),
    ("SoundsHeard", SOUND_DTYPE, (MAX_SOUNDS_HEARD,)),
], align=False)

OBSERVATION_DATA_SIZE = OBSERVATION_DTYPE.itemsize
OBSERVATION_ID_SIZE = 8
OBSERVATION_MESSAGE_SIZE = OBSERVATION_DATA_SIZE + OBSERVATION_ID_SIZE
EXPECTED_OBSERVATION_DATA_SIZE = 1056511
EXPECTED_OBSERVATION_MESSAGE_SIZE = 1056519

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

        observation_id = int.from_bytes(
            data[OBSERVATION_DATA_SIZE:], byteorder="little", signed=False
        )
        obs = np.frombuffer(
            data[:OBSERVATION_DATA_SIZE], dtype=OBSERVATION_DTYPE, count=1
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
