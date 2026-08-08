"""Semantic, finite observation encoding for the CB1 wire protocol."""

import numpy as np

import Types


# SearchBase/WorldGeneration.cs defines ordinary block health in [1, 15,000]
# and infinirock (block 14) at 420,133,760. Keep the ordinary range detailed
# and represent the separate extreme regime with its own spatial channel.
BLOCK_HEALTH_NORMAL_MAX = 15_000.0
BLOCK_HEALTH_WIRE_MAX = 420_133_760.0
BLOCK_HEALTH_EXTREME_THRESHOLD = BLOCK_HEALTH_NORMAL_MAX
BLOCK_HEALTH_LOG_MAX = np.log1p(BLOCK_HEALTH_NORMAL_MAX)
FLUID_TYPE_COUNT = 7
SLEEP_QUALITY_COUNT = 4
CURRENT_LAYER_COUNT = 10


GENERAL_FIELD_NAMES = (
    "Velocity", "IsRight", "MaxSpeed", "RelativeLookPos", "JumpCooldown",
    "Grounded", "TimeSinceGrounded", "StandingOn", "TimeRagdolled",
    "CrawlTime", "InWater", "LiquidSlipTime", "LiquidRagdollBar",
    "LiquidDrinkTime", "CanWalljumpLeft", "CanWalljumpRight",
    "AttackCooldown", "CrouchAmount", "Crouching", "IsClimbing",
    "ClimbableProgress", "ClimbVelocity", "HeartRate",
    "FibrillationProgress", "FibrillationForced", "FibrillationRising",
    "HasPulmonaryEmbolism", "BloodOxygen", "BloodVolume", "BloodPressure",
    "BloodVesselSize", "BloodViscosity", "TotalBleedSpeed",
    "InternalBleeding", "Hemothorax", "VenomTotal", "VenomCurrent",
    "RespiratoryRate", "Breathing", "Adrenaline", "CurAdrenaline",
    "StimulantMultiplier", "OnHardStimulants", "OpiateHappiness",
    "AntidepressantHappiness", "BrainGrowSickness", "UsedNeuralBooster",
    "MindWiped", "Caffeinated", "OverdoseIndex", "WeightOffset", "Hunger",
    "Thirst", "Stamina", "Energy", "Immunity", "TotalHappiness",
    "Dirtyness", "ClawHealth", "BrainHealth", "Consciousness", "Shock",
    "ReversedControls", "BrainDying", "PlayerDead", "StrokeAmount",
    "Temperature", "ClothingTemperature", "AveragePain", "PainShock",
    "HearingLoss", "BothHandsUnusable", "SicknessAmount", "SepticShock",
    "RadiationSickness", "CorpsesSeen", "TraumaAmount", "HorrifiedLevel",
    "FocusedLevel", "Disfigured", "EyeGone", "BothEyesGone",
    "TotalEncumberance", "OverEncumberance", "MaxEncumberance", "Sleeping",
    "CurSleep", "BadSleepAmount", "GoodSleepTime", "ForcedSleepQuality",
    "UsingSleepingBag", "CanTakeNap", "TriedRollingLastStand",
    "LastStandTime", "STR", "RES", "INT", "STRProgress", "RESProgress",
    "INTProgress", "LayerProgress", "CurrentLayer", "BestLayerDepth",
    "LayerTimeRemaining", "RadLineDisplacement", "PreviousAction",
    "PlayerTilePosition", "WorldDimensions",
)


BOOL_FIELDS = {
    "IsRight", "Grounded", "InWater", "CanWalljumpLeft",
    "CanWalljumpRight", "Crouching", "IsClimbing", "FibrillationForced",
    "FibrillationRising", "HasPulmonaryEmbolism", "Breathing",
    "OnHardStimulants", "UsedNeuralBooster", "MindWiped", "ReversedControls",
    "BrainDying", "PlayerDead", "BothHandsUnusable", "Disfigured", "EyeGone",
    "BothEyesGone", "Sleeping", "UsingSleepingBag", "CanTakeNap",
    "TriedRollingLastStand",
}

# Values with a meaningful zero and symmetric positive/negative extent.
SIGNED_SCALES = {
    "MaxSpeed": 20.0,
    "ClimbVelocity": 20.0,
    "BloodViscosity": 100.0,
    "TotalBleedSpeed": 10.0,
    "StimulantMultiplier": 2.0,
    "OpiateHappiness": 100.0,
    "AntidepressantHappiness": 100.0,
    "Caffeinated": 300.0,
    "WeightOffset": 100.0,
    "TotalHappiness": 100.0,
    "ClothingTemperature": 20.0,
    "CorpsesSeen": 100.0,
    "HorrifiedLevel": 100.0,
    "FocusedLevel": 100.0,
    "TotalEncumberance": 25.0,
    "MaxEncumberance": 25.0,
    "LastStandTime": 300.0,
}

# Known bounded or practically bounded values encoded to [0, 1].
UNIT_RANGES = {
    "JumpCooldown": (0.0, 2.0),
    "TimeSinceGrounded": (0.0, 5.0),
    "TimeRagdolled": (0.0, 60.0),
    "CrawlTime": (0.0, 2.0),
    "LiquidSlipTime": (0.0, 2.0),
    "LiquidRagdollBar": (0.0, 1.0),
    "LiquidDrinkTime": (0.0, 2.0),
    "AttackCooldown": (0.0, 2.0),
    "CrouchAmount": (0.0, 1.0),
    "ClimbableProgress": (0.0, 250.0),
    "HeartRate": (0.0, 320.0),
    "FibrillationProgress": (0.0, 100.0),
    "BloodOxygen": (0.0, 100.0),
    "BloodVolume": (-100.0, 200.0),
    "BloodPressure": (0.0, 250.0),
    "BloodVesselSize": (0.85, 1.15),
    "InternalBleeding": (0.0, 100.0),
    "Hemothorax": (0.0, 100.0),
    "VenomTotal": (0.0, 100.0),
    "VenomCurrent": (0.0, 100.0),
    "RespiratoryRate": (0.0, 100.0),
    "Adrenaline": (0.0, 100.0),
    "CurAdrenaline": (0.0, 100.0),
    "BrainGrowSickness": (0.0, 300.0),
    "OverdoseIndex": (0.0, 10.0),
    "Hunger": (-50.0, 125.0),
    "Thirst": (-50.0, 250.0),
    "Stamina": (0.0, 100.0),
    "Energy": (0.0, 100.0),
    "Immunity": (0.0, 200.0),
    "Dirtyness": (0.0, 100.0),
    "ClawHealth": (0.0, 100.0),
    "BrainHealth": (0.0, 100.0),
    "Consciousness": (0.0, 100.0),
    "Shock": (0.0, 100.0),
    "StrokeAmount": (0.0, 100.0),
    "AveragePain": (0.0, 100.0),
    "PainShock": (0.0, 1.0),
    "HearingLoss": (0.0, 100.0),
    "SicknessAmount": (0.0, 100.0),
    "SepticShock": (0.0, 100.0),
    "RadiationSickness": (0.0, 100.0),
    "TraumaAmount": (0.0, 100.0),
    "OverEncumberance": (0.0, 1.0),
    "BadSleepAmount": (0.0, 300.0),
    "GoodSleepTime": (0.0, 600.0),
    "STR": (0.0, 20.0),
    "RES": (0.0, 20.0),
    "INT": (0.0, 20.0),
    "STRProgress": (0.0, 1.0),
    "RESProgress": (0.0, 1.0),
    "INTProgress": (0.0, 1.0),
    "LayerProgress": (0.0, 1.0),
}

SPECIAL_FIELDS = {
    "Velocity", "RelativeLookPos", "StandingOn", "Temperature", "CurSleep",
    "ForcedSleepQuality", "CurrentLayer", "BestLayerDepth",
    "LayerTimeRemaining", "RadLineDisplacement", "PreviousAction",
    "PlayerTilePosition", "WorldDimensions",
}

_covered_fields = BOOL_FIELDS | set(SIGNED_SCALES) | set(UNIT_RANGES) | SPECIAL_FIELDS
if _covered_fields != set(GENERAL_FIELD_NAMES):
    missing = sorted(set(GENERAL_FIELD_NAMES) - _covered_fields)
    extra = sorted(_covered_fields - set(GENERAL_FIELD_NAMES))
    raise RuntimeError(f"Normalization schema mismatch; missing={missing}, extra={extra}")


def _finite(value):
    return np.nan_to_num(
        np.asarray(value, dtype=np.float32),
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )


def _signed(value, scale):
    return np.clip(_finite(value) / float(scale), -1.0, 1.0)


def _unit(value, low, high):
    return np.clip((_finite(value) - low) / (high - low), 0.0, 1.0)


def _one_hot(value, count, nullable=False):
    result = np.zeros(count + int(nullable), dtype=np.float32)
    index = int(np.asarray(value).item())
    if nullable:
        index = 0 if index < 0 else min(index, count - 1) + 1
    else:
        index = min(max(index, 0), count - 1)
    result[index] = 1.0
    return result


def _block_health(health):
    value = np.clip(_finite(health), 0.0, BLOCK_HEALTH_WIRE_MAX)
    value = np.minimum(value, BLOCK_HEALTH_NORMAL_MAX)
    return np.log1p(value) / BLOCK_HEALTH_LOG_MAX


def _block_health_extreme(health):
    value = np.clip(_finite(health), 0.0, BLOCK_HEALTH_WIRE_MAX)
    return (value > BLOCK_HEALTH_EXTREME_THRESHOLD).astype(np.float32)


def encode_general(obs):
    """Encode all general scalars by semantics into finite [-1, 1] values."""
    values = []
    world_width = max(float(obs["WorldDimensions"]["X"]), 1.0)
    world_height = max(float(obs["WorldDimensions"]["Y"]), 1.0)

    for field_name in GENERAL_FIELD_NAMES:
        value = obs[field_name]
        if field_name in BOOL_FIELDS:
            encoded = np.asarray([float(bool(value))], dtype=np.float32)
        elif field_name in SIGNED_SCALES:
            encoded = np.atleast_1d(_signed(value, SIGNED_SCALES[field_name]))
        elif field_name in UNIT_RANGES:
            encoded = np.atleast_1d(_unit(value, *UNIT_RANGES[field_name]))
        elif field_name == "Velocity":
            encoded = np.asarray([
                _signed(value["X"], 20.0),
                _signed(value["Y"], 30.0),
            ], dtype=np.float32)
        elif field_name == "RelativeLookPos":
            encoded = np.asarray([
                _signed(value["X"], Types.SIGHT_RANGE_X),
                _signed(value["Y"], Types.SIGHT_RANGE_Y),
            ], dtype=np.float32)
        elif field_name == "StandingOn":
            encoded = np.concatenate([
                np.atleast_1d(_block_health(value["Health"])),
                np.atleast_1d(_unit(value["Toxicity"], 0.0, 2.5)),
                _one_hot(value["SleepQuality"], SLEEP_QUALITY_COUNT),
            ])
        elif field_name == "Temperature":
            encoded = np.atleast_1d(_signed(_finite(value) - 37.0, 10.0))
        elif field_name == "CurSleep":
            encoded = _one_hot(value, SLEEP_QUALITY_COUNT)
        elif field_name == "ForcedSleepQuality":
            encoded = _one_hot(value, SLEEP_QUALITY_COUNT, nullable=True)
        elif field_name == "CurrentLayer":
            encoded = _one_hot(value, CURRENT_LAYER_COUNT)
        elif field_name == "BestLayerDepth":
            layer_height_meters = max((world_height - 3.1) * 0.3, 1.0)
            encoded = np.atleast_1d(_unit(value, 0.0, layer_height_meters))
        elif field_name == "LayerTimeRemaining":
            encoded = np.atleast_1d(_unit(value, 0.0, 1_000_000_000.0))
        elif field_name == "RadLineDisplacement":
            encoded = np.atleast_1d(_signed(value, world_height))
        elif field_name == "PreviousAction":
            encoded = np.asarray([
                _signed(value["MoveDirection"], 1.0),
                _unit(value["Jump"], 0.0, 1.0),
                _signed(value["VerticalMovement"], 1.0),
                _unit(value["Crouch"], 0.0, 1.0),
                _signed(value["LookDX"], 4.0),
                _signed(value["LookDY"], 5.0),
                _unit(value["Attack"], 0.0, 1.0),
            ], dtype=np.float32)
        elif field_name == "PlayerTilePosition":
            encoded = np.asarray([
                np.clip(2.0 * float(value["X"]) / world_width - 1.0, -1.0, 1.0),
                np.clip(2.0 * float(value["Y"]) / world_height - 1.0, -1.0, 1.0),
            ], dtype=np.float32)
        elif field_name == "WorldDimensions":
            # World size is useful context for the normalized absolute position.
            encoded = np.asarray([
                _unit(value["X"], 1.0, 2048.0),
                _unit(value["Y"], 1.0, 2048.0),
            ], dtype=np.float32)
        else:  # guarded by the schema assertion above
            raise KeyError(f"No normalization for {field_name}")

        values.append(np.asarray(encoded, dtype=np.float32).reshape(-1))

    result = np.concatenate(values).astype(np.float32, copy=False)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(result, -1.0, 1.0)


def encode_spatial(obs):
    """Jointly encode terrain, fluid categories, player center, and cursor."""
    blocks = obs["RelativeBlockMap"]
    fluids = np.asarray(obs["RelativeFluidMap"]["Type"], dtype=np.int64)

    health = _block_health(blocks["Health"])
    health_extreme = _block_health_extreme(blocks["Health"])
    toxicity = _unit(blocks["Toxicity"], 0.0, 2.5)
    sleep = np.asarray(blocks["SleepQuality"], dtype=np.int64)
    sleep = np.clip(sleep, 0, SLEEP_QUALITY_COUNT - 1)
    sleep_channels = np.eye(SLEEP_QUALITY_COUNT, dtype=np.float32)[sleep]

    fluids = np.clip(fluids, 0, FLUID_TYPE_COUNT - 1)
    fluid_channels = np.eye(FLUID_TYPE_COUNT, dtype=np.float32)[fluids]

    width = Types.SIGHT_RANGE_X * 2 + 1
    height = Types.SIGHT_RANGE_Y * 2 + 1
    center = np.zeros((width, height), dtype=np.float32)
    cursor = np.zeros_like(center)
    center[Types.SIGHT_RANGE_X, Types.SIGHT_RANGE_Y] = 1.0

    look_x = int(obs["RelativeLookPos"]["X"])
    look_y = int(obs["RelativeLookPos"]["Y"])
    cursor_x = Types.SIGHT_RANGE_X + look_x
    cursor_y = Types.SIGHT_RANGE_Y - look_y
    if 0 <= cursor_x < width and 0 <= cursor_y < height:
        cursor[cursor_x, cursor_y] = 1.0

    channels = [
        health,
        health_extreme,
        toxicity,
        *np.moveaxis(sleep_channels, -1, 0),
        *np.moveaxis(fluid_channels, -1, 0),
        center,
        cursor,
    ]
    result = np.stack(channels, axis=0).astype(np.float32, copy=False)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0))


SPATIAL_CHANNELS = 3 + SLEEP_QUALITY_COUNT + FLUID_TYPE_COUNT + 2
