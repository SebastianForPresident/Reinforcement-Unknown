import numpy as np

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

# TEMPORARY PROTOCOL EXPERIMENT: keep the unused observation definitions
# below, but omit them from the live wire dtype while the current policy does
# not consume them. This must match PPOBridge.IncludeUnusedObservations.
INCLUDE_UNUSED_OBSERVATIONS = False

OBSERVATION_DTYPE = np.dtype([
    ("RelativeBlockMap", BLOCK_DTYPE, (SIGHT_RANGE_X * 2 + 1, SIGHT_RANGE_Y * 2 + 1)),
    *([
        ("VisibleBuildings", BUILDING_DTYPE, (MAX_NEARBY_BUILDINGS,)),
        ("VisibleItems", WORLD_ITEM_DTYPE, (MAX_NEARBY_ITEMS,)),
    ] if INCLUDE_UNUSED_OBSERVATIONS else []),
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
    *([
        ("Inventory", ITEM_DTYPE, (25,)),
        ("Recipes", RECIPE_DTYPE, (132,)),
        ("Limbs", LIMB_DTYPE, (15,)),
    ] if INCLUDE_UNUSED_OBSERVATIONS else []),
    ("LayerProgress", FLOAT_DTYPE),
    ("CurrentLayer", BYTE_DTYPE),
    ("BestLayerDepth", SHORT_DTYPE),
    ("LayerTimeRemaining", INT_DTYPE),
    ("RadLineDisplacement", SHORT_DTYPE),
    *([
        ("SoundsHeard", SOUND_DTYPE, (MAX_SOUNDS_HEARD,)),
    ] if INCLUDE_UNUSED_OBSERVATIONS else []),
], align=False)
