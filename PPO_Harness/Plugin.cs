using BepInEx;
using UnityEngine;
using HarmonyLib;
using System.IO.Pipes;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using static HarmonyLib.AccessTools;
using Debug = UnityEngine.Debug;
using System.Linq;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Collections;
using UnityEngine.SceneManagement;

[BepInPlugin("sebastian.ppoharness", "PPO Harness", "0.2.0")]
public class PPO_Harness : BaseUnityPlugin
{
    private void Awake()
    {
        Logger.LogInfo("PPO Harness loaded");

        Harmony harmony = new Harmony("sebastian.ppoharness");
        harmony.PatchAll();

        Logger.LogInfo("Patches applied");

        Application.quitting += PPOBridge.Shutdown;
    }
}

class BinaryObservationWriter
{
    public const int ExpectedSize = 1056511;

    public int BytesWritten { get; private set; }
    public byte[] Buffer { get; } = new byte[ExpectedSize];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        BytesWritten = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(bool value)
    {
        Buffer[BytesWritten++] = value ? (byte)1 : (byte)0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(byte value)
    {
        Buffer[BytesWritten++] = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(sbyte value)
    {
        Buffer[BytesWritten++] = unchecked((byte)value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(short value)
    {
        Buffer[BytesWritten++] = unchecked((byte)value);
        Buffer[BytesWritten++] = unchecked((byte)(value >> 8));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(ushort value)
    {
        Buffer[BytesWritten++] = unchecked((byte)value);
        Buffer[BytesWritten++] = unchecked((byte)(value >> 8));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(int value)
    {
        Buffer[BytesWritten++] = unchecked((byte)value);
        Buffer[BytesWritten++] = unchecked((byte)(value >> 8));
        Buffer[BytesWritten++] = unchecked((byte)(value >> 16));
        Buffer[BytesWritten++] = unchecked((byte)(value >> 24));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(uint value)
    {
        Buffer[BytesWritten++] = unchecked((byte)value);
        Buffer[BytesWritten++] = unchecked((byte)(value >> 8));
        Buffer[BytesWritten++] = unchecked((byte)(value >> 16));
        Buffer[BytesWritten++] = unchecked((byte)(value >> 24));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(float value)
    {
        Write(unchecked((uint)BitConverter.SingleToInt32Bits(value)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Write(double value)
    {
        ulong bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
        Buffer[BytesWritten++] = unchecked((byte)bits);
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 8));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 16));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 24));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 32));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 40));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 48));
        Buffer[BytesWritten++] = unchecked((byte)(bits >> 56));
    }
}

public class PendingSound
{
    public AudioClip Clip;
    public Vector2 Position;
    public float Volume;
}

public class BlockObservation
{
    public float Health;
    public float Toxicity;
    public Body.SleepQuality SleepQuality;

    public short[] ItemPool;

    public BlockObservation()
    {
        ItemPool = [-1, -1, -1];
    }
}

public class FluidTileObservation
{
    public byte Type; // 0 none, 1 water, 2 lumalgae, 3 oil, 4 sap, 5 dirtywater, 6 magma
}

public class BuildingObservation
{
    public bool Exists;
    public Vector2IntObservation RelativePosition;
    public float Health;
    public short[] DropPool;

    public BuildingObservation()
    {
        RelativePosition = new();
        DropPool = [-1, -1, -1, -1, -1];
    }
}

public class LimbObservation
{
    public float SkinHealth;
    public float MuscleHealth;
    public float Pain;
    public float InfectionAmount;
    public float DisinfectionTime;

    public bool Dislocated;
    public bool Broken;
    public bool Splinted;
    public bool Infected;

    public float DislocationTimer;
    public float BoneHealTimer;

    public bool IsVital;
    public bool IsHead;
    public bool IsAbdomen;
    public bool IsLegLimb;
    public bool IsArm;

    public byte DistanceToHeart;

    public byte Shrapnel;
    
    public bool Dismembered;

    public float TotalBleedAmount;
}

public class QualityObservation
{
    public sbyte ID = -1;
    public int Amount;
}

public class LiquidObservation
{
    public sbyte ID = -1;
    public ushort Amount;
    public QualityObservation[] Qualities;

    public LiquidObservation()
    {
        Qualities = Observation.CreateArray<QualityObservation>(Observation.MAX_QUALITIES);
    }
}

public class ItemObservation
{
    public short ID = -1;
    public float Condition;

    public ItemObservation[] Contents;
    public LiquidObservation[] Liquids;
    public QualityObservation[] Qualities;

    public ItemObservation(bool allowContents)
    {
        if (allowContents)
        {
            Contents = new ItemObservation[Observation.MAX_BAG_ITEMS];

            for (int i = 0; i < Contents.Length; i++) Contents[i] = new ItemObservation(false);
        }
        else
        {
            Contents = [];
        }

        Liquids = Observation.CreateArray<LiquidObservation>(Observation.MAX_LIQUID_COMPONENTS);
        Qualities = Observation.CreateArray<QualityObservation>(Observation.MAX_QUALITIES);
    }
}

public class ItemRequirementObservation
{
    public short ID = -1;
    public float MinimumCondition;
}

public class RecipeResultObservation
{
    public short ID = -1;
    public float Condition;
    public byte Amount;
}

public class WorldItemObservation
{
    public Vector2IntObservation RelativePosition;
    public ItemObservation Item;

    public WorldItemObservation()
    {
        RelativePosition = new();
        Item = new(true);
    }
}

public class RecipeObservation
{
    public bool IsCraftable;

    public ItemRequirementObservation[] ItemRequirements;
    public QualityObservation[] QualityRequirements;

    public RecipeResultObservation Output;

    public RecipeObservation()
    {
        ItemRequirements = Observation.CreateArray<ItemRequirementObservation>(Observation.MAX_INGREDIENTS);
        QualityRequirements = Observation.CreateArray<QualityObservation>(Observation.MAX_QUALITIES);

        Output = new();
    }
}

public class Vector2Observation
{
    public float X;
    public float Y;
}

public class Vector2IntObservation
{
    public short X;
    public short Y;
}

public class SoundObservation
{
    public short ID = -1;
    public Vector2IntObservation RelativeTilePosition;
    public float Volume;

    public SoundObservation()
    {
        RelativeTilePosition = new();
    }
}

public class Observation
{
    public const int MAX_NEARBY_BUILDINGS = 16;
    public const int MAX_NEARBY_ITEMS = 16; // 16
    public const int SIGHT_RANGE_X = 42;
    public const int SIGHT_RANGE_Y = 24;
    public const int MAX_BAG_ITEMS = 32; // 32
    public const int MAX_INGREDIENTS = 16;
    public const int MAX_LIQUID_COMPONENTS = 16; // 16
    public const int MAX_QUALITIES = 8; // 8
    public const int MAX_SOUNDS_HEARD = 16; // 16
    public const int TOTAL_SLOTS = 25; // 6 item slots and wearables go up to index 24

    // Helper for initialization
    public static T[] CreateArray<T>(int count) where T : new()
    {
        T[] array = new T[count];

        for (int i = 0; i < array.Length; i++) array[i] = new T();

        return array;
    }

    // Surroundings
    public BlockObservation[,] RelativeBlockMap;
    public BuildingObservation[] VisibleBuildings;
    public WorldItemObservation[] VisibleItems;
    public FluidTileObservation[,] RelativeFluidMap;

    // Basic Locomotion
    public Vector2Observation Velocity;
    public bool IsRight;
    public float MaxSpeed;
    public Vector2IntObservation RelativeLookPos;

    // Jumping
    public float JumpCooldown;
    public bool Grounded;
    public float TimeSinceGrounded;
    public BlockObservation StandingOn;

    // Ragdolling
    public float TimeRagdolled;
    public float CrawlTime;
    
    // Water/Liquids
    public bool InWater;
    public float LiquidSlipTime;
    public float LiquidRagdollBar;
    public float LiquidDrinkTime;

    // Walljumping
    public bool CanWalljumpLeft;
    public bool CanWalljumpRight;

    // Attacking
    public float AttackCooldown;

    // Crouching
    public float CrouchAmount;
    public bool Crouching;

    // Climbing
    public bool IsClimbing;
    public float ClimbableProgress;
	public float ClimbVelocity;
    
    // Heart
    public float HeartRate;
    public float FibrillationProgress;
	public bool FibrillationForced;
    public bool FibrillationRising;
    public bool HasPulmonaryEmbolism;

    // SO MUCH BLOOD JESUS
	public float BloodOxygen;
	public float BloodVolume;
    public float BloodPressure;
	public float BloodVesselSize;
    public float BloodViscosity;
    public float TotalBleedSpeed;
    public float InternalBleeding;
    public float Hemothorax;
    public float VenomTotal; // Technically why your blood turns into marmalade
	public float VenomCurrent;
	
    // Automatic Breathing
	public float RespiratoryRate;
    public bool Breathing;
	
    // Adrenaline
	public float Adrenaline;
	public float CurAdrenaline;

    // DRUGS!!!!
    public float StimulantMultiplier;
    public bool OnHardStimulants;
	public float OpiateHappiness;
	public float AntidepressantHappiness;
    public float BrainGrowSickness;
    public bool UsedNeuralBooster;
    public bool MindWiped;
    public float Caffeinated;
    public byte OverdoseIndex;

    // Misc. Simple Stats
	public float WeightOffset;
	public float Hunger;
	public float Thirst;
	public float Stamina;
	public float Energy;
    public float Immunity;
    public float TotalHappiness;
    public float Dirtyness;
    public float ClawHealth;

    // Brain
	public float BrainHealth;
	public float Consciousness;
	public float Shock;
    public bool ReversedControls; // brain damage
    public bool BrainDying;
    public bool PlayerDead;
    public float StrokeAmount;

    // Temperature
	public float Temperature;
	public float ClothingTemperature;

    // Pain
	public float AveragePain;
    public float PainShock;
    public float HearingLoss;
    public bool BothHandsUnusable;
	
    // Sickness
    public float SicknessAmount;
    public float SepticShock;
    public float RadiationSickness;

    // Mental State
    public ushort CorpsesSeen;
    public float TraumaAmount;
    public float HorrifiedLevel;
	public float FocusedLevel;
    
    // Dismemberment
    public bool Disfigured;
    public bool EyeGone;
    public bool BothEyesGone;

    // Encumberance
    public float TotalEncumberance;
	public float OverEncumberance;
    public float MaxEncumberance;
    
    // Sleep
    public bool Sleeping;
    public Body.SleepQuality CurSleep;
	public float BadSleepAmount;
	public float GoodSleepTime;
    public Body.SleepQuality? ForcedSleepQuality;
    public bool UsingSleepingBag;
    public bool CanTakeNap;

    // Last Stand
    public bool TriedRollingLastStand;
    public float LastStandTime;

    // Skills
    public byte STR;
    public byte RES;
    public byte INT;
    public float STRProgress;
    public float RESProgress;
    public float INTProgress;

    // Inventory
    public ItemObservation[] Inventory;

    // Crafting
    public RecipeObservation[] Recipes;

    // Limbs
    public LimbObservation[] Limbs;

    // Progress
    public float LayerProgress; // normalized 0-1
    public byte CurrentLayer;
    public short BestLayerDepth;

    // Radline
    public int LayerTimeRemaining; // seconds
    public short RadLineDisplacement; // negative values mean you're behind the radline (bad)

    // Sounds
    public SoundObservation[] SoundsHeard;

    public Observation()
    {
        int width = SIGHT_RANGE_X * 2 + 1;
        int height = SIGHT_RANGE_Y * 2 + 1;

        // Surroundings
        RelativeBlockMap = new BlockObservation[width, height];
        for (int x = 0; x < RelativeBlockMap.GetLength(0); x++)
        {
            for (int y = 0; y < RelativeBlockMap.GetLength(1); y++)
            {
                RelativeBlockMap[x, y] = new BlockObservation();
            }
        }

        VisibleBuildings = CreateArray<BuildingObservation>(MAX_NEARBY_BUILDINGS);
        VisibleItems = CreateArray<WorldItemObservation>(MAX_NEARBY_ITEMS);
       
        RelativeFluidMap = new FluidTileObservation[width, height];
        for (int x = 0; x < RelativeBlockMap.GetLength(0); x++)
        {
            for (int y = 0; y < RelativeBlockMap.GetLength(1); y++)
            {
                RelativeFluidMap[x, y] = new FluidTileObservation();
            }
        }

        // Basic Locomotion
        Velocity = new();
        RelativeLookPos = new();

        // Jumping
        StandingOn = new();

        // Inventory
        Inventory = new ItemObservation[TOTAL_SLOTS]; // 6 item slots, rest are wearable up to index 24
        for (int i = 0; i < Inventory.Length; i++) Inventory[i] = new ItemObservation(true);

        // Crafting - remember to initialize recipes in Start() with the observation

        // Limbs
        Limbs = CreateArray<LimbObservation>(15);

        // Sounds
        SoundsHeard = CreateArray<SoundObservation>(MAX_SOUNDS_HEARD);
    }
}

public class Action
{
    public int MoveDirection;
    public int Jump;
    public int VerticalMovement;
    public int Crouch;

    public int LookDX;
    public int LookDY;

    public int Attack;
    public int Interact;
    public int TargetItemSlot;
    public int SelectedItemSlot;
    public int DropItem;
    public int MoveItem;
    public int SelectedBagIndex;

    public int UseItem;
    public int UseItemWorld;

    public int SelectedLimb;
    public int UseItemMedical;

    public int SelectedRecipe;

    public int FavoriteItem;
    public int SwitchMainHand;
    public int TrySleep;
    public int Ragdoll;
    public int Exercise;
    public int Bark;

    public int Throw;

    public int LiquidAmount;
    public int DrainLiquid;
    public int PullLiquidFromWorld;
}

public static class PPOBridge
{
    public static readonly string[] WearableSlots =
    {
        "arms", // 6
        "back", // 7
        "balaclava", // 8
        "bandolier", // 9
        "belt", // 10
        "blindfold", // 11
        "eyes", // 12
        "feet", // 13
        "hands", // 14
        "hat", // 15
        "knees", // 16
        "mouth", // 17
        "neck", // 18
        "outertorso", // 19
        "thigh", // 20
        "thighback", // 21
        "torso", // 22
        "torsofront", // 23
        "wraps" // 24
    };

    public static readonly string[] LimbSlots =
    {
        "Head", // 0

        "Upper Torso", // 1
        "Lower Torso", // 2

        "Front Upper Arm", // 3
        "Front Forearm", // 4
        "Front Hand", // 5

        "Back Upper Arm", // 6
        "Back Forearm", // 7
        "Back Hand", // 8

        "Front Thigh", // 9
        "Front Calf", // 10
        "Front Foot", // 11

        "Back Thigh", // 12
        "Back Calf", // 13
        "Back Foot" // 14
    };
    
    static bool shuttingDown = false;

    public static bool ControlEnabled = false;
    public static Observation CurrentObservation = new Observation();

    static NamedPipeClientStream obsPipe;
    static BinaryObservationWriter bufferWriter;
    static bool connected;

    static NamedPipeClientStream actionPipe;
    static StreamReader actionReader;
    static bool actionConnected;

    public static Action CurrentAction = new Action();

    public static Action LastAction = new Action();

    public static Action LatestAction = new Action(); // Reader Publishing Action, separate from the actor

    static Thread actionThread;
    static volatile bool resetRequested = false;
    static volatile bool shutdownRequested = false;
    static int resetSourceWorldInstanceId;

    private static GameObject lookDebugDot;
    
    // Minigame Junk
    public static float LockpickSpeed;
    public static LockpingMinigame LastLockpick;

    public static BandageMinigame LastBandage;
    public static float nextBandageSound;

    public static KeypadMinigame LastKeypad;
    public static float NextKeypadDigit;
    public static float KeypadDigitDelay;

    public static ShrapnelMinigame LastShrapnel;
    public static float NextShrapnelPull;
    public static float ShrapnelStartSound;

    public static DislocationMinigame LastDislocation;
    public static float NextDislocationHit;
    public static int DislocationHitsRemaining;
    public static float DislocationRemaining;

    public static SyringeMinigame LastSyringe;

    public static AEDMinigame LastAED;
    public static float AEDShockTime;
    public static float AEDAnalysisTime;
    public static bool AEDAnalyzed;

    public static ManualDefibMinigame LastManualDefib;
    public static float ManualDefibCharge;
    public static float ManualDefibShockTime;

    public static AmputationMinigame LastAmputation;
    public static float AmputationEndTime;

    const int NO_ITEM_SLOT = -1;

    public static float ThrowCharge;

    static float nextObsTime = 0f;
    
    static List<string> ItemNameList = new();

    static List<string> LiquidNameList = new();

    static List<string> QualityNameList = new();

    static List<string> SoundNameList = new();

    static Dictionary<ushort, short[]> BlockDropPools;

    static RecipeObservation[] RecipeDatabase;

    static bool StartRan = false;

    static int BestLayerDepth = 0;

    public static List<PendingSound> SoundEvents = new List<PendingSound>();

    static Observation obs = new();

    static volatile bool resetComplete = true;

    static readonly FieldRef<Body, float> jc = FieldRefAccess<Body, float>("jumpCooldown");
    static readonly FieldRef<Body, float> tsg = FieldRefAccess<Body, float>("timeSinceGrounded");
    static readonly FieldRef<Body, BlockInfo> so = FieldRefAccess<Body, BlockInfo>("standingOn");
    static readonly FieldRef<Body, float> tr = FieldRefAccess<Body, float>("timeRagdolled");
    static readonly FieldRef<Body, float> ct = FieldRefAccess<Body, float>("crawlTime");
    static readonly FieldRef<Body, float> tsl = FieldRefAccess<Body, float>("timeSinceSlidLeft");
    static readonly FieldRef<Body, float> tsr = FieldRefAccess<Body, float>("timeSinceSlidRight");

    static byte NarrowByte(int value, string field)
    {
        if ((uint)value > byte.MaxValue)
            throw new OverflowException($"{field} value {value} does not fit in a byte");
        return (byte)value;
    }

    static sbyte NarrowSByte(int value, string field)
    {
        if (value < sbyte.MinValue || value > sbyte.MaxValue)
            throw new OverflowException($"{field} value {value} does not fit in an sbyte");
        return (sbyte)value;
    }

    static ushort NarrowUShort(int value, string field)
    {
        if ((uint)value > ushort.MaxValue)
            throw new OverflowException($"{field} value {value} does not fit in a ushort");
        return (ushort)value;
    }

    static short NarrowShort(int value, string field)
    {
        if (value < short.MinValue || value > short.MaxValue)
            throw new OverflowException($"{field} value {value} does not fit in a short");
        return (short)value;
    }

    static short EncodeItemID(int index)
    {
        return index < 0 ? (short)-1 : NarrowShort(index, "item ID");
    }

    static sbyte EncodeSByteID(int index, string field)
    {
        return index < 0 ? (sbyte)-1 : NarrowSByte(index, field);
    }

    static short EncodeSoundID(int index)
    {
        return index < 0 ? (short)-1 : NarrowShort(index, "sound ID");
    }

    static void CopyAction(Action src, Action dst)
    {
        dst.MoveDirection = src.MoveDirection;
        dst.Jump = src.Jump;
        dst.VerticalMovement = src.VerticalMovement;
        dst.Crouch = src.Crouch;

        dst.LookDX = src.LookDX;
        dst.LookDY = src.LookDY;

        dst.Attack = src.Attack;
        dst.Interact = src.Interact;
        dst.TargetItemSlot = src.TargetItemSlot;
        dst.SelectedItemSlot = src.SelectedItemSlot;
        dst.DropItem = src.DropItem;
        dst.MoveItem = src.MoveItem;
        dst.SelectedBagIndex = src.SelectedBagIndex;

        dst.UseItem = src.UseItem;
        dst.UseItemWorld = src.UseItemWorld;

        dst.SelectedLimb = src.SelectedLimb;
        dst.UseItemMedical = src.UseItemMedical;

        dst.SelectedRecipe = src.SelectedRecipe;

        dst.FavoriteItem = src.FavoriteItem;
        dst.SwitchMainHand = src.SwitchMainHand;
        dst.TrySleep = src.TrySleep;
        dst.Ragdoll = src.Ragdoll;
        dst.Exercise = src.Exercise;
        dst.Bark = src.Bark;

        dst.Throw = src.Throw;

        dst.LiquidAmount = src.LiquidAmount;
        dst.DrainLiquid = src.DrainLiquid;
        dst.PullLiquidFromWorld = src.PullLiquidFromWorld;
    }

    static void CollectBlockObservation(BlockObservation obs, BlockInfo block, ushort index)
    {
        if (block == null)
        {
            obs.Health = 0;
            obs.SleepQuality = 0;
            obs.Toxicity = 0;

            obs.ItemPool[0] = -1;
            obs.ItemPool[1] = -1;
            obs.ItemPool[2] = -1;
            return;
        }

        obs.Health = block.health;
        obs.SleepQuality = block.sleep;
        obs.Toxicity = block.toxicity;

        if (BlockDropPools.TryGetValue(index, out short[] pool))
        {
            Array.Copy(pool, obs.ItemPool, obs.ItemPool.Length);
        }
        else
        {
            obs.ItemPool[0] = -1;
            obs.ItemPool[1] = -1;
            obs.ItemPool[2] = -1;
        }
    }

    static void CollectFluidTileObservation(FluidTileObservation obs, Vector2Int pos)
    {
        obs.Type = NarrowByte(FluidManager.main.GetLiquid(pos.x, pos.y), "fluid tile type");
    }
  
    static void FetchRelTileMap(BlockObservation[,] map, Vector2Int pos, Vector2Int maxDim)
    {
        int width = maxDim.x * 2 + 1;
        int height = maxDim.y * 2 + 1;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Vector2Int worldPos = pos + new Vector2Int(x - maxDim.x, maxDim.y - y);

                ushort blockId = WorldGeneration.world.GetBlock(worldPos);

                CollectBlockObservation(map[x, y], WorldGeneration.world.GetBlockInfo(blockId), blockId);
            }
        }
    }

    static void FetchRelFluidMap(FluidTileObservation[,] map, Vector2Int pos, Vector2Int maxDim)
    {
        int width = maxDim.x * 2 + 1;
        int height = maxDim.y * 2 + 1;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                Vector2Int tilePos = pos + new Vector2Int(x - maxDim.x, maxDim.y - y);

                CollectFluidTileObservation(map[x, y], tilePos);
            }
        }
    }

    static void FetchVisibleBuildings(BuildingObservation[] obs, Vector2 pos)
    {
        for (int i = 0; i < obs.Length; i++)
            obs[i].Exists = false;

        Collider2D[] hits = Physics2D.OverlapBoxAll(pos,
            new Vector2(Observation.SIGHT_RANGE_X * 2 + 1, Observation.SIGHT_RANGE_Y * 2 + 1),
            0f);

        Array.Sort(hits, (a, b) =>
        {
            float da = ((Vector2)a.transform.position - pos).sqrMagnitude;
            float db = ((Vector2)b.transform.position - pos).sqrMagnitude;
            return da.CompareTo(db);
        });

        int count = 0;

        foreach (Collider2D hit in hits)
        {
            BuildingEntity bldg = hit.transform.GetComponent<BuildingEntity>();

            if (bldg && count < obs.Length)
            {
                CollectBuildingObservation(obs[count], bldg, pos);
                obs[count].Exists = true;
                count++;
            }
        }
    }
    
    static void CollectBuildingObservation(BuildingObservation obs, BuildingEntity bldg, Vector2 pos)
    {
        obs.Health = bldg.health;
        Vector2Int relPos = WorldGeneration.world.WorldToBlockPos(bldg.transform.position) - WorldGeneration.world.WorldToBlockPos(pos);
        
        obs.RelativePosition.X = NarrowShort(relPos.x, "building relative X");
        obs.RelativePosition.Y = NarrowShort(relPos.y, "building relative Y");

        Array.Fill(obs.DropPool, (short)-1);

        int count = 0;

        foreach (ItemDrop item in bldg.itemsDropOnDestroy)
        {
            if (count < obs.DropPool.Length)
            {
                obs.DropPool[count] = EncodeItemID(ItemNameList.IndexOf(item.id));
                count++;
            }
        }
    }
    
    static void FetchVisibleItems(WorldItemObservation[] obs, Vector2 pos)
    {
        for (int i = 0; i < obs.Length; i++)
        {
            CollectItemObservation(obs[i].Item, null, true);
        }

        Collider2D[] hits = Physics2D.OverlapBoxAll(pos, new Vector2(Observation.SIGHT_RANGE_X * 2 + 1, Observation.SIGHT_RANGE_Y * 2 + 1), 0f);

        Array.Sort(hits, (a, b) =>
        {
            float da = ((Vector2)a.transform.position - pos).sqrMagnitude;
            float db = ((Vector2)b.transform.position - pos).sqrMagnitude;
            return da.CompareTo(db);
        });

        int count = 0;

        foreach (Collider2D hit in hits)
        {
            Item itm = hit.transform.GetComponent<Item>();

            if (itm && count < obs.Length)
            {
                CollectWorldItemObservation(obs[count], itm, pos); // Non -1 IDs basically mean it exists
                count++;
            }
        }
    }

    static void CollectWorldItemObservation(WorldItemObservation obs, Item itm, Vector2 pos)
    {
        Vector2Int relPos = WorldGeneration.world.WorldToBlockPos(itm.transform.position) - WorldGeneration.world.WorldToBlockPos(pos);
        
        obs.RelativePosition.X = NarrowShort(relPos.x, "world item relative X");
        obs.RelativePosition.Y = NarrowShort(relPos.y, "world item relative Y");
        
        CollectItemObservation(obs.Item, itm, true);
    }
    
    static void ClearQualityObservation(QualityObservation obs)
    {
        obs.ID = -1;
        obs.Amount = 0;
    }

    static void ClearLiquidObservation(LiquidObservation obs)
    {
        obs.ID = -1;
        obs.Amount = 0;

        foreach (QualityObservation quality in obs.Qualities) ClearQualityObservation(quality);
    }

    static void ClearItemObservation(ItemObservation obs)
    {
        obs.ID = -1;
        obs.Condition = 0;

        foreach (ItemObservation item in obs.Contents) ClearItemObservation(item);

        foreach (LiquidObservation liquid in obs.Liquids) ClearLiquidObservation(liquid);

        foreach (QualityObservation quality in obs.Qualities) ClearQualityObservation(quality);
    }

    static void ClearSoundObservation(SoundObservation obs)
    {
        obs.ID = -1;
        obs.RelativeTilePosition.X = 0;
        obs.RelativeTilePosition.Y = 0;
        obs.Volume = 0;
    }

    static void CollectItemObservation(ItemObservation obs, Item itm, bool allowContents)
    {
        if (!itm)
        {
            ClearItemObservation(obs);
            return;
        }

        obs.Condition = itm.condition;
        obs.ID = EncodeItemID(ItemNameList.IndexOf(itm.id));
        int count = 0;

        // Pull Container
        Container cont = itm.GetComponent<Container>();
        if (cont && allowContents)
        {
            foreach (Transform child in cont.transform)
            {
                if (count >= Observation.MAX_BAG_ITEMS) break;

                Item contItem = child.GetComponent<Item>();
                if (!contItem) continue;

                CollectItemObservation(obs.Contents[count], contItem, false);
                count++;
            }
        }

        for (int i = count; i < obs.Contents.Length; i++) ClearItemObservation(obs.Contents[i]);

        // Pull Liquid Container
        count = 0;
        WaterContainerItem watercont = itm.GetComponent<WaterContainerItem>();
        if (watercont)
        {
            foreach (LiquidStack liquid in watercont.stack)
            {
                if (count >= Observation.MAX_LIQUID_COMPONENTS) break;

                obs.Liquids[count].ID = EncodeSByteID(LiquidNameList.IndexOf(liquid.liquidId), "liquid ID");
                obs.Liquids[count].Amount = NarrowUShort(Mathf.RoundToInt(liquid.amount), "liquid amount");
                
                int qualityCount = 0;
                var liqQualities = Liquids.Registry[liquid.liquidId].GetScaledQualities(liquid.amount);
                for (int i = 0; i < Observation.MAX_QUALITIES && i < liqQualities.Count; i++)
                {
                    obs.Liquids[count].Qualities[i].ID = EncodeSByteID(QualityNameList.IndexOf(liqQualities[i].id), "quality ID");
                    obs.Liquids[count].Qualities[i].Amount = Mathf.RoundToInt(liqQualities[i].amount);
                    qualityCount++;
                }

                for (int i = qualityCount; i < obs.Liquids[count].Qualities.Length; i++) ClearQualityObservation(obs.Liquids[count].Qualities[i]);

                count++;
            }
        }
        
        for (int i = count; i < obs.Liquids.Length; i++) ClearLiquidObservation(obs.Liquids[i]);

        // Qualities
        var qualities = Item.GlobalItems[itm.id].qualities;
        count = 0;
        for (int i = 0; i < Observation.MAX_QUALITIES && i < qualities.Count; i++)
        {
            obs.Qualities[i].ID = EncodeSByteID(QualityNameList.IndexOf(qualities[i].id), "quality ID");
            obs.Qualities[i].Amount = Mathf.RoundToInt(qualities[i].amount);
            count++;
        }

        for (int i = count; i < obs.Qualities.Length; i++) ClearQualityObservation(obs.Qualities[i]);
    }
    
    static void GetInventory(ItemObservation[] obs, Body body)
    {
        for (int i = 0; i < Observation.TOTAL_SLOTS; i++)
        {
            if (i < 6) // Item Slot
            {
                CollectItemObservation(obs[i], body.GetItem(i), true);
            }
            else // Wearable Slot
            {
                CollectItemObservation(obs[i], body.GetWearableBySlotID(WearableSlots[i - 6]), true);
            }
        }
    }

    static void UpdateCraftableRecipes()
    {
        for (int i = 0; i < RecipeDatabase.Length; i++)
        {
            RecipeDatabase[i].IsCraftable = Recipes.recipes[i].GetItemsForRecipe() != null;
        }
    }
    
    static void GetLimbs(LimbObservation[] obs, Body body)
    {
        for (int i = 0; i < 15; i++)
        {
            Limb src = body.limbs[i];
            LimbObservation dst = obs[i];

            dst.SkinHealth = src.skinHealth;
            dst.MuscleHealth = src.muscleHealth;
            dst.Pain = src.pain;
            dst.InfectionAmount = src.infectionAmount;
            dst.DisinfectionTime = src.disinfectionTime;

            dst.Dislocated = src.dislocated;
            dst.Broken = src.broken;
            dst.Splinted = src.splinted;
            dst.Infected = src.infected;

            dst.DislocationTimer = src.dislocationTimer;
            dst.BoneHealTimer = src.boneHealTimer;

            dst.IsVital = src.isVital;
            dst.IsHead = src.isHead;
            dst.IsAbdomen = src.isAbdomen;
            dst.IsLegLimb = src.isLegLimb;
            dst.IsArm = src.isArm;

            dst.DistanceToHeart = NarrowByte(src.distanceToHeart, "limb distance to heart");

            dst.Shrapnel = NarrowByte(src.shrapnel, "limb shrapnel");

            dst.Dismembered = src.dismembered;

            dst.TotalBleedAmount = src.totalBleedAmount;
        }
    }
    
    static Observation CollectObservations()
    {
        // Auxiliary Preparation
        PlayerCamera pc = PlayerCamera.main;
        Body body = pc.body;

        var world = WorldGeneration.world;
        Vector2Int playerTile = world.WorldToBlockPos(body.transform.position);
        Vector2Int targetTile = playerTile + new Vector2Int(CurrentAction.LookDX,CurrentAction.LookDY);



        // The fun part (tm)

        // Surroundings
        FetchRelTileMap(obs.RelativeBlockMap, playerTile, new Vector2Int(Observation.SIGHT_RANGE_X, Observation.SIGHT_RANGE_Y));
        FetchVisibleBuildings(obs.VisibleBuildings, body.transform.position);
        FetchVisibleItems(obs.VisibleItems, body.transform.position);
        FetchRelFluidMap(obs.RelativeFluidMap, playerTile, new Vector2Int(Observation.SIGHT_RANGE_X, Observation.SIGHT_RANGE_Y));

        // Basic Locomotion
        obs.Velocity.X = body.rb.velocity.x;
        obs.Velocity.Y = body.rb.velocity.y;

        obs.IsRight = body.isRight;
        obs.MaxSpeed = body.maxSpeed;

        obs.RelativeLookPos.X = NarrowShort(LastAction.LookDX, "look X");
        obs.RelativeLookPos.Y = NarrowShort(LastAction.LookDY, "look Y");

        // Jumping
        obs.JumpCooldown = jc(body);
        obs.Grounded = body.grounded;
        obs.TimeSinceGrounded = tsg(body);
        CollectBlockObservation(obs.StandingOn, so(body), ushort.MaxValue);

        // Ragdolling
        obs.TimeRagdolled = tr(body);
        obs.CrawlTime = ct(body);
    
        // Water/Liquids
        obs.InWater = body.inWater;
        obs.LiquidSlipTime = body.liquidSlipTime;
        obs.LiquidRagdollBar = body.liquidRagdollBar;
        obs.LiquidDrinkTime = body.liquidDrinkTime;

        // Walljumping
        obs.CanWalljumpLeft = tsl(body) < 0.21f;
        obs.CanWalljumpRight = tsr(body) < 0.21f;

        // Attacking
        obs.AttackCooldown = body.attackCooldown;

        // Crouching
        obs.CrouchAmount = body.crouchAmount;
        obs.Crouching = body.crouching;

        // Climbing
        obs.IsClimbing = body.currentClimbable;
        obs.ClimbableProgress = body.climbableProgress;
        obs.ClimbVelocity = body.climbVelocity;
        
        // Heart
        obs.HeartRate = body.heartRate;
        obs.FibrillationProgress = body.fibrillationProgress;
        obs.FibrillationForced = body.fibrillationForced;
        obs.FibrillationRising = body.fibrillationRising;
        obs.HasPulmonaryEmbolism = body.hasPulmonaryEmbolism;

        // SO MUCH BLOOD JESUS
        obs.BloodOxygen = body.bloodOxygen;
        obs.BloodVolume = body.bloodVolume;
        obs.BloodPressure = body.bloodPressure;
        obs.BloodVesselSize = body.bloodVesselSize;
        obs.BloodViscosity = body.bloodViscosity;
        obs.TotalBleedSpeed = body.totalBleedSpeed;
        obs.InternalBleeding = body.internalBleeding;
        obs.Hemothorax = body.hemothorax;
        obs.VenomTotal = body.venomTotal; // Technically why your blood turns into marmalade
        obs.VenomCurrent = body.venomCurrent;
        
        // Automatic Breathing
        obs.RespiratoryRate = body.respiratoryRate;
        obs.Breathing = body.breathing;
        
        // Adrenaline
        obs.Adrenaline = body.adrenaline;
        obs.CurAdrenaline = body.curAdrenaline;

        // DRUGS!!!!
        obs.StimulantMultiplier = body.stimulantMultiplier;
        obs.OnHardStimulants = body.onHardStimulants;
        obs.OpiateHappiness = body.opiateHappiness;
        obs.AntidepressantHappiness = body.antidepressantHappiness;
        obs.BrainGrowSickness = body.brainGrowSickness;
        obs.UsedNeuralBooster = body.usedNeuralBooster;
        obs.MindWiped = body.mindWipe;
        obs.Caffeinated = body.caffeinated;
        obs.OverdoseIndex = NarrowByte(body.overdoseIndex, "overdose index");

        // Misc. Simple Stats
        obs.WeightOffset = body.weightOffset;
        obs.Hunger = body.hunger;
        obs.Thirst = body.thirst;
        obs.Stamina = body.stamina;
        obs.Energy = body.energy;
        obs.Immunity = body.immunity;
        obs.TotalHappiness = body.totalHappiness;
        obs.Dirtyness = body.dirtyness;
        obs.ClawHealth = body.clawHealth;

        // Brain
        obs.BrainHealth = body.brainHealth;
        obs.Consciousness = body.consciousness;
        obs.Shock = body.shock;
        obs.ReversedControls = body.reversedControls; // brain damage
        obs.BrainDying = body.brainDying;
        obs.PlayerDead = !body.alive;
        obs.StrokeAmount = body.strokeAmount;

        // Temperature
        obs.Temperature = body.temperature;
        obs.ClothingTemperature = body.clothingTemperature;

        // Pain
        obs.AveragePain = body.averagePain;
        obs.PainShock = body.painShock;
        obs.HearingLoss = body.hearingLoss;
        obs.BothHandsUnusable = body.bothHandsUnusable;
        
        // Sickness
        obs.SicknessAmount = body.sicknessAmount;
        obs.SepticShock = body.septicShock;
        obs.RadiationSickness = body.radiationSickness;

        // Mental State
        obs.CorpsesSeen = NarrowUShort(body.corpsesSeen, "corpses seen");
        obs.TraumaAmount = body.traumaAmount;
        obs.HorrifiedLevel = body.horrifiedLevel;
        obs.FocusedLevel = body.focusedLevel;
        
        // Dismemberment
        obs.Disfigured = body.disfigured;
        obs.EyeGone = body.eyeGone;
        obs.BothEyesGone = body.bothEyesGone;

        // Encumberance
        obs.TotalEncumberance = body.totalEncumberance;
        obs.OverEncumberance = body.overEncumberance;
        obs.MaxEncumberance = body.maxEncumberance;
        
        // Sleep
        obs.Sleeping = body.sleeping;
        obs.CurSleep = body.curSleep;
        obs.BadSleepAmount = body.badSleepAmount;
        obs.GoodSleepTime = body.goodSleepTime;
        obs.ForcedSleepQuality = body.forcedSleepQuality;
        obs.UsingSleepingBag = body.usingSleepingBag;
        obs.CanTakeNap = body.canTakeNap;

        // Last Stand
        obs.TriedRollingLastStand = body.triedRollingLastStand;
        obs.LastStandTime = body.lastStandTime;

        // Skills
        obs.STR = NarrowByte(body.skills.STR, "STR");
        obs.RES = NarrowByte(body.skills.RES, "RES");
        obs.INT = NarrowByte(body.skills.INT, "INT");
        obs.STRProgress = body.skills.expSTR / body.skills.maxSTR;
        obs.RESProgress = body.skills.expRES / body.skills.maxRES;
        obs.INTProgress = body.skills.expINT / body.skills.maxINT;

        // Inventory
        GetInventory(obs.Inventory, body);

        // Crafting
        UpdateCraftableRecipes();

        // Limbs
        GetLimbs(obs.Limbs, body);

        float layerHeightMeters = ((float)world.height - 3.1f) * 0.3f;

        // Progress
        obs.LayerProgress = Mathf.Clamp01(world.PlayerLayerDepthMeters() / layerHeightMeters);
        obs.CurrentLayer = NarrowByte(world.biomeDepth, "current layer");
        if (world.PlayerLayerDepthMeters() > BestLayerDepth) BestLayerDepth = Mathf.RoundToInt(world.PlayerLayerDepthMeters());
        obs.BestLayerDepth = NarrowShort(BestLayerDepth, "best layer depth");

        // Radline
        obs.LayerTimeRemaining = Mathf.RoundToInt(world.maxTimePerLayer - world.layerTimeSpent); // seconds
        if (obs.LayerTimeRemaining <= 0) obs.RadLineDisplacement = NarrowShort(Mathf.RoundToInt(world.PlayerLayerDepthMeters() - world.RadlineLayerDepthMeters()), "radline displacement"); // negative values mean you're behind the radline (bad)
        else obs.RadLineDisplacement = 10000; // Basically not a problem

        // Sound
        int max = Mathf.Min(SoundEvents.Count, obs.SoundsHeard.Length);

        for (int i = 0; i < max; i++)
        {
            obs.SoundsHeard[i].ID = EncodeSoundID(SoundNameList.IndexOf(SoundEvents[i].Clip.name));

            Vector2Int relPos = WorldGeneration.world.WorldToBlockPos(SoundEvents[i].Position) - playerTile;
            
            obs.SoundsHeard[i].RelativeTilePosition.X = NarrowShort(relPos.x, "sound relative X");
            obs.SoundsHeard[i].RelativeTilePosition.Y = NarrowShort(relPos.y, "sound relative Y");

            obs.SoundsHeard[i].Volume = SoundEvents[i].Volume;
        }

        for (int i = max; i < Observation.MAX_SOUNDS_HEARD; i++) ClearSoundObservation(obs.SoundsHeard[i]);

        SoundEvents.Clear(); // Flush sound candidates after using them for tick
        
        return obs;
    }

    static void WriteVector2(BinaryObservationWriter writer, Vector2Observation vector)
    {
        writer.Write(vector.X);
        writer.Write(vector.Y);
    }

    static void WriteVector2Int(BinaryObservationWriter writer, Vector2IntObservation vector)
    {
        writer.Write(vector.X);
        writer.Write(vector.Y);
    }

    static void WriteSound(BinaryObservationWriter writer, SoundObservation sound)
    {
        writer.Write(sound.ID);
        WriteVector2Int(writer, sound.RelativeTilePosition);
        writer.Write(sound.Volume);
    }

    static void WriteBlock(BinaryObservationWriter writer, BlockObservation block)
    {
        writer.Write(block.Health);
        writer.Write(block.Toxicity);
        writer.Write(NarrowSByte((int)block.SleepQuality, "block sleep quality"));

        foreach (short item in block.ItemPool)
            writer.Write(item);
    }

    static void WriteFluidTile(BinaryObservationWriter writer, FluidTileObservation tile)
    {
        writer.Write(tile.Type);
    }

    static void WriteBuilding(BinaryObservationWriter writer, BuildingObservation building)
    {
        writer.Write(building.Exists);
        WriteVector2Int(writer, building.RelativePosition);
        writer.Write(building.Health);

        foreach (short drop in building.DropPool)
            writer.Write(drop);
    }

    static void WriteQuality(BinaryObservationWriter writer, QualityObservation quality)
    {
        writer.Write(quality.ID);
        writer.Write(NarrowUShort(quality.Amount, "item/recipe quality amount"));
    }

    static void WriteLiquidQuality(BinaryObservationWriter writer, QualityObservation quality)
    {
        writer.Write(quality.ID);
        writer.Write(quality.Amount);
    }

    static void WriteLiquid(BinaryObservationWriter writer, LiquidObservation liquid)
    {
        writer.Write(liquid.ID);
        writer.Write(liquid.Amount);

        foreach (QualityObservation quality in liquid.Qualities)
            WriteLiquidQuality(writer, quality);
    }

    static void WriteItem(BinaryObservationWriter writer, ItemObservation item, bool allowContents)
    {
        writer.Write(item.ID);
        writer.Write(item.Condition);

        if (allowContents)
        {
            foreach (ItemObservation content in item.Contents)
                WriteItem(writer, content, false);
        }

        foreach (LiquidObservation liquid in item.Liquids)
            WriteLiquid(writer, liquid);

        foreach (QualityObservation quality in item.Qualities)
            WriteQuality(writer, quality);
    }

    static void WriteItemRequirement(BinaryObservationWriter writer, ItemRequirementObservation requirement)
    {
        writer.Write(requirement.ID);
        writer.Write(requirement.MinimumCondition);
    }

    static void WriteRecipeResult(BinaryObservationWriter writer, RecipeResultObservation result)
    {
        writer.Write(result.ID);
        writer.Write(result.Condition);
        writer.Write(result.Amount);
    }
    
    static void WriteWorldItem(BinaryObservationWriter writer, WorldItemObservation item)
    {
        WriteVector2Int(writer, item.RelativePosition);
        WriteItem(writer, item.Item, true);
    }

    static void WriteRecipe(BinaryObservationWriter writer, RecipeObservation recipe)
    {
        writer.Write(recipe.IsCraftable);

        foreach (ItemRequirementObservation requirement in recipe.ItemRequirements)
            WriteItemRequirement(writer, requirement);

        foreach (QualityObservation quality in recipe.QualityRequirements)
            WriteQuality(writer, quality);

        WriteRecipeResult(writer, recipe.Output);
    }

    static void WriteLimb(BinaryObservationWriter writer, LimbObservation limb)
    {
        writer.Write(limb.SkinHealth);
        writer.Write(limb.MuscleHealth);
        writer.Write(limb.Pain);
        writer.Write(limb.InfectionAmount);
        writer.Write(limb.DisinfectionTime);

        writer.Write(limb.Dislocated);
        writer.Write(limb.Broken);
        writer.Write(limb.Splinted);
        writer.Write(limb.Infected);

        writer.Write(limb.DislocationTimer);
        writer.Write(limb.BoneHealTimer);

        writer.Write(limb.IsVital);
        writer.Write(limb.IsHead);
        writer.Write(limb.IsAbdomen);
        writer.Write(limb.IsLegLimb);
        writer.Write(limb.IsArm);

        writer.Write(limb.DistanceToHeart);

        writer.Write(limb.Shrapnel);

        writer.Write(limb.Dismembered);

        writer.Write(limb.TotalBleedAmount);
    }

    static void WriteObservation(BinaryObservationWriter writer, Observation obs)
    {
        // Surroundings
        for (int x = 0; x < Observation.SIGHT_RANGE_X * 2 + 1; x++)
            for (int y = 0; y < Observation.SIGHT_RANGE_Y * 2 + 1; y++)
                WriteBlock(writer, obs.RelativeBlockMap[x, y]);

        foreach (BuildingObservation building in obs.VisibleBuildings)
            WriteBuilding(writer, building);

        foreach (WorldItemObservation item in obs.VisibleItems)
            WriteWorldItem(writer, item);

        for (int x = 0; x < Observation.SIGHT_RANGE_X * 2 + 1; x++)
            for (int y = 0; y < Observation.SIGHT_RANGE_Y * 2 + 1; y++)
                WriteFluidTile(writer, obs.RelativeFluidMap[x, y]);

        // Basic Locomotion
        WriteVector2(writer, obs.Velocity);
        writer.Write(obs.IsRight);
        writer.Write(obs.MaxSpeed);
        WriteVector2Int(writer, obs.RelativeLookPos);

        // Jumping
        writer.Write(obs.JumpCooldown);
        writer.Write(obs.Grounded);
        writer.Write(obs.TimeSinceGrounded);
        WriteBlock(writer, obs.StandingOn);

        // Ragdolling
        writer.Write(obs.TimeRagdolled);
        writer.Write(obs.CrawlTime);

        // Water/Liquids
        writer.Write(obs.InWater);
        writer.Write(obs.LiquidSlipTime);
        writer.Write(obs.LiquidRagdollBar);
        writer.Write(obs.LiquidDrinkTime);

        // Walljumping
        writer.Write(obs.CanWalljumpLeft);
        writer.Write(obs.CanWalljumpRight);

        // Attacking
        writer.Write(obs.AttackCooldown);

        // Crouching
        writer.Write(obs.CrouchAmount);
        writer.Write(obs.Crouching);

        // Climbing
        writer.Write(obs.IsClimbing);
        writer.Write(obs.ClimbableProgress);
        writer.Write(obs.ClimbVelocity);

        // Heart
        writer.Write(obs.HeartRate);
        writer.Write(obs.FibrillationProgress);
        writer.Write(obs.FibrillationForced);
        writer.Write(obs.FibrillationRising);
        writer.Write(obs.HasPulmonaryEmbolism);

        // Blood
        writer.Write(obs.BloodOxygen);
        writer.Write(obs.BloodVolume);
        writer.Write(obs.BloodPressure);
        writer.Write(obs.BloodVesselSize);
        writer.Write(obs.BloodViscosity);
        writer.Write(obs.TotalBleedSpeed);
        writer.Write(obs.InternalBleeding);
        writer.Write(obs.Hemothorax);
        writer.Write(obs.VenomTotal);
        writer.Write(obs.VenomCurrent);

        // Automatic Breathing
        writer.Write(obs.RespiratoryRate);
        writer.Write(obs.Breathing);

        // Adrenaline
        writer.Write(obs.Adrenaline);
        writer.Write(obs.CurAdrenaline);

        // Drugs
        writer.Write(obs.StimulantMultiplier);
        writer.Write(obs.OnHardStimulants);
        writer.Write(obs.OpiateHappiness);
        writer.Write(obs.AntidepressantHappiness);
        writer.Write(obs.BrainGrowSickness);
        writer.Write(obs.UsedNeuralBooster);
        writer.Write(obs.MindWiped);
        writer.Write(obs.Caffeinated);
        writer.Write(obs.OverdoseIndex);

        // Misc
        writer.Write(obs.WeightOffset);
        writer.Write(obs.Hunger);
        writer.Write(obs.Thirst);
        writer.Write(obs.Stamina);
        writer.Write(obs.Energy);
        writer.Write(obs.Immunity);
        writer.Write(obs.TotalHappiness);
        writer.Write(obs.Dirtyness);
        writer.Write(obs.ClawHealth);

        // Brain
        writer.Write(obs.BrainHealth);
        writer.Write(obs.Consciousness);
        writer.Write(obs.Shock);
        writer.Write(obs.ReversedControls);
        writer.Write(obs.BrainDying);
        writer.Write(obs.PlayerDead);
        writer.Write(obs.StrokeAmount);

        // Temperature
        writer.Write(obs.Temperature);
        writer.Write(obs.ClothingTemperature);

        // Pain
        writer.Write(obs.AveragePain);
        writer.Write(obs.PainShock);
        writer.Write(obs.HearingLoss);
        writer.Write(obs.BothHandsUnusable);

        // Sickness
        writer.Write(obs.SicknessAmount);
        writer.Write(obs.SepticShock);
        writer.Write(obs.RadiationSickness);

        // Mental
        writer.Write(obs.CorpsesSeen);
        writer.Write(obs.TraumaAmount);
        writer.Write(obs.HorrifiedLevel);
        writer.Write(obs.FocusedLevel);

        // Dismemberment
        writer.Write(obs.Disfigured);
        writer.Write(obs.EyeGone);
        writer.Write(obs.BothEyesGone);

        // Encumberance
        writer.Write(obs.TotalEncumberance);
        writer.Write(obs.OverEncumberance);
        writer.Write(obs.MaxEncumberance);

        // Sleep
        writer.Write(obs.Sleeping);
        writer.Write(NarrowSByte((int)obs.CurSleep, "current sleep quality"));
        writer.Write(obs.BadSleepAmount);
        writer.Write(obs.GoodSleepTime);
        writer.Write(obs.ForcedSleepQuality.HasValue
            ? NarrowSByte((int)obs.ForcedSleepQuality.Value, "forced sleep quality")
            : (sbyte)-1);
        writer.Write(obs.UsingSleepingBag);
        writer.Write(obs.CanTakeNap);

        // Last Stand
        writer.Write(obs.TriedRollingLastStand);
        writer.Write(obs.LastStandTime);

        // Skills
        writer.Write(obs.STR);
        writer.Write(obs.RES);
        writer.Write(obs.INT);
        writer.Write(obs.STRProgress);
        writer.Write(obs.RESProgress);
        writer.Write(obs.INTProgress);

        // Inventory
        foreach (ItemObservation item in obs.Inventory)
            WriteItem(writer, item, true);

        // Crafting
        foreach (RecipeObservation recipe in obs.Recipes)
            WriteRecipe(writer, recipe);

        // Limbs
        foreach (LimbObservation limb in obs.Limbs)
            WriteLimb(writer, limb);

        // Progress
        writer.Write(obs.LayerProgress);
        writer.Write(obs.CurrentLayer);
        writer.Write(obs.BestLayerDepth);

        // Radline
        writer.Write(obs.LayerTimeRemaining);
        writer.Write(obs.RadLineDisplacement);

        // Sounds
        foreach (SoundObservation sound in obs.SoundsHeard)
            WriteSound(writer, sound);
    }

    public static void Tick(Body body)
    {
        if (shutdownRequested)
        {
            shutdownRequested = false;
            Application.Quit();
            return;
        }

        if (resetRequested && resetComplete && WorldGeneration.world != null)
        {
            resetRequested = false;
            resetComplete = false;
            resetSourceWorldInstanceId = WorldGeneration.world.GetInstanceID();

            Debug.Log("PPO reset started.");
            WorldGeneration.world.StartCoroutine(Reset());

            return;
        }

        if (!resetComplete)
        {
            if (WorldGeneration.world != null &&
                WorldGeneration.world.GetInstanceID() != resetSourceWorldInstanceId &&
                WorldGeneration.world.worldExists &&
                !WorldGeneration.world.generatingWorld)
            {
                resetComplete = true;
                Debug.Log("PPO reset complete.");
            }

            return;
        }

        if (Time.unscaledTime < nextObsTime)
            return;

        if (Item.GlobalItems == null)
            return;
        
        if (!StartRan) Start();

        Stopwatch sw = Stopwatch.StartNew();

        CollectObservations();

        // Debug.Log($"Collect: {sw.Elapsed.TotalMilliseconds:F2}");
        sw.Restart();

        nextObsTime = Time.unscaledTime + .05f; // 20 Hz

        if (!connected)
        {
            try
            {
                Debug.Log("Connecting observation pipe...");
                obsPipe = new NamedPipeClientStream(".", "CasU_PPO_Pipe", PipeDirection.Out);
                bufferWriter = new BinaryObservationWriter();

                obsPipe.Connect(0);

                connected = true;
                Debug.Log("Observation pipe connected.");
            }
            catch
            {
                Debug.LogWarning("Failed to connect to observation pipe. Is the Server process running?");
                return;
            }
        }
        else
        {
            // sw = Stopwatch.StartNew();
            // string json = JsonConvert.SerializeObject(obs);
            // Debug.Log($"Serialize: {sw.Elapsed.TotalMilliseconds:F2}");
            // sw.Restart();
            // writer.WriteLine(json);
            // writer.Flush();
            // Debug.Log($"Pipe: {sw.Elapsed.TotalMilliseconds:F2}"); not yet!

            // BinaryObservationWriter counter = new();
            // WriteObservation(counter, obs);
            // Debug.Log($"Observation Size: {counter.BytesWritten} bytes"); // Byte size check

            sw = Stopwatch.StartNew();

            bufferWriter.Reset();

            WriteObservation(bufferWriter, obs);

            if (bufferWriter.BytesWritten != BinaryObservationWriter.ExpectedSize)
            {
                throw new InvalidDataException(
                    $"Observation writer produced {bufferWriter.BytesWritten} bytes; " +
                    $"expected {BinaryObservationWriter.ExpectedSize}"
                );
            }

            double serialize = sw.Elapsed.TotalMilliseconds;

            sw.Restart();

            obsPipe.Write(bufferWriter.Buffer, 0, bufferWriter.BytesWritten);

            double pipe = sw.Elapsed.TotalMilliseconds;

            // Debug.Log($"Serialize: {serialize:F2}  Pipe: {pipe:F2}");
        }
        if (!actionConnected)
        {
            try
            {
                Debug.Log("Connecting action pipe...");

                actionPipe = new NamedPipeClientStream(
                    ".",
                    "CasU_PPO_Action_Pipe",
                    PipeDirection.In
                );

                actionPipe.Connect(0);

                Debug.Log("Action pipe connected.");

                actionReader = new StreamReader(actionPipe);

                Debug.Log("About to StartActionReader()");
                StartActionReader();
                Debug.Log($"After StartActionReader alive={actionThread?.IsAlive}");

                actionConnected = true;
            }
            catch
            {
                Debug.LogWarning("Failed to connect to action pipe. Is the Server process running?");
            }
        }
    }

    public static void Start()
    {
        StartRan = true;
        foreach (string name in Item.GlobalItems.Keys.OrderBy(x => x))
        {
            ItemNameList.Add(name);
        }

        foreach (string name in Liquids.Registry.Keys.OrderBy(x => x))
        {
            LiquidNameList.Add(name);
        }

        AudioClip[] sounds = Resources.LoadAll<AudioClip>("Sounds");
        SoundNameList = sounds.Select(sound => sound.name).Distinct().OrderBy(name => name).ToList();

        HashSet<string> qualityNames = new(); // There is no global list of qualities, so we'll just found one ourselves. Use a hashset for redundants
        // Items
        foreach (ItemInfo info in Item.GlobalItems.Values)
        {
            if (info.qualities == null) continue;

            foreach (CraftingQuality q in info.qualities) qualityNames.Add(q.id);
        }

        // Liquids
        foreach (LiquidType liquid in Liquids.Registry.Values)
        {
            if (liquid.qualities == null) continue;

            foreach (CraftingQuality q in liquid.qualities) qualityNames.Add(q.id);
        }

        // Recipes
        foreach (Recipe recipe in Recipes.recipes)
        {
            foreach (RecipeItem item in recipe.items)
            {
                if (!item.specific) qualityNames.Add(item.quality.id);
            }
        }

        QualityNameList = qualityNames.OrderBy(x => x).ToList();

        if (ItemNameList.Count > short.MaxValue + 1)
            throw new InvalidOperationException($"Item registry has {ItemNameList.Count} entries; item IDs use signed shorts");
        if (LiquidNameList.Count > sbyte.MaxValue + 1)
            throw new InvalidOperationException($"Liquid registry has {LiquidNameList.Count} entries; liquid IDs use signed bytes");
        if (QualityNameList.Count > sbyte.MaxValue + 1)
            throw new InvalidOperationException($"Quality registry has {QualityNameList.Count} entries; quality IDs use signed bytes");
        if (SoundNameList.Count > short.MaxValue + 1)
            throw new InvalidOperationException($"Sound registry has {SoundNameList.Count} entries; sound IDs use signed shorts");

        BlockDropPools = new()
        {
            { 7,  new short[] { EncodeItemID(ItemNameList.IndexOf("glassshards")), -1, -1 } },
            { 3,  new short[] { EncodeItemID(ItemNameList.IndexOf("scrapmetal")), -1, -1 } },
            { 6,  new short[] { EncodeItemID(ItemNameList.IndexOf("scrapmetal")), -1, -1 } },
            { 10, new short[] { EncodeItemID(ItemNameList.IndexOf("scrapmetal")), -1, -1 } },
            { 8,  new short[] { EncodeItemID(ItemNameList.IndexOf("plasticchunk")), -1, -1 } },
            { 9,  new short[] { EncodeItemID(ItemNameList.IndexOf("plasticchunk")), -1, -1 } },
            { 11, new short[] { EncodeItemID(ItemNameList.IndexOf("woodscraps")), EncodeItemID(ItemNameList.IndexOf("stick")), EncodeItemID(ItemNameList.IndexOf("woodpanel")) } },
            { 24, new short[] { EncodeItemID(ItemNameList.IndexOf("woodscraps")), EncodeItemID(ItemNameList.IndexOf("stick")), EncodeItemID(ItemNameList.IndexOf("woodpanel")) } },
            { 34, new short[] { EncodeItemID(ItemNameList.IndexOf("rawcopper")), -1, -1 } },
            { 35, new short[] { EncodeItemID(ItemNameList.IndexOf("ilmenitechunk")), -1, -1 } },
        };

        RecipeDatabase = new RecipeObservation[Recipes.recipes.Count];
        for (int i = 0; i < Recipes.recipes.Count; i++)
        {
            RecipeDatabase[i] = new RecipeObservation();
            RecipeObservation rec = RecipeDatabase[i];
            Recipe refRec = Recipes.recipes[i];

            rec.Output.ID = EncodeItemID(ItemNameList.IndexOf(refRec.result.id));
            rec.Output.Condition = refRec.result.resultCondition;

            for (int j = 0; j < refRec.items.Count; j++)
            {
                if (refRec.items[j].specific)
                {
                    rec.ItemRequirements[j].ID = EncodeItemID(ItemNameList.IndexOf(refRec.items[j].specificId));
                    rec.ItemRequirements[j].MinimumCondition = refRec.items[j].minimumCondition;
                }
            }

            rec.Output.Amount = NarrowByte(refRec.result.amount, "recipe output amount");
        }
    
        obs.Recipes = RecipeDatabase;
    }
    
    static void StartActionReader()
    {
        actionThread = new Thread(() =>
        {
            while (!shuttingDown)
            {
                try
                {
                    string line = actionReader.ReadLine();
                    if (!string.IsNullOrEmpty(line))
                    {
                        if (line == "RESET")
                        {
                            resetRequested = true;
                            continue;
                        }
                        if (line == "SHUTDOWN")
                        {
                            shutdownRequested = true;
                            continue;
                        }
                        string[] parts = line.Split(',');
                        if (parts.Length >= 28)
                        {
                            LatestAction.MoveDirection = int.Parse(parts[0]);
                            LatestAction.Jump = int.Parse(parts[1]);
                            LatestAction.VerticalMovement = int.Parse(parts[2]);
                            LatestAction.Crouch = int.Parse(parts[3]);
                            LatestAction.LookDX = int.Parse(parts[4]);
                            LatestAction.LookDY = int.Parse(parts[5]);
                            LatestAction.Attack = int.Parse(parts[6]);
                            LatestAction.Interact = int.Parse(parts[7]);
                            LatestAction.TargetItemSlot = int.Parse(parts[8]);
                            LatestAction.SelectedItemSlot = int.Parse(parts[9]);
                            LatestAction.DropItem = int.Parse(parts[10]);
                            LatestAction.MoveItem = int.Parse(parts[11]);
                            LatestAction.SelectedBagIndex = int.Parse(parts[12]);
                            LatestAction.UseItem = int.Parse(parts[13]);
                            LatestAction.UseItemWorld = int.Parse(parts[14]);
                            LatestAction.SelectedLimb = int.Parse(parts[15]);
                            LatestAction.UseItemMedical = int.Parse(parts[16]);
                            LatestAction.SelectedRecipe = int.Parse(parts[17]);
                            LatestAction.FavoriteItem = int.Parse(parts[18]);
                            LatestAction.SwitchMainHand = int.Parse(parts[19]);
                            LatestAction.TrySleep = int.Parse(parts[20]);
                            LatestAction.Ragdoll = int.Parse(parts[21]);
                            LatestAction.Exercise = int.Parse(parts[22]);
                            LatestAction.Bark = int.Parse(parts[23]);
                            LatestAction.Throw = int.Parse(parts[24]);
                            LatestAction.LiquidAmount = int.Parse(parts[25]);
                            LatestAction.DrainLiquid = int.Parse(parts[26]);
                            LatestAction.PullLiquidFromWorld = int.Parse(parts[27]);
                        }
                    }
                }
                catch
                {
                    break;
                }
            }
        });
        actionThread.IsBackground = true;
        actionThread.Start();
    }

    public static void ApplyPPOActions(PlayerCamera playerCamera)
    {
        if (!resetComplete || resetRequested)
            return;

        // Before doing anything, update Actions:
        CopyAction(CurrentAction, LastAction);
        CopyAction(LatestAction, CurrentAction);

        Body body = playerCamera.body;

        var world = WorldGeneration.world;
        Vector2Int playerTile = world.WorldToBlockPos(body.transform.position);
        Vector2Int targetTile = playerTile + new Vector2Int(CurrentAction.LookDX,CurrentAction.LookDY);
        Vector2 targetWorldPos = world.BlockToWorldPos(targetTile);

        if (!MinigameBase.main.currentMinigame)
        {
            body.moveDir.x = 0;
            body.moveDir.y = 0;

            // Apply Jump Input
            if ((LastAction.Jump == 0 && CurrentAction.Jump == 1) || (PlayerCamera.autoJump && CurrentAction.Jump == 1))
            {
                playerCamera.wantsJumpInput = true;
            }
            if (LastAction.Jump == 1 && CurrentAction.Jump == 0)
            {
                playerCamera.wantsJumpInput = false;
                body.endedJump = true;
            }
            if (playerCamera.wantsJumpInput)
            {
                body.Jump();
                body.endedJump = false;
            }
            // Apply Horizontal Movement
            body.moveDir.x = CurrentAction.MoveDirection;
            body.moveDir.y = CurrentAction.VerticalMovement;
            body.crouching = CurrentAction.Crouch == 1;

            // Apply Look Direction
            body.targetLookPos = targetWorldPos;

            if (lookDebugDot == null)
            {
                lookDebugDot = new GameObject("PPO Look Debug Dot");
                lookDebugDot.layer = LayerMask.NameToLayer("Default");

                var sr = lookDebugDot.AddComponent<SpriteRenderer>();
                sr.sprite = playerCamera.placeSquare.sprite;
                sr.color = Color.red;
                sr.sortingOrder = 10000;
            }

            lookDebugDot.transform.position = body.targetLookPos;
            lookDebugDot.transform.localScale = Vector3.one * 0.5f;
        }

        // Attack
        Item item = body.GetItem(body.handSlot);
		if (CurrentAction.Attack == 1 && (!item || !item.Stats.usableWithLMB || item.Stats.autoAttack) && !MinigameBase.main.currentMinigame)
		{
			body.UseItemInHand();
		}
		else if (CurrentAction.Attack == 1 && LastAction.Attack == 0 && (!item || !item.Stats.usableWithLMB || !item.Stats.autoAttack) && !MinigameBase.main.currentMinigame)
		{
			if (body.allowUseItem)
			{
				body.UseItemInHand();
				return;
			}
			body.talker.Talk(Locale.GetCharacter("refuse"));
			playerCamera.UseFailUnhappiness();
		}

        // Craft
        if (CurrentAction.SelectedRecipe > -1 && LastAction.SelectedRecipe == -1)
        {
            Debug.Log("Crafting table!");
            Recipes.recipes[CurrentAction.SelectedRecipe].TryMake();
        }

        // Interact
        if (CurrentAction.Interact == 1 && LastAction.Interact == 0)
        {
            if (MinigameBase.main.currentMinigame && MinigameBase.main.currentMinigame.CanExit())
            {
                MinigameBase.main.EndMinigame();
            }
            else
            {
                // Prioritize items over structure usage
                Collider2D hit = Physics2D.OverlapBox(targetWorldPos, Vector2.one, 0);
                Item itemFound = hit?.GetComponent<Item>();
                if (itemFound)
                {
                    if (itemFound.TryGetComponent<Container>(out var cont) && CurrentAction.SelectedBagIndex > -1)
                    {
                        if (CurrentAction.SelectedBagIndex < cont.itemCount)
                        {
                            itemFound = itemFound.transform.GetChild(CurrentAction.SelectedBagIndex).GetComponent<Item>();
                            cont.UnloadItem(itemFound);
                        }   
                        else
                        {
                            itemFound = null; // Equivalent to dragging air out of your bag on the ground
                        }
                    }
                    if (itemFound && itemFound.Stats.wearable)
                    {
                        body.WearWearable(itemFound);
                    }
                    else
                    {
                        if (itemFound) 
                        {
                            if (CurrentAction.TargetItemSlot > 5)
                            {
                                Item slotWear = body.GetWearableBySlotID(WearableSlots[CurrentAction.TargetItemSlot-6]);
                                if ((bool)slotWear && slotWear.TryGetComponent<Container>(out var component2))
                                {
                                    component2.UnloadItem(itemFound);
                                    component2.LoadItem(itemFound);
                                    playerCamera.PlayBackpackSound();
                                }
                            }
                            else
                            {
                                Item slotItem = body.GetItem(CurrentAction.TargetItemSlot);
                                if ((bool)slotItem && slotItem.TryGetComponent<Container>(out var component2))
                                {
                                    component2.UnloadItem(itemFound);
                                    component2.LoadItem(itemFound);
                                    playerCamera.PlayBackpackSound();
                                }
                                else
                                {
                                    body.DropItem(CurrentAction.TargetItemSlot);
                                    body.PickUpItem(itemFound, CurrentAction.TargetItemSlot);
                                }
                            }
                        }
                    }
                }
                else
                {
                    UsableObject usable = hit?.GetComponent<UsableObject>();
                    usable?.gameObject.SendMessage("OnUse");
                }
            }
        }
        else if (CurrentAction.UseItem == 1 && LastAction.UseItem == 0 && CurrentAction.SelectedItemSlot != NO_ITEM_SLOT && CurrentAction.SelectedItemSlot <= 5 && !MinigameBase.main.currentMinigame)
        {
            Item inventoryItem = body.GetItem(CurrentAction.SelectedItemSlot);
            if (inventoryItem)
            {
                body.UseItem(inventoryItem);
            }
        }
        else if (CurrentAction.UseItemWorld == 1 && LastAction.UseItemWorld == 0 && !MinigameBase.main.currentMinigame)
        {
            // Use from cursor
            Collider2D hit = Physics2D.OverlapBox(targetWorldPos, Vector2.one, 0);
            Item itemFound = hit?.GetComponent<Item>();
            if (itemFound)
            {
                body.UseItem(itemFound);
            }
        }
        else if (CurrentAction.UseItemMedical == 1 && LastAction.UseItemMedical == 0 && !MinigameBase.main.currentMinigame)
        {
            Limb limb = body.limbs[CurrentAction.SelectedLimb];
            if (CurrentAction.SelectedItemSlot == NO_ITEM_SLOT)
            {
                Debug.Log($"I have no tool but I will procedure regardless");
                playerCamera.selectedLimb = limb;
                playerCamera.WoundSpecialAction();
            }
            else if (CurrentAction.SelectedItemSlot <= 5)
            {
                Item medItem = body.GetItem(CurrentAction.SelectedItemSlot);
                if (medItem)
                {
                    Debug.Log($"Okay, I have a {body.GetItem(CurrentAction.SelectedItemSlot)}. I will use on my {limb} until this dumbass PPO decides to stop");
                    playerCamera.selectedLimb = limb;
                    playerCamera.ApplyWoundItem(medItem);
                }
            }
        }
        else if (CurrentAction.DropItem == 1 && LastAction.DropItem == 0 && CurrentAction.SelectedItemSlot != NO_ITEM_SLOT &&!MinigameBase.main.currentMinigame)
        {
            if(CurrentAction.SelectedItemSlot > 5) // Wearable slot
            {
                Item wearable = body.GetWearableBySlotID(WearableSlots[CurrentAction.SelectedItemSlot - 6]);

                if (wearable)
                {
                    body.DropWearable(wearable);
                }
            }
            else // Item slot
            {
                body.DropItem(CurrentAction.SelectedItemSlot);
            }
        }
        else if (CurrentAction.MoveItem == 1 && LastAction.MoveItem == 0 && !MinigameBase.main.currentMinigame)
        {
            TryPerformInventoryAction(CurrentAction.SelectedItemSlot, CurrentAction.TargetItemSlot, CurrentAction.SelectedBagIndex, playerCamera, body);
        }
        else if (CurrentAction.SwitchMainHand == 1 && LastAction.SwitchMainHand == 0)
        {
            playerCamera.SwitchHands();
        }
        else if (CurrentAction.Bark == 1 && LastAction.Bark == 0)
        {
            body.GetComponent<PantSound>().Bark();
        }
        else if (CurrentAction.Ragdoll == 1)
        {
            body.Ragdoll();
        }
        else if (CurrentAction.FavoriteItem == 1 && LastAction.FavoriteItem == 0 && CurrentAction.SelectedItemSlot != NO_ITEM_SLOT)
        {
            Item curItem = CurrentAction.SelectedItemSlot < 6 ? body.GetItem(CurrentAction.SelectedItemSlot): body.GetWearableBySlotID(WearableSlots[CurrentAction.SelectedItemSlot - 6]);

            if (curItem) curItem.favourited = !curItem.favourited;
        }
        else if (CurrentAction.TrySleep == 1 && LastAction.TrySleep == 0)
        {
            body.TakeANap();
        }
        else if (CurrentAction.Exercise > -1 && LastAction.Exercise == -1)
        {
            body.StartCoroutine(body.DoWorkout((Body.WorkoutType)CurrentAction.Exercise));
        }
        if (CurrentAction.Throw == 1)
        {
            ThrowCharge += Time.deltaTime;
        }
        else if (CurrentAction.Throw == 0 && LastAction.Throw == 1)
        {
            if (ThrowCharge < 0.15f)
                body.DropItem(body.handSlot);
            else
                body.ThrowItem(ThrowCharge * 2f);

            ThrowCharge = 0f;
        }
        else if (CurrentAction.DrainLiquid == 1 && CurrentAction.SelectedItemSlot != NO_ITEM_SLOT)
        {
            Item selectedItem = CurrentAction.SelectedItemSlot < 6 ? body.GetItem(CurrentAction.SelectedItemSlot) : body.GetWearableBySlotID(WearableSlots[CurrentAction.SelectedItemSlot - 6]);
            if (!selectedItem) return;
            WaterContainerItem container = selectedItem.GetComponent<WaterContainerItem>();
            
            if (container)
            {
                container.Drain(container.CalculateDrain(0.2f * Time.deltaTime * container.Capacity));
            }
        }
        else if (CurrentAction.PullLiquidFromWorld == 1 && LastAction.PullLiquidFromWorld == 0)
        {
            Item targetItem = CurrentAction.TargetItemSlot < 6 ? body.GetItem(CurrentAction.TargetItemSlot) : body.GetWearableBySlotID(WearableSlots[CurrentAction.TargetItemSlot - 6]);
            if (!targetItem) return;
            WaterContainerItem transferTo = targetItem.GetComponent<WaterContainerItem>();
            if (!transferTo) return;
            Collider2D hit = Physics2D.OverlapBox(targetWorldPos, Vector2.one, 0);
            Item itemFound = hit?.GetComponent<Item>();
            WaterContainerItem transferFrom = null;
            if (itemFound)
            {
                if (itemFound.TryGetComponent<WaterContainerItem>(out var cont))
                {
                    transferFrom = cont;
                }
            } 
            if (transferFrom == null) return;
            body.CombineLiquids(transferTo, transferFrom, CurrentAction.LiquidAmount);
        }
    }

    public static bool TryPerformInventoryAction(int selSlot, int tarSlot, int bagIndex, PlayerCamera pc, Body body)
	{
        var containerOperation = false;

        Item dragItem = null;
        Container cont = null;
        Container tarCont = null;
        Item item = null;
        if (selSlot == NO_ITEM_SLOT || tarSlot == NO_ITEM_SLOT)
        {
            return false;
        }
        // I won't fail you, john wearable!!
        if (selSlot > 5)
        {
            dragItem = body.GetWearableBySlotID(WearableSlots[selSlot-6]);
            cont = dragItem?.GetComponent<Container>();
            if (cont && bagIndex > -1 && bagIndex < cont.itemCount)
            {
                dragItem = dragItem.transform.GetChild(bagIndex).GetComponent<Item>();
                containerOperation = true;
            }
        }
        else if (selSlot != NO_ITEM_SLOT)
        {
            dragItem = body.GetItem(selSlot);
            cont = dragItem?.GetComponent<Container>();
            if (cont && bagIndex > -1 && bagIndex < cont.itemCount)
            {
                dragItem = dragItem.transform.GetChild(bagIndex).GetComponent<Item>();
                containerOperation = true;
            }
        }
        if (tarSlot > 5)
        {
            item = body.GetWearableBySlotID(WearableSlots[tarSlot-6]);
            tarCont = item?.GetComponent<Container>();
        }
        else
        {
            item = body.GetItem(tarSlot);
            tarCont = item?.GetComponent<Container>();
        }
        if (!dragItem && !item)
        {
            return false;
        }
        if (item && dragItem)
        {
            if (item == dragItem)
            {
                return true;
            }
            if ((bool)item && (bool)item.battery)
            {
                if (dragItem.Stats.HasTag("tool"))
                {
                    item.battery.UnloadBattery();
                    return true;
                }
                if (dragItem.Stats.HasTag("battery"))
                {
                    item.battery.LoadBattery(dragItem);
                    return true;
                }
            }
            if ((bool)item && item.TryGetComponent<Container>(out var component2))
            {
                if (cont && cont.itemCount > 0) // We have a bag with stuff in it
                {
                    // Do a full transfer
                    List<Item> list = new List<Item>();
                    foreach (Transform item2 in dragItem.transform)
                    {
                        if (item2.TryGetComponent<Item>(out var component3))
                        {
                            list.Add(component3);
                        }
                    }
                    bool flag = false;
                    foreach (Item item3 in list)
                    {
                        if (component2.CanHoldItem(item3))
                        {
                            dragItem.container.UnloadItem(item3);
                            component2.LoadItem(item3);
                            flag = true;
                        }
                    }
                    if (flag)
                    {
                        pc.PlayBackpackSound();
                        return true;
                    }
                }
                else // anything else
                {
                    component2.UnloadItem(dragItem);
                    component2.LoadItem(dragItem);
                    pc.PlayBackpackSound();
                }
                return true;
            }
            if (item.TryGetComponent<WaterContainerItem>(out var transferTo) && dragItem.TryGetComponent<WaterContainerItem>(out var transferFrom))
            {
                body.CombineLiquids(transferTo, transferFrom, CurrentAction.LiquidAmount);
                return true;
            }
            if ((bool)item && body.CanCombine(item, dragItem))
            {
                Debug.Log("Combine");
                body.CombineItems(item, dragItem);
                return true;
            }
        }
		if (tarSlot < 6 && selSlot < 6 && !containerOperation) // 0-5 are inventory slots, beyond that's wearables. This is slot-based and eats shit when you try to remove something from a container
		{
			body.SwapSlots(tarSlot, selSlot); // the PPO can only currently work with slots, not external inventory interactions
			return true;
		}
        else if (containerOperation)
        {

            if (dragItem && dragItem.Stats.wearable)
            {
                body.WearWearable(dragItem);
            }
            else if (tarCont)
            {
                tarCont.UnloadItem(dragItem);
                tarCont.LoadItem(dragItem);
                pc.PlayBackpackSound();
            }
            else if (tarSlot <= 5)
            {
                body.DropItem(tarSlot); // Just like the pick up action, need to kick the current item out
                body.PickUpItem(dragItem, tarSlot);
            }
            else return false;
            return true;
        }
        return false;
	}
    
    public static IEnumerator Reset()
	{
		resetComplete = false;
		yield return WorldGeneration.world.Clear();
		Time.timeScale = 1f;
		SceneManager.LoadScene(SceneManager.GetActiveScene().name);
		yield break;
	}
    
    public static void Shutdown()
    {
        shuttingDown = true;

        try
        {
            obsPipe?.Close();

            actionReader?.Close();
            actionPipe?.Close();
        }
        catch {}
    }
}

[HarmonyPatch(typeof(Body))]
[HarmonyPatch("FixedUpdate")]
public static class BodyFixedUpdatePatch
{   
    static void Postfix(Body __instance)
    {
        PPOBridge.Tick(__instance);
    }
}


[HarmonyPatch(typeof(PlayerCamera))]
[HarmonyPatch("HandleInput")]
public static class HandleInputPatch
{
    static bool Prefix(PlayerCamera __instance)
    {

        if (Input.GetKeyDown(KeyCode.F8))
        {
            PPOBridge.ControlEnabled = !PPOBridge.ControlEnabled;
            Debug.Log($"Human Control Enabled: {PPOBridge.ControlEnabled}");
        }
        if (PPOBridge.ControlEnabled)
        {
            return true; // Allow normal input handling
        }
        else
        {
            PPOBridge.ApplyPPOActions(__instance);
        }
        return false; // Skip this!
    }
}

[HarmonyPatch(typeof(PlayerCamera))]
[HarmonyPatch("OpenCraftScreen")]
public static class OpenCraftScreenPatch
{
    static bool Prefix()
    {
        if (PPOBridge.ControlEnabled)
        {
            return true; // Allow normal input handling
        }
        return false; // Skip this!
    }
}

[HarmonyPatch(typeof(PlayerCamera))]
[HarmonyPatch("TryPickupFromWorld")]
public static class TryPickupFromWorldPatch
{
    static bool Prefix()
    {
        if (PPOBridge.ControlEnabled)
        {
            return true; // Allow normal input handling
        }
        return false; // Skip this!
    }
}

[HarmonyPatch(typeof(PlayerCamera))]
[HarmonyPatch("HandleTradeMenu")]
public static class HandleTradeMenuPatch
{
    static bool Prefix()
    {
        if (PPOBridge.ControlEnabled)
        {
            return true; // Allow normal input handling
        }
        return false; // Skip this!
    }
}

[HarmonyPatch(typeof(EPdaScript), "Use")]
class EPDAPatch
{
    static bool Prefix(EPdaScript __instance)
    {
        if (PPOBridge.ControlEnabled)
            return true;

        if (!PlayerCamera.main.body.mindWipe)
        {
            if (!__instance.hasBeenRead)
            {
                __instance.hasBeenRead = true;

                PlayerCamera.main.body.skills.AddExp(2, 35f);

                PlayerCamera.main.DoAlert(
                    Locale.GetOther("epdalearn")
                );
            }
        }

        return false;
    }
}

[HarmonyPatch(typeof(Body), "TakeANap")]
class TakeANapPatch
{
    static readonly AccessTools.FieldRef<Body, bool> movingAllowed = AccessTools.FieldRefAccess<Body, bool>("movingAllowed");

    static bool Prefix(Body __instance)
    {
        if (PPOBridge.ControlEnabled)
            return true;

        
        if (__instance.canTakeNap)
        {
            __instance.DropItem(0);
            __instance.DropItem(1);
            __instance.DropItem(2);
            if ( __instance.sicknessAmount > 30f ||  __instance.totalHappiness < -50f ||  __instance.temperature < 34.5f ||  __instance.temperature > 38.5f)
            {
                __instance.StartCoroutine(AltNapCoroutine(__instance));
            }
            else
            {
                __instance.StartCoroutine(NapCoroutine(__instance));
            }
        }

        return false;
    }

    static System.Collections.IEnumerator NapCoroutine(Body __instance)
	{
		__instance.bodyAnimator.Play("ExperimentLayDown");
		__instance.armsAnimator.Play("ArmsLayDown");
		movingAllowed(__instance) = false;
		yield return new WaitForSeconds(0.3f);
		__instance.eyeCloseTime = 0.8f;
		__instance.eatTime = 0.7f;
		Sound.Play("stretch", __instance.transform.position, twoDimensional: false, pitchShift: false, null, 0.5f); // is base. __instance.
		yield return new WaitForSeconds(1.65f);
		movingAllowed(__instance) = true;
		__instance.consciousness = 10f;
		__instance.sleeping = true;
	}

	static System.Collections.IEnumerator AltNapCoroutine(Body __instance)
	{
		__instance.bodyAnimator.Play("ExperimentLayDownAlt");
		__instance.armsAnimator.Play("ArmsLayDownAlt");
		movingAllowed(__instance) = false;
		yield return new WaitForSeconds(0.4f);
		__instance.eyeCloseTime = 0.8f;
		yield return new WaitForSeconds(0.55f);
		movingAllowed(__instance) = true;
		__instance.consciousness = 10f;
		__instance.sleeping = true;
	}
}

[HarmonyPatch(typeof(Sound), nameof(Sound.Play), new Type[] { typeof(AudioClip), typeof(Vector2), typeof(bool), typeof(bool), typeof(Transform), typeof(float), typeof(float), typeof(bool), typeof(bool) })]
class SoundPlayPatch
{
    static void Prefix(AudioClip clip, Vector2 pos, float volume)
    {
        if (clip == null) return;
        PPOBridge.SoundEvents.Add(new PendingSound
        {
            Clip = clip,
            Position = pos,
            Volume = volume
        });
    }
}

[HarmonyPatch(typeof(MinigameBase))]
[HarmonyPatch("ActiveMinigameUpdate")]
public static class MinigameUpdatePatch
{
    static readonly FieldRef<SelfHarmMinigame, float> CutTime = AccessTools.FieldRefAccess<SelfHarmMinigame, float>("cutTime");
    static readonly FieldRef<SelfHarmMinigame, bool> Cutting = AccessTools.FieldRefAccess<SelfHarmMinigame, bool>("cutting");
    static readonly FieldRef<ShrapnelMinigame, Limb> Shrapnel_Limb = AccessTools.FieldRefAccess<ShrapnelMinigame, Limb>("limb");
    static readonly FieldRef<ShrapnelMinigame, List<RectTransform>> ObjectsRef = AccessTools.FieldRefAccess<ShrapnelMinigame, List<RectTransform>>("objects");
    static readonly FieldRef<DislocationMinigame, Limb> Dislocation_Limb = AccessTools.FieldRefAccess<DislocationMinigame, Limb>("limb");

    static bool Prefix()
    {
        if (PPOBridge.ControlEnabled)
        {
            return true;
        }

        // Many Minigames! Minigames!!!! I LOVE MINIFJHGLJSDLFJSKLDF
        if (MinigameBase.main.currentMinigame is LockpingMinigame lp) {HandleLockping(lp); return false;}
        if (MinigameBase.main.currentMinigame is BandageMinigame b) {HandleBandage(b); return false;}
        if (MinigameBase.main.currentMinigame is HandCrankMinigame hc) {HandleHandCrank(hc); return false;}
        if (MinigameBase.main.currentMinigame is SelfHarmMinigame sh) {HandleSelfHarm(sh); return false;}
        if (MinigameBase.main.currentMinigame is KeypadMinigame k) {HandleKeypad(k); return false;}
        if (MinigameBase.main.currentMinigame is ShrapnelMinigame s) {HandleShrapnel(s); return false;}
        if (MinigameBase.main.currentMinigame is DislocationMinigame d) {HandleDislocation(d); return false;}
        if (MinigameBase.main.currentMinigame is SyringeMinigame sy) {HandleSyringe(sy); return false;}
        if (MinigameBase.main.currentMinigame is AEDMinigame aed) {HandleAED(aed); return false;}
        if (MinigameBase.main.currentMinigame is ManualDefibMinigame md) {HandleManualDefib(md); return false;}
        if (MinigameBase.main.currentMinigame is AmputationMinigame a) {HandleAmputation(a); return false;}

        return true;
    }

    static void HandleLockping(LockpingMinigame lp)
    {
        if (lp != PPOBridge.LastLockpick) 
        { 
            PPOBridge.LastLockpick = lp; 
            float duration = UnityEngine.Random.Range(4f, 8f); 
            PPOBridge.LockpickSpeed = 1f / duration; 
        }
        lp.lockProgress += Time.deltaTime * PPOBridge.LockpickSpeed * MinigamePrecisionFactor(PlayerCamera.main.body);
        // Debug.Log(lp.lockProgress);

        foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
        {
            if (source.clip.name == "lockpickLoop")
            {
                source.volume = 1f;
            }
        }

        if (lp.lockProgress >= 1f)
        {
            PPOBridge.LastLockpick = null;

            lp.toDestroy.health = 0f;
            MinigameBase.main.EndMinigame();
        }
    }

    static void HandleBandage(BandageMinigame b)
    {
        if (b != PPOBridge.LastBandage)
        {
            PPOBridge.LastBandage = b;
            PPOBridge.nextBandageSound = Time.time;
        }

        if (Time.time >= PPOBridge.nextBandageSound)
        {
            Sound.Play(
                "bandage",
                Minigame.game.currentItem.transform.position
            );

            PPOBridge.nextBandageSound = Time.time + 1.6f;
        }

        b.OnUse(MinigameSpeedFactor(PlayerCamera.main.body) / 36f);
    }

    static void HandleHandCrank(HandCrankMinigame hc)
    {
        foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
        {
            source.volume = 0.5f;
            source.pitch = 1f;
        }

        Minigame.game.currentItem.battery.DrainCharge(-Time.deltaTime * 0.04f * MinigameSpeedFactor(PlayerCamera.main.body));
        Minigame.game.body.stamina -= Time.deltaTime * 18f * MinigameSpeedFactor(PlayerCamera.main.body);
    }

    static void HandleSelfHarm(SelfHarmMinigame sh)
    {
        if (!Cutting(sh))
        {
            sh.StartCut();
        }

        CutTime(sh) -= Time.deltaTime;

        if (Cutting(sh) && CutTime(sh) < 0f)
        {
            sh.EndCut();
        }
    }

    static void HandleKeypad(KeypadMinigame k)
    {
        if (k != PPOBridge.LastKeypad)
        {
            PPOBridge.LastKeypad = k;
            PPOBridge.NextKeypadDigit = Time.time;
        }

        if (Time.time >= PPOBridge.NextKeypadDigit)
        {
            char next = k.match[k.current.Length];

            k.current += next;

            Sound.Play(
                "beep" + next,
                Vector2.zero,
                twoDimensional: true,
                pitchShift: false,
                null,
                1f,
                1f,
                noReverb: true
            );

            PPOBridge.KeypadDigitDelay = UnityEngine.Random.Range(0.25f, 1.0f);
            PPOBridge.NextKeypadDigit = Time.time + PPOBridge.KeypadDigitDelay;
        }

        if (k.current == k.match)
        {
            PPOBridge.LastKeypad = null;

            k.toDestroy.health = 0f;
            Minigame.game.EndMinigame();
        }
    }

    static void HandleShrapnel(ShrapnelMinigame s)
    {
        Limb limb = Shrapnel_Limb(s);

        if (s != PPOBridge.LastShrapnel)
        {
            PPOBridge.LastShrapnel = s;
            PPOBridge.NextShrapnelPull = Time.time + UnityEngine.Random.Range(.5f, 1.5f);
            PPOBridge.ShrapnelStartSound = Time.time;
        }

        if (Time.time >= PPOBridge.NextShrapnelPull)
        {
            limb.shrapnel--;

            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 0f;
            }
            float mult = s.hasTweezers ? 0.5f : 1f;

            PPOBridge.NextShrapnelPull = Time.time + UnityEngine.Random.Range(.75f, 1.75f) * mult * MinigamePrecisionFactor(limb.body);
            PPOBridge.ShrapnelStartSound = Time.time + UnityEngine.Random.Range(.25f, 1.0f) * mult * MinigamePrecisionFactor(limb.body);
        }

        if (Time.time >= PPOBridge.ShrapnelStartSound)
        {
            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 1f;
            }
        }

        if (limb.shrapnel <= 0)
        {
            PPOBridge.LastShrapnel = null;
            MinigameBase.main.EndMinigame();
        }
    }

    static void HandleDislocation(DislocationMinigame d)
    {
        Limb limb = Dislocation_Limb(d);

        if (d != PPOBridge.LastDislocation)
        {
            PPOBridge.LastDislocation = d;

            PPOBridge.DislocationRemaining = limb.dislocationTimer;

            PPOBridge.DislocationHitsRemaining =
                Mathf.Clamp(
                    Mathf.RoundToInt(limb.dislocationTimer / 20f),
                    3,
                    8
                );

            PPOBridge.NextDislocationHit = Time.time + UnityEngine.Random.Range(0.5f, 1.5f) * (1f / MinigameSpeedFactor(limb.body));
        }

        if (limb.body.averagePain > 75f)
        {
            PPOBridge.LastDislocation = null;
            MinigameBase.main.EndMinigame();
            return;
        }

        if (Time.time >= PPOBridge.NextDislocationHit)
        {
            Sound.Play(
                "boneHit",
                Vector2.zero,
                twoDimensional: true,
                pitchShift: true,
                null,
                0.4f
            );

            if (d.hasWrench)
            {
                limb.pain += UnityEngine.Random.Range(4f, 10f);
            }
            else
            {
                limb.pain += UnityEngine.Random.Range(15f, 24f);

                if (UnityEngine.Random.value > 0.995f)
                {
                    limb.BreakBone();

                    PPOBridge.LastDislocation = null;
                    MinigameBase.main.EndMinigame();
                    return;
                }
            }

            float reduction = MinigameSpeedFactor(limb.body) * PPOBridge.DislocationRemaining / PPOBridge.DislocationHitsRemaining * UnityEngine.Random.Range(.8f, 1.2f);

            PPOBridge.DislocationRemaining -= reduction;
            PPOBridge.DislocationRemaining = Mathf.Clamp(PPOBridge.DislocationRemaining, 0f, 100f);
            PPOBridge.DislocationHitsRemaining--;
            
            limb.dislocationTimer = PPOBridge.DislocationRemaining;

            PPOBridge.NextDislocationHit = Time.time + UnityEngine.Random.Range(0.4f, 1.0f)  * (1f / MinigameSpeedFactor(limb.body));
        }

        if (PPOBridge.DislocationRemaining <= 3f || PPOBridge.DislocationHitsRemaining <= 0)
        {
            PPOBridge.LastDislocation = null;

            limb.UnDislocate();
            MinigameBase.main.EndMinigame();
        }
    }

    static void HandleSyringe(SyringeMinigame sy)
    {
        if (sy != PPOBridge.LastSyringe)
        {
            PPOBridge.LastSyringe = sy;

            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 1f;
            }
        }

        Item item = Minigame.game.currentItem;

        if (!item)
        {
            PPOBridge.LastSyringe = null;
            MinigameBase.main.EndMinigame();
            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 0f;
            }
            return;
        }

        sy.OnUse(Time.deltaTime);

        if (item.condition <= 0f)
        {
            PPOBridge.LastSyringe = null;
            MinigameBase.main.EndMinigame();
            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 0f;
            }
            return;
        }
    }

    static void HandleAED(AEDMinigame a)
    {
        Item item = Minigame.game.currentItem;

        if (a != PPOBridge.LastAED)
        {
            PPOBridge.LastAED = a;

            PPOBridge.AEDShockTime = Time.time + 7.75f; // 4.25 analysis + 3.5 charge
            PPOBridge.AEDAnalysisTime = Time.time + 4.25f;

            PPOBridge.AEDAnalyzed = false;

            Sound.Play("aedstart", Vector2.zero, true, false);

            item.battery.DrainCharge(0.01f);
        }

        if (!item || !item.battery.hasCharge)
        {
            PPOBridge.LastAED = null;
            MinigameBase.main.EndMinigame();
            return;
        }

        if (!PPOBridge.AEDAnalyzed && Time.time >= PPOBridge.AEDAnalysisTime)
        {
            PPOBridge.AEDAnalyzed = true;
            item.battery.DrainCharge(0.02f);

            if (a.limb.body.fibrillationProgress <= 0f && a.limb == a.limb.body.limbs[1])
            {
                PPOBridge.LastAED = null;
                MinigameBase.main.EndMinigame();
                Sound.Play("aedfail", Vector2.zero, true, false);
                return;
            }
            else
            {
                Sound.Play("aedstart", Vector2.zero, true, false);
                Sound.Play("aedcharge", Vector2.zero, true, false);
            }
        }

        if (Time.time >= PPOBridge.AEDShockTime)
        {
            item.Defibrillate(new Item.DefibInfo
            {
                chance = 1f,
                limb = a.limb
            });

            item.battery.DrainCharge(0.14f);

            PPOBridge.LastAED = null;
            MinigameBase.main.EndMinigame();
        }
    }

    static void HandleManualDefib(ManualDefibMinigame md)
    {
        Item item = Minigame.game.currentItem;

        if (md != PPOBridge.LastManualDefib)
        {
            PPOBridge.LastManualDefib = md;

            PPOBridge.ManualDefibCharge =
                Mathf.Clamp(
                    md.limb.body.fibrillationProgress * 2f,
                    10f,
                    200f
                );

            PPOBridge.ManualDefibShockTime =
                Time.time + PPOBridge.ManualDefibCharge / 40f;
        }

        if (!item || !item.battery.hasCharge)
        {
            PPOBridge.LastManualDefib = null;
            MinigameBase.main.EndMinigame();
            return;
        }

        item.battery.DrainCharge(Time.deltaTime / 800f);

        if (Time.time >= PPOBridge.ManualDefibShockTime)
        {
            item.Defibrillate(new Item.DefibInfo
            {
                chance = 1f,
                limb = md.limb
            });

            item.battery.DrainCharge(
                PPOBridge.ManualDefibCharge / 4000f
            );

            Sound.Play(
                "manualdefib",
                Vector2.zero,
                true,
                false
            );

            PPOBridge.LastManualDefib = null;
            MinigameBase.main.EndMinigame();
        }
    }

    static void HandleAmputation(AmputationMinigame a)
    {
        Limb limb = a.limb;

        if (a != PPOBridge.LastAmputation)
        {
            PPOBridge.LastAmputation = a;

            PPOBridge.AmputationEndTime = Time.time + UnityEngine.Random.Range(4f, 8f) * MinigameSpeedFactor(limb.body);

            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 1f;
            }
        }

        limb.body.adrenaline = Mathf.Max(limb.body.adrenaline, 60f);

        limb.pain += 6f * Time.deltaTime;
        limb.skinHealth -= 12f * Time.deltaTime;
        limb.muscleHealth -= 12f * Time.deltaTime;
        limb.bleedAmount += 6f * Time.deltaTime;

        if (Time.time >= PPOBridge.AmputationEndTime)
        {
            limb.Dismember();

            foreach (Limb connected in limb.connectedLimbs)
            {
                connected.infected = false;
                connected.infectionAmount = 0f;
                connected.SetDisinfect(300f);
                connected.bleedAmount *= 0.5f;
            }

            limb.body.traumaAmount -= 20f;

            foreach (AudioSource source in Minigame.game.spawnedMiniGame.GetComponents<AudioSource>())
            {
                source.volume = 0f;
            }

            PPOBridge.LastAmputation = null;
            MinigameBase.main.EndMinigame();
        }
    }

    static float MinigameSpeedFactor(Body body)
    {
        float speedFactor = (4f + body.skills.STRFrom10 * 0.3f) / 7f * body.consciousness * 0.01f;

        return speedFactor;
    }

    static float MinigamePrecisionFactor(Body body)
    {
        float shake =
            body.averagePain *
            Mathf.Clamp01(1f - body.skills.RESFrom10 * 0.06f);

        return Mathf.Clamp(1f - shake * 0.01f, 0.1f, 1f);
    }
}
