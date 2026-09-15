using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using CB1.Counterfactual;
using Newtonsoft.Json;
using UnityEngine;

public static partial class PPOBridge
{
    static readonly bool CounterfactualMode = ConfiguredBoolean("PPO_COUNTERFACTUAL", false);
    // The checkpoint-preparation experiment deliberately leaves the process
    // in the transport-detached hold. Ordinary detach/attach tests retain
    // automatic reconnect behavior unless this explicit opt-in is set.
    static readonly bool CounterfactualFinalDetach =
        ConfiguredBoolean("PPO_COUNTERFACTUAL_FINAL_DETACH", false);
    static bool counterfactualInsideBoundary;
    static bool counterfactualPoisoned;
    static string counterfactualOutput;
    static int counterfactualReportIndex;
    static SemanticSnapshot counterfactualSnapshot;
    static bool counterfactualPostRestoreBoundaryPending;
    // Transport-only hold for the checkpoint experiment. This is deliberately
    // excluded from the causal snapshot: it controls only when the two local
    // protocol streams are re-established while the policy boundary remains
    // paused.
    static bool counterfactualTransportDetached;
    static long counterfactualReconnectNotBefore;
    static readonly Queue<string> CounterfactualCommands = new Queue<string>();

    // Transport state stays monotonic and OUTSIDE the rewound simulation.
    // Opt-in owns the connection: ordinary Server.py cannot start training here.
    static bool CounterfactualOwnsConnection(string line)
    {
        if (!CounterfactualMode)
        {
            if (!line.StartsWith("CF_", StringComparison.Ordinal)) return false;
            SendActionAcknowledgement("CF " + JsonConvert.SerializeObject(new { ok = false, error = "PPO_COUNTERFACTUAL=1 is required" }));
            return true;
        }
        if (line == "SHUTDOWN") return false;
        lock (ActionLock)
        {
            if (CounterfactualCommands.Count >= 4)
            {
                actionReaderFailureMessage = "Counterfactual command queue overflow";
                actionReaderFailed = true;
            }
            else CounterfactualCommands.Enqueue(line);
            System.Threading.Monitor.PulseAll(ActionLock);
        }
        return true;
    }
    static bool CounterfactualRuntimeReadyToConnect(Body body)
    {
        var w = WorldGeneration.world;
        return w != null && w.worldExists && !w.generatingWorld &&
            (w.loadingObject == null || !w.loadingObject.activeSelf) &&
            body != null && body.alive && body.conscious &&
            CounterfactualPresentationReady();
    }
    static void AssertCounterfactualBoundary()
    {
        if (!CounterfactualMode || !counterfactualInsideBoundary || !awaitingPolicyAction ||
            !resetComplete || initialObservationPending || resetRequested || shutdownRequested ||
            worldSeedApplicationPending || resetObservationAcknowledgementPending ||
            ppoPauseRequested || ppoPaused || latestActionSequence != activeActionSequence)
            throw new InvalidOperationException("Not a quiescent, fully published policy boundary");
        if (ControlEnabled || !ForceBodyUpdatePerPhysicsTick || NaturalStamina ||
            IncludeUnusedObservations || IncludeLimbObservations)
            throw new InvalidOperationException("Unsupported CB1 configuration");
        var w = WorldGeneration.world;
        var b = PlayerCamera.main == null ? null : PlayerCamera.main.body;
        var failures = new List<string>();
        if (w == null) failures.Add("world=null");
        else
        {
            if (!w.worldExists) failures.Add("worldExists=false");
            if (w.generatingWorld) failures.Add("generatingWorld=true");
            if (w.loadingObject != null && w.loadingObject.activeSelf) failures.Add("loadingObject=active");
            if (w.earthquakeIntensity != 0f) failures.Add("earthquakeIntensity!=0");
            if (w.earthquakeTime > 0f) failures.Add("earthquakeTime>0");
            if (w.maxTimePerLayer != CompletionProtocolLayerTimeLimitSeconds)
                failures.Add("completionProtocol=not-installed");
        }
        if (b == null) failures.Add("body=null");
        else
        {
            if (!b.alive) failures.Add("body.alive=false");
            if (!b.conscious) failures.Add("body.conscious=false");
        }
        if (MinigameBase.main == null) failures.Add("minigameManager=null");
        else if (MinigameBase.main.currentMinigame) failures.Add("minigame=active");
        if (RadiationLine.line == null) failures.Add("radiationLine=null");
        else if (RadiationLine.line.active) failures.Add("radiationLine=active");
        if (failures.Count > 0)
            throw new InvalidOperationException("CB1 runtime guards failed: " + string.Join(", ", failures));
    }
    static readonly string[] CounterfactualHarnessFields = {
        "CurrentAction", "LastAction", "LatestAction", "latestActionSequence", "activeActionSequence",
        "newMacrostepAction", "macrostepPhysicsTicks", "macrostepSimulationDeltaTime",
        "macrostepDeepestLayerProgress", "macrostepAttackSwingsExecuted", "macrostepAttackContactClawHealthLoss",
        "nextObservationId", "awaitingPolicyAction", "initialObservationPending", "policyBoundaryPaused",
        "policyResumeTimeScale", "ppoPauseRequested", "ppoPaused", "ppoResumeActionReceived", "prePPOPauseTimeScale",
        "resetRequested", "requestedResetToken", "activeResetToken", "requestedWorldSeed", "activeWorldSeed",
        "appliedWorldSeed", "seededWorldInstanceId", "worldSeedApplicationPending",
        "resetObservationAcknowledgementPending", "resetSourceWorldInstanceId", "resetSourceBodyInstanceId",
        "generationFinishedWorldInstanceId",
        "resetComplete", "BestLayerDepth", "ThrowCharge", "configuredLayerWorldInstanceId",
        // The serialized observation wire packet is captured separately. These
        // remaining observation-related roots are causal bookkeeping, not DTO
        // scratch: sound candidates, IDs, actions, depth, and damage tracking
        // are consumed by the next boundary or by CollectObservations.
        "SoundEvents", "BlockDamageByPosition",
        "StartRan", "ItemNameList", "LiquidNameList", "QualityNameList", "SoundNameList",
        "worldGenerationExceptionCount"
    };
    static SemanticSnapshot CaptureCounterfactualState(bool detailed)
    {
        long startedAt = CounterfactualTiming.Start();
        try
        {
            AssertCounterfactualBoundary();
            if (counterfactualPostRestoreBoundaryPending && counterfactualSnapshot != null)
            {
                counterfactualSnapshot.WriteAnimatorDiagnostic("first_policy_boundary_after_restore");
                counterfactualPostRestoreBoundaryPending = false;
            }
            long observationCopyStartedAt = CounterfactualTiming.Start();
            byte[] observation = new byte[bufferWriter.BytesWritten];
            Array.Copy(bufferWriter.Buffer, observation, observation.Length);
            CounterfactualTiming.End("capture.observation_copy", observationCopyStartedAt);
            return new SemanticSnapshot(counterfactualOutput, activeWorldSeed, observation, graph => {
                foreach (var name in CounterfactualHarnessFields)
                {
                    var f = typeof(PPOBridge).GetField(name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                    if (f == null) throw new MissingFieldException(typeof(PPOBridge).FullName, name);
                    graph.Field(null, f, "harness/" + name);
                }
            }, detailed);
        }
        finally { CounterfactualTiming.End("CaptureCounterfactualState", startedAt); }
    }
    static void RestoreCounterfactualState(SemanticSnapshot snapshot)
    {
        AssertCounterfactualBoundary();
        if (snapshot == null) throw new InvalidOperationException("No captured state");
        snapshot.Restore();
        snapshot.WriteAnimatorDiagnostic("after_restore");
        counterfactualPostRestoreBoundaryPending = true;
        // Restore the exact baseline wire packet without recollecting it:
        // CollectObservations flushes sound events and updates depth bookkeeping.
        bufferWriter.LoadSerialized(snapshot.BaselineObservationWire);
        // No packet is sent here. The isolated client rewinds its own baseline
        // observation; the next packet will follow the next ordinary physics step.
    }
    static object CounterfactualReport(SemanticSnapshot s, bool detailed)
    {
        string filename = (counterfactualReportIndex++).ToString("D6") + ".json";
        long reportStartedAt = CounterfactualTiming.Start();
        using (var f = new StreamWriter(new FileStream(Path.Combine(counterfactualOutput, filename),
            FileMode.CreateNew, FileAccess.Write, FileShare.Read)))
        using (var writer = new JsonTextWriter(f))
        {
            var serializer = JsonSerializer.CreateDefault();
            long observationHashStartedAt = CounterfactualTiming.Start();
            string observationWireHash = SemanticSnapshot.Hash(s.BaselineObservationWire);
            string observationHash = SemanticSnapshot.Hash(s.BaselineObservationWire);
            CounterfactualTiming.End("observation.hash", observationHashStartedAt);
            if (detailed)
                serializer.Serialize(writer, new {
                    schema = SemanticSnapshot.Schema, state = s.State, arrays = s.Arrays,
                    coverage_gaps = s.Gaps, engine_clock = s.Clock,
                    state_sha256 = s.StateHash,
                    seed = s.Seed,
                    observation_wire_sha256 = observationWireHash,
                    observation_sha256 = observationHash,
                    observation_base64 = Convert.ToBase64String(s.BaselineObservationWire),
                    managed_reference_identity_nodes = s.Graph.UniqueReferenceIdentities,
                    managed_reference_identity_reuses = s.Graph.ReusedReferenceIdentities,
                    // A field match never silently promotes this candidate to trusted A/B.
                    scientifically_validated = false
                });
            else
                serializer.Serialize(writer, new {
                    schema = SemanticSnapshot.Schema, state_sections = s.SectionHashes,
                    coverage_gap_count = s.Gaps.Count, state_sha256 = s.StateHash,
                    observation_wire_sha256 = observationWireHash,
                    observation_sha256 = observationHash,
                    observation_base64 = Convert.ToBase64String(s.BaselineObservationWire),
                    managed_reference_identity_nodes = s.Graph.UniqueReferenceIdentities,
                    managed_reference_identity_reuses = s.Graph.ReusedReferenceIdentities,
                    scientifically_validated = false
                });
        }
        CounterfactualTiming.End("CounterfactualReport.json", reportStartedAt);
        return new { ok = true, file = filename, coverage_gaps = s.Gaps.Count, scientifically_validated = false };
    }
    static void ProcessCounterfactualCommands()
    {
        if (!CounterfactualMode) return;
        CounterfactualTiming.ProcessInvocation();
        string command;
        lock (ActionLock)
        {
            if (CounterfactualCommands.Count == 0) return;
            command = CounterfactualCommands.Dequeue();
        }
        string commandName = command.Split(' ')[0];
        if (commandName == "CF_STEP") CounterfactualTiming.StartPolicyStep();
        CounterfactualTiming.Invocation(commandName);
        long commandStartedAt = CounterfactualTiming.BeginCommand(commandName);
        object reply;
        bool detachAfterReply = false;
        try
        {
            if (counterfactualPoisoned) throw new InvalidOperationException("Diagnostic session invalidated; restart the game");
            string[] p = command.Split(' ');
            if (p[0] == "CF_HELLO" && p.Length == 2)
            {
                if (counterfactualOutput != null) throw new InvalidOperationException("Session already initialized");
                AssertCounterfactualBoundary();
                string token = p[1];
                if (token.Length < 8 || token.Length > 64 || token.Any(c => !(c >= 'a' && c <= 'z') && !(c >= '0' && c <= '9') && c != '-'))
                    throw new InvalidOperationException("Invalid diagnostic session token");
                string configuredRoot = ConfiguredValue(
                    "PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT",
                    "--ppo-counterfactual-diagnostics-root",
                    "counterfactual_diagnostics");
                string dir = Path.IsPathRooted(configuredRoot)
                    ? Path.Combine(configuredRoot, token, "unity")
                    : Path.Combine(Directory.GetCurrentDirectory(), configuredRoot, token, "unity");
                if (Directory.Exists(dir)) throw new IOException("Diagnostic directory already exists");
                Directory.CreateDirectory(Path.Combine(dir, "arrays"));
                counterfactualOutput = dir;
                CounterfactualTiming.Configure(dir);
                CaptureCounterfactualStartupDiagnostics("protocol_ready");
                reply = new { ok = true, schema = SemanticSnapshot.Schema,
                    dll_sha256 = SemanticSnapshot.FileHash(typeof(PPOBridge).Assembly.Location),
                    game_assembly_sha256 = SemanticSnapshot.FileHash(typeof(Body).Assembly.Location),
                    unity_version = Application.unityVersion,
                    time_scale = TrainingTimeScale, gameplay_hz = GameplayUpdateRate,
                    fixed_delta_time = Time.fixedDeltaTime, natural_stamina = NaturalStamina,
                    protocol = Observation.PROTOCOL_VERSION, observation_bytes = BinaryObservationWriter.ExpectedSize,
                    python_participates = false, scientific_status = "A/A required before A/B" };
            }
            else
            {
                if (counterfactualOutput == null) throw new InvalidOperationException("CF_HELLO required");
                if (new DirectoryInfo(counterfactualOutput).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length) > 2L * 1024 * 1024 * 1024)
                    throw new IOException("Diagnostic disk budget exceeded (2 GiB)");
                AssertCounterfactualBoundary();
                if (command == "CF_CAPTURE")
                {
                    if (counterfactualSnapshot != null) throw new InvalidOperationException("Snapshot already captured");
                    counterfactualSnapshot = CaptureCounterfactualState(true);
                    reply = CounterfactualReport(counterfactualSnapshot, true);
                }
                else if (command == "CF_RESTORE")
                {
                    try { RestoreCounterfactualState(counterfactualSnapshot); }
                    catch { counterfactualPoisoned = true; throw; }
                    reply = CounterfactualReport(CaptureCounterfactualState(true), true);
                }
                else if (command == "CF_STATE_DETAIL")
                {
                    // Prefix replay uses this only when a cheap full-coverage
                    // fingerprint mismatch needs an exact path-level report.
                    // It never restores or changes simulation state.
                    reply = CounterfactualReport(CaptureCounterfactualState(true), true);
                }
                else if (command == "CF_STATE") reply = CounterfactualReport(CaptureCounterfactualState(false), false);
                else if (command == "CF_DETACH")
                {
                    if (counterfactualTransportDetached)
                        throw new InvalidOperationException("Counterfactual transport is already detached");
                    AssertCounterfactualBoundary();
                    counterfactualTransportDetached = true;
                    counterfactualReconnectNotBefore =
                        System.Diagnostics.Stopwatch.GetTimestamp() +
                        System.Diagnostics.Stopwatch.Frequency / 2;
                    reply = new {
                        ok = true,
                        detached = true,
                        hold = true,
                        reconnect_after_ms = 500,
                        final_detach = CounterfactualFinalDetach,
                        reset_used = false
                    };
                    detachAfterReply = true;
                }
                else if (command == "CF_ATTACH")
                {
                    if (!counterfactualTransportDetached)
                        throw new InvalidOperationException("No counterfactual transport detach is pending");
                    if (!connected || !actionConnected)
                        throw new InvalidOperationException("Both counterfactual channels must reconnect before CF_ATTACH");
                    AssertCounterfactualBoundary();
                    counterfactualTransportDetached = false;
                    reply = new { ok = true, attached = true, reset_used = false };
                }
                else if (p[0] == "CF_RESET" && p.Length == 2)
                {
                    if (counterfactualSnapshot != null) throw new InvalidOperationException("Cannot reset a captured session");
                    if (!int.TryParse(p[1], out int seed) || seed < 0) throw new InvalidOperationException("Invalid seed");
                    requestedWorldSeed = seed; requestedResetToken++; resetRequested = true;
                    reply = new { ok = true, reset_token = requestedResetToken, seed };
                }
                else if (p[0] == "CF_STEP" && p.Length == 8)
                {
                    int[] v = p.Skip(1).Select(int.Parse).ToArray();
                    int[] sizes = { 3, 2, 3, 2, 9, 11, 2 };
                    if (v.Where((x, i) => x < 0 || x >= sizes[i]).Any()) throw new InvalidOperationException("Action outside CB1 MultiDiscrete");
                    lock (ActionLock)
                    {
                        LatestAction = new Action {
                            MoveDirection = v[0] - 1, Jump = v[1], VerticalMovement = v[2] - 1,
                            Crouch = v[3], LookDX = v[4] - 4, LookDY = v[5] - 5, Attack = v[6],
                            SelectedItemSlot = -1, SelectedBagIndex = -1, SelectedRecipe = -1, Exercise = -1
                        };
                        latestActionSequence = activeActionSequence + 1;
                    }
                    reply = new { ok = true, action_sequence = latestActionSequence };
                }
                else throw new InvalidOperationException("Unsupported command; ordinary training/action traffic is forbidden in counterfactual mode");
            }
        }
        catch (Exception e)
        {
            reply = new { ok = false, error = e.ToString(), poisoned = counterfactualPoisoned };
        }
        SendActionAcknowledgement("CF " + JsonConvert.SerializeObject(reply));
        if (detachAfterReply)
            DetachCounterfactualConnections();
        CounterfactualTiming.EndCommand(commandStartedAt);
    }
}
