using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using CB1.Counterfactual;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

// The normal menu remains the source of truth for run setup.  This file only
// waits for that menu's own PreRunScript, applies the same preset/value model
// the controls use, and invokes its ordinary StartRun button method.
public static partial class PPOBridge
{
    static bool autoStartAttempted;
    static string autoStartStatusPath;
    static int autoStartDiagnosticIndex;
    static bool autoStartCandidateCaptured;
    static string autoStartPhase = "not_started";

    public static bool AutoStartEnabled
    {
        get { return CounterfactualMode || ConfiguredBoolean("PPO_AUTOSTART", false) || HasCommandLineFlag("--ppo-autostart"); }
    }

    public static IEnumerator RunAutoStart()
    {
        if (!AutoStartEnabled || autoStartAttempted)
            yield break;

        autoStartAttempted = true;
        autoStartCandidateCaptured = false;
        autoStartPhase = "loading_configuration";
        autoStartStatusPath = ConfiguredValue(
            "PPO_AUTOSTART_STATUS_FILE",
            "--ppo-autostart-status-file",
            Path.Combine(Directory.GetCurrentDirectory(), "counterfactual_autostart_status.json"));

        AutoStartOptions options;
        try
        {
            options = AutoStartOptions.Load();
            WriteAutoStartStatus("configuring", null, options);
            Debug.Log("AUTO_START_CONFIG " + options.Describe());
        }
        catch (Exception ex)
        {
            FailAutoStart("invalid configuration: " + Unwrap(ex).Message, null);
            yield break;
        }

        float deadline = Time.realtimeSinceStartup + options.TimeoutSeconds;
        autoStartPhase = "waiting_for_menu";
        PreRunScript preRun = null;
        Dictionary<string, object> settings = null;
        while (Time.realtimeSinceStartup < deadline)
        {
            preRun = FindPreRunScript();
            settings = preRun == null ? null : ReadSettings(preRun);
            if (preRun != null && settings != null && settings.Count > 0)
                break;
            yield return null;
        }

        if (preRun == null || settings == null || settings.Count == 0)
        {
            FailAutoStart("PreRunScript/runSettings did not become ready before timeout: " + StartupDiagnostics(preRun), preRun, options);
            yield break;
        }

        try
        {
            autoStartPhase = "applying_configuration";
            ApplyConfiguration(preRun, settings, options);
            CaptureAutoStartDiagnostics("after_config_applied", preRun, options);
            SetPrivateBoolean(preRun, "didIntro", true);
            InvokeRequired(preRun, "UpdateAllSettingDisplays");
            WriteAutoStartStatus("ready", null, options);
            Debug.Log("AUTO_START_READY " + StartupDiagnostics(preRun));
        }
        catch (Exception ex)
        {
            FailAutoStart("run configuration failed: " + Unwrap(ex), preRun, options);
            yield break;
        }

        // ScreenCapture.CaptureScreenshot is an end-of-frame operation.  Give
        // the configuration milestone one frame to materialize before the
        // next milestone is requested, otherwise Unity may drop one of two
        // screenshots queued from the same frame.
        yield return null;

        try
        {
            autoStartPhase = "invoking_start_run";
            InvokeRequired(preRun, "StartRun");
            CaptureAutoStartDiagnostics("after_start_run_invoked", preRun, options);
            WriteAutoStartStatus("triggered", null, options);
            Debug.Log("AUTO_START_TRIGGERED");
        }
        catch (Exception ex)
        {
            FailAutoStart("run-start invocation failed: " + Unwrap(ex), preRun, options);
            yield break;
        }

        while (Time.realtimeSinceStartup < deadline)
        {
            autoStartPhase = "waiting_for_run_ready";
            Body body = PlayerCamera.main == null ? null : PlayerCamera.main.body;
            if (CounterfactualRuntimeReadyToConnect(body))
            {
                List<string> wrongState;
                if (!autoStartCandidateCaptured)
                {
                    CaptureAutoStartDiagnostics("autostart_complete_candidate", preRun, options);
                    autoStartCandidateCaptured = true;
                }
                if (CounterfactualRunReady(preRun, body, out wrongState))
                {
                    CaptureAutoStartDiagnostics("complete", preRun, options);
                    WriteAutoStartStatus("complete", null, options);
                    autoStartPhase = "complete";
                    Debug.Log("AUTO_START_COMPLETE " + StartupDiagnostics(preRun));
                    yield break;
                }

                // A world/body that exists is not sufficient.  If a known
                // blocking UI or gameplay mode is present, fail with the
                // preserved screenshot/state report instead of allowing the
                // protocol to run against a different entry state.
                if (wrongState.Any(IsBlockingEntryState))
                {
                    FailAutoStart("AUTO_START_WRONG_STATE: " + string.Join(", ", wrongState), preRun, options);
                    yield break;
                }
            }
            yield return null;
        }

        FailAutoStart("simulation did not become counterfactual-ready before timeout: " + StartupDiagnostics(preRun), preRun, options);
    }

    // Called by the plugin's ordinary Unity Update loop. This is deliberately
    // throttled and counterfactual-only: it proves whether frames continue to
    // execute while startup is waiting, without changing startup state.
    public static void RecordAutoStartHeartbeat()
    {
        if (!AutoStartEnabled || !autoStartAttempted) return;
        float now = Time.realtimeSinceStartup;
        if (now < nextAutoStartHeartbeat) return;
        nextAutoStartHeartbeat = now + 5f;
        Debug.Log("AUTO_START_HEARTBEAT phase=" + autoStartPhase +
            " frame=" + Time.frameCount + " realtime=" + now.ToString("F3", CultureInfo.InvariantCulture) +
            " " + StartupDiagnostics(FindPreRunScript()));
    }

    static float nextAutoStartHeartbeat;

    sealed class AutoStartOptions
    {
        public string Preset;
        public Dictionary<string, JToken> Overrides;
        public string Source;
        public float TimeoutSeconds;

        public static AutoStartOptions Load()
        {
            string preset = ConfiguredValue("PPO_AUTOSTART_PRESET", "--ppo-autostart-preset", null);
            string settingsText = Environment.GetEnvironmentVariable("PPO_AUTOSTART_SETTINGS");
            string settingsFile = ConfiguredValue("PPO_AUTOSTART_SETTINGS_FILE", "--ppo-autostart-settings-file", null);
            string commandSettings = CommandLineValue("--ppo-autostart-settings");
            if (!string.IsNullOrWhiteSpace(commandSettings))
            {
                if (!string.IsNullOrWhiteSpace(settingsText))
                    throw new InvalidOperationException("PPO_AUTOSTART_SETTINGS and --ppo-autostart-settings were both supplied");
                settingsText = commandSettings;
            }
            if (!string.IsNullOrWhiteSpace(settingsFile))
            {
                if (!string.IsNullOrWhiteSpace(settingsText))
                    throw new InvalidOperationException("settings JSON and settings file were both supplied");
                settingsText = File.ReadAllText(settingsFile);
            }

            var overrides = new Dictionary<string, JToken>(StringComparer.Ordinal);
            if (!string.IsNullOrWhiteSpace(settingsText))
            {
                JObject obj = JObject.Parse(settingsText);
                foreach (var property in obj.Properties())
                    overrides.Add(property.Name, property.Value);
            }

            string timeoutText = ConfiguredValue("PPO_AUTOSTART_TIMEOUT_SECONDS", "--ppo-autostart-timeout-seconds", "180");
            float timeout;
            if (!float.TryParse(timeoutText, NumberStyles.Float, CultureInfo.InvariantCulture, out timeout) ||
                float.IsNaN(timeout) || float.IsInfinity(timeout) || timeout <= 0f)
                throw new InvalidOperationException("PPO_AUTOSTART_TIMEOUT_SECONDS must be a finite positive number");

            return new AutoStartOptions {
                Preset = string.IsNullOrWhiteSpace(preset) ? null : preset,
                Overrides = overrides,
                Source = string.IsNullOrWhiteSpace(settingsText) ? "menu-defaults" : "json",
                TimeoutSeconds = timeout
            };
        }

        public string Describe()
        {
            string values = JsonConvert.SerializeObject(Overrides.ToDictionary(p => p.Key, p => p.Value));
            return "enabled=1 preset=" + (Preset ?? "<current>") +
                " settings_source=" + Source + " overrides=" + values +
                " timeout_seconds=" + TimeoutSeconds.ToString("F1", CultureInfo.InvariantCulture);
        }
    }

    static void ApplyConfiguration(PreRunScript preRun, Dictionary<string, object> settings, AutoStartOptions options)
    {
        if (options.Preset != null)
        {
            // ApplyPreset is the same method used by the preset UI.  The game
            // currently throws from RunSettings.GetPreset for an unknown name;
            // that exception is deliberately allowed to hard-fail startup.
            InvokeRequired(preRun, "ApplyPreset", options.Preset);
            Dictionary<string, object> afterPreset = ReadSettings(preRun);
            if (afterPreset == null || afterPreset.Count == 0)
                throw new InvalidOperationException("ApplyPreset left runSettings unavailable");
            settings = afterPreset;
        }

        var converted = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var pair in options.Overrides)
        {
            object current;
            if (!settings.TryGetValue(pair.Key, out current))
                throw new InvalidOperationException("unknown run setting '" + pair.Key + "'; available=" + string.Join(",", settings.Keys.OrderBy(k => k, StringComparer.Ordinal)));
            object convertedValue = ConvertSetting(pair.Key, pair.Value, current);
            ValidateSetting(preRun, pair.Key, convertedValue);
            converted[pair.Key] = convertedValue;
        }
        foreach (var pair in converted)
            settings[pair.Key] = pair.Value;

        Debug.Log("AUTO_START_SETTINGS_APPLIED keys=" + string.Join(",", settings.Keys.OrderBy(k => k, StringComparer.Ordinal)));
    }

    static object ConvertSetting(string name, JToken token, object current)
    {
        if (current == null)
            throw new InvalidOperationException("setting '" + name + "' has no initialized type");
        Type type = current.GetType();
        try
        {
            object value;
            if (type == typeof(float)) value = token.ToObject<float>();
            else if (type == typeof(double)) value = token.ToObject<double>();
            else if (type == typeof(int)) value = token.ToObject<int>();
            else if (type == typeof(long)) value = token.ToObject<long>();
            else if (type == typeof(bool))
            {
                if (token.Type != JTokenType.Boolean) throw new InvalidDataException("must be boolean");
                value = token.ToObject<bool>();
            }
            else if (type == typeof(string))
            {
                if (token.Type != JTokenType.String) throw new InvalidDataException("must be string");
                value = token.ToObject<string>();
            }
            else if (type.IsEnum) value = Enum.Parse(type, token.ToString(), true);
            else value = token.ToObject(type);

            if (value is float f && (float.IsNaN(f) || float.IsInfinity(f))) throw new InvalidDataException("must be finite");
            if (value is double d && (double.IsNaN(d) || double.IsInfinity(d))) throw new InvalidDataException("must be finite");
            return value;
        }
        catch (Exception ex)
        {
            throw new InvalidDataException("setting '" + name + "' has invalid value " + token.ToString(Formatting.None) + ": " + Unwrap(ex).Message);
        }
    }

    static void ValidateSetting(PreRunScript preRun, string name, object value)
    {
        FieldInfo displaysField = typeof(PreRunScript).GetField("runSettingObjects",
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        var displays = displaysField == null ? null : displaysField.GetValue(preRun) as System.Collections.IEnumerable;
        if (displays == null) return;
        foreach (object display in displays)
        {
            if (display == null) continue;
            FieldInfo associatedField = display.GetType().GetField("associated",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            object descriptor = associatedField == null ? null : associatedField.GetValue(display);
            if (descriptor == null) continue;
            FieldInfo nameField = descriptor.GetType().GetField("name",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (!string.Equals(nameField == null ? null : nameField.GetValue(descriptor) as string,
                name, StringComparison.Ordinal)) continue;

            if (descriptor.GetType().Name == "RunSettingFloat" && value is IConvertible)
            {
                double number = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                FieldInfo limitsField = descriptor.GetType().GetField("limits",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                object limits = limitsField == null ? null : limitsField.GetValue(descriptor);
                double minimum;
                double maximum;
                if (TryNumericMember(limits, "min", out minimum) && TryNumericMember(limits, "max", out maximum) &&
                    (number < minimum || number > maximum))
                    throw new InvalidDataException("setting '" + name + "' is outside its allowed range [" + minimum.ToString(CultureInfo.InvariantCulture) + "," + maximum.ToString(CultureInfo.InvariantCulture) + "]");
                FieldInfo wholeNumField = descriptor.GetType().GetField("wholeNum",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (wholeNumField != null && (bool)wholeNumField.GetValue(descriptor) && number != Math.Truncate(number))
                    throw new InvalidDataException("setting '" + name + "' must be a whole number");
            }
            else if (descriptor.GetType().Name == "RunSettingDropdown")
            {
                FieldInfo choicesField = descriptor.GetType().GetField("choices",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                string[] choices = choicesField == null ? null : choicesField.GetValue(descriptor) as string[];
                if (choices != null)
                {
                    if (value is int index && (index < 0 || index >= choices.Length))
                        throw new InvalidDataException("setting '" + name + "' dropdown index is outside its choices");
                    if (value is string choice && !choices.Contains(choice, StringComparer.Ordinal))
                        throw new InvalidDataException("setting '" + name + "' is not a valid dropdown choice: " + choice);
                }
            }
            return;
        }
    }

    static bool TryNumericMember(object target, string name, out double value)
    {
        value = 0d;
        if (target == null) return false;
        Type type = target.GetType();
        FieldInfo field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        object raw = field == null ? null : field.GetValue(target);
        if (raw == null)
        {
            PropertyInfo property = type.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            raw = property == null ? null : property.GetValue(target, null);
        }
        if (!(raw is IConvertible)) return false;
        value = Convert.ToDouble(raw, CultureInfo.InvariantCulture);
        return true;
    }

    static PreRunScript FindPreRunScript()
    {
        return Resources.FindObjectsOfTypeAll<PreRunScript>()
            .Where(x => x != null && x.gameObject != null && x.gameObject.scene.IsValid() && x.gameObject.scene.isLoaded)
            .OrderByDescending(x => x.gameObject.activeInHierarchy)
            .FirstOrDefault();
    }

    static Dictionary<string, object> ReadSettings(PreRunScript preRun)
    {
        FieldInfo field = typeof(PreRunScript).GetField("runSettings", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        return field == null ? null : field.GetValue(preRun) as Dictionary<string, object>;
    }

    static void SetPrivateBoolean(object target, string name, bool value)
    {
        FieldInfo field = target.GetType().GetField(name,
            BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
        if (field == null || field.FieldType != typeof(bool)) throw new MissingFieldException(target.GetType().FullName, name);
        field.SetValue(field.IsStatic ? null : target, value);
    }

    static void InvokeRequired(object target, string name, params object[] arguments)
    {
        MethodInfo method = target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (method == null) throw new MissingMethodException(target.GetType().FullName, name);
        method.Invoke(target, arguments);
    }

    static string StartupDiagnostics(PreRunScript preRun)
    {
        WorldGeneration world = WorldGeneration.world;
        Body body = PlayerCamera.main == null ? null : PlayerCamera.main.body;
        return "scene=" + SceneManager.GetActiveScene().name +
            " preRun=" + (preRun == null ? "null" : preRun.name) +
            " world=" + (world == null ? "null" : world.GetInstanceID().ToString()) +
            " worldExists=" + (world != null && world.worldExists) +
            " generating=" + (world != null && world.generatingWorld) +
            " loading=" + (world != null && world.loadingObject != null && world.loadingObject.activeSelf) +
            " body=" + (body == null ? "null" : body.GetInstanceID().ToString()) +
            " alive=" + (body != null && body.alive) + " conscious=" + (body != null && body.conscious);
    }

    static bool CounterfactualRunReady(PreRunScript preRun, Body body, out List<string> failures)
    {
        failures = CounterfactualRunReadyFailures(preRun, body);
        return failures.Count == 0;
    }

    static List<string> CounterfactualRunReadyFailures(PreRunScript preRun, Body body)
    {
        var failures = new List<string>();
        Scene activeScene = SceneManager.GetActiveScene();
        if (!activeScene.IsValid() || !activeScene.isLoaded || !string.Equals(activeScene.name, "SampleScene", StringComparison.Ordinal))
            failures.Add("active_scene=" + (activeScene.IsValid() ? activeScene.name : "invalid"));

        WorldGeneration world = WorldGeneration.world;
        if (world == null) failures.Add("world_manager=null");
        else
        {
            if (!world.worldExists) failures.Add("world_exists=false");
            if (world.generatingWorld) failures.Add("generating_world=true");
            if (world.loadingObject != null && world.loadingObject.activeSelf) failures.Add("loading_object=active");
        }

        PlayerCamera player = PlayerCamera.main;
        if (player == null) failures.Add("player_camera=null");
        else
        {
            if (!player.enabled || !player.gameObject.activeInHierarchy) failures.Add("player_controller=inactive");
            if (body == null) failures.Add("body=null");
            else
            {
                if (!body.enabled || !body.gameObject.activeInHierarchy) failures.Add("body=inactive");
                if (!body.alive) failures.Add("body.alive=false");
                if (!body.conscious) failures.Add("body.conscious=false");
            }
            if (player.mainView == null || !player.mainView.activeInHierarchy) failures.Add("main_view=inactive");
            if (player.woundView != null && player.woundView.activeInHierarchy) failures.Add("medical.wound_view=active");
            if (player.craftingPanel != null && player.craftingPanel.activeInHierarchy) failures.Add("inventory.crafting_panel=active");
            if (player.tradeMenu != null && player.tradeMenu.activeInHierarchy) failures.Add("inventory.trade_menu=active");
            if (player.containerMenu != null && player.containerMenu.activeInHierarchy) failures.Add("inventory.container_menu=active");
            if (player.gunMenu != null && player.gunMenu.activeInHierarchy) failures.Add("inventory.gun_menu=active");
            if (player.radialMenu != null && player.radialMenu.gameObject.activeInHierarchy) failures.Add("inventory.radial_menu=active");
            if (player.radialOpen) failures.Add("inventory.radial_open=true");
            if (player.didDeathScreen) failures.Add("death_screen=true");
            if (player.selfDestructActive) failures.Add("self_destruct=true");
        }

        if (preRun != null) failures.Add("pre_run_still_present");
        if (MinigameBase.main == null) failures.Add("minigame_manager=null");
        else if (MinigameBase.main.currentMinigame) failures.Add("medical.minigame=" + MinigameBase.main.currentMinigame.GetType().FullName);
        if (PauseHandler.paused) failures.Add("pause_menu=active");
        if (SettingsMenu.instance != null) failures.Add("settings_menu=active");
        if (LiquidTransfer.instance != null) failures.Add("liquid_transfer=active");

        if (GlobalDark.main == null) failures.Add("global_dark=null");
        else
        {
            if (GlobalDark.main.IsDarkening()) failures.Add("scene_transition=darkening");
            if (GlobalDark.main.image != null && GlobalDark.main.image.color.a > 0.01f)
                failures.Add("scene_transition=overlay_alpha_" + GlobalDark.main.image.color.a.ToString("F3", CultureInfo.InvariantCulture));
        }
        return failures;
    }

    static bool CounterfactualPresentationReady()
    {
        return GlobalDark.main == null || GlobalDark.main.image == null ||
            GlobalDark.main.image.color.a <= 0.01f;
    }

    static bool IsBlockingEntryState(string failure)
    {
        return failure.StartsWith("medical.", StringComparison.Ordinal) ||
            failure.StartsWith("inventory.", StringComparison.Ordinal) ||
            failure.StartsWith("pause_menu", StringComparison.Ordinal) ||
            failure.StartsWith("settings_menu", StringComparison.Ordinal) ||
            failure.StartsWith("liquid_transfer", StringComparison.Ordinal) ||
            failure.StartsWith("death_screen", StringComparison.Ordinal) ||
            failure.StartsWith("self_destruct", StringComparison.Ordinal);
    }

    static void FailAutoStart(string message, PreRunScript preRun, AutoStartOptions options = null)
    {
        autoStartPhase = "failed";
        string diagnostic = message + " | " + StartupDiagnostics(preRun);
        CaptureAutoStartDiagnostics("failure_before_shutdown", preRun, options, diagnostic);
        WriteAutoStartStatus("failed", diagnostic, options);
        Debug.LogError("AUTO_START_FAILED " + diagnostic);
        Application.Quit(86);
    }

    // Unity-native startup evidence.  ScreenCapture writes from inside the
    // process; no desktop/session capture or automation is involved.  The
    // state report is intentionally small and explicit rather than a managed
    // graph dump, so a bad startup state can be inspected without adding a
    // second source of simulation state.
    public static void CaptureCounterfactualStartupDiagnostics(string label)
    {
        CaptureAutoStartDiagnostics(label, FindPreRunScript(), null, null);
    }

    static void CaptureAutoStartDiagnostics(string label, PreRunScript preRun, AutoStartOptions options, string reason = null)
    {
        string directory;
        try
        {
            directory = StartupDiagnosticsDirectory();
            Directory.CreateDirectory(directory);
            int index = autoStartDiagnosticIndex++;
            string stem = "startup_" + index.ToString("D4", CultureInfo.InvariantCulture) + "_" + SanitizeDiagnosticLabel(label);
            string screenshotPath = Path.Combine(directory, stem + ".png");
            string statePath = Path.Combine(directory, stem + ".json");
            var report = BuildStartupStateReport(label, preRun, options, reason, screenshotPath, statePath);
            File.WriteAllText(statePath, JsonConvert.SerializeObject(report, Formatting.Indented));
            Debug.Log("AUTO_START_STATE " + statePath);
            try
            {
                ScreenCapture.CaptureScreenshot(screenshotPath);
                Debug.Log("AUTO_START_SCREENSHOT " + screenshotPath);
            }
            catch (Exception ex)
            {
                Debug.LogError("AUTO_START_SCREENSHOT_FAILED path=" + screenshotPath + " error=" + ex.Message);
            }
        }
        catch (Exception ex)
        {
            // Diagnostics must never turn a valid simulation failure into an
            // opaque secondary exception.  The original failure remains in
            // the status/log, while this reports the capture problem.
            Debug.LogError("AUTO_START_DIAGNOSTICS_FAILED label=" + label + " error=" + ex);
        }
    }

    static string StartupDiagnosticsDirectory()
    {
        string baseDirectory = null;
        if (!string.IsNullOrWhiteSpace(autoStartStatusPath))
            baseDirectory = Path.GetDirectoryName(Path.GetFullPath(autoStartStatusPath));
        if (string.IsNullOrWhiteSpace(baseDirectory) && !string.IsNullOrWhiteSpace(counterfactualOutput))
            baseDirectory = Path.GetDirectoryName(Path.GetFullPath(counterfactualOutput));
        if (string.IsNullOrWhiteSpace(baseDirectory))
            baseDirectory = Path.Combine(Directory.GetCurrentDirectory(), "counterfactual_startup_diagnostics");
        return Path.Combine(baseDirectory, "startup");
    }

    static string SanitizeDiagnosticLabel(string label)
    {
        if (string.IsNullOrWhiteSpace(label)) return "event";
        return new string(label.Select(c => char.IsLetterOrDigit(c) || c == '_' || c == '-' ? c : '_').ToArray());
    }

    static Dictionary<string, object> BuildStartupStateReport(string label, PreRunScript preRun, AutoStartOptions options, string reason, string screenshotPath, string statePath)
    {
        WorldGeneration world = WorldGeneration.world;
        PlayerCamera player = PlayerCamera.main;
        Body body = player == null ? null : player.body;
        MinigameBase minigame = MinigameBase.main;
        PauseHandler pause = PauseHandler.main;
        List<string> runReadyFailures = CounterfactualRunReadyFailures(preRun, body);
        var report = new Dictionary<string, object>(StringComparer.Ordinal) {
            ["schema"] = "cb1.autostart-diagnostics.v1",
            ["label"] = label,
            ["reason"] = reason,
            ["timestamp_utc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            ["screenshot_path"] = screenshotPath,
            ["state_path"] = statePath,
            ["frame"] = Time.frameCount,
            ["active_scenes"] = ActiveScenes(),
            ["run_state"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["world_type"] = world == null ? null : world.GetType().FullName,
                ["world_exists"] = world != null && world.worldExists,
                ["generating_world"] = world != null && world.generatingWorld,
                ["loading_object"] = GameObjectState(world == null ? null : world.loadingObject),
                ["world_biome_override"] = world == null ? null : world.biomeOverride.ToString(),
                ["world_layer_time_spent"] = world == null ? (object)null : world.layerTimeSpent,
                ["world_generation_time"] = world == null ? (object)null : world.genTimePassed,
                ["body_exists"] = body != null,
                ["body_alive"] = body != null && body.alive,
                ["body_conscious"] = body != null && body.conscious,
                ["body_active"] = body != null && body.gameObject.activeInHierarchy,
                ["body_enabled"] = body != null && body.enabled,
                ["minigame_manager"] = minigame == null ? null : minigame.GetType().FullName,
                ["current_minigame"] = minigame == null || !minigame.currentMinigame ? null : minigame.currentMinigame.GetType().FullName,
                ["pause_handler"] = pause == null ? null : pause.GetType().FullName,
                ["paused"] = PauseHandler.paused,
                ["settings_menu"] = SettingsMenu.instance == null ? null : SettingsMenu.instance.GetType().FullName,
                ["liquid_transfer"] = LiquidTransfer.instance == null ? null : LiquidTransfer.instance.GetType().FullName
            },
            ["ui"] = UiState(preRun, player, minigame, pause),
            ["cameras"] = CameraState(),
            ["autostart"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["preset"] = options == null ? null : options.Preset,
                ["settings_source"] = options == null ? null : options.Source,
                ["requested_overrides"] = options == null ? null : options.Overrides.ToDictionary(p => p.Key, p => (object)p.Value),
                ["resolved_settings"] = preRun == null ? null : SerializableSettings(ReadSettings(preRun)),
                ["pre_run_exists"] = preRun != null,
                ["pre_run_name"] = preRun == null ? null : preRun.name,
                ["pre_run_current_preset"] = preRun == null ? (object)null : preRun.currentPreset,
                ["did_intro"] = ReadMember(preRun, "didIntro"),
                ["run_settings_screen"] = GameObjectState(preRun == null ? null : preRun.runSettingsScreen),
                ["tutorial_warning"] = GameObjectState(preRun == null ? null : preRun.tutorialWarning)
            },
            ["seed"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["requested"] = requestedWorldSeed,
                ["active"] = activeWorldSeed,
                ["applied"] = appliedWorldSeed,
                ["pending"] = worldSeedApplicationPending
            },
            ["simulation"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["time"] = Time.time,
                ["unscaled_time"] = Time.unscaledTime,
                ["fixed_time"] = Time.fixedTime,
                ["time_scale"] = Time.timeScale,
                ["fixed_delta_time"] = Time.fixedDeltaTime
            },
            ["protocol"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["counterfactual_mode"] = CounterfactualMode,
                ["output_ready"] = !string.IsNullOrWhiteSpace(counterfactualOutput),
                ["connected"] = connected,
                ["action_connected"] = actionConnected,
                ["runtime_ready"] = CounterfactualRuntimeReadyToConnect(body),
                ["expected_world_manager"] = world != null,
                ["expected_player_controller"] = player != null && body != null
            },
            ["run_ready"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["predicate"] = runReadyFailures.Count == 0,
                ["failures"] = runReadyFailures.ToArray()
            },
            ["global_dark"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["exists"] = GlobalDark.main != null,
                ["is_darkening"] = GlobalDark.main != null && GlobalDark.main.IsDarkening(),
                ["overlay_alpha"] = GlobalDark.main == null || GlobalDark.main.image == null ? (object)null : GlobalDark.main.image.color.a
            },
            ["top_level_canvases"] = ActiveCanvasState()
        };
        return report;
    }

    static List<Dictionary<string, object>> ActiveScenes()
    {
        var result = new List<Dictionary<string, object>>();
        for (int i = 0; i < SceneManager.sceneCount; i++)
        {
            Scene scene = SceneManager.GetSceneAt(i);
            result.Add(new Dictionary<string, object>(StringComparer.Ordinal) {
                ["build_index"] = scene.buildIndex, ["name"] = scene.name,
                ["path"] = scene.path, ["is_loaded"] = scene.isLoaded,
                ["is_active"] = scene == SceneManager.GetActiveScene()
            });
        }
        return result;
    }

    static Dictionary<string, object> UiState(PreRunScript preRun, PlayerCamera player, MinigameBase minigame, PauseHandler pause)
    {
        var result = new Dictionary<string, object>(StringComparer.Ordinal) {
            ["pre_run_main_canvas"] = CanvasState(preRun == null ? null : preRun.mainCanvas),
            ["player_main_canvas"] = CanvasState(player == null ? null : player.mainCanvas),
            ["world_canvas"] = CanvasState(player == null ? null : player.worldCanvas),
            ["global_dark_canvas"] = CanvasState(GlobalDark.main == null ? null : GlobalDark.main.canvas),
            ["pre_run_panels"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["content_warning"] = GameObjectState(preRun == null ? null : preRun.contentWarning),
                ["run_settings_screen"] = GameObjectState(preRun == null ? null : preRun.runSettingsScreen),
                ["tutorial_warning"] = GameObjectState(preRun == null ? null : preRun.tutorialWarning),
                ["credits_screen"] = GameObjectState(preRun == null ? null : preRun.creditsScreen)
            },
            ["player_panels"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["main_view"] = GameObjectState(player == null ? null : player.mainView),
                ["wound_view_medical"] = GameObjectState(player == null ? null : player.woundView),
                ["crafting_panel"] = GameObjectState(player == null ? null : player.craftingPanel),
                ["trade_menu"] = GameObjectState(player == null ? null : player.tradeMenu),
                ["container_menu"] = GameObjectState(player == null ? null : player.containerMenu),
                ["gun_menu"] = GameObjectState(player == null ? null : player.gunMenu),
                ["last_stand_panel"] = GameObjectState(player == null ? null : player.lastStandPanel),
                ["radial_menu"] = GameObjectState(player == null || player.radialMenu == null ? null : player.radialMenu.gameObject),
                ["radial_open_flag"] = player == null ? (object)null : player.radialOpen,
                ["death_screen_flag"] = player == null ? (object)null : player.didDeathScreen,
                ["self_destruct_flag"] = player == null ? (object)null : player.selfDestructActive,
                ["time_scale_mode"] = player == null ? null : player.curTimeScale.ToString()
            },
            ["minigame"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["manager"] = minigame == null ? null : GameObjectState(minigame.gameObject),
                ["current_type"] = minigame == null || !minigame.currentMinigame ? null : minigame.currentMinigame.GetType().FullName,
                ["spawned_screen"] = minigame == null || !minigame.spawnedMiniGame ? null : GameObjectState(minigame.spawnedMiniGame.gameObject),
                ["screen"] = minigame == null || minigame.minigameScreen == null ? null : GameObjectState(minigame.minigameScreen.gameObject)
            },
            ["pause"] = new Dictionary<string, object>(StringComparer.Ordinal) {
                ["handler"] = pause == null ? null : GameObjectState(pause.gameObject),
                ["paused"] = PauseHandler.paused,
                ["pause_container"] = GameObjectState(pause == null ? null : pause.pauseContainer)
            }
        };
        return result;
    }

    static List<Dictionary<string, object>> ActiveCanvasState()
    {
        var result = new List<Dictionary<string, object>>();
        foreach (Canvas canvas in Resources.FindObjectsOfTypeAll<Canvas>()
            .Where(x => x != null && x.gameObject != null && x.enabled && x.gameObject.activeInHierarchy)
            .OrderBy(x => TransformPath(x.transform), StringComparer.Ordinal))
        {
            result.Add(new Dictionary<string, object>(StringComparer.Ordinal) {
                ["type"] = canvas.GetType().FullName,
                ["name"] = canvas.name,
                ["path"] = TransformPath(canvas.transform),
                ["render_mode"] = canvas.renderMode.ToString(),
                ["sorting_order"] = canvas.sortingOrder,
                ["root"] = TransformPath(canvas.transform.root),
                ["direct_active_children"] = Enumerable.Range(0, canvas.transform.childCount)
                    .Select(i => canvas.transform.GetChild(i)).Where(t => t.gameObject.activeSelf)
                    .Select(t => TransformPath(t)).ToArray()
            });
        }
        return result;
    }

    static List<Dictionary<string, object>> CameraState()
    {
        return Camera.allCameras.Where(camera => camera != null).Select(camera => new Dictionary<string, object>(StringComparer.Ordinal) {
            ["name"] = camera.name, ["path"] = TransformPath(camera.transform),
            ["tag"] = camera.tag, ["enabled"] = camera.enabled,
            ["active_in_hierarchy"] = camera.gameObject.activeInHierarchy,
            ["is_main"] = camera == Camera.main,
            ["position"] = new [] { camera.transform.position.x, camera.transform.position.y, camera.transform.position.z },
            ["rotation"] = new [] { camera.transform.rotation.x, camera.transform.rotation.y, camera.transform.rotation.z, camera.transform.rotation.w }
        }).ToList();
    }

    static Dictionary<string, object> CanvasState(Canvas canvas)
    {
        return canvas == null ? null : new Dictionary<string, object>(StringComparer.Ordinal) {
            ["type"] = canvas.GetType().FullName, ["name"] = canvas.name,
            ["path"] = TransformPath(canvas.transform), ["enabled"] = canvas.enabled,
            ["active_self"] = canvas.gameObject.activeSelf,
            ["active_in_hierarchy"] = canvas.gameObject.activeInHierarchy
        };
    }

    static Dictionary<string, object> GameObjectState(GameObject gameObject)
    {
        if (gameObject == null) return null;
        return new Dictionary<string, object>(StringComparer.Ordinal) {
            ["name"] = gameObject.name, ["path"] = TransformPath(gameObject.transform),
            ["active_self"] = gameObject.activeSelf, ["active_in_hierarchy"] = gameObject.activeInHierarchy,
            ["scene"] = gameObject.scene.name,
            ["component_types"] = gameObject.GetComponents<Component>().Where(c => c != null).Select(c => c.GetType().FullName).OrderBy(x => x, StringComparer.Ordinal).ToArray()
        };
    }

    static Dictionary<string, object> SerializableSettings(Dictionary<string, object> settings)
    {
        if (settings == null) return null;
        return settings.ToDictionary(pair => pair.Key, pair => (object)SerializableValue(pair.Value), StringComparer.Ordinal);
    }

    static object SerializableValue(object value)
    {
        if (value == null || value is string || value is bool || value is int || value is long || value is float || value is double || value is decimal)
            return value;
        if (value is Enum) return value.ToString();
        return value.ToString();
    }

    static object ReadMember(object target, string name)
    {
        if (target == null) target = typeof(PreRunScript);
        Type type = target is Type ? (Type)target : target.GetType();
        BindingFlags flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        FieldInfo field = type.GetField(name, flags);
        if (field != null) return SerializableValue(field.GetValue(field.IsStatic ? null : target));
        PropertyInfo property = type.GetProperty(name, flags);
        if (property != null && property.GetIndexParameters().Length == 0)
        {
            try { return SerializableValue(property.GetValue(property.GetMethod.IsStatic ? null : target, null)); } catch { }
        }
        return null;
    }

    static string TransformPath(Transform transform)
    {
        if (transform == null) return null;
        var parts = new List<string>();
        for (Transform current = transform; current != null; current = current.parent)
            parts.Add(current.name);
        parts.Reverse();
        return string.Join("/", parts);
    }

    static void WriteAutoStartStatus(string status, string error, AutoStartOptions options)
    {
        if (string.IsNullOrWhiteSpace(autoStartStatusPath)) return;
        try
        {
            string fullPath = Path.GetFullPath(autoStartStatusPath);
            string directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            var payload = new {
                status,
                error,
                process_pid = System.Diagnostics.Process.GetCurrentProcess().Id,
                timestamp_utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                scene = SceneManager.GetActiveScene().name,
                preset = options == null ? null : options.Preset,
                settings_source = options == null ? null : options.Source,
                settings = options == null ? null : options.Overrides.ToDictionary(p => p.Key, p => p.Value),
                timeout_seconds = options == null ? 0f : options.TimeoutSeconds
            };
            File.WriteAllText(fullPath, JsonConvert.SerializeObject(payload, Formatting.Indented));
        }
        catch (Exception ex)
        {
            Debug.LogError("AUTO_START_STATUS_WRITE_FAILED " + ex.Message);
        }
    }

    static string ConfiguredValue(string envName, string argumentName, string fallback)
    {
        string env = Environment.GetEnvironmentVariable(envName);
        if (!string.IsNullOrWhiteSpace(env)) return env;
        string arg = CommandLineValue(argumentName);
        return string.IsNullOrWhiteSpace(arg) ? fallback : arg;
    }

    static string CommandLineValue(string name)
    {
        string prefix = name + "=";
        foreach (string arg in Environment.GetCommandLineArgs())
        {
            if (arg.StartsWith(prefix, StringComparison.Ordinal)) return arg.Substring(prefix.Length);
        }
        return null;
    }

    static bool HasCommandLineFlag(string name)
    {
        return Environment.GetCommandLineArgs().Any(arg => string.Equals(arg, name, StringComparison.Ordinal));
    }

    static Exception Unwrap(Exception ex)
    {
        while (ex is TargetInvocationException && ex.InnerException != null) ex = ex.InnerException;
        return ex;
    }
}
