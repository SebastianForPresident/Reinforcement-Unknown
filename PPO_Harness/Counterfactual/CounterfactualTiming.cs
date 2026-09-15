using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace CB1.Counterfactual
{
    // Measurement only. This ledger never gates, skips, or changes a
    // counterfactual operation; it records where the Unity main thread spent
    // wall-clock time and how often each path was entered.
    public static class CounterfactualTiming
    {
        sealed class Metric
        {
            public long Ticks;
            public long MaxTicks;
            public int Calls;

            public void Add(long ticks)
            {
                Ticks += ticks;
                Calls++;
                if (ticks > MaxTicks) MaxTicks = ticks;
            }
        }

        static readonly Dictionary<string, int> frameInvocations = new Dictionary<string, int>(StringComparer.Ordinal);
        static readonly Dictionary<string, int> policyInvocations = new Dictionary<string, int>(StringComparer.Ordinal);
        static Dictionary<string, Metric> commandMetrics;
        static string commandName;
        static long commandStartedAt;
        static string output;
        static long lastRenderedFrameAt;
        static long renderedFrame;
        static long policyStep;
        static int processInvocationsThisFrame;

        public static long Start() => Stopwatch.GetTimestamp();
        public static double Milliseconds(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        public static void Configure(string directory)
        {
            output = directory;
            lastRenderedFrameAt = Stopwatch.GetTimestamp();
            renderedFrame = 0;
            policyStep = 0;
            frameInvocations.Clear();
            policyInvocations.Clear();
            Emit("TIMING_SESSION configured=1");
        }

        public static void ProcessInvocation()
        {
            processInvocationsThisFrame++;
            Increment(frameInvocations, "ProcessCounterfactualCommands");
        }

        public static void StartPolicyStep()
        {
            policyStep++;
            policyInvocations.Clear();
            Increment(policyInvocations, "ProcessCounterfactualCommands");
        }

        public static void Invocation(string name)
        {
            Increment(frameInvocations, name);
            if (policyStep > 0) Increment(policyInvocations, name);
        }

        public static void Report(string text)
        {
            Emit("REPORT " + text);
        }

        public static long BeginCommand(string name)
        {
            commandName = name;
            commandStartedAt = Start();
            commandMetrics = new Dictionary<string, Metric>(StringComparer.Ordinal);
            return commandStartedAt;
        }

        public static void End(string name, long startedAt)
        {
            if (startedAt == 0) return;
            long elapsed = Stopwatch.GetTimestamp() - startedAt;
            if (commandMetrics == null)
            {
                Emit("EVENT metric=" + name +
                    " calls=1 total_ms=" + Milliseconds(elapsed).ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                return;
            }
            if (!commandMetrics.TryGetValue(name, out var metric))
            {
                metric = new Metric();
                commandMetrics.Add(name, metric);
            }
            metric.Add(elapsed);
        }

        public static void EndCommand(long startedAt)
        {
            if (startedAt == 0 || commandName == null) return;
            long elapsed = Stopwatch.GetTimestamp() - startedAt;
            var lines = new List<string> {
                "COMMAND command=" + commandName +
                " policy_step=" + policyStep +
                " rendered_frame=" + renderedFrame +
                " total_ms=" + Milliseconds(elapsed).ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
            };
            foreach (var pair in commandMetrics.OrderBy(pair => pair.Key, StringComparer.Ordinal))
            {
                lines.Add("  " + pair.Key + " calls=" + pair.Value.Calls +
                    " total_ms=" + Milliseconds(pair.Value.Ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                    " max_ms=" + Milliseconds(pair.Value.MaxTicks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
            }
            lines.Add("  policy_invocations=" + FormatCounts(policyInvocations));
            Emit(string.Join(Environment.NewLine, lines));
            commandMetrics = null;
            commandName = null;
            commandStartedAt = 0;
        }

        public static void RenderedFrame()
        {
            if (output == null) return;
            long now = Stopwatch.GetTimestamp();
            long gap = now - lastRenderedFrameAt;
            renderedFrame++;
            if (Milliseconds(gap) >= 100.0 || processInvocationsThisFrame > 0 || frameInvocations.Count > 0)
            {
                Emit("FRAME frame=" + renderedFrame +
                    " gap_ms=" + Milliseconds(gap).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                    " process_calls=" + processInvocationsThisFrame +
                    " invocations=" + FormatCounts(frameInvocations));
            }
            lastRenderedFrameAt = now;
            processInvocationsThisFrame = 0;
            frameInvocations.Clear();
        }

        static void Increment(Dictionary<string, int> counts, string name)
        {
            counts.TryGetValue(name, out int count);
            counts[name] = count + 1;
        }

        static string FormatCounts(Dictionary<string, int> counts) => counts.Count == 0
            ? "none"
            : string.Join(",", counts.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => pair.Key + "=" + pair.Value));

        static void Emit(string text)
        {
            if (output == null) return;
            try { File.AppendAllText(Path.Combine(output, "timing.log"), text + Environment.NewLine); }
            catch { }
            UnityEngine.Debug.Log("[CB1 TIMING] " + text.Replace(Environment.NewLine, " | "));
        }
    }
}
