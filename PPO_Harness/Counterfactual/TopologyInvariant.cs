using System;
using System.Collections.Generic;
using System.Linq;

namespace CB1.Counterfactual
{
    // The topology contract is intentionally pure so its symmetry can be
    // regression-tested without constructing Unity objects.
    public sealed class TopologyInvariantRecord
    {
        public string Identity;
        public string Classification;
    }

    public static class TopologyInvariant
    {
        public const string PresentationUiLifecycle = "presentation-ui-lifecycle";

        public static bool Participates(TopologyInvariantRecord record) =>
            record != null && record.Classification != "derived-rebuildable" &&
            record.Classification != "demonstrably-noncausal";

        public static bool LifecycleParticipates(TopologyInvariantRecord record) =>
            Participates(record) && record.Classification != PresentationUiLifecycle;

        public static bool LifecycleEquivalent(IEnumerable<TopologyInvariantRecord> baseline,
            IEnumerable<TopologyInvariantRecord> live)
        {
            var expected = baseline.Where(LifecycleParticipates).Select(r => r.Identity)
                .OrderBy(identity => identity, StringComparer.Ordinal);
            var actual = live.Where(LifecycleParticipates).Select(r => r.Identity)
                .OrderBy(identity => identity, StringComparer.Ordinal);
            return expected.SequenceEqual(actual, StringComparer.Ordinal);
        }

        public static bool Equivalent(IEnumerable<TopologyInvariantRecord> baseline,
            IEnumerable<TopologyInvariantRecord> live)
        {
            var expected = baseline.Where(Participates).Select(r => r.Identity)
                .OrderBy(identity => identity, StringComparer.Ordinal);
            var actual = live.Where(Participates).Select(r => r.Identity)
                .OrderBy(identity => identity, StringComparer.Ordinal);
            return expected.SequenceEqual(actual, StringComparer.Ordinal);
        }
    }
}
