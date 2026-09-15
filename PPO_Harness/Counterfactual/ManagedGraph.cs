using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;

namespace CB1.Counterfactual
{
    // No Unity calls or constructors during capture/restore. Nodes retain the original
    // identity; stored field/collection values never point into mutable live containers.
    public sealed class ManagedGraph
    {
        sealed class Identity : IEqualityComparer<object>
        {
            public new bool Equals(object a, object b) => ReferenceEquals(a, b);
            public int GetHashCode(object a) => RuntimeHelpers.GetHashCode(a);
        }
        sealed class Node
        {
            public object Target;
            public string Path;
            public Array PrimitiveArray;
            public object[] Values;
            public FieldInfo[] Fields;
            public string Kind;
        }
        sealed class Record
        {
            public Type Type;
            public FieldInfo[] Fields;
            public object[] Values;
        }
        readonly Dictionary<object, Node> seen = new Dictionary<object, Node>(new Identity());
        readonly Dictionary<object, object> reboundUnityObjects = new Dictionary<object, object>(new Identity());
        readonly List<Node> nodes = new List<Node>();
        readonly List<(object target, FieldInfo field, object value)> roots = new List<(object, FieldInfo, object)>();
        readonly Func<object, string> externalIdentity;
        readonly Func<Type, bool> ownedType;
        readonly Action<string, Array> arrayEvidence;
        readonly Action<string, string> digestEvidence;
        readonly Func<object, FieldInfo, string, bool> fieldFilter;
        readonly bool retainSnapshot;
        readonly Dictionary<object, string> digestSeen = new Dictionary<object, string>(new Identity());
        // FieldInfo metadata is immutable for the lifetime of the loaded
        // assemblies. Cache the deterministic inherited/name-sorted traversal
        // plan once per runtime type; this removes repeated reflection
        // discovery from every object and every CF_STATE while preserving the
        // same field set and order.
        static readonly Dictionary<Type, FieldInfo[]> FieldPlanCache =
            new Dictionary<Type, FieldInfo[]>();
        static readonly Dictionary<Type, FieldInfo[]> DeclaredFieldPlanCache =
            new Dictionary<Type, FieldInfo[]>();
        int digestNodes;
        long arrayBytes;
        public int UniqueReferenceIdentities { get; private set; }
        public int ReusedReferenceIdentities { get; private set; }
        public int DigestValuesVisited { get; private set; }
        public int DigestFieldsVisited { get; private set; }
        public int DigestObjectsVisited { get; private set; }
        public int DigestValuesEmitted { get; private set; }
        public int RootFieldsVisited { get; private set; }
        public long DigestPathBytes { get; private set; }
        public long DigestValueBytes { get; private set; }
        public long DigestFieldAccessTicks { get; private set; }
        public long DigestCaptureTicks { get; private set; }
        public int DigestCaptureCalls { get; private set; }
        public long DigestArrayBytes => arrayBytes;
        public int DigestNodeCount => digestNodes;
        public int ExcludedFieldCount { get; private set; }
        // Safety ceilings for malformed/cyclic state, not simulation limits.
        // Large preallocated DTO graphs are classified out by their owning
        // boundary; this ceiling remains a guard for genuinely covered state.
        const int MaxDepth = 128;
        const int MaxNodes = 1_000_000;
        public readonly SortedDictionary<string, string> Evidence = new SortedDictionary<string, string>(StringComparer.Ordinal);
        public readonly SortedSet<string> Gaps = new SortedSet<string>(StringComparer.Ordinal);
        public ManagedGraph(Func<object, string> externalIdentity, Func<Type, bool> ownedType,
            Action<string, Array> arrayEvidence, bool retainSnapshot = true,
            Action<string, string> digestEvidence = null,
            Func<object, FieldInfo, string, bool> fieldFilter = null)
        {
            this.externalIdentity = externalIdentity;
            this.ownedType = ownedType;
            this.arrayEvidence = arrayEvidence;
            this.retainSnapshot = retainSnapshot;
            this.digestEvidence = digestEvidence;
            this.fieldFilter = fieldFilter;
        }
        public static IEnumerable<FieldInfo> Fields(Type type)
        {
            if (type == null) return new FieldInfo[0];
            if (FieldPlanCache.TryGetValue(type, out var cached)) return cached;
            var fields = new List<FieldInfo>();
            for (var t = type; t != null && t != typeof(object); t = t.BaseType)
                fields.AddRange(t.GetFields(BindingFlags.Instance | BindingFlags.Public |
                    BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                    .OrderBy(f => f.Name, StringComparer.Ordinal));
            cached = fields.ToArray();
            FieldPlanCache[type] = cached;
            return cached;
        }

        public static IEnumerable<FieldInfo> DeclaredFields(Type type)
        {
            if (type == null) return new FieldInfo[0];
            if (DeclaredFieldPlanCache.TryGetValue(type, out var cached)) return cached;
            cached = type.GetFields(BindingFlags.Instance | BindingFlags.Public |
                BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                .OrderBy(f => f.Name, StringComparer.Ordinal).ToArray();
            DeclaredFieldPlanCache[type] = cached;
            return cached;
        }
        public void Field(object target, FieldInfo field, string path)
        {
            if (!IncludeField(target, field, path))
            {
                ExcludedFieldCount++;
                return;
            }
            RootFieldsVisited++;
            if (!retainSnapshot)
            {
                long digestStartedAt = Stopwatch.GetTimestamp();
                long fieldReadStartedAt = Stopwatch.GetTimestamp();
                object fieldValue = field.GetValue(target);
                DigestFieldAccessTicks += Stopwatch.GetTimestamp() - fieldReadStartedAt;
                CaptureDigest(fieldValue, path, 0);
                DigestCaptureTicks += Stopwatch.GetTimestamp() - digestStartedAt;
                DigestCaptureCalls++;
                return;
            }
            roots.Add((target, field, Capture(field.GetValue(target), path, 0)));
        }

        public void RebindUnityObject(object original, object replacement)
        {
            if (original == null || replacement == null || ReferenceEquals(original, replacement)) return;
            reboundUnityObjects[original] = replacement;
            for (int i = 0; i < nodes.Count; i++)
                if (ReferenceEquals(nodes[i].Target, original)) nodes[i].Target = replacement;
            for (int i = 0; i < roots.Count; i++)
                if (ReferenceEquals(roots[i].target, original))
                    roots[i] = (replacement, roots[i].field, roots[i].value);
        }
        void Emit(string path, string value)
        {
            DigestValuesEmitted++;
            if (digestEvidence != null)
            {
                // The fingerprint sink already measures and encodes these
                // strings. Avoid a second GetByteCount pass for every emitted
                // value in the normal (non-detailed) CF_STATE path.
                digestEvidence(path, value);
            }
            else
            {
                DigestPathBytes += Encoding.UTF8.GetByteCount(path);
                DigestValueBytes += Encoding.UTF8.GetByteCount(value ?? "");
            }
        }
        void CaptureDigest(object value, string path, int depth)
        {
            DigestValuesVisited++;
            if (depth > MaxDepth)
                throw new InvalidOperationException(
                    $"Managed snapshot fingerprint budget exceeded at {path} " +
                    $"(depth={depth}/{MaxDepth}, nodes={digestNodes}/{MaxNodes})");
            if (value == null) { Emit(path, "null"); return; }
            Type type = value.GetType();
            if (type.IsPrimitive || type.IsEnum || value is string || value is decimal)
            { Emit(path, Scalar(value)); return; }
            string external = externalIdentity(value);
            if (external != null) { Emit(path, external); return; }
            if (value is Type || value is MemberInfo)
            { Emit(path, value.ToString()); return; }
            // Repeated managed references are emitted as references and do not
            // consume traversal budget. The identity map is shared across all
            // roots, so aliases are not recursively walked a second time.
            if (!type.IsValueType && digestSeen.TryGetValue(value, out var prior))
            {
                ReusedReferenceIdentities++;
                Emit(path, "ref:" + prior);
                return;
            }
            if (++digestNodes > MaxNodes)
                throw new InvalidOperationException(
                    $"Managed snapshot fingerprint budget exceeded at {path} " +
                    $"(depth={depth}/{MaxDepth}, nodes={digestNodes}/{MaxNodes})");
            DigestObjectsVisited++;
            if (type.IsValueType)
            {
                Emit(path, "type:" + type.FullName);
                foreach (var f in Fields(type))
                {
                    string fieldPath = path + "/" + f.Name;
                    if (!IncludeField(value, f, fieldPath)) continue;
                    DigestFieldsVisited++;
                    long fieldReadStartedAt = Stopwatch.GetTimestamp();
                    object fieldValue = f.GetValue(value);
                    DigestFieldAccessTicks += Stopwatch.GetTimestamp() - fieldReadStartedAt;
                    CaptureDigest(fieldValue, fieldPath, depth + 1);
                }
                return;
            }
            digestSeen.Add(value, path);
            UniqueReferenceIdentities++;
            Emit(path, "type:" + type.FullName);
            if (value is Array a)
            {
                Emit(path + "/shape", string.Join(",", Enumerable.Range(0, a.Rank)
                    .Select(i => a.GetLowerBound(i) + ":" + a.GetLength(i))));
                if (type.GetElementType().IsPrimitive) arrayEvidence(path, a);
                else
                {
                    int i = 0;
                    foreach (var x in a) CaptureDigest(x, path + "/" + i++, depth + 1);
                }
            }
            else if (value is IDictionary dict)
            {
                int i = 0;
                foreach (DictionaryEntry e in dict)
                {
                    CaptureDigest(e.Key, path + "/key/" + i, depth + 1);
                    CaptureDigest(e.Value, path + "/value/" + i++, depth + 1);
                }
                var comparer = type.GetProperty("Comparer")?.GetValue(value);
                if (comparer != null && Fields(comparer.GetType()).Any())
                    CaptureDigest(comparer, path + "/comparer", depth + 1);
            }
            else if (value is IList list && !list.IsReadOnly && !list.IsFixedSize)
            {
                int i = 0;
                foreach (var x in list) CaptureDigest(x, path + "/" + i++, depth + 1);
            }
            else if (value is Delegate del)
            {
                int i = 0;
                foreach (var d in del.GetInvocationList())
                {
                    Emit(path + "/method/" + i, d.Method.DeclaringType.FullName + "." + d.Method.Name);
                    CaptureDigest(d.Target, path + "/target/" + i++, depth + 1);
                }
            }
            else if (ownedType(type))
            {
                foreach (var f in Fields(type))
                {
                    string fieldPath = path + "/" + f.DeclaringType.Name + "." + f.Name;
                    if (!IncludeField(value, f, fieldPath)) continue;
                    DigestFieldsVisited++;
                    long fieldReadStartedAt = Stopwatch.GetTimestamp();
                    object fieldValue = f.GetValue(value);
                    DigestFieldAccessTicks += Stopwatch.GetTimestamp() - fieldReadStartedAt;
                    CaptureDigest(fieldValue, fieldPath, depth + 1);
                }
            }
            else
            {
                Gaps.Add(path + ": opaque managed " + type.FullName);
            }
        }
        object Capture(object value, string path, int depth)
        {
            if (depth > MaxDepth || nodes.Count > MaxNodes)
                throw new InvalidOperationException(
                    $"Managed snapshot graph budget exceeded at {path} " +
                    $"(depth={depth}/{MaxDepth}, nodes={nodes.Count}/{MaxNodes})");
            if (value == null) { Evidence[path] = "null"; return null; }
            Type type = value.GetType();
            if (type.IsPrimitive || type.IsEnum || value is string || value is decimal)
            {
                Evidence[path] = Scalar(value);
                return value;
            }
            string external = externalIdentity(value);
            if (external != null) { Evidence[path] = external; return value; }
            if (value is Type || value is MemberInfo)
            { Evidence[path] = value.ToString(); return value; }
            if (type.IsValueType)
            {
                var fs = Fields(type).ToArray();
                return new Record { Type = type, Fields = fs,
                    Values = fs.Select(f => Capture(f.GetValue(value), path + "/" + f.Name, depth + 1)).ToArray() };
            }
            if (seen.TryGetValue(value, out var prior))
            {
                ReusedReferenceIdentities++;
                Evidence[path] = "ref:" + prior.Path;
                return prior;
            }
            var n = new Node { Target = value, Path = path };
            seen.Add(value, n); nodes.Add(n); UniqueReferenceIdentities++;
            Evidence[path] = "type:" + type.FullName;
            if (value is Array a)
            {
                n.Kind = "array";
                Evidence[path + "/shape"] = string.Join(",", Enumerable.Range(0, a.Rank)
                    .Select(i => a.GetLowerBound(i) + ":" + a.GetLength(i)));
                if (type.GetElementType().IsPrimitive)
                {
                    arrayBytes += Buffer.ByteLength(a);
                    if (arrayBytes > 128L * 1024 * 1024)
                        throw new InvalidOperationException("Managed snapshot array budget exceeded (128 MiB)");
                    n.PrimitiveArray = (Array)a.Clone();
                    arrayEvidence(path, n.PrimitiveArray);
                }
                else n.Values = a.Cast<object>().Select((x, i) => Capture(x, path + "/" + i, depth + 1)).ToArray();
            }
            else if (value is IDictionary dict)
            {
                n.Kind = "dictionary";
                var vals = new List<object>(); int i = 0;
                foreach (DictionaryEntry e in dict)
                {
                    vals.Add(Capture(e.Key, path + "/key/" + i, depth + 1));
                    vals.Add(Capture(e.Value, path + "/value/" + i++, depth + 1));
                }
                n.Values = vals.ToArray();
                // Comparers may themselves carry state; standard stateless comparers
                // are preserved by retaining this exact dictionary object.
                var comparer = type.GetProperty("Comparer")?.GetValue(value);
                if (comparer != null && Fields(comparer.GetType()).Any())
                    Capture(comparer, path + "/comparer", depth + 1);
            }
            else if (value is IList list && !list.IsReadOnly && !list.IsFixedSize)
            {
                n.Kind = "list";
                n.Values = list.Cast<object>().Select((x, i) => Capture(x, path + "/" + i, depth + 1)).ToArray();
            }
            else if (value is Delegate del)
            {
                n.Kind = "opaque";
                int i = 0;
                foreach (var d in del.GetInvocationList())
                {
                    Evidence[path + "/method/" + i] = d.Method.DeclaringType.FullName + "." + d.Method.Name;
                    Capture(d.Target, path + "/target/" + i++, depth + 1);
                }
            }
            else if (ownedType(type))
            {
                n.Kind = "object";
                n.Fields = Fields(type).Where(f => IncludeField(value, f,
                    path + "/" + f.DeclaringType.Name + "." + f.Name)).ToArray();
                n.Values = n.Fields.Select(f => Capture(f.GetValue(value),
                    path + "/" + f.DeclaringType.Name + "." + f.Name, depth + 1)).ToArray();
            }
            else
            {
                n.Kind = "opaque";
                Gaps.Add(path + ": opaque managed " + type.FullName);
            }
            return n;
        }
        bool IncludeField(object target, FieldInfo field, string path)
        {
            return field == null || fieldFilter == null || fieldFilter(target, field, path);
        }
        object Resolve(object v)
        {
            if (v != null && reboundUnityObjects.TryGetValue(v, out var rebound)) return rebound;
            if (v is Node n) return n.Target;
            if (v is Record r)
            {
                object boxed = Activator.CreateInstance(r.Type);
                for (int i = 0; i < r.Fields.Length; i++) r.Fields[i].SetValue(boxed, Resolve(r.Values[i]));
                return boxed;
            }
            return v;
        }
        public void Restore()
        {
            // Restore key objects before rebuilding dictionary hash buckets.
            foreach (var n in nodes.Where(n => n.Kind == "object"))
                for (int i = 0; i < n.Fields.Length; i++)
                    if (!n.Fields[i].IsInitOnly) n.Fields[i].SetValue(n.Target, Resolve(n.Values[i]));
                    else if (!Equals(n.Fields[i].GetValue(n.Target), Resolve(n.Values[i])))
                        throw new InvalidOperationException("Readonly identity changed: " + n.Path);
            foreach (var n in nodes)
            {
                if (n.Kind == "array")
                {
                    var a = (Array)n.Target;
                    if (n.PrimitiveArray != null) Array.Copy(n.PrimitiveArray, a, a.Length);
                    else
                    {
                        int[] indices = Enumerable.Range(0, a.Rank).Select(a.GetLowerBound).ToArray();
                        for (int i = 0; i < n.Values.Length; i++)
                        {
                            a.SetValue(Resolve(n.Values[i]), indices);
                            for (int j = a.Rank - 1; j >= 0; j--)
                            {
                                if (++indices[j] <= a.GetUpperBound(j)) break;
                                indices[j] = a.GetLowerBound(j);
                            }
                        }
                    }
                }
                else if (n.Kind == "list")
                {
                    var list = (IList)n.Target; list.Clear();
                    foreach (var v in n.Values) list.Add(Resolve(v));
                }
            }
            foreach (var n in nodes.Where(n => n.Kind == "dictionary"))
            {
                var dict = (IDictionary)n.Target; dict.Clear();
                for (int i = 0; i < n.Values.Length; i += 2) dict.Add(Resolve(n.Values[i]), Resolve(n.Values[i + 1]));
            }
            foreach (var r in roots)
                if (!r.field.IsInitOnly) r.field.SetValue(r.target, Resolve(r.value));
                else if (!Equals(r.field.GetValue(r.target), Resolve(r.value)))
                    throw new InvalidOperationException("Readonly root changed: " + r.field.Name);
        }
        public static string Scalar(object value)
        {
            if (value is float f) return "f32:" + BitConverter.SingleToInt32Bits(f).ToString("x8");
            if (value is double d) return "f64:" + BitConverter.DoubleToInt64Bits(d).ToString("x16");
            return value.GetType().FullName + ":" + Convert.ToString(value, CultureInfo.InvariantCulture);
        }
    }
}
