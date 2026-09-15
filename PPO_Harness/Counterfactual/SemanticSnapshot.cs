using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Tilemaps;
using Object = UnityEngine.Object;

namespace CB1.Counterfactual
{
    // Diagnostic schema 1. Coverage gaps are part of the result, never a warning
    // that can be overridden into scientific acceptance.
    public sealed class SemanticSnapshot
    {
        public const int Schema = 1;
        public readonly ManagedGraph Graph;
        public readonly SortedDictionary<string, string> State = new SortedDictionary<string, string>(StringComparer.Ordinal);
        public readonly SortedDictionary<string, object> Arrays = new SortedDictionary<string, object>(StringComparer.Ordinal);
        public readonly SortedDictionary<string, string> SectionHashes = new SortedDictionary<string, string>(StringComparer.Ordinal);
        public readonly SortedSet<string> Gaps = new SortedSet<string>(StringComparer.Ordinal);
        public readonly Dictionary<string, object> Clock = new Dictionary<string, object>();
        public sealed class CaptureDiagnostics
        {
            public int scene_gameobjects_visited;
            public int scene_components_visited;
            public int causal_gameobjects;
            public int causal_components;
            public int topology_entries;
            public long topology_path_bytes;
            public int native_components_visited;
            public int native_property_lookups;
            public int native_property_reads;
            public int managed_root_fields;
            public int managed_values_visited;
            public int managed_objects_visited;
            public int managed_fields_visited;
            public int managed_values_emitted;
            public int managed_unique_references;
            public int managed_reference_reuses;
            public int managed_excluded_fields;
            public long managed_path_bytes;
            public long managed_value_bytes;
            public long managed_array_bytes;
            public long managed_field_access_ticks;
            public long tilemap_cells_examined;
            public int tilemaps_examined;
            public long array_hash_bytes;
            public long native_reflection_ticks;
            public int contact_pairs;
            public long fingerprint_input_bytes;
            public long fingerprint_add_ticks;
            public long fingerprint_finish_ticks;
            public long gc_total_memory_delta;
            public int gc_collections_gen0;
            public int gc_collections_gen1;
            public int gc_collections_gen2;
        }
        public readonly CaptureDiagnostics Diagnostics = new CaptureDiagnostics();
        public string StateHash { get; private set; }
        readonly List<NativeRecord> native = new List<NativeRecord>();
        readonly List<TileRecord> tilemaps = new List<TileRecord>();
        readonly Component[] components;
        readonly Dictionary<int, Component> snapshotComponentObjects;
        readonly Dictionary<int, Component> currentComponentObjects;
        readonly Dictionary<int, TopologyEntry> allGameObjectTopology;
        readonly Dictionary<int, TopologyEntry> allComponentTopology;
        readonly Dictionary<int, TopologyEntry> gameObjectTopology;
        readonly Dictionary<int, TopologyEntry> componentTopology;
        static int topologyDiagnosticIndex;
        static TopologyCache topologyCache;
        readonly UnityEngine.Random.State rng;
        readonly float timeScale, fixedDeltaTime, maximumDeltaTime;
        readonly string output;
        readonly bool detailed;
        readonly Fingerprint fingerprint;
        readonly byte[] arrayHashScratch = new byte[1024 * 1024];
        // Primitive arrays are hashed sequentially on Unity's main thread.
        // Reusing the algorithm removes hundreds of SHA object allocations;
        // Initialize() resets the same SHA-256 computation for each array.
        readonly SHA256 arraySha = SHA256.Create();
        static readonly byte[] EmptyHashInput = new byte[0];
        // Native property values each start a fresh identity scope in the
        // existing digest semantics. Reuse the dictionary storage between
        // those scopes instead of allocating one map for every property.
        readonly Dictionary<object, string> nativeDigestSeen =
            new Dictionary<object, string>(new UnityIdentity());
        long nativeReflectionTicks;
        readonly List<string> contactPairPreview = new List<string>();
        public readonly int Seed;
        static readonly Dictionary<int, string> StableUnityIdentities = new Dictionary<int, string>();
        static int restoreDiagnosticIndex;
        // Serialized baseline wire data is independent from managed snapshot
        // coverage. Restore loads these bytes; the next publish reconstructs
        // the observation DTO scratch graph and emits a fresh packet.
        public readonly byte[] BaselineObservationWire;

        public sealed class TopologyEntry
        {
            public string kind;
            public int unity_instance_id;
            public string type;
            public int game_object_instance_id;
            public string game_object_name;
            public string scene;
            public string transform_path;
            public int parent_instance_id;
            public string parent_transform_path;
            public int component_ordinal;
            public string stable_identity;
            public string registry_membership;
            public string classification;
            public string classification_basis;
            public string lifecycle_classification;
            public string lifecycle_classification_basis;
        }
        public sealed class TopologyChange
        {
            public string change;
            public TopologyEntry entry;
        }

        sealed class TopologyCache
        {
            public readonly GameObject[] AllGameObjects;
            public readonly Component[] AllComponents;
            public readonly GameObject[] CausalGameObjects;
            public readonly Component[] CausalComponents;
            public readonly HashSet<int> PresentationIds;
            public readonly Dictionary<int, TopologyEntry> AllGameObjectTopology;
            public readonly Dictionary<int, TopologyEntry> AllComponentTopology;
            public readonly Dictionary<int, TopologyEntry> GameObjectTopology;
            public readonly Dictionary<int, TopologyEntry> ComponentTopology;
            public readonly Dictionary<int, string> ComponentPaths;

            public TopologyCache(GameObject[] allGameObjects, Component[] allComponents,
                HashSet<int> presentationIds, GameObject[] causalGameObjects, Component[] causalComponents,
                Dictionary<int, TopologyEntry> allGameObjectTopology,
                Dictionary<int, TopologyEntry> allComponentTopology,
                Dictionary<int, TopologyEntry> gameObjectTopology,
                Dictionary<int, TopologyEntry> componentTopology)
            {
                AllGameObjects = allGameObjects;
                AllComponents = allComponents;
                PresentationIds = presentationIds;
                CausalGameObjects = causalGameObjects;
                CausalComponents = causalComponents;
                AllGameObjectTopology = allGameObjectTopology;
                AllComponentTopology = allComponentTopology;
                GameObjectTopology = gameObjectTopology;
                ComponentTopology = componentTopology;
                ComponentPaths = componentTopology.ToDictionary(
                    pair => pair.Key, pair => "scene/" + pair.Value.stable_identity);
            }

            public bool Matches(GameObject[] liveGameObjects, Component[] liveComponents)
            {
                if (liveGameObjects.Length != CausalGameObjects.Length ||
                    liveComponents.Length != CausalComponents.Length) return false;
                for (int i = 0; i < CausalGameObjects.Length; i++)
                    if (!ReferenceEquals(CausalGameObjects[i], liveGameObjects[i])) return false;
                for (int i = 0; i < CausalComponents.Length; i++)
                    if (!ReferenceEquals(CausalComponents[i], liveComponents[i])) return false;

                // Validate metadata that affects filtering, stable hash paths,
                // or the existing topology classification. No live value is
                // cached; this is only a read-only plan check.
                foreach (var go in liveGameObjects)
                {
                    if (!GameObjectTopology.TryGetValue(go.GetInstanceID(), out var entry)) return false;
                    if (go.name != entry.game_object_name || go.scene.path != entry.scene ||
                        ParentId(go.transform) != entry.parent_instance_id ||
                        TransformPath(go.transform) != entry.transform_path) return false;
                }
                foreach (var component in liveComponents)
                {
                    if (!ComponentTopology.TryGetValue(component.GetInstanceID(), out var entry)) return false;
                    if (component.GetType().FullName != entry.type ||
                        component.gameObject.GetInstanceID() != entry.game_object_instance_id ||
                        RegistryMembership(component) != entry.registry_membership) return false;
                }
                return true;
            }
        }

        sealed class NativeRecord
        {
            public Component Target;
            public readonly List<(PropertyInfo property, object value)> Properties = new List<(PropertyInfo, object)>();
            // The exact GlobalDark description TMP submesh is a renderer-owned
            // presentation leaf. Its active/enabled bits are regenerated by
            // the UI/text system and are therefore canonicalized, while the
            // component itself remains in topology and all other lifecycle
            // checks remain strict.
            public bool TrackLifecycle = true;
            public bool RecreatedForRestore;
            public bool Active;
            public int Layer;
            public Transform Parent;
            public bool? Enabled;
            public bool Sleeping;
            public Vector2[][] PolygonPaths;
            public AnimatorStateInfo[] AnimatorStates;
            public AnimatorStateInfo[] AnimatorNextStates;
            public AnimatorTransitionInfo[] AnimatorTransitions;
            public bool[] AnimatorInTransitions;
            public float[] LayerWeights;
            public AnimatorControllerParameter[] Parameters;
            public object[] ParameterValues;
            public ContactPoint2D[] Contacts;
        }
        sealed class AnimatorDiagnostic
        {
            public string identity;
            public string stable_identity;
            public string hierarchy;
            public bool active;
            public bool enabled;
            public string speed;
            public string apply_root_motion;
            public string update_mode;
            public string culling_mode;
            public string fire_events;
            public string runtime_controller;
            public string avatar;
            public Dictionary<string, string> optional_properties;
            public object[] layers;
            public object[] parameters;
        }
        sealed class PhysicsContactDiagnostic
        {
            public string identity;
            public string hierarchy;
            public int contact_count;
            public object[] contacts;
        }
        sealed class TileRecord
        {
            public Tilemap Map;
            public BoundsInt Bounds;
            public TileBase[] Tiles;
            public Matrix4x4[] Matrices;
            public TileFlags[] Flags;
        }
        sealed class RestoreMetrics
        {
            public int ChangedTilemaps;
            public int ChangedCells;
            public int FullTilemapRewrites;
            public int TilemapCollidersRebuilt;
            public int CompositeCollidersRebuilt;
            public int NativeChangedComponents;
            public int NativeFieldWrites;
            public int NativeFieldSkips;
            public readonly HashSet<Tilemap> ChangedTilemapObjects = new HashSet<Tilemap>();
            public readonly HashSet<Component> ChangedNativeObjects = new HashSet<Component>();
        }
        sealed class Fingerprint
        {
            struct SectionKey : IEquatable<SectionKey>
            {
                public readonly int Hash;
                public readonly int Length;

                public SectionKey(int hash, int length)
                {
                    Hash = hash; Length = length;
                }

                public bool Equals(SectionKey other) => Hash == other.Hash && Length == other.Length;
                public override bool Equals(object value) => value is SectionKey && Equals((SectionKey)value);
                public override int GetHashCode() => Hash * 397 ^ Length;
            }

            sealed class SectionEntry
            {
                public string Name;
                public BufferedHash Hash;
            }

            sealed class BufferedHash : IDisposable
            {
                const int BufferSize = 64 * 1024;
                readonly SHA256 hash = SHA256.Create();
                readonly byte[] buffer = new byte[BufferSize];
                int count;

                public void Append(byte[] source, int offset, int length)
                {
                    while (length > 0)
                    {
                        int copied = Math.Min(length, buffer.Length - count);
                        Buffer.BlockCopy(source, offset, buffer, count, copied);
                        offset += copied;
                        length -= copied;
                        count += copied;
                        if (count == buffer.Length) Flush();
                    }
                }

                void Flush()
                {
                    if (count == 0) return;
                    hash.TransformBlock(buffer, 0, count, buffer, 0);
                    count = 0;
                }

                public byte[] Finish()
                {
                    Flush();
                    hash.TransformFinalBlock(new byte[0], 0, 0);
                    return hash.Hash;
                }

                public void Dispose() { hash.Dispose(); }
            }

            readonly BufferedHash whole = new BufferedHash();
            readonly Dictionary<SectionKey, List<SectionEntry>> sectionIndex =
                new Dictionary<SectionKey, List<SectionEntry>>();
            readonly List<SectionEntry> sections = new List<SectionEntry>();
            byte[] utf8Scratch = new byte[256];
            public long InputBytes { get; private set; }
            public long PathBytes { get; private set; }
            public long ValueBytes { get; private set; }
            public long AddTicks { get; private set; }
            public long FinishTicks { get; private set; }

            static bool IsAscii(string text)
            {
                for (int i = 0; i < text.Length; i++)
                    if (text[i] > 0x7f) return false;
                return true;
            }

            static int WriteUtf8(string text, bool ascii, byte[] destination, int offset)
            {
                if (ascii)
                {
                    for (int i = 0; i < text.Length; i++)
                        destination[offset++] = (byte)text[i];
                    return offset;
                }
                return offset + Encoding.UTF8.GetBytes(text, 0, text.Length, destination, offset);
            }

            static SectionEntry SectionFor(string path,
                Dictionary<SectionKey, List<SectionEntry>> index, List<SectionEntry> entries)
            {
                int first = path.IndexOf('/');
                int end = first < 0 ? path.Length : path.IndexOf('/', first + 1);
                if (end < 0) end = path.Length;
                int hash = unchecked((int)2166136261u);
                for (int i = 0; i < end; i++)
                    hash = unchecked((hash ^ path[i]) * 16777619);
                var key = new SectionKey(hash, end);
                if (index.TryGetValue(key, out var candidates))
                    foreach (var candidate in candidates)
                    {
                        bool equal = true;
                        for (int i = 0; i < end; i++)
                            if (candidate.Name[i] != path[i]) { equal = false; break; }
                        if (equal) return candidate;
                    }

                var created = new SectionEntry {
                    Name = path.Substring(0, end), Hash = new BufferedHash()
                };
                if (candidates == null)
                {
                    candidates = new List<SectionEntry>();
                    index.Add(key, candidates);
                }
                candidates.Add(created);
                entries.Add(created);
                return created;
            }

            public void Add(string path, string value)
            {
                long startedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                value = value ?? "";
                bool valueAscii = IsAscii(value);
                bool pathAscii = IsAscii(path);
                int pathBytes = pathAscii ? path.Length : Encoding.UTF8.GetByteCount(path);
                int valueBytes = valueAscii ? value.Length : Encoding.UTF8.GetByteCount(value);
                int length = pathBytes + 1 + valueBytes + 1;
                PathBytes += pathBytes;
                ValueBytes += valueBytes;
                if (utf8Scratch.Length < length)
                    Array.Resize(ref utf8Scratch, Math.Max(length, utf8Scratch.Length * 2));
                int offset = WriteUtf8(path, pathAscii, utf8Scratch, 0);
                utf8Scratch[offset++] = (byte)'=';
                offset = WriteUtf8(value, valueAscii, utf8Scratch, offset);
                utf8Scratch[offset++] = (byte)'\n';
                InputBytes += offset;
                whole.Append(utf8Scratch, 0, offset);
                var section = SectionFor(path, sectionIndex, sections);
                section.Hash.Append(utf8Scratch, 0, offset);
                AddTicks += System.Diagnostics.Stopwatch.GetTimestamp() - startedAt;
            }
            public string Finish(out SortedDictionary<string, string> sectionHashes)
            {
                long startedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                sectionHashes = new SortedDictionary<string, string>(StringComparer.Ordinal);
                foreach (var pair in sections)
                {
                    sectionHashes[pair.Name] = Hex(pair.Hash.Finish());
                    pair.Hash.Dispose();
                }
                string result = Hex(whole.Finish()); whole.Dispose();
                FinishTicks += System.Diagnostics.Stopwatch.GetTimestamp() - startedAt;
                return result;
            }
            static string Hex(byte[] bytes) => BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
        }
        static readonly string[] BodyProperties = {
            "bodyType", "simulated", "useAutoMass", "mass", "drag", "angularDrag", "gravityScale",
            "constraints", "collisionDetectionMode", "interpolation", "sleepMode", "position", "rotation",
            "velocity", "angularVelocity" };
        static readonly string[] ColliderProperties = {
            "enabled", "isTrigger", "offset", "density", "sharedMaterial", "usedByEffector", "usedByComposite" };
        static readonly Dictionary<Type, Dictionary<string, PropertyInfo>> NativePropertyCache =
            new Dictionary<Type, Dictionary<string, PropertyInfo>>();
        static readonly object NativePropertyCacheLock = new object();
        public static string Identity(object value)
        {
            if (!(value is Object obj)) return null;
            if (obj == null) return "unity:destroyed";
            int instanceId = obj.GetInstanceID();
            if (StableUnityIdentities.TryGetValue(instanceId, out var stable)) return stable;
            // Runtime instance IDs are process-local and therefore cannot be
            // part of a fresh-process replay fingerprint. Scene objects are
            // registered with hierarchy identities before values are hashed;
            // this fallback is for Unity assets and other unregistered
            // objects, for which type/name is the stable observable identity.
            string name = obj.name;
            return "unity:" + obj.GetType().FullName + ":" +
                (string.IsNullOrEmpty(name) ? "<unnamed>" : name);
        }

        static string RandomFingerprint(UnityEngine.Random.State state)
        {
            var values = typeof(UnityEngine.Random.State).GetFields(
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                .OrderBy(f => f.Name, StringComparer.Ordinal)
                .Select(f => f.Name + "=" + Convert.ToString(f.GetValue(state),
                    System.Globalization.CultureInfo.InvariantCulture));
            return Hash(Encoding.UTF8.GetBytes(string.Join("|", values)));
        }

        static void RegisterStableIdentity(Object obj, string stableIdentity)
        {
            if (obj != null) StableUnityIdentities[obj.GetInstanceID()] = "stable:" + stableIdentity;
        }
        static Component[] AllSceneComponents()
        {
            long startedAt = CounterfactualTiming.Start();
            try
            {
                return Resources.FindObjectsOfTypeAll<Component>()
                    .Where(c => c != null && c.gameObject.scene.IsValid() && c.gameObject.scene.isLoaded)
                    .OrderBy(c => c.GetInstanceID()).ToArray();
            }
            finally { CounterfactualTiming.End("scene.components_enumeration", startedAt); }
        }

        static GameObject[] AllSceneGameObjects()
        {
            long startedAt = CounterfactualTiming.Start();
            try
            {
                return Resources.FindObjectsOfTypeAll<GameObject>()
                    .Where(go => go != null && go.scene.IsValid() && go.scene.isLoaded)
                    .OrderBy(go => go.GetInstanceID()).ToArray();
            }
            finally { CounterfactualTiming.End("scene.gameobjects_enumeration", startedAt); }
        }

        static HashSet<int> PresentationGameObjectIds(GameObject[] objects)
        {
            var result = new HashSet<int>();
            foreach (var go in objects)
                if (IsPresentationOnly(go)) result.Add(go.GetInstanceID());
            return result;
        }

        static Component[] CausalComponents(Component[] components, HashSet<int> presentationIds) =>
            components.Where(c => c != null && !presentationIds.Contains(c.gameObject.GetInstanceID())).ToArray();

        static bool IsPresentationOnly(GameObject go)
        {
            if (go == null) return false;
            if (go.name == "PPO Look Debug Dot") return true;
            if (IsProvenPresentationUiLeaf(go)) return true;
            var all = go.GetComponents<Component>().Where(c => c != null).ToArray();
            // SearchBase/ArrowGuide has no gameplay readers: Awake stores the
            // original position and Update derives only a visual sine-wave
            // offset from Time.time. Keep this exact owner/component rule
            // narrow; arbitrary transforms remain causal.
            if (all.Any(c => c is ArrowGuide))
                return all.All(c => c is Transform || c is ArrowGuide || c is SpriteRenderer);
            bool namedPresentation = go.name.StartsWith("Dust", StringComparison.Ordinal) ||
                go.name == "ItemBreakParticle(Clone)";
            bool hasPresentationComponent = all.Any(IsPresentationComponent);
            if (!namedPresentation && !hasPresentationComponent) return false;
            return all.All(c => c is Transform || c is AudioSource || c is PlayedSound ||
                IsParticleComponent(c) || c is SpriteRenderer);
        }

        static bool IsParticleComponent(Component component)
        {
            string type = component == null ? "" : component.GetType().FullName;
            return type == "UnityEngine.ParticleSystem" || type == "UnityEngine.ParticleSystemRenderer";
        }

        static bool IsPresentationComponent(Component component) =>
            component is PlayedSound || IsParticleComponent(component);

        static bool IsPresentationOnly(Component component) =>
            component != null && IsPresentationOnly(component.gameObject);

        // Evidence-bound lifecycle exception. This is deliberately narrower
        // than "all TMP", "all UI", or "all DontDestroyOnLoad" objects:
        // Unity.TextMeshPro.dll owns TMP_SubMeshUI; the observed path is the
        // description renderer below GlobalDark/MouseOver/DescBack/Description;
        // and the leaf has no gameplay registry or physics identity. The
        // the exact proven presentation leaf is canonicalized out of topology
        // and lifecycle together, so creation/destruction is symmetric.
        static bool IsProvenPresentationLifecycle(Component component)
        {
            if (IsProvenGlobalDarkMouseOverContainer(component)) return true;
            if (component == null || component.GetType().FullName != "TMPro.TMP_SubMeshUI") return false;
            var leaf = component.transform;
            if (leaf == null || !leaf.name.StartsWith("TMP UI SubObject", StringComparison.Ordinal)) return false;
            var description = leaf.parent;
            var back = description == null ? null : description.parent;
            var mouseOver = back == null ? null : back.parent;
            var globalDark = mouseOver == null ? null : mouseOver.parent;
            bool knownTextLeaf = description != null &&
                ((description.name == "Description" && back != null && back.name == "DescBack") ||
                 (description.name == "Title" && back != null && back.name == "TitleBack"));
            if (!knownTextLeaf ||
                mouseOver == null || mouseOver.name != "MouseOver" ||
                globalDark == null || globalDark.name != "GlobalDark(Clone)") return false;
            // Keep the source-level evidence check explicit: if this leaf is
            // ever registered as gameplay state, or acquires a physics/body
            // identity, the exception stops matching and the hard guard wins.
            if (RegistryMembership(component) != "none-observed" || component is Body ||
                component is Item || component is BuildingEntity || component is Rigidbody2D ||
                component is Collider2D || component is Joint2D) return false;
            return globalDark.GetComponent("GlobalDark") != null;
        }

        // The latest restore artifact identifies the direct MouseOver
        // RectTransform itself, not one of its TMP leaves. Keep this rule
        // exact: it requires the observed GlobalDark/MouseOver hierarchy,
        // no gameplay/physics component on the object, and a UI-only owner.
        // This does not exempt the GlobalDark controller or arbitrary UI.
        static bool IsProvenGlobalDarkMouseOverContainer(Component component)
        {
            if (component == null || component.GetType() != typeof(RectTransform)) return false;
            var mouseOver = component.transform;
            var globalDark = mouseOver == null ? null : mouseOver.parent;
            if (mouseOver == null || mouseOver.name != "MouseOver" || globalDark == null ||
                globalDark.name != "GlobalDark(Clone)" || globalDark.parent != null ||
                globalDark.GetComponent("GlobalDark") == null) return false;
            if (RegistryMembership(component) != "none-observed") return false;
            var ownerComponents = mouseOver.gameObject.GetComponents<Component>().Where(c => c != null).ToArray();
            return ownerComponents.Length > 0 && ownerComponents.All(c => c is Transform);
        }

        static string PresentationLifecycleBasis(Component component)
        {
            if (IsProvenGlobalDarkMouseOverContainer(component))
                return "exact DontDestroyOnLoad GlobalDark(Clone)/MouseOver RectTransform container; owner has only Transform and no gameplay registry or physics identity";
            return "exact GlobalDark(Clone)/MouseOver/DescBack/Description-or-Title/TMP UI SubObject TMP_SubMeshUI renderer leaf; no gameplay registry or physics identity";
        }

        static string PresentationLifecycleKey(Component component)
        {
            return IsProvenGlobalDarkMouseOverContainer(component)
                ? "GlobalDark/MouseOver/RectTransform"
                : "GlobalDark/MouseOver/TMP_SubMeshUI";
        }

        // Evidence-bound topology exclusion for the same presentation text
        // leaves as the lifecycle rule. Only the exact GlobalDark title and
        // description TMP-generated leaves qualify; arbitrary TMP/UI objects
        // remain causal and continue to trip topology guards.
        static bool IsProvenPresentationUiLeaf(GameObject go)
        {
            if (go == null || !go.name.StartsWith("TMP UI SubObject", StringComparison.Ordinal)) return false;
            var leaf = go.transform;
            var text = leaf == null ? null : leaf.parent;
            var back = text == null ? null : text.parent;
            var mouseOver = back == null ? null : back.parent;
            var globalDark = mouseOver == null ? null : mouseOver.parent;
            bool knownTextLeaf = text != null &&
                ((text.name == "Description" && back != null && back.name == "DescBack") ||
                 (text.name == "Title" && back != null && back.name == "TitleBack"));
            if (!knownTextLeaf || mouseOver == null || mouseOver.name != "MouseOver" ||
                globalDark == null || globalDark.name != "GlobalDark(Clone)" ||
                globalDark.GetComponent("GlobalDark") == null) return false;
            var components = go.GetComponents<Component>().Where(c => c != null).ToArray();
            return components.Length > 0 && components.All(c =>
                c is Transform || c.GetType().FullName == "UnityEngine.CanvasRenderer" ||
                c.GetType().FullName == "UnityEngine.UI.LayoutElement" ||
                c.GetType().FullName == "TMPro.TMP_SubMeshUI");
        }

        public static Component[] SceneComponents()
        {
            long startedAt = CounterfactualTiming.Start();
            var allObjects = AllSceneGameObjects();
            var presentationIds = PresentationGameObjectIds(allObjects);
            try { return CausalComponents(AllSceneComponents(), presentationIds); }
            finally { CounterfactualTiming.End("scene.causal_component_filter", startedAt); }
        }

        static GameObject[] SceneGameObjects()
        {
            long startedAt = CounterfactualTiming.Start();
            var allObjects = AllSceneGameObjects();
            var presentationIds = PresentationGameObjectIds(allObjects);
            try { return allObjects.Where(go => !presentationIds.Contains(go.GetInstanceID())).ToArray(); }
            finally { CounterfactualTiming.End("scene.causal_gameobject_filter", startedAt); }
        }

        static string TransformPath(Transform transform)
        {
            if (transform == null) return "<none>";
            var parts = new List<string>();
            for (var current = transform; current != null; current = current.parent)
                parts.Add(current.name + "[" + current.GetSiblingIndex() + "]");
            parts.Reverse();
            return string.Join("/", parts);
        }

        static int ParentId(Transform transform) => transform != null && transform.parent != null
            ? transform.parent.gameObject.GetInstanceID() : 0;

        static string ParentPath(Transform transform) => transform != null && transform.parent != null
            ? TransformPath(transform.parent) : "<root>";

        static int ComponentOrdinal(Component component)
        {
            if (component == null || component.gameObject == null) return -1;
            var all = component.gameObject.GetComponents<Component>();
            for (int i = 0; i < all.Length; i++)
                if (ReferenceEquals(all[i], component)) return i;
            return -1;
        }

        static string RegistryMembership(Component component)
        {
            var memberships = new List<string>();
            if (component is WorldGeneration && ReferenceEquals(WorldGeneration.world, component))
                memberships.Add("WorldGeneration.world");
            if (component is Body && PlayerCamera.main != null &&
                ReferenceEquals(PlayerCamera.main.body, component))
                memberships.Add("PlayerCamera.main.body");
            if (component is PlayerCamera && ReferenceEquals(PlayerCamera.main, component))
                memberships.Add("PlayerCamera.main");
            if (component is RadiationLine && ReferenceEquals(RadiationLine.line, component))
                memberships.Add("RadiationLine.line");
            if (component is MinigameBase && ReferenceEquals(MinigameBase.main, component))
                memberships.Add("MinigameBase.main");
            if (component is Item item && Item.GlobalItems != null && !string.IsNullOrEmpty(item.id) &&
                Item.GlobalItems.ContainsKey(item.id))
                memberships.Add("Item.GlobalItems[" + item.id + "]");
            return memberships.Count == 0 ? "none-observed" : string.Join(",", memberships);
        }

        static Tuple<string, string> Classify(string name, Component[] components, string registry)
        {
            if (name == "PPO Look Debug Dot")
                return Tuple.Create("demonstrably-noncausal",
                    "created only by PPOBridge.ApplyPPOActions and only assigned transform/render properties");
            if (name.StartsWith("Dust", StringComparison.Ordinal) ||
                name == "ItemBreakParticle(Clone)" || components.Any(IsPresentationComponent))
                return Tuple.Create("derived-rebuildable",
                    "presentation-only particle/audio object with no gameplay registry membership");
            if (registry != "none-observed")
                return Tuple.Create("causal", "member of an observed gameplay singleton or item registry");
            if (components.Any(c => c is Body || c is Item || c is BuildingEntity ||
                c is WorldGeneration || c is Rigidbody2D || c is Collider2D || c is Joint2D))
                return Tuple.Create("causal", "gameplay entity or physics component");
            if (components.Any(c => c != null && c.GetType().Assembly == typeof(Body).Assembly))
                return Tuple.Create("causal", "game assembly component; no noncausal proof");
            return Tuple.Create("causal", "conservative default; no noncausal proof");
        }

        static string HashCategory(Component component)
        {
            if (component is WorldGeneration) return "world";
            if (component is Tilemap) return "tilemaps";
            if (component is FluidManager) return "fluids";
            if (component is Body) return "body";
            if (component is Item || component is BuildingEntity) return "entities_items";
            return "other";
        }

        static TopologyEntry DescribeGameObject(GameObject go)
        {
            var all = go.GetComponents<Component>().Where(c => c != null).ToArray();
            string registry = string.Join(",", all.Select(RegistryMembership)
                .Where(x => x != "none-observed").Distinct());
            if (string.IsNullOrEmpty(registry)) registry = "none-observed";
            var classification = Classify(go.name, all, registry);
            string scene = go.scene.path;
            string path = TransformPath(go.transform);
            return new TopologyEntry {
                kind = "GameObject", unity_instance_id = go.GetInstanceID(), type = "UnityEngine.GameObject",
                game_object_instance_id = go.GetInstanceID(), game_object_name = go.name, scene = scene,
                transform_path = path, parent_instance_id = ParentId(go.transform),
                parent_transform_path = ParentPath(go.transform), component_ordinal = -1,
                stable_identity = "scene:" + scene + "|hierarchy:" + path,
                registry_membership = registry, classification = classification.Item1,
                classification_basis = classification.Item2
            };
        }

        static TopologyEntry DescribeComponent(Component component)
        {
            string registry = RegistryMembership(component);
            var ownerComponents = component.gameObject.GetComponents<Component>()
                .Where(c => c != null).ToArray();
            var classification = Classify(component.gameObject.name, ownerComponents, registry);
            bool presentationLifecycle = IsProvenPresentationLifecycle(component);
            string scene = component.gameObject.scene.path;
            string path = TransformPath(component.transform);
            int ordinal = ComponentOrdinal(component);
            return new TopologyEntry {
                kind = "Component", unity_instance_id = component.GetInstanceID(),
                type = component.GetType().FullName,
                game_object_instance_id = component.gameObject.GetInstanceID(),
                game_object_name = component.gameObject.name, scene = scene,
                transform_path = path, parent_instance_id = ParentId(component.transform),
                parent_transform_path = ParentPath(component.transform), component_ordinal = ordinal,
                stable_identity = "scene:" + scene + "|hierarchy:" + path +
                    "|component_ordinal:" + ordinal + "|component_type:" + component.GetType().FullName,
                registry_membership = registry, classification = classification.Item1,
                classification_basis = classification.Item2,
                lifecycle_classification = presentationLifecycle ? TopologyInvariant.PresentationUiLifecycle : "causal",
                lifecycle_classification_basis = presentationLifecycle
                    ? PresentationLifecycleBasis(component)
                    : "active/enabled lifecycle remains causal unless an exact evidence-bound presentation rule matches"
            };
        }

        static List<TopologyChange> CompareTopology(
            IDictionary<int, TopologyEntry> baseline, IDictionary<int, TopologyEntry> live)
        {
            var changes = new List<TopologyChange>();
            foreach (var pair in baseline.OrderBy(p => p.Key))
                if (!live.ContainsKey(pair.Key))
                    changes.Add(new TopologyChange { change = "destroyed", entry = pair.Value });
            foreach (var pair in live.OrderBy(p => p.Key))
                if (!baseline.ContainsKey(pair.Key))
                    changes.Add(new TopologyChange { change = "created", entry = pair.Value });
            return changes;
        }

        public SemanticSnapshot(string output, int seed, byte[] observation,
            System.Action<ManagedGraph> captureHarness, bool detailed = true)
        {
            long snapshotStartedAt = CounterfactualTiming.Start();
            long gcMemoryBefore = GC.GetTotalMemory(false);
            int gc0Before = GC.CollectionCount(0);
            int gc1Before = GC.CollectionCount(1);
            int gc2Before = GC.CollectionCount(2);
            this.output = output; this.detailed = detailed;
            fingerprint = detailed ? null : new Fingerprint();
            Seed = seed; BaselineObservationWire = (byte[])observation.Clone();
            string rngBeforeFingerprint = RandomFingerprint(UnityEngine.Random.state);
            long sceneDiscoveryStartedAt = CounterfactualTiming.Start();
            var allBaselineComponents = AllSceneComponents();
            var allBaselineGameObjects = AllSceneGameObjects();
            CounterfactualTiming.End("capture.scene_discovery", sceneDiscoveryStartedAt);
            long topologyStartedAt = CounterfactualTiming.Start();
            var baselinePresentationIds = PresentationGameObjectIds(allBaselineGameObjects);
            var causalComponents = CausalComponents(allBaselineComponents, baselinePresentationIds);
            var causalGameObjectPlan = allBaselineGameObjects
                .Where(go => !baselinePresentationIds.Contains(go.GetInstanceID())).ToArray();
            TopologyCache plan = detailed ? null : topologyCache;
            bool topologyCacheHit = plan != null && plan.Matches(causalGameObjectPlan, causalComponents);
            if (!topologyCacheHit)
            {
                var allGameObjectPlan = detailed
                    ? allBaselineGameObjects.ToDictionary(go => go.GetInstanceID(), DescribeGameObject)
                    : causalGameObjectPlan.ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
                var allComponentPlan = detailed
                    ? allBaselineComponents.ToDictionary(c => c.GetInstanceID(), DescribeComponent)
                    : causalComponents.ToDictionary(c => c.GetInstanceID(), DescribeComponent);
                var gameObjectPlan = causalGameObjectPlan.ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
                var componentPlan = causalComponents.ToDictionary(c => c.GetInstanceID(), DescribeComponent);
                plan = new TopologyCache(allBaselineGameObjects, allBaselineComponents,
                    baselinePresentationIds, causalGameObjectPlan, causalComponents,
                    allGameObjectPlan, allComponentPlan, gameObjectPlan, componentPlan);
                topologyCache = plan;
            }
            allGameObjectTopology = plan.AllGameObjectTopology;
            allComponentTopology = plan.AllComponentTopology;
            components = plan.CausalComponents;
            var baselineGameObjects = plan.CausalGameObjects;
            gameObjectTopology = plan.GameObjectTopology;
            componentTopology = plan.ComponentTopology;
            snapshotComponentObjects = components.ToDictionary(c => c.GetInstanceID(), c => c);
            currentComponentObjects = new Dictionary<int, Component>(snapshotComponentObjects);
            foreach (var go in baselineGameObjects)
                RegisterStableIdentity(go, gameObjectTopology[go.GetInstanceID()].stable_identity);
            foreach (var component in components)
                RegisterStableIdentity(component, componentTopology[component.GetInstanceID()].stable_identity);
            CounterfactualTiming.End("capture.topology_fingerprint", topologyStartedAt);
            CounterfactualTiming.Report("capture:topology_cache=" + (topologyCacheHit ? "hit" : "miss"));
            Diagnostics.scene_components_visited = allBaselineComponents.Length;
            Diagnostics.scene_gameobjects_visited = allBaselineGameObjects.Length;
            Diagnostics.causal_components = components.Length;
            Diagnostics.causal_gameobjects = baselineGameObjects.Length;
            Diagnostics.topology_entries = allGameObjectTopology.Count + allComponentTopology.Count +
                gameObjectTopology.Count + componentTopology.Count;
            foreach (var entry in allGameObjectTopology.Values.Concat(allComponentTopology.Values)
                .Concat(gameObjectTopology.Values).Concat(componentTopology.Values))
                Diagnostics.topology_path_bytes += Encoding.UTF8.GetByteCount(
                    (entry.transform_path ?? "") + (entry.stable_identity ?? ""));
            Diagnostics.native_components_visited = components.Length;
            rng = UnityEngine.Random.state;
            timeScale = Time.timeScale; fixedDeltaTime = Time.fixedDeltaTime; maximumDeltaTime = Time.maximumDeltaTime;
            Clock["time"] = Time.time; Clock["fixedTime"] = Time.fixedTime;
            Clock["timeSinceLevelLoad"] = Time.timeSinceLevelLoad;
            Clock["unscaledTime"] = Time.unscaledTime; Clock["realtimeSinceStartup"] = Time.realtimeSinceStartup;
            Clock["frameCount"] = Time.frameCount;
            Clock["deltaTime"] = Time.deltaTime; Clock["unscaledDeltaTime"] = Time.unscaledDeltaTime;
            Graph = new ManagedGraph(Identity, t => t.Assembly == typeof(Body).Assembly ||
                t.Assembly == typeof(PPOBridge).Assembly, CaptureArray, detailed,
                (path, value) => fingerprint?.Add(path, value),
                IncludeManagedField);
            var staticTypes = new HashSet<Type>();
            foreach (var c in components)
            {
                // The topology cache has already validated the component
                // identity, type, hierarchy, and stable path. Reuse the
                // canonical root path instead of allocating it every state.
                string path = plan.ComponentPaths[c.GetInstanceID()];
                AddValue(path + "/name", c.gameObject.name);
                AddValue(path + "/scene", c.gameObject.scene.path);
                string category = HashCategory(c);
                long nativeStartedAt = CounterfactualTiming.Start();
                CaptureNative(c, path);
                CounterfactualTiming.End("hash." + category + ".native", nativeStartedAt);
                if (c.GetType().Assembly != typeof(Body).Assembly) continue;
                long managedStartedAt = CounterfactualTiming.Start();
                for (var t = c.GetType(); t != null && t.Assembly == typeof(Body).Assembly; t = t.BaseType)
                {
                    staticTypes.Add(t);
                    foreach (var f in ManagedGraph.DeclaredFields(t))
                        Graph.Field(c, f, path + "/" + t.Name + "." + f.Name);
                }
                CounterfactualTiming.End("hash." + category + ".managed", managedStartedAt);
            }
            // These registries are initialized by the existing game startup path.
            // Do not enumerate/read every assembly static: that could run new .cctors.
            foreach (Type t in new[] { typeof(Item), typeof(WorldGeneration), typeof(Climbable),
                typeof(FluidManager), typeof(LiquidAffect), typeof(Sound), typeof(Skills),
                typeof(SaveSystem), typeof(PauseHandler), typeof(PlayerCamera), typeof(RadiationLine) })
                staticTypes.Add(t);
            foreach (var t in staticTypes.OrderBy(t => t.FullName, StringComparer.Ordinal))
                foreach (var f in t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic |
                    BindingFlags.DeclaredOnly).Where(f => !f.IsLiteral)
                    .OrderBy(f => f.Name, StringComparer.Ordinal))
                    Graph.Field(null, f, "static/" + t.FullName + "/" + f.Name);
            if (detailed) WriteAnimatorDiagnostic("capture");
            if (detailed) WritePhysicsContactDiagnostic("capture");
            long harnessStartedAt = CounterfactualTiming.Start();
            captureHarness(Graph);
            CounterfactualTiming.End("capture.harness", harnessStartedAt);
            foreach (var pair in Graph.Evidence) State.Add(pair.Key, pair.Value);
            foreach (var gap in Graph.Gaps) Gaps.Add(gap);
            long rngStartedAt = CounterfactualTiming.Start();
            AddValue("rng/unity", rng);
            CounterfactualTiming.End("capture.rng", rngStartedAt);
            AddValue("clock/timeScale", timeScale);
            AddValue("clock/fixedDeltaTime", fixedDeltaTime);
            AddValue("clock/maximumDeltaTime", maximumDeltaTime);
            Gaps.Add("Engine contact manifolds, warm-start impulses, broadphase and fixed-step accumulator are not restorable through this adapter.");
            Gaps.Add("Absolute Unity clocks are observed but not rewound; game code reads Time.time/timeSinceLevelLoad/unscaledTime.");
            Gaps.Add("Active coroutine PCs, Invoke queues, delayed Destroy and animation events are not enumerable/restorable.");
            Gaps.Add("Non-scene native assets and non-live-type game statics require a further coverage audit.");
            SortedDictionary<string, string> digestSections = null;
            if (detailed)
            {
                long stateHashStartedAt = CounterfactualTiming.Start();
                StateHash = Hash(Encoding.UTF8.GetBytes(string.Join("\n", State
                    .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => pair.Key + "=" + pair.Value))));
                foreach (var group in State.GroupBy(pair => Section(pair.Key), StringComparer.Ordinal))
                    SectionHashes[group.Key] = Hash(Encoding.UTF8.GetBytes(string.Join("\n", group
                        .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                        .Select(pair => pair.Key + "=" + pair.Value))));
                CounterfactualTiming.End("capture.state_hash", stateHashStartedAt);
            }
            else
            {
                long fingerprintFinishStartedAt = CounterfactualTiming.Start();
                StateHash = fingerprint.Finish(out digestSections);
                CounterfactualTiming.End("capture.fingerprint_finalize", fingerprintFinishStartedAt);
            }
            if (!detailed) foreach (var pair in digestSections) SectionHashes.Add(pair.Key, pair.Value);
            Diagnostics.managed_root_fields = Graph.RootFieldsVisited;
            Diagnostics.managed_values_visited = Graph.DigestValuesVisited;
            Diagnostics.managed_objects_visited = Graph.DigestObjectsVisited;
            Diagnostics.managed_fields_visited = Graph.DigestFieldsVisited;
            Diagnostics.managed_values_emitted = Graph.DigestValuesEmitted;
            Diagnostics.managed_unique_references = Graph.UniqueReferenceIdentities;
            Diagnostics.managed_reference_reuses = Graph.ReusedReferenceIdentities;
            Diagnostics.managed_excluded_fields = Graph.ExcludedFieldCount;
            Diagnostics.managed_path_bytes = Graph.DigestPathBytes;
            Diagnostics.managed_value_bytes = Graph.DigestValueBytes;
            Diagnostics.managed_array_bytes = Graph.DigestArrayBytes;
            Diagnostics.managed_field_access_ticks = Graph.DigestFieldAccessTicks;
            Diagnostics.native_reflection_ticks = nativeReflectionTicks;
            Diagnostics.contact_pairs = contactPairPreview.Count;
            if (fingerprint != null)
            {
                Diagnostics.fingerprint_input_bytes = fingerprint.InputBytes;
                Diagnostics.managed_path_bytes = fingerprint.PathBytes;
                Diagnostics.managed_value_bytes = fingerprint.ValueBytes;
                Diagnostics.fingerprint_add_ticks = fingerprint.AddTicks;
                Diagnostics.fingerprint_finish_ticks = fingerprint.FinishTicks;
            }
            CounterfactualTiming.Report("managed_graph_digest_calls=" + Graph.DigestCaptureCalls +
                " managed_graph_digest_ms=" + CounterfactualTiming.Milliseconds(Graph.DigestCaptureTicks)
                    .ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
            Diagnostics.gc_total_memory_delta = GC.GetTotalMemory(false) - gcMemoryBefore;
            Diagnostics.gc_collections_gen0 = GC.CollectionCount(0) - gc0Before;
            Diagnostics.gc_collections_gen1 = GC.CollectionCount(1) - gc1Before;
            Diagnostics.gc_collections_gen2 = GC.CollectionCount(2) - gc2Before;
            CounterfactualTiming.Report("capture:scene_gameobjects=" + Diagnostics.scene_gameobjects_visited +
                " scene_components=" + Diagnostics.scene_components_visited +
                " causal_gameobjects=" + Diagnostics.causal_gameobjects +
                " causal_components=" + Diagnostics.causal_components +
                " topology_entries=" + Diagnostics.topology_entries +
                " topology_path_bytes=" + Diagnostics.topology_path_bytes +
                " native_components=" + Diagnostics.native_components_visited +
                " native_property_lookups=" + Diagnostics.native_property_lookups +
                " native_property_reads=" + Diagnostics.native_property_reads +
                " native_reflection_ms=" + CounterfactualTiming.Milliseconds(Diagnostics.native_reflection_ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                " contact_pairs=" + Diagnostics.contact_pairs +
                " managed_root_fields=" + Diagnostics.managed_root_fields +
                " managed_values=" + Diagnostics.managed_values_visited +
                " managed_objects=" + Diagnostics.managed_objects_visited +
                " managed_fields=" + Diagnostics.managed_fields_visited +
                " managed_emitted=" + Diagnostics.managed_values_emitted +
                " managed_unique_refs=" + Diagnostics.managed_unique_references +
                " managed_ref_reuses=" + Diagnostics.managed_reference_reuses +
                " managed_excluded_fields=" + Diagnostics.managed_excluded_fields +
                " managed_path_bytes=" + Diagnostics.managed_path_bytes +
                " managed_value_bytes=" + Diagnostics.managed_value_bytes +
                " managed_array_bytes=" + Diagnostics.managed_array_bytes +
                " managed_field_access_ms=" + CounterfactualTiming.Milliseconds(Diagnostics.managed_field_access_ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                " tilemaps=" + Diagnostics.tilemaps_examined +
                " tilemap_cells=" + Diagnostics.tilemap_cells_examined +
                " array_hash_bytes=" + Diagnostics.array_hash_bytes +
                " fingerprint_input_bytes=" + Diagnostics.fingerprint_input_bytes +
                " fingerprint_add_ms=" + CounterfactualTiming.Milliseconds(Diagnostics.fingerprint_add_ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                " fingerprint_finish_ms=" + CounterfactualTiming.Milliseconds(Diagnostics.fingerprint_finish_ticks).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                " gc_memory_delta=" + Diagnostics.gc_total_memory_delta +
                " gc0=" + Diagnostics.gc_collections_gen0 +
                " gc1=" + Diagnostics.gc_collections_gen1 +
                " gc2=" + Diagnostics.gc_collections_gen2);
            string rngAfterFingerprint = RandomFingerprint(UnityEngine.Random.state);
            CounterfactualTiming.Report("capture:rng_before=" + rngBeforeFingerprint +
                " rng_captured=" + RandomFingerprint(rng) +
                " rng_after=" + rngAfterFingerprint +
                " rng_unchanged=" + (rngBeforeFingerprint == rngAfterFingerprint ? "1" : "0"));
            if (!UnityEngine.Random.state.Equals(rng))
                throw new InvalidOperationException("CAPTURE_SIDE_EFFECT: Unity RNG changed during read-only capture");
            CounterfactualTiming.End("SemanticSnapshot", snapshotStartedAt);
        }
        static string Section(string path)
        {
            int first = path.IndexOf('/');
            if (first < 0) return path;
            int second = path.IndexOf('/', first + 1);
            return second < 0 ? path : path.Substring(0, second);
        }
        static bool IncludeManagedField(object target, FieldInfo field, string path)
        {
            // Evidence-bound presentation exclusions. These values are not
            // consulted by gameplay: Limb.randDripMult is read only by
            // Limb.FurBloodUpdate to drive WaterDrip particle emission, and
            // ConsoleScript's log buffers are diagnostic/UI history.
            if (target is Limb && field.Name == "randDripMult") return false;
            // Body eye timers are decremented by Body but consumed only by
            // FacialExpression to select visual eye sprites. Their writers
            // include gameplay events, but the values have no gameplay reader
            // or physics/contact effect, so they are derived presentation
            // state rather than causal body state.
            if (target is Body && (field.Name == "eyeCloseTime" ||
                field.Name == "eyePanicTime" || field.Name == "eyeScareTime")) return false;
            // Body.overrideLookTime is the unscaled lifetime of the visual
            // LookTarget override. Its only readers are Body.peekingBehind
            // and Body's visual look-placement code; it has no gameplay,
            // physics, inventory, or navigation reader.
            if (target is Body && field.Name == "overrideLookTime") return false;
            if (target is ConsoleScript &&
                (field.Name == "logs" || field.Name == "lastErr")) return false;

            // Startup timing audit: these fields are not simulation inputs.
            // genTimePassed is written only by the generation loading UI;
            // realTimeElapsed is consumed only by SaveSystem.SaveGame when
            // writing persistent run metadata. Saving is outside the replay
            // action/protocol contract, so wall-clock generation time must not
            // make an otherwise identical entry state fail A/A.
            if (target is WorldGeneration &&
                (field.Name == "genTimePassed" || field.Name == "realTimeElapsed")) return false;

            // PlayerCamera.alertTime is written by DoAlert and read only by
            // HandleAlertPopupOpacity to render the alert overlay.
            if (target is PlayerCamera && field.Name == "alertTime") return false;

            // RPCManager.timer only schedules Discord presence polling; it is
            // external presentation/network bookkeeping and has no gameplay
            // reader.
            if (target is RPCManager && field.Name == "timer") return false;

            // LiquidAffect.lastSplash is only the cooldown around splash
            // audio (the water damage/physics path does not consume it).
            if (target == null && field.DeclaringType == typeof(LiquidAffect) &&
                field.Name == "lastSplash") return false;

            // Keep every other managed field covered by the conservative
            // default.
            return true;
        }
        void CaptureArray(string path, Array array)
        {
            // Content-addressed blobs retain fallback diffs without keeping a
            // trajectory's worth of worlds in RAM.
            int count = Buffer.ByteLength(array);
            Diagnostics.array_hash_bytes += count;
            long arrayHashStartedAt = CounterfactualTiming.Start();
            string hash;
            byte[] bytes = null;
            if (detailed)
            {
                bytes = new byte[count];
                Buffer.BlockCopy(array, 0, bytes, 0, count);
                hash = Hash(bytes);
            }
            else hash = HashPrimitiveArray(array, count, arrayHashScratch);
            CounterfactualTiming.End("hash.array_bytes", arrayHashStartedAt);
            string file = Path.Combine(output, "arrays", hash + ".bin");
            if (detailed && !File.Exists(file))
            {
                using (var s = new FileStream(file, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                    s.Write(bytes, 0, bytes.Length);
            }
            if (detailed)
            {
                State[path + "/sha256"] = hash;
                Arrays[path] = new { sha256 = hash, bytes = count,
                    element = array.GetType().GetElementType().FullName,
                    shape = Enumerable.Range(0, array.Rank).Select(array.GetLength).ToArray(),
                    lower_bounds = Enumerable.Range(0, array.Rank).Select(array.GetLowerBound).ToArray() };
            }
            else fingerprint.Add(path + "/sha256", hash);
        }
        string HashPrimitiveArray(Array array, int count, byte[] scratch)
        {
            arraySha.Initialize();
            int offset = 0;
            while (offset < count)
            {
                int length = Math.Min(scratch.Length, count - offset);
                Buffer.BlockCopy(array, offset, scratch, 0, length);
                arraySha.TransformBlock(scratch, 0, length, scratch, 0);
                offset += length;
            }
            arraySha.TransformFinalBlock(EmptyHashInput, 0, 0);
            return BitConverter.ToString(arraySha.Hash).Replace("-", "").ToLowerInvariant();
        }
        public static string Hash(byte[] bytes)
        {
            using (var sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }
        public static string FileHash(string path)
        {
            using (var sha = SHA256.Create()) using (var file = File.OpenRead(path))
                return BitConverter.ToString(sha.ComputeHash(file)).Replace("-", "").ToLowerInvariant();
        }
        void AddContactField(string path, string value)
        {
            if (detailed) State[path] = value;
            else fingerprint.Add(path, value);
        }
        void AddContact(string path, ContactPoint2D contact)
        {
            // ContactPoint2D's reflected m_* object fields contain Unity
            // instance IDs. Those IDs are process-local, so comparing them
            // across fresh A/A processes creates a false divergence even when
            // the contact pair and all physical measurements are identical.
            // Canonicalize only the identity-bearing fields to the already
            // registered stable scene/component identities; retain enabled,
            // point, normal, separation, and both impulses/relative velocity.
            AddContactField(path + "/m_Collider", Identity(contact.collider));
            AddContactField(path + "/m_Enabled", ManagedGraph.Scalar(contact.enabled ? 1 : 0));
            AddContactField(path + "/m_Normal/x", ManagedGraph.Scalar(contact.normal.x));
            AddContactField(path + "/m_Normal/y", ManagedGraph.Scalar(contact.normal.y));
            AddContactField(path + "/m_NormalImpulse", ManagedGraph.Scalar(contact.normalImpulse));
            AddContactField(path + "/m_OtherCollider", Identity(contact.otherCollider));
            AddContactField(path + "/m_OtherRigidbody", Identity(contact.otherRigidbody));
            AddContactField(path + "/m_Point/x", ManagedGraph.Scalar(contact.point.x));
            AddContactField(path + "/m_Point/y", ManagedGraph.Scalar(contact.point.y));
            AddContactField(path + "/m_RelativeVelocity/x", ManagedGraph.Scalar(contact.relativeVelocity.x));
            AddContactField(path + "/m_RelativeVelocity/y", ManagedGraph.Scalar(contact.relativeVelocity.y));
            AddContactField(path + "/m_Rigidbody", Identity(contact.rigidbody));
            AddContactField(path + "/m_Separation", ManagedGraph.Scalar(contact.separation));
            AddContactField(path + "/m_TangentImpulse", ManagedGraph.Scalar(contact.tangentImpulse));
        }
        void AddValue(string path, object value)
        {
            if (!detailed)
            {
                if (value == null) { fingerprint.Add(path, "null"); return; }
                string fastIdentity = Identity(value);
                if (fastIdentity != null) { fingerprint.Add(path, fastIdentity); return; }
                if (value is ContactPoint2D fastContact) { AddContact(path, fastContact); return; }
                Type scalarType = value.GetType();
                if (scalarType.IsPrimitive || scalarType.IsEnum || value is string || value is decimal)
                { fingerprint.Add(path, ManagedGraph.Scalar(value)); return; }
                nativeDigestSeen.Clear();
                DigestValue(path, value, nativeDigestSeen, 0);
                return;
            }
            if (value == null) { State[path] = "null"; return; }
            string identity = Identity(value);
            if (identity != null) { State[path] = identity; return; }
            if (value is ContactPoint2D contact) { AddContact(path, contact); return; }
            Type t = value.GetType();
            if (t.IsPrimitive || t.IsEnum || value is string || value is decimal)
            { State[path] = ManagedGraph.Scalar(value); return; }
            if (value is Array a)
            { int i = 0; foreach (var x in a) AddValue(path + "/" + i++, x); return; }
            foreach (var f in ManagedGraph.Fields(t)) AddValue(path + "/" + f.Name, f.GetValue(value));
        }
        void DigestValue(string path, object value, Dictionary<object, string> seen, int depth)
        {
            if (value == null) { fingerprint.Add(path, "null"); return; }
            if (depth > 128) throw new InvalidOperationException("Native fingerprint depth exceeded: " + path);
            string identity = Identity(value);
            if (identity != null) { fingerprint.Add(path, identity); return; }
            if (value is ContactPoint2D contact) { AddContact(path, contact); return; }
            Type t = value.GetType();
            if (t.IsPrimitive || t.IsEnum || value is string || value is decimal)
            { fingerprint.Add(path, ManagedGraph.Scalar(value)); return; }
            if (!t.IsValueType && seen.TryGetValue(value, out var prior))
            { fingerprint.Add(path, "ref:" + prior); return; }
            if (!t.IsValueType) seen[value] = path;
            fingerprint.Add(path, "type:" + t.FullName);
            if (value is Array a)
            {
                fingerprint.Add(path + "/shape", string.Join(",", Enumerable.Range(0, a.Rank)
                    .Select(i => a.GetLowerBound(i) + ":" + a.GetLength(i))));
                if (t.GetElementType().IsPrimitive) CaptureArray(path, a);
                else { int i = 0; foreach (var x in a) DigestValue(path + "/" + i++, x, seen, depth + 1); }
                return;
            }
            foreach (var f in ManagedGraph.Fields(t))
                DigestValue(path + "/" + f.Name, f.GetValue(value), seen, depth + 1);
        }
        sealed class UnityIdentity : IEqualityComparer<object>
        {
            public new bool Equals(object left, object right) => ReferenceEquals(left, right);
            public int GetHashCode(object value) => RuntimeHelpers.GetHashCode(value);
        }
        static PropertyInfo NativeProperty(Type type, string name)
        {
            lock (NativePropertyCacheLock)
            {
                if (!NativePropertyCache.TryGetValue(type, out var byName))
                {
                    byName = new Dictionary<string, PropertyInfo>(StringComparer.Ordinal);
                    NativePropertyCache.Add(type, byName);
                }
                if (!byName.TryGetValue(name, out var property))
                {
                    property = type.GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
                    byName.Add(name, property);
                }
                return property;
            }
        }
        void Props(NativeRecord n, string path, params string[] properties)
        {
            foreach (string name in properties)
            {
                Diagnostics.native_property_lookups++;
                long reflectionStartedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                var p = NativeProperty(n.Target.GetType(), name);
                if (p == null || !p.CanRead || !p.CanWrite)
                    throw new MissingMemberException(n.Target.GetType().FullName, name);
                var value = p.GetValue(n.Target);
                nativeReflectionTicks += System.Diagnostics.Stopwatch.GetTimestamp() - reflectionStartedAt;
                Diagnostics.native_property_reads++;
                if (value is Array a) value = a.Clone();
                if (detailed) n.Properties.Add((p, value));
                AddValue(path + "/native/" + name, value);
            }
        }
        void CaptureNative(Component c, string path)
        {
            bool trackLifecycle = !IsProvenPresentationLifecycle(c);
            var n = new NativeRecord { Target = c, TrackLifecycle = trackLifecycle,
                Active = c.gameObject.activeSelf, Layer = c.gameObject.layer,
                Parent = c.transform.parent, Enabled = c is Behaviour b ? (bool?)b.enabled : null };
            if (detailed) native.Add(n);
            if (trackLifecycle)
            {
                AddValue(path + "/activeSelf", n.Active);
                AddValue(path + "/enabled", n.Enabled);
            }
            else
            {
                // Emit a stable canonical value in both detailed snapshots
                // and lightweight digests; never hash the volatile renderer
                // lifecycle bits in either direction.
                AddValue(path + "/lifecycle", TopologyInvariant.PresentationUiLifecycle + ":" +
                    PresentationLifecycleKey(c));
            }
            AddValue(path + "/layer", n.Layer); AddValue(path + "/parent", n.Parent);
            if (c is Transform)
                Props(n, path, "localPosition", "localRotation", "localScale");
            else if (c is Rigidbody2D rb)
            {
                Props(n, path, BodyProperties); n.Sleeping = rb.IsSleeping();
                AddValue(path + "/sleeping", n.Sleeping);
                var contacts = new ContactPoint2D[64];
                int count = rb.GetContacts(contacts);
                n.Contacts = detailed ? contacts.Take(count).ToArray() : null;
                if (detailed)
                {
                    for (int contactIndex = 0; contactIndex < count; contactIndex++)
                    {
                        var contact = contacts[contactIndex];
                        string pair = path + "->collider=" + Identity(contact.collider) +
                            "|otherCollider=" + Identity(contact.otherCollider) +
                            "|otherRigidbody=" + Identity(contact.rigidbody);
                        contactPairPreview.Add(pair);
                    }
                }
                AddValue(path + "/contacts", contacts.Take(count).ToArray());
                if (count == contacts.Length) Gaps.Add(path + ": contact diagnostic buffer potentially truncated");
            }
            else if (c is Collider2D)
            {
                Props(n, path, ColliderProperties);
                if (c is BoxCollider2D) Props(n, path, "size", "edgeRadius", "autoTiling");
                else if (c is CircleCollider2D) Props(n, path, "radius");
                else if (c is CapsuleCollider2D) Props(n, path, "size", "direction");
                else if (c is EdgeCollider2D) Props(n, path, "points", "edgeRadius");
                else if (c is PolygonCollider2D poly)
                {
                    n.PolygonPaths = Enumerable.Range(0, poly.pathCount).Select(poly.GetPath).ToArray();
                    AddValue(path + "/polygonPaths", n.PolygonPaths);
                }
                else if (!(c is TilemapCollider2D) && !(c is CompositeCollider2D)) Gaps.Add(path + ": collider adapter missing");
            }
            else if (c is HingeJoint2D)
                Props(n, path, "enabled", "connectedBody", "enableCollision", "breakForce", "breakTorque",
                    "autoConfigureConnectedAnchor", "anchor", "connectedAnchor", "useMotor", "motor", "useLimits", "limits");
            else if (c is Joint2D) Gaps.Add(path + ": joint adapter missing");
            else if (c is Animator animator)
            {
                Props(n, path, "speed", "applyRootMotion", "updateMode", "cullingMode", "fireEvents");
                n.AnimatorStates = new AnimatorStateInfo[animator.layerCount];
                n.AnimatorNextStates = new AnimatorStateInfo[animator.layerCount];
                n.AnimatorTransitions = new AnimatorTransitionInfo[animator.layerCount];
                n.AnimatorInTransitions = new bool[animator.layerCount];
                n.LayerWeights = new float[animator.layerCount];
                for (int i = 0; i < animator.layerCount; i++)
                {
                    n.AnimatorStates[i] = animator.GetCurrentAnimatorStateInfo(i);
                    n.AnimatorNextStates[i] = animator.GetNextAnimatorStateInfo(i);
                    n.AnimatorTransitions[i] = animator.GetAnimatorTransitionInfo(i);
                    n.LayerWeights[i] = animator.GetLayerWeight(i);
                    AddValue(path + "/animator/state/" + i, n.AnimatorStates[i]);
                    AddValue(path + "/animator/weight/" + i, n.LayerWeights[i]);
                    bool inTransition = animator.IsInTransition(i);
                    n.AnimatorInTransitions[i] = inTransition;
                    AddValue(path + "/animator/inTransition/" + i, inTransition);
                    AddValue(path + "/animator/nextState/" + i, n.AnimatorNextStates[i]);
                    AddValue(path + "/animator/transition/" + i, n.AnimatorTransitions[i]);
                    if (inTransition) Gaps.Add(path + ": animator transition active at layer " + i);
                }
                n.Parameters = animator.parameters;
                n.ParameterValues = new object[n.Parameters.Length];
                for (int i = 0; i < n.Parameters.Length; i++)
                {
                    var p = n.Parameters[i];
                    switch (p.type)
                    {
                        case AnimatorControllerParameterType.Float: n.ParameterValues[i] = animator.GetFloat(p.nameHash); break;
                        case AnimatorControllerParameterType.Int: n.ParameterValues[i] = animator.GetInteger(p.nameHash); break;
                        case AnimatorControllerParameterType.Bool: n.ParameterValues[i] = animator.GetBool(p.nameHash); break;
                        default:
                            // Unity exposes trigger definitions and names but
                            // no non-mutating API for a trigger's pending bit.
                            Gaps.Add(path + ": unobservable Animator trigger " + p.name);
                            break;
                    }
                    AddValue(path + "/animator/parameter/" + p.name, n.ParameterValues[i]);
                    AddValue(path + "/animator/parameterType/" + p.name, p.type);
                }
            }
            else if (c is Tilemap map)
            {
                var bounds = map.cellBounds;
                long size = (long)bounds.size.x * bounds.size.y * bounds.size.z;
                if (size > 2_000_000) throw new InvalidOperationException("Tilemap capture size budget exceeded");
                Diagnostics.tilemaps_examined++;
                Diagnostics.tilemap_cells_examined += size;
                if (!detailed)
                {
                    var tiles = map.GetTilesBlock(bounds);
                    CaptureArray(path + "/tiles", tiles.Select(t => t == null ? 0 : t.GetInstanceID()).ToArray());
                    var matrices = new float[size * 16];
                    var flags = new int[size];
                    int digestIndex = 0;
                    foreach (var pos in bounds.allPositionsWithin)
                    {
                        Matrix4x4 tileMatrix = map.GetTransformMatrix(pos);
                        for (int j = 0; j < 16; j++) matrices[digestIndex * 16 + j] = tileMatrix[j];
                        flags[digestIndex++] = (int)map.GetTileFlags(pos);
                    }
                    CaptureArray(path + "/tileMatrices", matrices);
                    CaptureArray(path + "/tileFlags", flags);
                    AddValue(path + "/bounds", bounds);
                    return;
                }
                var tr = new TileRecord { Map = map, Bounds = bounds, Tiles = map.GetTilesBlock(bounds),
                    Matrices = new Matrix4x4[size], Flags = new TileFlags[size] };
                int i = 0;
                foreach (var pos in bounds.allPositionsWithin)
                { tr.Matrices[i] = map.GetTransformMatrix(pos); tr.Flags[i++] = map.GetTileFlags(pos); }
                tilemaps.Add(tr);
                CaptureArray(path + "/tiles", tr.Tiles.Select(t => t == null ? 0 : t.GetInstanceID()).ToArray());
                var matrix = new float[tr.Matrices.Length * 16];
                for (int k = 0; k < tr.Matrices.Length; k++) for (int j = 0; j < 16; j++) matrix[k * 16 + j] = tr.Matrices[k][j];
                CaptureArray(path + "/tileMatrices", matrix);
                CaptureArray(path + "/tileFlags", tr.Flags.Select(f => (int)f).ToArray());
                AddValue(path + "/bounds", bounds);
            }
            else if (c is Renderer) Props(n, path, "enabled");
            else if (c is Camera) Props(n, path, "orthographic", "orthographicSize", "fieldOfView", "aspect");
            else if (c.GetType().Assembly != typeof(Body).Assembly && !IsProvenPresentationLifecycle(c))
                Gaps.Add(path + ": native/third-party component not fully captured: " + c.GetType().FullName);
        }

        string WriteTopologyDiagnostic(Component[] ignoredLiveComponents)
        {
            long startedAt = CounterfactualTiming.Start();
            var allLiveObjects = AllSceneGameObjects();
            var allLiveComponents = AllSceneComponents();
            var liveObjects = allLiveObjects.ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
            var liveComponentMap = allLiveComponents.ToDictionary(c => c.GetInstanceID(), DescribeComponent);
            var livePresentationIds = PresentationGameObjectIds(allLiveObjects);
            var causalLiveObjects = allLiveObjects.Where(go => !livePresentationIds.Contains(go.GetInstanceID()))
                .ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
            var causalLiveComponents = CausalComponents(allLiveComponents, livePresentationIds)
                .ToDictionary(c => c.GetInstanceID(), DescribeComponent);
            var allChanges = CompareTopology(allGameObjectTopology, liveObjects);
            allChanges.AddRange(CompareTopology(allComponentTopology, liveComponentMap));
            var changes = CompareTopology(gameObjectTopology, causalLiveObjects);
            changes.AddRange(CompareTopology(componentTopology, causalLiveComponents));
            string filename = "topology_diff_" + (topologyDiagnosticIndex++).ToString("D4") + ".json";
            string path = Path.Combine(output, filename);
            var payload = new {
                schema = Schema, baseline_gameobjects = gameObjectTopology.Count,
                live_gameobjects = causalLiveObjects.Count, baseline_components = componentTopology.Count,
                live_components = causalLiveComponents.Count, all_baseline_gameobjects = allGameObjectTopology.Count,
                all_live_gameobjects = liveObjects.Count, all_baseline_components = allComponentTopology.Count,
                all_live_components = liveComponentMap.Count, changes, all_changes = allChanges,
                causal_changes = changes.Count(c => c.entry.classification == "causal"),
                derived_rebuildable_changes = changes.Count(c => c.entry.classification == "derived-rebuildable"),
                demonstrably_noncausal_changes = changes.Count(c => c.entry.classification == "demonstrably-noncausal"),
                presentation_changes = allChanges.Count(c => c.entry.classification != "causal"),
                exclusion_decision = "none; topology guard remains enforced until this evidence is reviewed"
            };
            File.WriteAllText(path, JsonConvert.SerializeObject(payload, Formatting.Indented));
            CounterfactualTiming.End("topology.diagnostic", startedAt);
            return filename;
        }

        void RestoreProgress(string stage)
        {
            string line = DateTime.UtcNow.ToString("O") + " " + stage + Environment.NewLine;
            Debug.Log("[CB1] CF_RESTORE " + stage);
            try { File.AppendAllText(Path.Combine(output, "restore_progress.log"), line); }
            catch (Exception e) { Debug.LogWarning("[CB1] CF_RESTORE progress write failed: " + e.Message); }
        }

        static bool SameTopology(
            IDictionary<int, TopologyEntry> baselineObjects,
            IDictionary<int, TopologyEntry> liveObjects,
            IDictionary<int, TopologyEntry> baselineComponents,
            IDictionary<int, TopologyEntry> liveComponents)
        {
            // GameObject IDs must remain identical: causal GameObject
            // recreation is deliberately unsupported. Component IDs may
            // change when a missing component is recreated, so compare its
            // surviving GameObject/type identity instead.
            var expected = baselineObjects.Values
                .Select(e => new TopologyInvariantRecord {
                    Identity = "GameObject|" + e.unity_instance_id,
                    Classification = e.classification
                })
                .Concat(baselineComponents.Values.Select(e => new TopologyInvariantRecord {
                    Identity = "Component|" + e.game_object_instance_id + "|" + e.type,
                    Classification = e.classification
                }));
            var actual = liveObjects.Values
                .Select(e => new TopologyInvariantRecord {
                    Identity = "GameObject|" + e.unity_instance_id,
                    Classification = e.classification
                })
                .Concat(liveComponents.Values.Select(e => new TopologyInvariantRecord {
                    Identity = "Component|" + e.game_object_instance_id + "|" + e.type,
                    Classification = e.classification
                }));
            return TopologyInvariant.Equivalent(expected, actual);
        }

        void RestoreTopology()
        {
            long startedAt = CounterfactualTiming.Start();
            RestoreProgress("topology:start");
            var liveObjects = SceneGameObjects().ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
            var liveComponents = SceneComponents();
            var liveComponentMap = liveComponents.ToDictionary(c => c.GetInstanceID(), DescribeComponent);
            var changes = CompareTopology(gameObjectTopology, liveObjects);
            changes.AddRange(CompareTopology(componentTopology, liveComponentMap));

            // A causal GameObject cannot be reconstructed without a stable
            // prefab/registry identity. Keep the guard hard in that case.
            if (changes.Any(c => c.entry.kind == "GameObject"))
                throw new InvalidOperationException("RESTORE_BLOCKED_TOPOLOGY_CAUSAL_GAMEOBJECT");
            if (changes.Any(c => c.entry.kind == "Component" && c.change == "created"))
                throw new InvalidOperationException("RESTORE_BLOCKED_TOPOLOGY_CAUSAL_COMPONENT");

            // Recreate missing components on surviving snapshot GameObjects.
            // The managed graph and native records are rebound before any
            // restore write, so fields pointing at the old component reconnect.
            RestoreProgress("topology:validated");
            var afterCleanup = SceneComponents();
            var afterIds = new HashSet<int>(afterCleanup.Select(c => c.GetInstanceID()));
            foreach (var entry in componentTopology.Values.OrderBy(e => e.unity_instance_id))
            {
                if (afterIds.Contains(entry.unity_instance_id)) continue;
                var go = SceneGameObjects().FirstOrDefault(x => x.GetInstanceID() == entry.game_object_instance_id);
                if (go == null || entry.classification != "causal")
                    throw new InvalidOperationException(
                        "RESTORE_BLOCKED_TOPOLOGY_UNRESTORABLE: missing " + entry.stable_identity);
                if (!currentComponentObjects.TryGetValue(entry.unity_instance_id, out var previous))
                    throw new InvalidOperationException(
                        "RESTORE_BLOCKED_TOPOLOGY_NO_ORIGINAL_COMPONENT: " + entry.stable_identity);
                Type type = typeof(Body).Assembly.GetType(entry.type, false);
                if (type == null || !typeof(Component).IsAssignableFrom(type))
                    throw new InvalidOperationException(
                        "RESTORE_BLOCKED_TOPOLOGY_NO_STABLE_TYPE: " + entry.type);
                RestoreProgress("topology:add-component:" + entry.type);
                var replacement = go.AddComponent(type);
                // Hold a recreated behaviour disabled until Graph.Restore has
                // rebound its managed dependencies. This prevents Start from
                // dereferencing a not-yet-restored field.
                if (replacement is Behaviour replacementBehaviour)
                    replacementBehaviour.enabled = false;
                // A causal component may be a legitimate one-shot object
                // that destroys itself again after the first restore. Rebind
                // both the current replacement and the original snapshot
                // identity so repeated restores never retain a destroyed
                // intermediate wrapper.
                Graph.RebindUnityObject(previous, replacement);
                if (!ReferenceEquals(previous, snapshotComponentObjects[entry.unity_instance_id]))
                    Graph.RebindUnityObject(snapshotComponentObjects[entry.unity_instance_id], replacement);
                for (int i = 0; i < components.Length; i++)
                    if (ReferenceEquals(components[i], previous)) components[i] = replacement;
                foreach (var record in native)
                    if (ReferenceEquals(record.Target, previous))
                    {
                        record.Target = replacement;
                        record.RecreatedForRestore = true;
                    }
                RegisterStableIdentity(replacement, entry.stable_identity);
                currentComponentObjects[entry.unity_instance_id] = replacement;
                afterIds.Add(replacement.GetInstanceID());
            }

            var finalObjects = SceneGameObjects().ToDictionary(go => go.GetInstanceID(), DescribeGameObject);
            var finalComponents = SceneComponents().ToDictionary(c => c.GetInstanceID(), DescribeComponent);
            RestoreProgress("topology:final-check");
            if (!SameTopology(gameObjectTopology, finalObjects, componentTopology, finalComponents))
                throw new InvalidOperationException("RESTORE_BLOCKED_TOPOLOGY_UNRESTORED");
            RestoreProgress("topology:complete");
            CounterfactualTiming.End("restore.topology", startedAt);
        }

        static bool SameTile(TileBase left, TileBase right)
        {
            if (left == null || right == null) return left == null && right == null;
            return left.GetInstanceID() == right.GetInstanceID();
        }

        static bool SameMatrix(Matrix4x4 left, Matrix4x4 right) => left.Equals(right);

        static bool SameTilemapBounds(BoundsInt left, BoundsInt right) =>
            left.xMin == right.xMin && left.yMin == right.yMin && left.zMin == right.zMin &&
            left.size == right.size;

        static bool Contains(BoundsInt bounds, Vector3Int position) =>
            position.x >= bounds.xMin && position.x < bounds.xMax &&
            position.y >= bounds.yMin && position.y < bounds.yMax &&
            position.z >= bounds.zMin && position.z < bounds.zMax;

        static int TileIndex(BoundsInt bounds, Vector3Int position) =>
            ((position.z - bounds.zMin) * bounds.size.y + (position.y - bounds.yMin)) * bounds.size.x +
            (position.x - bounds.xMin);

        static BoundsInt Union(BoundsInt left, BoundsInt right)
        {
            var min = new Vector3Int(Math.Min(left.xMin, right.xMin), Math.Min(left.yMin, right.yMin),
                Math.Min(left.zMin, right.zMin));
            var max = new Vector3Int(Math.Max(left.xMax, right.xMax), Math.Max(left.yMax, right.yMax),
                Math.Max(left.zMax, right.zMax));
            return new BoundsInt(min, max - min);
        }

        static int CountDifferentCells(TileRecord snapshot, BoundsInt liveBounds, TileBase[] liveTiles)
        {
            int different = 0;
            foreach (var position in Union(snapshot.Bounds, liveBounds).allPositionsWithin)
            {
                bool inSnapshot = Contains(snapshot.Bounds, position);
                bool inLive = Contains(liveBounds, position);
                TileBase expectedTile = inSnapshot ? snapshot.Tiles[TileIndex(snapshot.Bounds, position)] : null;
                TileBase currentTile = inLive ? liveTiles[TileIndex(liveBounds, position)] : null;
                Matrix4x4 expectedMatrix = inSnapshot ? snapshot.Matrices[TileIndex(snapshot.Bounds, position)] : Matrix4x4.identity;
                Matrix4x4 currentMatrix = inLive ? snapshot.Map.GetTransformMatrix(position) : Matrix4x4.identity;
                TileFlags expectedFlags = inSnapshot ? snapshot.Flags[TileIndex(snapshot.Bounds, position)] : TileFlags.None;
                TileFlags currentFlags = inLive ? snapshot.Map.GetTileFlags(position) : TileFlags.None;
                if (!SameTile(currentTile, expectedTile) || !SameMatrix(currentMatrix, expectedMatrix) ||
                    currentFlags != expectedFlags) different++;
            }
            return different;
        }

        static bool SameOrDescendant(Transform candidate, Transform ancestor)
        {
            if (candidate == null || ancestor == null) return false;
            for (var current = candidate; current != null; current = current.parent)
                if (ReferenceEquals(current, ancestor)) return true;
            return false;
        }

        static bool SemanticallyEqual(object left, object right)
        {
            if (ReferenceEquals(left, right)) return true;
            if (left == null || right == null) return false;
            if (left is Object leftObject && right is Object rightObject)
                return Identity(leftObject) == Identity(rightObject);
            if (left is Array leftArray && right is Array rightArray)
            {
                if (leftArray.Length != rightArray.Length || leftArray.Rank != rightArray.Rank) return false;
                for (int i = 0; i < leftArray.Length; i++)
                    if (!SemanticallyEqual(leftArray.GetValue(i), rightArray.GetValue(i))) return false;
                return true;
            }
            return left.Equals(right);
        }

        bool RestoreTilemaps(RestoreMetrics metrics)
        {
            long startedAt = CounterfactualTiming.Start();
            foreach (var t in tilemaps)
            {
                var currentBounds = t.Map.cellBounds;
                bool boundsChanged = !SameTilemapBounds(currentBounds, t.Bounds);
                TileBase[] currentTiles = t.Map.GetTilesBlock(t.Bounds);
                int changedCells = 0;
                if (boundsChanged)
                {
                    // A bounds change means cells outside the captured bounds
                    // may exist. Count the actual union difference for the
                    // diagnostic, but clear once and restore the complete
                    // captured tilemap for correctness.
                    changedCells = CountDifferentCells(t, currentBounds, t.Map.GetTilesBlock(currentBounds));
                }
                else
                {
                    int index = 0;
                    foreach (var pos in t.Bounds.allPositionsWithin)
                    {
                        bool different = !SameTile(currentTiles[index], t.Tiles[index]) ||
                            !SameMatrix(t.Map.GetTransformMatrix(pos), t.Matrices[index]) ||
                            t.Map.GetTileFlags(pos) != t.Flags[index];
                        if (different) changedCells++;
                        index++;
                    }
                }
                if (changedCells == 0) continue;

                metrics.ChangedTilemaps++;
                metrics.ChangedCells += changedCells;
                metrics.ChangedTilemapObjects.Add(t.Map);
                if (boundsChanged)
                {
                    metrics.FullTilemapRewrites++;
                    t.Map.ClearAllTiles();
                    t.Map.SetTilesBlock(t.Bounds, t.Tiles);
                }
                int i = 0;
                foreach (var pos in t.Bounds.allPositionsWithin)
                {
                    bool different = boundsChanged || !SameTile(currentTiles[i], t.Tiles[i]) ||
                        !SameMatrix(t.Map.GetTransformMatrix(pos), t.Matrices[i]) ||
                        t.Map.GetTileFlags(pos) != t.Flags[i];
                    if (different)
                    {
                        t.Map.SetTileFlags(pos, TileFlags.None);
                        if (!boundsChanged && !SameTile(currentTiles[i], t.Tiles[i]))
                            t.Map.SetTile(pos, t.Tiles[i]);
                        t.Map.SetTransformMatrix(pos, t.Matrices[i]);
                        t.Map.SetTileFlags(pos, t.Flags[i]);
                    }
                    i++;
                }
            }
            CounterfactualTiming.End("restore.tilemaps", startedAt);
            return metrics.ChangedTilemaps > 0;
        }

        string WriteRestoreDiagnostic(RestoreMetrics metrics, TilemapCollider2D[] tilemapColliders,
            CompositeCollider2D[] compositeColliders, bool physicsInputsChanged)
        {
            string filename = "restore_diff_" + (restoreDiagnosticIndex++).ToString("D4") + ".json";
            string path = Path.Combine(output, filename);
            var rebuiltTilemapColliders = tilemapColliders.Where(c =>
                metrics.ChangedTilemapObjects.Any(map => map != null &&
                    (SameOrDescendant(map.transform, c.transform) || SameOrDescendant(c.transform, map.transform))) ||
                metrics.ChangedNativeObjects.Contains(c)).Select(DescribeComponent).ToArray();
            var rebuiltCompositeColliders = compositeColliders.Where(c =>
                metrics.ChangedTilemapObjects.Any(map => map != null &&
                    (SameOrDescendant(map.transform, c.transform) || SameOrDescendant(c.transform, map.transform))) ||
                metrics.ChangedNativeObjects.OfType<Collider2D>().Any(other =>
                    !ReferenceEquals(other, c) && SameOrDescendant(other.transform, c.transform)) ||
                metrics.ChangedNativeObjects.Contains(c)).Select(DescribeComponent).ToArray();
            var payload = new {
                schema = Schema,
                changed_tilemaps = metrics.ChangedTilemapObjects.OrderBy(map => TransformPath(map.transform))
                    .Select(DescribeComponent).ToArray(),
                changed_cells = metrics.ChangedCells,
                full_tilemap_rewrites = metrics.FullTilemapRewrites,
                tilemap_colliders_rebuilt = rebuiltTilemapColliders,
                composite_colliders_rebuilt = rebuiltCompositeColliders,
                native_changed_components = metrics.ChangedNativeObjects.OrderBy(c => TransformPath(c.transform))
                    .ThenBy(c => c.GetType().FullName, StringComparer.Ordinal)
                    .Select(DescribeComponent).ToArray(),
                native_changed_component_count = metrics.NativeChangedComponents,
                native_noop_component_count = native.Count - metrics.NativeChangedComponents,
                native_field_writes = metrics.NativeFieldWrites,
                native_field_skips = metrics.NativeFieldSkips,
                physics_sync = physicsInputsChanged
            };
            try { File.WriteAllText(path, JsonConvert.SerializeObject(payload, Formatting.Indented)); }
            catch (Exception e) { Debug.LogWarning("[CB1] restore diff write failed: " + e.Message); }
            return filename;
        }

        public void Restore()
        {
            // All foreseeable support checks occur BEFORE the first write.
            long restoreStartedAt = CounterfactualTiming.Start();
            string rngBeforeRestore = RandomFingerprint(UnityEngine.Random.state);
            string rngSnapshot = RandomFingerprint(rng);
            RestoreProgress("restore:start");
            string animatorBefore = WriteAnimatorDiagnostic("before_restore");
            RestoreProgress("restore:animator-before:" + animatorBefore);
            if (detailed) WritePhysicsContactDiagnostic("before_restore");
            var live = SceneComponents();
            if (!componentTopology.Keys.OrderBy(id => id).SequenceEqual(
                live.Select(c => c.GetInstanceID()).OrderBy(id => id)))
            {
                RestoreProgress("restore:topology-diff");
                long diagnosticStartedAt = CounterfactualTiming.Start();
                string diagnostic = WriteTopologyDiagnostic(live);
                CounterfactualTiming.End("restore.topology_diagnostic", diagnosticStartedAt);
                RestoreProgress("restore:diagnostic-written:" + diagnostic);
                try { RestoreTopology(); }
                catch (InvalidOperationException ex)
                {
                    throw new InvalidOperationException(
                        "RESTORE_BLOCKED_TOPOLOGY: topology diagnostics=" + diagnostic + "; " + ex.Message);
                }
            }
            RestoreProgress("restore:lifecycle-start:" + native.Count);
            for (int nativeIndex = 0; nativeIndex < native.Count; nativeIndex++)
            {
                var n = native[nativeIndex];
                if ((nativeIndex & 255) == 0)
                    RestoreProgress("restore:lifecycle:" + nativeIndex + "/" + native.Count + ":" +
                        (n.Target == null ? "<null>" : n.Target.GetType().FullName));
                if (n.Target == null)
                    throw new InvalidOperationException(
                        "RESTORE_BLOCKED_LIFECYCLE_MISSING_TARGET: native index " + nativeIndex);
                if (n.TrackLifecycle && n.Target.gameObject.activeSelf != n.Active)
                    throw new InvalidOperationException("RESTORE_BLOCKED_LIFECYCLE: active/enabled change on " + Identity(n.Target));
                if (n.TrackLifecycle && !n.RecreatedForRestore && n.Enabled.HasValue &&
                    ((Behaviour)n.Target).enabled != n.Enabled.Value)
                    throw new InvalidOperationException("RESTORE_BLOCKED_LIFECYCLE: active/enabled change on " + Identity(n.Target));
                if (n.Target is Animator a)
                {
                    int transitionLayer = Enumerable.Range(0, a.layerCount).FirstOrDefault(i => a.IsInTransition(i));
                    if (a.layerCount > 0 && a.IsInTransition(transitionLayer))
                    {
                        // A transition that existed at capture is an engine
                        // internal state which this adapter cannot recreate
                        // through the public Animator API. A transition that
                        // began after capture is different: RestoreNative can
                        // attempt a direct Play + zero-delta evaluation and
                        // then prove the complete captured public state.
                        if (n.AnimatorInTransitions != null &&
                            transitionLayer < n.AnimatorInTransitions.Length &&
                            n.AnimatorInTransitions[transitionLayer])
                        {
                            string blocked = WriteAnimatorDiagnostic("blocked_transition");
                            throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR_TRANSITION: " + Identity(a) +
                                " layer=" + transitionLayer + " baseline_also_in_transition=true diagnostics=" + blocked);
                        }
                        CounterfactualTiming.Report("animator:transition-reconstruction-deferred identity=" +
                            Identity(a) + " layer=" + transitionLayer + " baseline_in_transition=false");
                    }
                }
            }
            RestoreProgress("restore:lifecycle-complete");
            if (Gaps.Any(g => g.Contains(": animator transition active") || g.Contains(": unobservable Animator trigger")))
                throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR: snapshot contains a transition or unobservable trigger");
            RestoreProgress("restore:managed-graph");
            Graph.Restore();
            RestoreProgress("restore:managed-graph-complete");
            var metrics = new RestoreMetrics();
            foreach (var n in native.Where(n => n.RecreatedForRestore && n.Enabled.HasValue))
                ((Behaviour)n.Target).enabled = n.Enabled.Value;
            // Do not call WorldGeneration.UpdateChunk: it invokes seating
            // listeners. Restore only cells whose captured semantic values
            // differ, with a full rewrite reserved for a changed cell bounds.
            RestoreTilemaps(metrics);
            RestoreProgress("restore:tilemaps-complete");
            long nativeStartedAt = CounterfactualTiming.Start();
            foreach (var n in native.Where(n => n.Target is Transform)) RestoreNative(n, metrics);
            foreach (var n in native.Where(n => !(n.Target is Transform) && !(n.Target is Rigidbody2D))) RestoreNative(n, metrics);
            foreach (var n in native.Where(n => n.Target is Rigidbody2D)) RestoreNative(n, metrics);
            CounterfactualTiming.End("restore.native", nativeStartedAt);
            RestoreProgress("restore:native-complete");
            var tilemapColliders = components.OfType<TilemapCollider2D>().ToArray();
            var compositeColliders = components.OfType<CompositeCollider2D>().ToArray();
            long colliderStartedAt = CounterfactualTiming.Start();
            RestoreProgress("restore:colliders-start:tilemap=" + tilemapColliders.Length +
                ":composite=" + compositeColliders.Length);
            for (int i = 0; i < tilemapColliders.Length; i++)
            {
                var collider = tilemapColliders[i];
                bool tilemapChanged = metrics.ChangedTilemapObjects.Any(map =>
                    map != null && (SameOrDescendant(map.transform, collider.transform) ||
                        SameOrDescendant(collider.transform, map.transform)));
                bool colliderChanged = metrics.ChangedNativeObjects.Contains(collider);
                if (!tilemapChanged && !colliderChanged) continue;
                if ((i & 15) == 0) RestoreProgress("restore:tilemap-collider:" + i + "/" + tilemapColliders.Length);
                tilemapColliders[i].ProcessTilemapChanges();
                metrics.TilemapCollidersRebuilt++;
            }
            RestoreProgress("restore:tilemap-colliders-complete");
            for (int i = 0; i < compositeColliders.Length; i++)
            {
                var composite = compositeColliders[i];
                bool inputChanged = metrics.ChangedTilemapObjects.Any(map =>
                    map != null && (SameOrDescendant(map.transform, composite.transform) ||
                        SameOrDescendant(composite.transform, map.transform))) ||
                    metrics.ChangedNativeObjects.OfType<Collider2D>().Any(collider =>
                        !ReferenceEquals(collider, composite) &&
                        SameOrDescendant(collider.transform, composite.transform));
                bool compositeChanged = metrics.ChangedNativeObjects.Contains(composite);
                if (!inputChanged && !compositeChanged) continue;
                if ((i & 15) == 0) RestoreProgress("restore:composite-collider:" + i + "/" + compositeColliders.Length);
                composite.GenerateGeometry();
                metrics.CompositeCollidersRebuilt++;
            }
            RestoreProgress("restore:composite-colliders-complete");
            RestoreProgress("restore:colliders-complete");
            CounterfactualTiming.End("restore.collider_rebuild", colliderStartedAt);
            long physicsStartedAt = CounterfactualTiming.Start();
            foreach (var n in native.Where(n => n.Target is Rigidbody2D))
            {
                var body = (Rigidbody2D)n.Target;
                bool sleepingChanged = body.IsSleeping() != n.Sleeping;
                if (!sleepingChanged) { metrics.NativeFieldSkips++; continue; }
                if (n.Sleeping) body.Sleep(); else body.WakeUp();
                metrics.NativeFieldWrites++;
                metrics.ChangedNativeObjects.Add(body);
            }
            bool physicsInputsChanged = metrics.ChangedNativeObjects.Any(c =>
                c is Transform || c is Rigidbody2D || c is Collider2D) ||
                metrics.ChangedTilemapObjects.Count > 0;
            if (physicsInputsChanged) Physics2D.SyncTransforms();
            CounterfactualTiming.End("restore.physics_sync", physicsStartedAt);
            if (detailed)
            {
                WritePhysicsContactDiagnostic("after_physics_sync");
                string contactDifference = FirstContactDifference();
                if (contactDifference != null)
                    throw new InvalidOperationException("RESTORE_BLOCKED_ENGINE_CONTACT_STATE: " + contactDifference +
                        "; Unity exposes Rigidbody2D contacts for observation but provides no public API to restore " +
                        "contact manifold warm-start impulses/broadphase state");
            }
            metrics.NativeChangedComponents = metrics.ChangedNativeObjects.Count;
            int nativeNoOpComponents = native.Count - metrics.NativeChangedComponents;
            int skippedComponentWrites = Math.Max(0, nativeNoOpComponents);
            string restoreStats = "restore:stats changed_tilemaps=" + metrics.ChangedTilemaps + "/" + tilemaps.Count +
                " changed_cells=" + metrics.ChangedCells +
                " full_tilemap_rewrites=" + metrics.FullTilemapRewrites +
                " tilemap_colliders_rebuilt=" + metrics.TilemapCollidersRebuilt + "/" + tilemapColliders.Length +
                " composite_colliders_rebuilt=" + metrics.CompositeCollidersRebuilt + "/" + compositeColliders.Length +
                " native_changed_components=" + metrics.NativeChangedComponents + "/" + native.Count +
                " native_noop_components=" + skippedComponentWrites +
                " native_field_writes=" + metrics.NativeFieldWrites +
                " native_field_skips=" + metrics.NativeFieldSkips +
                " physics_sync=" + (physicsInputsChanged ? "1" : "0");
            string restoreDiagnostic = WriteRestoreDiagnostic(metrics, tilemapColliders, compositeColliders,
                physicsInputsChanged);
            RestoreProgress(restoreStats + " diagnostic=" + restoreDiagnostic);
            CounterfactualTiming.Report(restoreStats);
            Time.fixedDeltaTime = fixedDeltaTime; Time.maximumDeltaTime = maximumDeltaTime; Time.timeScale = timeScale;
            string rngBeforeFinalSet = RandomFingerprint(UnityEngine.Random.state);
            UnityEngine.Random.state = rng;
            string rngAfterFinalSet = RandomFingerprint(UnityEngine.Random.state);
            CounterfactualTiming.Report("restore:rng_before=" + rngBeforeRestore +
                " rng_snapshot=" + rngSnapshot +
                " rng_before_final_set=" + rngBeforeFinalSet +
                " rng_after_final_set=" + rngAfterFinalSet +
                " restored_exact=" + (rngSnapshot == rngAfterFinalSet ? "1" : "0"));
            RestoreProgress("restore:complete");
            CounterfactualTiming.End("SemanticSnapshot.Restore", restoreStartedAt);
        }

        public string WriteAnimatorDiagnostic(string phase)
        {
            if (string.IsNullOrWhiteSpace(output)) return null;
            var records = new List<AnimatorDiagnostic>();
            foreach (var n in native)
            {
                var animator = n.Target as Animator;
                if (animator == null) continue;
                string stable = null;
                TopologyEntry topology;
                if (componentTopology != null && componentTopology.TryGetValue(animator.GetInstanceID(), out topology))
                    stable = topology.stable_identity;
                var layers = new List<object>();
                for (int i = 0; i < animator.layerCount; i++)
                {
                    AnimatorStateInfo current = animator.GetCurrentAnimatorStateInfo(i);
                    AnimatorStateInfo next = animator.GetNextAnimatorStateInfo(i);
                    AnimatorTransitionInfo transition = animator.GetAnimatorTransitionInfo(i);
                    layers.Add(new {
                        index = i,
                        weight = animator.GetLayerWeight(i),
                        in_transition = animator.IsInTransition(i),
                        current = StateInfoDiagnostic(current),
                        next = StateInfoDiagnostic(next),
                        transition = TransitionInfoDiagnostic(transition)
                    });
                }
                var parameters = new List<object>();
                foreach (var parameter in animator.parameters)
                {
                    object runtime = null;
                    bool observable = false;
                    string observationError = null;
                    try
                    {
                        switch (parameter.type)
                        {
                            case AnimatorControllerParameterType.Float: runtime = animator.GetFloat(parameter.nameHash); observable = true; break;
                            case AnimatorControllerParameterType.Int: runtime = animator.GetInteger(parameter.nameHash); observable = true; break;
                            case AnimatorControllerParameterType.Bool: runtime = animator.GetBool(parameter.nameHash); observable = true; break;
                            case AnimatorControllerParameterType.Trigger:
                                observationError = "Unity Animator has no non-mutating trigger-pending-state query";
                                break;
                        }
                    }
                    catch (Exception ex) { observationError = ex.GetType().Name + ": " + ex.Message; }
                    parameters.Add(new {
                        name = parameter.name,
                        name_hash = parameter.nameHash,
                        type = parameter.type.ToString(),
                        default_float = parameter.defaultFloat,
                        default_int = parameter.defaultInt,
                        default_bool = parameter.defaultBool,
                        runtime,
                        runtime_observable = observable,
                        observation_error = observationError
                    });
                }
                records.Add(new AnimatorDiagnostic {
                    identity = Identity(animator), stable_identity = stable,
                    hierarchy = TransformPath(animator.transform),
                    active = animator.gameObject.activeSelf, enabled = animator.enabled,
                    speed = animator.speed.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
                    apply_root_motion = animator.applyRootMotion.ToString(),
                    update_mode = animator.updateMode.ToString(),
                    culling_mode = animator.cullingMode.ToString(),
                    fire_events = animator.fireEvents.ToString(),
                    runtime_controller = Identity(animator.runtimeAnimatorController),
                    avatar = Identity(animator.avatar),
                    optional_properties = OptionalAnimatorProperties(animator),
                    layers = layers.ToArray(), parameters = parameters.ToArray()
                });
            }
            string filename = "animator_" + phase + "_" + (restoreDiagnosticIndex++).ToString("D4") + ".json";
            string path = Path.Combine(output, filename);
            var payload = new {
                schema = Schema, phase, frame = Time.frameCount,
                realtime = Time.realtimeSinceStartup, fixed_time = Time.fixedTime,
                animator_count = records.Count, animators = records.ToArray()
            };
            try { File.WriteAllText(path, JsonConvert.SerializeObject(payload, Formatting.Indented)); }
            catch (Exception ex) { Debug.LogWarning("[CB1] animator diagnostic write failed: " + ex.Message); }
            CounterfactualTiming.Report("animator:" + phase + " file=" + filename + " count=" + records.Count);
            return filename;
        }

        public string WritePhysicsContactDiagnostic(string phase)
        {
            if (string.IsNullOrWhiteSpace(output)) return null;
            var records = new List<PhysicsContactDiagnostic>();
            foreach (var n in native)
            {
                var body = n.Target as Rigidbody2D;
                if (body == null) continue;
                var contacts = new ContactPoint2D[64];
                int count = body.GetContacts(contacts);
                records.Add(new PhysicsContactDiagnostic {
                    identity = Identity(body), hierarchy = TransformPath(body.transform),
                    contact_count = count,
                    contacts = contacts.Take(count).Select(ContactDiagnostic).ToArray()
                });
            }
            string filename = "physics_contacts_" + phase + "_" + (restoreDiagnosticIndex++).ToString("D4") + ".json";
            string path = Path.Combine(output, filename);
            var payload = new {
                schema = Schema, phase, frame = Time.frameCount,
                realtime = Time.realtimeSinceStartup, fixed_time = Time.fixedTime,
                rigidbody_count = records.Count, total_contacts = records.Sum(r => r.contact_count),
                rigidbodies = records.ToArray()
            };
            try { File.WriteAllText(path, JsonConvert.SerializeObject(payload, Formatting.Indented)); }
            catch (Exception ex)
            {
                CounterfactualTiming.Report("physics-contacts-write-failed phase=" + phase +
                    " error=" + ex.GetType().Name + ":" + ex.Message);
                Debug.LogWarning("[CB1] physics contact diagnostic write failed: " + ex.Message);
            }
            CounterfactualTiming.Report("physics-contacts:" + phase + " file=" + filename +
                " rigidbodies=" + records.Count + " contacts=" + records.Sum(r => r.contact_count));
            return filename;
        }

        object ContactDiagnostic(ContactPoint2D contact)
        {
            return new {
                collider = Identity(contact.collider), other_collider = Identity(contact.otherCollider),
                rigidbody = Identity(contact.rigidbody), point = contact.point.ToString(),
                normal = contact.normal.ToString(), separation = ManagedGraph.Scalar(contact.separation),
                normal_impulse = ManagedGraph.Scalar(contact.normalImpulse),
                tangent_impulse = ManagedGraph.Scalar(contact.tangentImpulse),
                relative_velocity = contact.relativeVelocity.ToString()
            };
        }

        string FirstContactDifference()
        {
            foreach (var n in native)
            {
                var body = n.Target as Rigidbody2D;
                if (body == null || n.Contacts == null) continue;
                var current = new ContactPoint2D[64];
                int count = body.GetContacts(current);
                if (count != n.Contacts.Length)
                    return Identity(body) + " hierarchy=" + TransformPath(body.transform) +
                        " contact_count baseline=" + n.Contacts.Length + " live=" + count;
                for (int i = 0; i < count; i++)
                {
                    var expected = n.Contacts[i];
                    var actual = current[i];
                    if (Identity(expected.collider) != Identity(actual.collider) ||
                        Identity(expected.otherCollider) != Identity(actual.otherCollider) ||
                        Identity(expected.rigidbody) != Identity(actual.rigidbody) ||
                        !expected.point.Equals(actual.point) || !expected.normal.Equals(actual.normal) ||
                        !expected.separation.Equals(actual.separation) ||
                        !expected.normalImpulse.Equals(actual.normalImpulse) ||
                        !expected.tangentImpulse.Equals(actual.tangentImpulse) ||
                        !expected.relativeVelocity.Equals(actual.relativeVelocity))
                    {
                        return Identity(body) + " hierarchy=" + TransformPath(body.transform) +
                            " contact_index=" + i +
                            " normal_impulse baseline=" + ManagedGraph.Scalar(expected.normalImpulse) +
                            " live=" + ManagedGraph.Scalar(actual.normalImpulse) +
                            " tangent_impulse baseline=" + ManagedGraph.Scalar(expected.tangentImpulse) +
                            " live=" + ManagedGraph.Scalar(actual.tangentImpulse);
                    }
                }
            }
            return null;
        }

        static object StateInfoDiagnostic(AnimatorStateInfo state)
        {
            return new {
                full_path_hash = state.fullPathHash, short_name_hash = state.shortNameHash,
                tag_hash = state.tagHash, normalized_time = state.normalizedTime,
                length = state.length, speed = state.speed,
                speed_multiplier = state.speedMultiplier, loop = state.loop
            };
        }

        static object TransitionInfoDiagnostic(AnimatorTransitionInfo transition)
        {
            return new {
                full_path_hash = transition.fullPathHash, name_hash = transition.nameHash,
                user_name_hash = transition.userNameHash, normalized_time = transition.normalizedTime,
                duration = transition.duration, any_state = transition.anyState,
                is_name_empty = transition.IsName("")
            };
        }

        static Dictionary<string, string> OptionalAnimatorProperties(Animator animator)
        {
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string name in new[] { "isInitialized", "hasBoundPlayables", "hasRootMotion", "isHuman",
                "isMatchingTarget", "recorderMode", "recorderStartTime", "recorderStopTime", "playbackTime",
                "stabilizeFeet", "feetPivotActive", "layersAffectMassCenter", "pivotWeight", "deltaPosition",
                "deltaRotation", "rootPosition", "rootRotation" })
            {
                try
                {
                    PropertyInfo property = typeof(Animator).GetProperty(name,
                        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    if (property != null) result[name] = Convert.ToString(property.GetValue(animator, null),
                        System.Globalization.CultureInfo.InvariantCulture);
                }
                catch (Exception ex) { result[name] = "<error:" + ex.GetType().Name + ">"; }
            }
            return result;
        }
        static void RestoreNative(NativeRecord n, RestoreMetrics metrics)
        {
            bool changed = false;
            if (n.Target is Transform t && !ReferenceEquals(t.parent, n.Parent))
            {
                t.SetParent(n.Parent, false);
                changed = true; metrics.NativeFieldWrites++;
            }
            else if (n.Target is Transform) metrics.NativeFieldSkips++;
            if (n.Target.gameObject.layer != n.Layer)
            {
                n.Target.gameObject.layer = n.Layer;
                changed = true; metrics.NativeFieldWrites++;
            }
            else metrics.NativeFieldSkips++;
            foreach (var p in n.Properties)
            {
                object current = p.property.GetValue(n.Target);
                if (SemanticallyEqual(current, p.value)) { metrics.NativeFieldSkips++; continue; }
                p.property.SetValue(n.Target, p.value);
                changed = true; metrics.NativeFieldWrites++;
            }
            if (n.PolygonPaths != null)
            {
                var poly = (PolygonCollider2D)n.Target;
                bool pathsChanged = poly.pathCount != n.PolygonPaths.Length;
                if (!pathsChanged)
                    for (int i = 0; i < n.PolygonPaths.Length && !pathsChanged; i++)
                        pathsChanged = !SemanticallyEqual(poly.GetPath(i), n.PolygonPaths[i]);
                if (pathsChanged)
                {
                    poly.pathCount = n.PolygonPaths.Length;
                    for (int i = 0; i < n.PolygonPaths.Length; i++) poly.SetPath(i, n.PolygonPaths[i]);
                    changed = true; metrics.NativeFieldWrites++;
                }
                else metrics.NativeFieldSkips++;
            }
            if (n.AnimatorStates != null)
            {
                var a = (Animator)n.Target;
                for (int i = 0; i < n.Parameters.Length; i++)
                {
                    var p = n.Parameters[i];
                    object current = p.type == AnimatorControllerParameterType.Float ? (object)a.GetFloat(p.nameHash) :
                        p.type == AnimatorControllerParameterType.Int ? a.GetInteger(p.nameHash) :
                        p.type == AnimatorControllerParameterType.Bool ? (object)a.GetBool(p.nameHash) : null;
                    if (SemanticallyEqual(current, n.ParameterValues[i])) { metrics.NativeFieldSkips++; continue; }
                    if (p.type == AnimatorControllerParameterType.Float) a.SetFloat(p.nameHash, (float)n.ParameterValues[i]);
                    else if (p.type == AnimatorControllerParameterType.Int) a.SetInteger(p.nameHash, (int)n.ParameterValues[i]);
                    else if (p.type == AnimatorControllerParameterType.Bool) a.SetBool(p.nameHash, (bool)n.ParameterValues[i]);
                    changed = true; metrics.NativeFieldWrites++;
                }
                for (int i = 0; i < n.AnimatorStates.Length; i++)
                {
                    bool weightChanged = !a.GetLayerWeight(i).Equals(n.LayerWeights[i]);
                    var currentState = a.GetCurrentAnimatorStateInfo(i);
                    bool transitionChanged = a.IsInTransition(i) ||
                        (n.AnimatorInTransitions != null && n.AnimatorInTransitions[i]);
                    bool stateChanged = !currentState.Equals(n.AnimatorStates[i]);
                    if (weightChanged) { a.SetLayerWeight(i, n.LayerWeights[i]); changed = true; metrics.NativeFieldWrites++; }
                    else metrics.NativeFieldSkips++;
                    if (stateChanged || transitionChanged)
                    {
                        AnimatorStateInfo targetState = n.AnimatorStates[i];
                        CounterfactualTiming.Report("animator:play identity=" + Identity(a) +
                            " layer=" + i + " target_hash=" + targetState.fullPathHash +
                            " target_normalized=" + targetState.normalizedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture) +
                            " current_hash=" + currentState.fullPathHash +
                            " current_normalized=" + currentState.normalizedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                        a.Play(targetState.fullPathHash, i, targetState.normalizedTime);
                        AnimatorStateInfo scheduledState = a.GetCurrentAnimatorStateInfo(i);
                        CounterfactualTiming.Report("animator:play-immediate identity=" + Identity(a) +
                            " layer=" + i + " hash=" + scheduledState.fullPathHash +
                            " normalized=" + scheduledState.normalizedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                        // Animator.Play schedules the state change in this
                        // Unity build. A zero-delta evaluation materializes
                        // the requested state without advancing simulation
                        // time; verify the public API actually reconstructed
                        // every captured AnimatorStateInfo field.
                        a.Update(0f);
                        AnimatorStateInfo evaluatedState = a.GetCurrentAnimatorStateInfo(i);
                        CounterfactualTiming.Report("animator:update-zero identity=" + Identity(a) +
                            " layer=" + i + " hash=" + evaluatedState.fullPathHash +
                            " normalized=" + evaluatedState.normalizedTime.ToString("R", System.Globalization.CultureInfo.InvariantCulture) +
                            " in_transition=" + a.IsInTransition(i));
                        if (!evaluatedState.Equals(targetState))
                            throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR_RECONSTRUCTION: " + Identity(a) +
                                " layer=" + i + " target=" + targetState.fullPathHash + "/" + targetState.normalizedTime +
                                " evaluated=" + evaluatedState.fullPathHash + "/" + evaluatedState.normalizedTime);
                        if (n.AnimatorInTransitions != null && n.AnimatorInTransitions[i] != a.IsInTransition(i))
                            throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR_RECONSTRUCTION: " + Identity(a) +
                                " layer=" + i + " baseline_in_transition=" + n.AnimatorInTransitions[i] +
                                " evaluated_in_transition=" + a.IsInTransition(i));
                        if (n.AnimatorNextStates != null &&
                            !a.GetNextAnimatorStateInfo(i).Equals(n.AnimatorNextStates[i]))
                            throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR_RECONSTRUCTION: " + Identity(a) +
                                " layer=" + i + " next-state differs after zero-delta evaluation");
                        if (n.AnimatorTransitions != null &&
                            !AnimatorTransitionEquals(a.GetAnimatorTransitionInfo(i), n.AnimatorTransitions[i]))
                            throw new InvalidOperationException("RESTORE_BLOCKED_ANIMATOR_RECONSTRUCTION: " + Identity(a) +
                                " layer=" + i + " transition-info differs after zero-delta evaluation");
                        changed = true; metrics.NativeFieldWrites++;
                    }
                    else metrics.NativeFieldSkips++;
                }
            }
            if (changed) metrics.ChangedNativeObjects.Add(n.Target);
        }

        static bool AnimatorTransitionEquals(AnimatorTransitionInfo left, AnimatorTransitionInfo right)
        {
            return left.fullPathHash == right.fullPathHash &&
                left.nameHash == right.nameHash &&
                left.userNameHash == right.userNameHash &&
                left.normalizedTime.Equals(right.normalizedTime) &&
                left.duration.Equals(right.duration) &&
                left.anyState == right.anyState &&
                left.IsName("") == right.IsName("");
        }
    }
}
