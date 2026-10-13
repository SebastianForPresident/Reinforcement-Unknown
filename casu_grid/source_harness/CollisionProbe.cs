using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace CB1.Counterfactual
{
    /// <summary>
    /// Opt-in, read-only collision metadata export. Kept outside the semantic
    /// snapshot hash and CB1 response so ordinary protocol behavior is intact.
    /// </summary>
    internal static class CollisionProbe
    {
        const int MaxCompositePathPoints = 250000;
        static StreamWriter playerTickWriter;
        static StreamWriter playerPostPhysicsWriter;
        static ContactPoint2D[] playerContacts = new ContactPoint2D[128];

        static object PlayerLiquidSamples(Body body)
        {
            WorldGeneration world = WorldGeneration.world;
            FluidManager fluids = FluidManager.main;
            if (body == null || world == null || fluids == null) return null;

            object Sample(string kind, int index, Transform transform, Rigidbody2D rigidbody)
            {
                Vector2 position = transform == null ? Vector2.zero : (Vector2)transform.position;
                Vector2Int cell = world.WorldToBlockPos(position);
                var liquid = fluids.WaterInfo(cell);
                return new {
                    kind, index, position = new[] { position.x, position.y },
                    velocity = rigidbody == null ? null : new[] { rigidbody.velocity.x, rigidbody.velocity.y },
                    cell = new[] { cell.x, cell.y },
                    liquid_type = liquid.type, buoyancy = liquid.buoyancy, drag = liquid.drag
                };
            }

            var samples = new List<object>();
            samples.Add(Sample("body", -1, body.transform, body.rb));
            if (body.limbs != null)
            {
                for (int i = 0; i < body.limbs.Length; i++)
                {
                    Limb limb = body.limbs[i];
                    if (limb != null)
                        samples.Add(Sample("limb", i, limb.transform, limb.rb));
                }
            }
            return new { in_water = body.inWater, liquid_slip_time = body.liquidSlipTime, samples };
        }


        static T? SafeValue<T>(Func<T> read) where T : struct
        {
            try { return read(); }
            catch { return null; }
        }

        static object Vec2(object value)
        {
            if (value is Vector2 v) return new[] { v.x, v.y };
            return value;
        }

        static object SafeExtra(Body body)
        {
            try { return ExtraPlayerState(body); }
            catch (Exception e) { return new { probe_error = e.GetType().Name + ": " + e.Message }; }
        }

        /// <summary>Additional private controller state used for exact controller replay.</summary>
        static object ExtraPlayerState(Body body)
        {
            var camera = PlayerCamera.main;
            return new {
                sliding_left = InstanceField(body, "slidingLeft"),
                sliding_right = InstanceField(body, "slidingRight"),
                time_since_slid_left = InstanceField(body, "timeSinceSlidLeft"),
                time_since_slid_right = InstanceField(body, "timeSinceSlidRight"),
                time_slid_for = InstanceField(body, "timeSlidfor"),
                ground_normal_dir = Vec2(InstanceField(body, "groundNormalDir")),
                last_time_step_velocity = Vec2(InstanceField(body, "lastTimeStepVelocity")),
                last_jumped_on_right_wall = InstanceField(body, "lastJumpedOnRightWall"),
                first_wall_jump = InstanceField(body, "firstWallJump"),
                stand_lerp_time = InstanceField(body, "standLerpTime"),
                accel_rot = InstanceField(body, "accelRot"),
                crouching = body.crouching,
                wall_slide_slowdown_raw = InstanceField(body, "wallSlideSlowdown"),
                actual_wall_slide_slowdown = SafeValue(() => body.actualWallSlideSlowdown),
                actual_stamina_strength = SafeValue(() => body.actualStaminaStrength),
                energy = InstanceField(body, "energy"),
                stamina = InstanceField(body, "stamina"),
                force_walk = InstanceField(body, "forceWalk"),
                moving_allowed = InstanceField(body, "movingAllowed"),
                liquid_slip_time = InstanceField(body, "liquidSlipTime"),
                max_speed_raw = InstanceField(body, "maxSpeed"),
                move_force_raw = InstanceField(body, "moveForce"),
                jump_speed_raw = InstanceField(body, "jumpSpeed"),
                slowdown_amount_raw = InstanceField(body, "slowdownAmount"),
                climbing = InstanceField(body, "currentClimbable") != null,
                orig_col_size = Vec2(InstanceField(body, "origColSize")),
                target_look_pos = Vec2(InstanceField(body, "targetLookPos")),
                harmer_time_was_still = SafeValue(() => body.harmer.timeWasStill),
                leg_speed_mult = SafeValue(() => body.legSpeedMult),
                actual_jump_speed = SafeValue(() => body.actualJumpSpeed),
                wants_jump_input = SafeValue(() => camera.wantsJumpInput),
                body_position = new[] { body.transform.position.x, body.transform.position.y },
                rb_position = body.rb == null ? null : new[] { body.rb.position.x, body.rb.position.y },
                rb_rotation = body.rb == null ? 0f : body.rb.rotation,
                visual_body_offset = Vec2(InstanceField(body, "visualBodyOffset")),
                in_water = SafeValue(() => body.inWater),
                was_water = SafeValue(() => body.bodyAffect.wasWater)
            };
        }

        static object PlayerBodyConfiguration(Body body)
        {
            if (body == null || body.rb == null || body.col == null) return null;
            return new {
                rb_mass = body.rb.mass, rb_drag = body.rb.drag, rb_angular_drag = body.rb.angularDrag,
                rb_gravity_scale = body.rb.gravityScale, rb_constraints = body.rb.constraints.ToString(),
                rb_collision_detection = body.rb.collisionDetectionMode.ToString(),
                rb_interpolation = body.rb.interpolation.ToString(),
                rb_sleep_mode = body.rb.sleepMode.ToString(), rb_body_type = body.rb.bodyType.ToString(),
                rb_inertia = body.rb.inertia, rb_center_of_mass = new[] { body.rb.centerOfMass.x, body.rb.centerOfMass.y },
                rb_shared_material = Material(body.rb.sharedMaterial),
                col_shared_material = Material(body.col.sharedMaterial),
                col_effective_friction = body.col.friction, col_effective_bounciness = body.col.bounciness,
                col_density = body.col.density, col_size = new[] { body.col.size.x, body.col.size.y },
                col_offset = new[] { body.col.offset.x, body.col.offset.y },
                col_edge_radius = body.col.edgeRadius, col_layer = body.gameObject.layer,
                col_is_trigger = body.col.isTrigger, col_used_by_composite = body.col.usedByComposite,
                col_contact_capture_layers = body.col.contactCaptureLayers.value,
                col_layer_override_priority = body.col.layerOverridePriority,
                physics2d_gravity = new[] { Physics2D.gravity.x, Physics2D.gravity.y },
                velocity_iterations = Physics2D.velocityIterations,
                position_iterations = Physics2D.positionIterations,
                max_linear_correction = Physics2D.maxLinearCorrection,
                baumgarte_scale = Physics2D.baumgarteScale,
                baumgarte_toi_scale = Physics2D.baumgarteTOIScale,
                velocity_threshold = Physics2D.velocityThreshold,
                default_contact_offset = Physics2D.defaultContactOffset,
                max_translation_speed = Physics2D.maxTranslationSpeed,
                time_to_sleep = Physics2D.timeToSleep,
                fixed_delta_time = Time.fixedDeltaTime
            };
        }

        static object InstanceField(object instance, string name)
        {
            if (instance == null) return null;
            var field = instance.GetType().GetField(name,
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (field == null) return null;
            try { return field.GetValue(instance); }
            catch { return null; }
        }

        /// <summary>
        /// Optional per-fixed-tick standing/body/contact trace. This is called
        /// after Body.FixedUpdate, immediately before Unity's Physics2D step;
        /// every entry states that sampling boundary explicitly.
        /// </summary>
        public static void RecordPlayerFixedUpdate(Body body, long fixedIndex,
            ulong actionSequence, int macrostepPhysicsTicks)
        {
            if (Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_TICKS") != "1" ||
                body == null || PlayerCamera.main == null || PlayerCamera.main.body != body)
                return;

            if (playerTickWriter == null)
            {
                string output = Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_TICKS_OUTPUT");
                string root = Environment.GetEnvironmentVariable("PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT");
                if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output) ||
                    string.IsNullOrWhiteSpace(root) || !Path.IsPathRooted(root))
                    throw new InvalidOperationException(
                        "Player tick trace requires absolute output and diagnostics root paths.");
                string fullOutput = Path.GetFullPath(output);
                string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
                if (!fullOutput.StartsWith(fullRoot, StringComparison.Ordinal))
                    throw new InvalidOperationException("Player tick trace output escapes staged diagnostics root.");
                Directory.CreateDirectory(Path.GetDirectoryName(fullOutput));
                playerTickWriter = new StreamWriter(new FileStream(fullOutput,
                    FileMode.CreateNew, FileAccess.Write, FileShare.Read));
                playerTickWriter.WriteLine(JsonConvert.SerializeObject(new {
                    schema = "cb1.player-fixed-tick.v1", record_type = "header",
                    sampling_boundary = "after Body.FixedUpdate and before Physics2D simulation",
                    unity_fixed_delta_time = Time.fixedDeltaTime,
                    physics2d_gravity = new[] { Physics2D.gravity.x, Physics2D.gravity.y },
                    body_source = "SearchBase/Body.cs standing controller",
                    body_configuration = PlayerBodyConfiguration(body)
                }));
                playerTickWriter.Flush();
            }

            int contactCount = body.rb == null ? 0 : body.rb.GetContacts(playerContacts);
            var contacts = new List<object>(Math.Min(contactCount, playerContacts.Length));
            for (int i = 0; i < Math.Min(contactCount, playerContacts.Length); i++)
            {
                ContactPoint2D contact = playerContacts[i];
                contacts.Add(new {
                    point = new[] { contact.point.x, contact.point.y },
                    normal = new[] { contact.normal.x, contact.normal.y },
                    separation = contact.separation,
                    relative_velocity = new[] { contact.relativeVelocity.x, contact.relativeVelocity.y },
                    collider = contact.collider == null ? null : Hierarchy(contact.collider.transform),
                    other_collider = contact.otherCollider == null ? null : Hierarchy(contact.otherCollider.transform),
                    other_layer = contact.otherCollider == null ? -1 : contact.otherCollider.gameObject.layer
                });
            }
            var hit = body.col == null ? default(RaycastHit2D) : Physics2D.BoxCast(
                (Vector2)body.transform.position + body.col.offset,
                body.standing ? body.col.size : new Vector2(body.col.size.x, 0.25f),
                0f, Vector2.down, body.standing ? body.col.edgeRadius + 0.2f : 3.5f,
                LayerMask.GetMask("Ground"));
            object jumpCooldown = InstanceField(body, "jumpCooldown");
            object timeSinceGrounded = InstanceField(body, "timeSinceGrounded");
            object endedJump = InstanceField(body, "endedJump");
            object slippery = InstanceField(body, "slippery");
            object crouchAmount = InstanceField(body, "crouchAmount");
            object standingOn = InstanceField(body, "standingOn");
            var p = body.transform.position;
            var v = body.rb == null ? Vector2.zero : body.rb.velocity;
            playerTickWriter.WriteLine(JsonConvert.SerializeObject(new {
                record_type = "player_tick", fixed_index = fixedIndex,
                frame = Time.frameCount, fixed_time = Time.fixedTime,
                action_sequence = actionSequence,
                macrostep_physics_ticks = macrostepPhysicsTicks,
                position = new[] { p.x, p.y }, velocity = new[] { v.x, v.y },
                mass = body.rb == null ? 0f : body.rb.mass,
                drag = body.rb == null ? 0f : body.rb.drag,
                gravity_scale = body.rb == null ? 0f : body.rb.gravityScale,
                standing = body.standing, grounded = body.grounded,
                move_dir = new[] { body.moveDir.x, body.moveDir.y },
                ended_jump = endedJump, jump_cooldown = jumpCooldown,
                time_since_grounded = timeSinceGrounded,
                max_speed = body.actualMaxSpeed, move_force = body.actualMoveForce,
                jump_speed = body.actualJumpSpeed, crouch_amount = crouchAmount,
                collider_size = body.col == null ? null : new[] { body.col.size.x, body.col.size.y },
                collider_offset = body.col == null ? null : new[] { body.col.offset.x, body.col.offset.y },
                collider_edge_radius = body.col == null ? 0f : body.col.edgeRadius,
                slippery, standing_on = standingOn == null ? null : standingOn.ToString(),
                ground_box_cast = hit.collider == null ? null : new {
                    point = new[] { hit.point.x, hit.point.y },
                    normal = new[] { hit.normal.x, hit.normal.y }, distance = hit.distance,
                    collider = Hierarchy(hit.collider.transform)
                },
                contact_buffer_capacity = playerContacts.Length,
                contacts_truncated = contactCount >= playerContacts.Length,
                contact_count = contactCount, contacts,
                liquid = PlayerLiquidSamples(body),
                extra = SafeExtra(body)
            }));
            playerTickWriter.Flush();
        }


        static StreamWriter playerPreFixedWriter;

        /// <summary>Optional state at entry to Body.FixedUpdate (after Update + action application).</summary>
        public static void RecordPlayerPreFixed(Body body, long fixedIndex,
            ulong actionSequence, int macrostepPhysicsTicks)
        {
            if (Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_PREFIXED") != "1" ||
                body == null || PlayerCamera.main == null || PlayerCamera.main.body != body)
                return;
            if (playerPreFixedWriter == null)
            {
                string output = Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_PREFIXED_OUTPUT");
                string root = Environment.GetEnvironmentVariable("PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT");
                if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output) ||
                    string.IsNullOrWhiteSpace(root) || !Path.IsPathRooted(root))
                    throw new InvalidOperationException("Pre-fixed trace needs absolute diagnostic paths.");
                string fullOutput = Path.GetFullPath(output);
                string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
                if (!fullOutput.StartsWith(fullRoot, StringComparison.Ordinal))
                    throw new InvalidOperationException("Pre-fixed trace output escapes staged diagnostics root.");
                Directory.CreateDirectory(Path.GetDirectoryName(fullOutput));
                playerPreFixedWriter = new StreamWriter(new FileStream(fullOutput,
                    FileMode.CreateNew, FileAccess.Write, FileShare.Read));
                playerPreFixedWriter.WriteLine(JsonConvert.SerializeObject(new {
                    schema = "cb1.player-pre-fixed.v1",
                    sampling_boundary = "entry to Body.FixedUpdate (after Body.Update and held-action application)",
                    body_configuration = PlayerBodyConfiguration(body)
                }));
            }
            var p = body.transform.position;
            var v = body.rb == null ? Vector2.zero : body.rb.velocity;
            playerPreFixedWriter.WriteLine(JsonConvert.SerializeObject(new {
                record_type = "pre_fixed_tick", fixed_index = fixedIndex,
                action_sequence = actionSequence, macrostep_physics_ticks = macrostepPhysicsTicks,
                position = new[] { p.x, p.y }, velocity = new[] { v.x, v.y },
                standing = body.standing, grounded = body.grounded,
                move_dir = new[] { body.moveDir.x, body.moveDir.y },
                ended_jump = InstanceField(body, "endedJump"),
                jump_cooldown = InstanceField(body, "jumpCooldown"),
                time_since_grounded = InstanceField(body, "timeSinceGrounded"),
                crouch_amount = InstanceField(body, "crouchAmount"),
                slippery = InstanceField(body, "slippery"),
                standing_on = InstanceField(body, "standingOn") == null ? null : InstanceField(body, "standingOn").ToString(),
                max_speed = body.actualMaxSpeed, move_force = body.actualMoveForce,
                gravity_scale = body.rb == null ? 0f : body.rb.gravityScale,
                collider_size = body.col == null ? null : new[] { body.col.size.x, body.col.size.y },
                collider_offset = body.col == null ? null : new[] { body.col.offset.x, body.col.offset.y },
                extra = SafeExtra(body)
            }));
            playerPreFixedWriter.Flush();
        }



        static object ColliderGeometry(Collider2D c)
        {
            Func<Vector2, float[]> world = p => {
                Vector3 w = c.transform.TransformPoint(p);
                return new[] { w.x, w.y };
            };
            switch (c)
            {
                case BoxCollider2D box:
                {
                    Vector2 h = box.size * 0.5f;
                    return new {
                        kind = "box", size = new[] { box.size.x, box.size.y }, offset = new[] { box.offset.x, box.offset.y },
                        edge_radius = box.edgeRadius,
                        world_corners = new[] {
                            world(box.offset + new Vector2(-h.x, -h.y)), world(box.offset + new Vector2(h.x, -h.y)),
                            world(box.offset + new Vector2(h.x, h.y)), world(box.offset + new Vector2(-h.x, h.y)) },
                        lossy_scale = new[] { c.transform.lossyScale.x, c.transform.lossyScale.y },
                        rotation_z = c.transform.eulerAngles.z };
                }
                case PolygonCollider2D poly:
                {
                    var paths = new List<float[][]>();
                    for (int i = 0; i < poly.pathCount; i++)
                        paths.Add(poly.GetPath(i).Select(point => world(point + poly.offset)).ToArray());
                    return new { kind = "polygon", path_count = poly.pathCount, world_paths = paths };
                }
                case CircleCollider2D circle:
                {
                    Vector3 center = c.transform.TransformPoint(circle.offset);
                    return new { kind = "circle", radius = circle.radius, world_center = new[] { center.x, center.y },
                        lossy_scale = new[] { c.transform.lossyScale.x, c.transform.lossyScale.y } };
                }
                case CapsuleCollider2D capsule:
                {
                    return new { kind = "capsule", size = new[] { capsule.size.x, capsule.size.y },
                        direction = capsule.direction.ToString(), offset = new[] { capsule.offset.x, capsule.offset.y },
                        world_center = world(capsule.offset), rotation_z = c.transform.eulerAngles.z };
                }
                case EdgeCollider2D edge:
                    return new { kind = "edge", edge_radius = edge.edgeRadius,
                        world_points = edge.points.Select(point => world(point + edge.offset)).ToArray() };
                case CompositeCollider2D composite:
                    return new { kind = "composite", path_count = composite.pathCount };
                default:
                    return new { kind = c.GetType().Name };
            }
        }

        /// <summary>One JSON line per non-tile Collider2D with its geometry and physics role.</summary>
        public static void DumpAllNonTileColliders(string path)
        {
            Body player = PlayerCamera.main == null ? null : PlayerCamera.main.body;
            using (var writer = new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)))
            {
                int tilemapCount = 0, written = 0;
                foreach (Collider2D c in UnityEngine.Object.FindObjectsOfType<Collider2D>())
                {
                    if (c is TilemapCollider2D) { tilemapCount++; continue; }
                    Body owner = c.GetComponentInParent<Body>();
                    Rigidbody2D rb = c.attachedRigidbody;
                    Bounds b = c.bounds;
                    writer.WriteLine(JsonConvert.SerializeObject(new {
                        hierarchy = Hierarchy(c.transform), layer = c.gameObject.layer,
                        layer_name = LayerMask.LayerToName(c.gameObject.layer),
                        tag = c.gameObject.tag, type = c.GetType().Name, enabled = c.enabled,
                        active_in_hierarchy = c.gameObject.activeInHierarchy, is_trigger = c.isTrigger,
                        used_by_composite = c.usedByComposite,
                        body_type = rb == null ? "NoRigidbody(static)" : rb.bodyType.ToString(),
                        rb_simulated = rb == null ? (bool?)null : rb.simulated,
                        rb_mass = rb == null ? (float?)null : rb.mass,
                        is_player_body = player != null && owner == player,
                        is_any_body = owner != null,
                        friction = c.friction, bounciness = c.bounciness,
                        shared_material = Material(c.sharedMaterial),
                        bounds_center = new[] { b.center.x, b.center.y }, bounds_size = new[] { b.size.x, b.size.y },
                        geometry = ColliderGeometry(c)
                    }));
                    written++;
                }
                writer.WriteLine(JsonConvert.SerializeObject(new {
                    record_type = "summary", tilemap_collider_components_skipped = tilemapCount,
                    non_tile_colliders_written = written,
                    ignored_layer_pairs_note = "see collision_detail export for the layer matrix"
                }));
            }
        }

        static StreamWriter occupancyWriter;
        static bool occupancyFullDone;

        static string OccupancyPath(string variable)
        {
            string output = Environment.GetEnvironmentVariable(variable);
            string root = Environment.GetEnvironmentVariable("PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT");
            if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output) ||
                string.IsNullOrWhiteSpace(root) || !Path.IsPathRooted(root))
                throw new InvalidOperationException(variable + " requires absolute diagnostic paths.");
            string fullOutput = Path.GetFullPath(output);
            string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!fullOutput.StartsWith(fullRoot, StringComparison.Ordinal))
                throw new InvalidOperationException(variable + " escapes staged diagnostics root.");
            Directory.CreateDirectory(Path.GetDirectoryName(fullOutput));
            return fullOutput;
        }

        /// <summary>
        /// Opt-in read-only Physics2D occupancy audit: does a layer-6 collider
        /// overlap each grid cell center (and four inset corner probes)?
        /// PPO_DIAG_OCC_FULL=1 writes the whole 1024x1024 world once;
        /// PPO_DIAG_OCC_REGION="x0,y0,w,h" with PPO_DIAG_OCC_EVERY=N writes a region
        /// snapshot every N fixed updates (the first fixed update always).
        /// </summary>
        public static void RecordOccupancy(long fixedIndex)
        {
            WorldGeneration world = WorldGeneration.world;
            if (world == null || PlayerCamera.main == null) return;
            // Default: audit once the first policy macrostep is under way. Before the
            // first action latch the Physics2D world is still the pre-reset one.
            int.TryParse(Environment.GetEnvironmentVariable("PPO_DIAG_OCC_FULL_AT"), out int fullAt);
            bool fullReady = fullAt > 0
                ? fixedIndex >= fullAt
                : PPOBridge.DiagnosticActionSequence >= 1 && PPOBridge.DiagnosticMacrostepPhysicsTicks >= 3;
            if (Environment.GetEnvironmentVariable("PPO_DIAG_OCC_FULL") == "1" && !occupancyFullDone &&
                fullReady)
            {
                occupancyFullDone = true;
                int width = (int)world.width, height = (int)world.height;
                var bytes = new byte[width * height];
                for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    Vector2 c = world.BlockToWorldPos(new Vector2Int(x, y));
                    byte flags = 0;
                    if (Physics2D.OverlapPoint(c, 1 << 6) != null) flags |= 1;
                    if (Physics2D.OverlapPoint(c + new Vector2(-0.4f, -0.4f), 1 << 6) != null) flags |= 2;
                    if (Physics2D.OverlapPoint(c + new Vector2(0.4f, -0.4f), 1 << 6) != null) flags |= 4;
                    if (Physics2D.OverlapPoint(c + new Vector2(-0.4f, 0.4f), 1 << 6) != null) flags |= 8;
                    if (Physics2D.OverlapPoint(c + new Vector2(0.4f, 0.4f), 1 << 6) != null) flags |= 16;
                    bytes[x * height + y] = flags;
                }
                File.WriteAllBytes(OccupancyPath("PPO_DIAG_OCC_FULL_OUTPUT"), bytes);
                DumpAllNonTileColliders(OccupancyPath("PPO_DIAG_OCC_FULL_OUTPUT") + ".colliders.jsonl");
            }
            string region = Environment.GetEnvironmentVariable("PPO_DIAG_OCC_REGION");
            if (string.IsNullOrWhiteSpace(region)) return;
            int.TryParse(Environment.GetEnvironmentVariable("PPO_DIAG_OCC_EVERY"), out int every);
            if (!(fixedIndex == 1 || (every > 0 && fixedIndex % every == 0))) return;
            var parts = region.Split(',');
            if (parts.Length != 4) return;
            int x0 = int.Parse(parts[0]), y0 = int.Parse(parts[1]), w = int.Parse(parts[2]), h = int.Parse(parts[3]);
            if (occupancyWriter == null)
                occupancyWriter = new StreamWriter(new FileStream(OccupancyPath("PPO_DIAG_OCC_OUTPUT"),
                    FileMode.CreateNew, FileAccess.Write, FileShare.Read));
            var rows = new string[h];
            for (int row = 0; row < h; row++)
            {
                int y = y0 + h - 1 - row;
                var chars = new char[w];
                for (int col = 0; col < w; col++)
                {
                    Vector2 c = world.BlockToWorldPos(new Vector2Int(x0 + col, y));
                    chars[col] = Physics2D.OverlapPoint(c, 1 << 6) == null ? '.' : '#';
                }
                rows[row] = new string(chars);
            }
            occupancyWriter.WriteLine(JsonConvert.SerializeObject(new {
                fixed_index = fixedIndex, region = new[] { x0, y0, w, h },
                action_sequence = PPOBridge.DiagnosticActionSequence,
                macrostep_physics_ticks = PPOBridge.DiagnosticMacrostepPhysicsTicks,
                note = "rows are top-to-bottom (y descending), columns x ascending", rows
            }));
            occupancyWriter.Flush();
        }

        public static void ClosePlayerTickTrace()
        {
            if (playerTickWriter != null)
            {
                playerTickWriter.Flush();
                playerTickWriter.Dispose();
                playerTickWriter = null;
            }
            if (playerPostPhysicsWriter != null)
            {
                playerPostPhysicsWriter.Flush();
                playerPostPhysicsWriter.Dispose();
                playerPostPhysicsWriter = null;
            }
            if (playerPreFixedWriter != null)
            {
                playerPreFixedWriter.Flush();
                playerPreFixedWriter.Dispose();
                playerPreFixedWriter = null;
            }
        }

        /// <summary>Optional resolved-contact sample immediately after Physics2D.</summary>
        public static void RecordPlayerPostPhysics(Body body, long fixedIndex,
            ulong actionSequence, int macrostepPhysicsTicks)
        {
            if (Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_POST_PHYSICS") != "1" ||
                body == null || PlayerCamera.main == null || PlayerCamera.main.body != body)
                return;
            if (playerPostPhysicsWriter == null)
            {
                string output = Environment.GetEnvironmentVariable("PPO_DIAG_PLAYER_POST_PHYSICS_OUTPUT");
                string root = Environment.GetEnvironmentVariable("PPO_COUNTERFACTUAL_DIAGNOSTICS_ROOT");
                if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output) ||
                    string.IsNullOrWhiteSpace(root) || !Path.IsPathRooted(root))
                    throw new InvalidOperationException("Post-physics trace needs absolute diagnostic paths.");
                string fullOutput = Path.GetFullPath(output);
                string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
                if (!fullOutput.StartsWith(fullRoot, StringComparison.Ordinal))
                    throw new InvalidOperationException("Post-physics trace output escapes staged diagnostics root.");
                Directory.CreateDirectory(Path.GetDirectoryName(fullOutput));
                playerPostPhysicsWriter = new StreamWriter(new FileStream(fullOutput,
                    FileMode.CreateNew, FileAccess.Write, FileShare.Read));
                playerPostPhysicsWriter.WriteLine(JsonConvert.SerializeObject(new {
                    schema = "cb1.player-post-physics.v1",
                    sampling_boundary = "immediately after Unity Physics2DFixedUpdate"
                }));
            }
            int count = body.rb == null ? 0 : body.rb.GetContacts(playerContacts);
            var contacts = new List<object>(Math.Min(count, playerContacts.Length));
            for (int i = 0; i < Math.Min(count, playerContacts.Length); i++)
            {
                ContactPoint2D c = playerContacts[i];
                contacts.Add(new {
                    point = new[] { c.point.x, c.point.y },
                    normal = new[] { c.normal.x, c.normal.y },
                    separation = c.separation,
                    relative_velocity = new[] { c.relativeVelocity.x, c.relativeVelocity.y },
                    collider = c.collider == null ? null : Hierarchy(c.collider.transform),
                    other_collider = c.otherCollider == null ? null : Hierarchy(c.otherCollider.transform),
                    other_layer = c.otherCollider == null ? -1 : c.otherCollider.gameObject.layer
                });
            }
            playerPostPhysicsWriter.WriteLine(JsonConvert.SerializeObject(new {
                record_type = "post_physics_tick", fixed_index = fixedIndex,
                action_sequence = actionSequence, macrostep_physics_ticks = macrostepPhysicsTicks,
                position = new[] { body.transform.position.x, body.transform.position.y },
                velocity = body.rb == null ? null : new[] { body.rb.velocity.x, body.rb.velocity.y },
                grounded = body.grounded,
                contact_count = count, contacts_truncated = count >= playerContacts.Length, contacts,
                liquid = PlayerLiquidSamples(body),
                extra = SafeExtra(body)
            }));
            playerPostPhysicsWriter.Flush();
        }

        static object ReadStatic(Type type, string name)
        {
            var property = type.GetProperty(name, BindingFlags.Public | BindingFlags.Static);
            if (property == null || !property.CanRead) return null;
            try { return property.GetValue(null, null); }
            catch { return null; }
        }

        static string Hierarchy(Transform transform)
        {
            var names = new Stack<string>();
            for (var current = transform; current != null; current = current.parent)
                names.Push(current.name);
            return string.Join("/", names.ToArray());
        }

        static object Material(PhysicsMaterial2D material)
        {
            if (material == null) return null;
            return new { name = material.name, friction = material.friction,
                bounciness = material.bounciness };
        }

        static object Composite(CompositeCollider2D collider, ref int remainingPoints)
        {
            int pathCount = collider.pathCount;
            var paths = new List<object>();
            var pointCountMethod = collider.GetType().GetMethod("GetPathPointCount");
            var getPathMethod = collider.GetType().GetMethod("GetPath", new[] { typeof(int), typeof(Vector2[]) });
            bool truncated = false;
            for (int i = 0; i < pathCount; i++)
            {
                if (pointCountMethod == null || getPathMethod == null) break;
                int count = (int)pointCountMethod.Invoke(collider, new object[] { i });
                if (count < 0 || count > remainingPoints) { truncated = true; break; }
                var points = new Vector2[count];
                getPathMethod.Invoke(collider, new object[] { i, points });
                var numericPoints = points.Select(point => new[] { point.x, point.y }).ToArray();
                paths.Add(new { path_index = i, point_count = count, points = numericPoints });
                remainingPoints -= count;
            }
            var chunk = collider.GetComponent<ChunkScript>();
            return new {
                hierarchy = Hierarchy(collider.transform),
                chunk_xy = chunk == null ? null : new[] { chunk.pos.x, chunk.pos.y },
                active = collider.gameObject.activeSelf, enabled = collider.enabled,
                layer = collider.gameObject.layer, is_trigger = collider.isTrigger,
                density = collider.density, offset = new[] { collider.offset.x, collider.offset.y },
                shared_material = Material(collider.sharedMaterial),
                geometry_type = collider.geometryType.ToString(),
                generation_type = collider.generationType.ToString(),
                path_count = pathCount, paths = paths, path_data_truncated = truncated
            };
        }

        static object TilemapCollider(TilemapCollider2D collider)
        {
            var chunk = collider.GetComponent<ChunkScript>();
            return new {
                hierarchy = Hierarchy(collider.transform),
                chunk_xy = chunk == null ? null : new[] { chunk.pos.x, chunk.pos.y },
                active = collider.gameObject.activeSelf, enabled = collider.enabled,
                layer = collider.gameObject.layer, is_trigger = collider.isTrigger,
                density = collider.density, offset = new[] { collider.offset.x, collider.offset.y },
                used_by_composite = collider.usedByComposite,
                used_by_effector = collider.usedByEffector,
                extrusion_factor = collider.extrusionFactor,
                use_delaunay_mesh = collider.useDelaunayMesh,
                maximum_tile_change_count = collider.maximumTileChangeCount,
                shared_material = Material(collider.sharedMaterial)
            };
        }

        static object TileTypes()
        {
            var world = WorldGeneration.world;
            if (world == null) return new { error = "WorldGeneration.world is null" };
            var tilesField = typeof(WorldGeneration).GetField("worldBlocks",
                BindingFlags.Instance | BindingFlags.NonPublic);
            var chunksField = typeof(WorldGeneration).GetField("chunks",
                BindingFlags.Instance | BindingFlags.NonPublic);
            var blocks = tilesField == null ? null : tilesField.GetValue(world) as ushort[,];
            var chunks = chunksField == null ? null : chunksField.GetValue(world) as Tilemap[,];
            if (blocks == null || chunks == null)
                return new { error = "world block/chunk arrays unavailable" };
            var counts = new Dictionary<ushort, long>();
            var candidates = new Dictionary<ushort, (int x, int y, int empty_neighbors)>();
            int width = blocks.GetLength(0), height = blocks.GetLength(1);
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                ushort id = blocks[x, y];
                counts[id] = counts.TryGetValue(id, out long count) ? count + 1 : 1;
                int emptyNeighbors = 0;
                for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && blocks[nx, ny] == 0)
                        emptyNeighbors++;
                }
                if (!candidates.TryGetValue(id, out var candidate) || emptyNeighbors > candidate.empty_neighbors)
                    candidates[id] = (x, y, emptyNeighbors);
            }
            var samples = new Dictionary<ushort, object>();
            foreach (var pair in candidates)
            {
                ushort id = pair.Key;
                int x = pair.Value.x, y = pair.Value.y;
                if (id >= world.tiles.Length) continue;
                Tilemap map = chunks[x / WorldGeneration.CHUNKSIZE, y / WorldGeneration.CHUNKSIZE];
                if (map == null) continue;
                var local = new Vector3Int(x % WorldGeneration.CHUNKSIZE,
                    y % WorldGeneration.CHUNKSIZE, 0);
                TileBase tile = world.tiles[id];
                string colliderType = tile is Tile tileAsset ? tileAsset.colliderType.ToString() : null;
                var center = world.BlockToWorldPos(new Vector2Int(x, y));
                string[] overlap = new string[27];
                for (int row = 0; row < 27; row++)
                {
                    var chars = new char[27];
                    for (int col = 0; col < 27; col++)
                    {
                        float px = center.x + (col - 13) * 0.04f;
                        float py = center.y + (row - 13) * 0.04f;
                        chars[col] = Physics2D.OverlapPoint(new Vector2(px, py), 1 << 6) == null ? '.' : '#';
                    }
                    overlap[row] = new string(chars);
                }
                samples[id] = new {
                    id, name = tile == null ? "<null>" : tile.name,
                    type = tile == null ? "<null>" : tile.GetType().FullName,
                    asset_collider_type = colliderType,
                    grid_xy = new[] { x, y }, world_center_xy = new[] { center.x, center.y },
                    chunk_xy = new[] { x / WorldGeneration.CHUNKSIZE, y / WorldGeneration.CHUNKSIZE },
                    empty_neighbors_of_8 = pair.Value.empty_neighbors,
                    neighboring_block_ids = new[] {
                        blocks[Math.Max(0,x-1),y], blocks[Math.Min(width-1,x+1),y],
                        blocks[x,Math.Max(0,y-1)], blocks[x,Math.Min(height-1,y+1)] },
                    layer6_overlap_pattern_0p04_cell_spacing_from_minus0p52 = overlap
                };
            }
            return new {
                width, height, block_type_count = counts.Count,
                block_types = counts.OrderBy(pair => pair.Key).Select(pair => new {
                    id = pair.Key, occupied_cells = pair.Value, sample = samples[pair.Key]
                }).ToArray()
            };
        }

        static object GlobalPhysics()
        {
            var ignoredLayers = new List<object>();
            for (int a = 0; a < 32; a++)
            for (int b = a; b < 32; b++)
                if (Physics2D.GetIgnoreLayerCollision(a, b))
                    ignoredLayers.Add(new[] { a, b });
            return new {
                gravity = new[] { Physics2D.gravity.x, Physics2D.gravity.y },
                default_contact_offset = Physics2D.defaultContactOffset,
                velocity_iterations = Physics2D.velocityIterations,
                position_iterations = Physics2D.positionIterations,
                queries_hit_triggers = Physics2D.queriesHitTriggers,
                queries_start_in_colliders = Physics2D.queriesStartInColliders,
                auto_sync_transforms = Physics2D.autoSyncTransforms,
                simulation_mode = ReadStatic(typeof(Physics2D), "simulationMode"),
                default_physics_material = Material(ReadStatic(typeof(Physics2D),
                    "defaultPhysicsMaterial") as PhysicsMaterial2D),
                ignored_layer_pairs = ignoredLayers
            };
        }

        public static string Write(string output, IEnumerable<Component> components)
        {
            int remainingPoints = MaxCompositePathPoints;
            var colliders = components.Where(c => c != null).ToArray();
            var tilemaps = colliders.OfType<TilemapCollider2D>().OrderBy(c => Hierarchy(c.transform))
                .Select(TilemapCollider).ToArray();
            var composites = colliders.OfType<CompositeCollider2D>().OrderBy(c => Hierarchy(c.transform))
                .Select(c => Composite(c, ref remainingPoints)).ToArray();
            string filename = "collision_detail_" + Time.frameCount.ToString("D8") + ".json";
            string path = Path.Combine(output, filename);
            var payload = new {
                schema = "cb1.collision-detail.v1", frame = Time.frameCount,
                fixed_time = Time.fixedTime, realtime = Time.realtimeSinceStartup,
                capture_kind = "read-only live component and generated tile metadata",
                composite_path_point_budget = MaxCompositePathPoints,
                remaining_composite_path_points = remainingPoints,
                tilemap_colliders = tilemaps, composite_colliders = composites,
                tile_types = TileTypes(), physics2d = GlobalPhysics(),
                ignore_collision_source_audit = new {
                    source_tree_search = "Physics2D.IgnoreCollision calls only pair limb colliders in Body.cs",
                    terrain_specific_ignore_pairs_found = false
                }
            };
            File.WriteAllText(path, JsonConvert.SerializeObject(payload, Formatting.Indented));
            return filename;
        }
    }
}
