# Layer 1 collision measurement handoff

The canonical fixture records component presence and configuration only. It is
not collider geometry, and it does not establish that the sibling composite
components actually compose the tilemaps. Preserve this distinction when
building the next CasU capture.

## Small measurement set

Before building all 256 Isaac chunk colliders, capture and replay these cases:

1. One representative tile each for empty space, soil, rock, and steel, plus
   the door/tile with any known special behavior. Record each Tile asset's
   `ColliderType`, material, and owning tilemap.
2. A two-cell floor spanning a 64-cell chunk seam, with contacts from above
   and both sides. Capture the effective CompositeCollider2D paths and identify
   which chunk owns each path.
3. A controlled deletion of one tile on an otherwise flat floor. Record the
   next physics-step contact, tilemap/composite refresh timing, and any stale
   contact frame.
4. Any tile-specific `Physics2D.IgnoreCollision` exception found in the player
   and terrain runtime setup.

For every case, retain the CasU hierarchy, collider enabled/trigger/layer,
offset, density, material, `usedByComposite`, generated paths, contact points,
and the exact frame at which the tile mutation occurred. Use a disposable
world; do not modify the frozen seed export.

## Isaac translation questions

- Which tile shapes can be represented by planar convex polygons, and what
  plane/thickness convention matches the game's 2D contacts?
- Should chunks use one collider mesh each or a shared baked representation?
- How will a changed tile update local geometry without rebuilding unrelated
  chunks?
- What are the rebuild latency and memory costs for a seam edit and a broad
  64×64 edit?
- Which Unity material/contact parameters have a meaningful Isaac equivalent?

The Mission 9 capture reports 256 enabled TilemapCollider2D and 256 enabled
CompositeCollider2D components on layer 6, non-trigger, unit density, zero
offset, null shared material. It reports `usedByComposite=false` on the
captured collider configuration. Tile asset shapes, effective composite paths,
ignore-collision pairs, and refresh behavior remain unresolved.
