# Casualties Unknown: Accessible Living-Entity Report

This report is based on the shipped world-generation paths, runtime behaviour code, and serialized prefab values in `SearchBase` and `CasualtiesUnknown_Data`. It includes living actors and organisms that ordinary procedural play can actually produce. It excludes corpses, machinery, purely mineral hazards, tutorial/debug-only actors, and orphaned or otherwise inaccessible resources.

## Roster at a glance

| Entity | Role | Health | Ordinary source |
|---|---:|---:|---|
| Cave Tick | swarm predator | 15 each | Layers 2 and 5; 16 emerge from one nest trigger |
| Shade Crawler | skittish ambush predator | 100 | Layers 1, 2, 3, and 5; also multiplied by the Infested modifier |
| Sand Stalker | aggressive predator | 400 | Layers 3 and 5 |
| Young Thornback | excavating predator | 280 | Layer 5 |
| Overgrown Tick | aggressive venomous predator | 300 | Layer 5 |
| Elder Thornback | boss-class excavating predator | 6,000 | Exactly three are instantiated on Layer 5 |
| Manifestation (`crystalenemy`) | summoned/lunging enemy | 450 | Produced by an accessible Mimic crystal effect |
| Trader 1 | Sawian NPC/trader | 1,000 | Random lifepods |
| Trader 2 | Sawian NPC/trader | 900 | Random lifepods |
| Trader 3 | Sawian NPC/trader | 1,250 | Random lifepods |
| Leadbush | passive animal masquerading as a bush | 120 | Layers 1, 2, and 5 on soil |
| Grabber Plant | stationary living restraint hazard | N/A (no `BuildingEntity` health) | Layer 5 |
| Cactus | stationary living contact hazard | destructible; loses 30 health when it strikes | Layers 3 and 4 |
| Banana Plant | stationary living trip hazard | N/A (no `BuildingEntity` health) | Layer 5 on soil |

Layer numbers above are the player-facing layer number: internal `biomeDepth + 1`.

## Shared animal logic

The five ordinary mobile fauna use the same `SpiderHandler` logic; the Elder Thornback uses its specialized derivative. They search for the nearest `Body` or `Limb` inside their sight radius, choose pursuit versus retreat on each movement decision, and bite on contact after a cooldown. The serialized `aggro` value is effectively the probability of choosing pursuit while healthy: the code retreats when `Random.value > aggro` or health is below the retreat threshold.

A bite chooses a limb and applies separate skin, muscle, pain, bleeding, venom, and infection effects after armour reduction. Taking damage can stun an animal; declining health also reduces its movement force. Death awards Survival experience equal to maximum health divided by 100 and invokes the prefab's configured drops.

### Cave Tick

- **Health:** 15.
- **Behaviour:** Fast, extremely short-cycle swarm attacker. It always chooses pursuit (`aggro = 1`) and does not have a low-health retreat threshold.
- **Mobility:** speed 49, force 100, movement interval 0.4–1.0 seconds, sight 25.
- **Bite cadence:** 0.5 seconds; retreats for 0.5 seconds after biting.
- **Per-bite injury:** 3 skin, 0.2 muscle, 25 pain, 0.5 bleeding, 0.25 venom; no infection chance.
- **Spawn:** The world places a Cave Tick nest trigger on Layers 2 and 5. Touching it causes **16 Cave Ticks** to emerge over 1.6 seconds. Generator coefficients are 0.15–0.20 × trap rarity on Layer 2 and 0.15–0.16 × trap rarity on Layer 5.
- **Combat meaning:** Individually trivial health, collectively strong action interruption, pain, and cumulative bleeding. The nest is the encounter, not a lone tick.

### Shade Crawler

- **Health:** 100; retreats automatically below 40.
- **Behaviour:** Deliberately skittish. With `aggro = 0.15`, only about 15% of healthy movement decisions pursue; the remainder move away. It has a long five-second post-bite retreat.
- **Mobility:** speed 30, force 500, movement interval 1.0–6.5 seconds, sight 8.
- **Bite cadence:** 5 seconds.
- **Per-bite injury:** 15 skin, 60 pain, 8 bleeding, 20 venom, and 10% base infection chance; no muscle damage.
- **Spawn:** Layers 1–2 at 0.40–0.42 × trap rarity, Layer 3 at 0.20 × trap rarity, and Layer 5 at 0.45–0.50 × trap rarity. The accessible **Infested** layer modifier additionally distributes roughly 1.7–1.8 per generation unit.
- **Combat meaning:** It is less a sustained melee opponent than a retreating poison/bleed ambusher. A combat head should not interpret disengagement as safety or defeat.

### Sand Stalker

- **Internal prefab:** `wallbiter`.
- **Health:** 400; retreats automatically below 80.
- **Behaviour:** Highly aggressive (`aggro = 0.95`) and substantially heavier than a crawler.
- **Mobility:** speed 17, force 2,800, movement interval 0.6–3.0 seconds, sight 30.
- **Bite cadence:** 1.5 seconds; 1.5-second retreat after contact.
- **Per-bite injury:** 30 skin, 15 muscle, 50 pain, 10 bleeding, and 20% base infection chance; no venom.
- **Spawn:** Layer 3 at 0.12–0.13 × trap rarity and Layer 5 at 0.10–0.11 × trap rarity. It is not distributed on Layer 4 despite sharing the arid generation branch with Layer 3.
- **Combat meaning:** A durable, direct physical threat. Unlike the Shade Crawler, its normal policy is to close distance.

### Young Thornback

- **Health:** 280; retreats automatically below 80.
- **Behaviour:** Highly aggressive (`aggro = 0.95`) and able to excavate toward the player.
- **Mobility:** speed 18, force 3,200, movement interval 0.6–3.0 seconds, sight 40.
- **Bite cadence:** 1.5 seconds; 1.5-second retreat after contact.
- **Per-bite injury:** 24 skin, 15 muscle, 40 pain, 6 bleeding, and 5% base infection chance; no venom.
- **Terrain damage:** 30 damage to blocks on its burrow cycle; non-animal structures take 15.
- **Spawn:** Layer 5 at 0.24–0.26 × trap rarity.
- **Combat meaning:** Terrain is not a reliable separator. Its ability to alter the route is directly relevant to both navigation feasibility and combat escape estimates.

### Overgrown Tick

- **Health:** 300.
- **Behaviour:** Always pursues (`aggro = 1`) and never enters a low-health retreat state.
- **Mobility:** speed 17, force 200, movement interval 0.5–1.75 seconds, sight 35.
- **Bite cadence:** 1.5 seconds; 1.5-second retreat after contact.
- **Per-bite injury:** 30 skin, 15 muscle, 50 pain, 10 bleeding, 14 venom, and 5% base infection chance.
- **Spawn:** Layer 5 at 0.10–0.12 × trap rarity.
- **Combat meaning:** Much tougher than a Cave Tick and combines ordinary tissue damage with venom. It does not burrow, so hard terrain separation still matters.

### Elder Thornback

- **Health:** 6,000.
- **Population:** Exactly **three** are instantiated on Layer 5, at random horizontal positions and vertical positions between the layer bottom and at least 20 units below the player at generation time.
- **Behaviour:** Boss-class hunter with 80-unit sight, `aggro = 1`, very high physical force, and terrain excavation. Proximity also forces wakefulness/energy, adrenaline, horror, stamina, and threat presentation onto the player.
- **Base mobility:** speed 4, force 20,000, movement interval 3–4 seconds. It becomes faster and more frequent below 3,750 health, then escalates again below 2,500.
- **Terrain damage:** 1,000 damage to blocks per burrow strike and 500 to non-animal structures. Collided loose items are destroyed.
- **Bite cadence:** 0.5 seconds.
- **Bite result:** This does **not** use the ordinary numeric bite package. It selects an intact non-core limb and can dismember it outright. Its fallback sequence progresses through disfigurement/eye removal, catastrophic torso damage (skin zero, 100 bleeding, 100 pain), and ultimately setting brain health to zero.
- **Death effect:** Nearby victory gives a major happiness and stimulation payoff and plays dedicated defeat music.
- **Combat meaning:** Health alone understates it. The important labels for medical arbitration are potential immediate limb loss, eye loss/disfigurement, catastrophic haemorrhage, and fatal brain damage. Navigation also needs to know that most ordinary terrain is not a durable blocker.

## Manifestation

`crystalenemy` is a living/hostile combat entity created at runtime rather than distributed directly by world generation.

- **Health:** 450.
- **Source:** A generated crystal has a 30% gate before receiving one or more weighted special effects. **Mimic** has weight 8 in a total weight pool of 139 for each effect draw. A Mimic crystal creates one or two Manifestations when touched by a body or struck. Because effects can be repeated through the draw loop, this is not cleanly equivalent to a single global spawn percentage.
- **Behaviour:** Floats toward a point 12 units above the player, rotates to aim, telegraphs a ray, and then lunges along that line. It begins with a random 2–5 second attack delay.
- **Attack:** One intact limb takes 50 skin, 35 muscle, 60 pain, and 15 bleeding after armour reduction; wearables take 0.4 damage. The target gains 70 adrenaline, is set to 100 stamina, screams, and ragdolls.
- **Recovery cycle:** The Manifestation lodges into the first ground surface hit. While the player is conscious it frees itself after 2.25 seconds plus one additional second per other active Manifestation, then attacks again after 0.5 to `4 + (enemyCount - 1)` seconds.
- **Combat meaning:** The telegraph makes evasion spatially learnable. Multiple manifestations deliberately slow one another's re-attack cycle, but each remains capable of a severe targeted wound and forced ragdoll.

## Traders

All three trader prefabs are accessible Sawian NPCs. A trader may replace the chest inside any procedurally generated lifepod according to the run's `traderchance`; the ordinary preset is 33%, with other shipped presets using 10% or 50%. The chosen variant is uniform among `trader1`, `trader2`, and `trader3`.

They begin as social/economic actors, generate inventories, respond to the player's condition, reputation, speech, appearance, and visible gun, and can provide a free dressing when first meeting a bleeding player. They become hostile at hostility 100. Low reputation causes hostility to rise while the player remains within 12 units; attacking one normally makes hostility immediate.

Once hostile, a trader attempts a swing roughly every 0.78 seconds with ±0.5 seconds of random timing whenever the player is within 8 units. The hit itself only connects within 6.5 units.

| Variant | Health | Damage multiplier | Special attack property |
|---|---:|---:|---|
| Trader 1 (`character = 0`) | 1,000 | 1.0× | none |
| Trader 2 (`character = 1`) | 900 | 0.9× | lower physical damage |
| Trader 3 (`character = 2`) | 1,250 | 1.3× | 10% chance to dismember a struck non-core limb |

At 1.0×, a connected swing deals 20 muscle, 24–40 skin, 45 pain, 0–12 bleeding, and 0.3 wearable damage after armour handling, then sets adrenaline to 100. Apply the table multiplier to those values.

At **below 200 health**, a trader is functionally defeated: they collapse, drop inventory, cease interaction, and become described as dead/corpse. This is distinct from the prefab's nominal health reaching zero. That distinction matters for target termination and medical/social state labels.

## Living environmental organisms

These organisms are accessible and have mechanical consequences, but they are not autonomous combatants.

### Leadbush

- **Health:** 120.
- **Nature:** The locale identifies it as an animal; its behaviour animates breathing while it remains rooted and non-hostile.
- **Spawn:** On soil on Layers 1–2 at 2.0–2.2 generation units and Layer 5 at 1.1–1.2.
- **Threat:** None. It is relevant as a destructible living object, not a combat target.

### Grabber Plant

- **Spawn:** Layer 5 at 0.40–0.50 × trap rarity.
- **Behaviour:** Its articulated tip wanders, avoids ground, and grabs a random player limb when the tip comes within 3.2 units, provided its five-second grab cooldown has elapsed.
- **Effect:** Immediately ragdolls and makes the player scream. For two seconds it drags the selected limb to its tip, holds shock at least 20, and maintains eye panic.
- **Direct tissue damage:** None in its own script. Its danger is restraint, forced ragdoll, displacement, and exposure to other threats.
- **Combat meaning:** “No damage” must not make it invisible to arbitration. This is a crowd-control/environmental hazard and a likely medical risk multiplier rather than a wound source.

### Cactus

- **Spawn:** Layers 3 and 4 at 1.4–1.6 generation units on the arid substrate/soil conditions.
- **Contact effect:** On collision with a body it sets shock to 30, kicks the body away at up to roughly 15 velocity, and applies 30 pain, 10 skin damage, and 2 bleeding to a random limb.
- **Self-damage:** Each successful strike removes 30 health from the cactus itself.
- **Combat meaning:** It is both a route hazard and a possible weapon through knockback positioning.

### Banana Plant

- **Spawn:** Layer 5 on soil at 1.9–2.0 × trap rarity.
- **Trigger:** A body entering horizontally faster than 5 units/second.
- **Effect:** Forces ragdoll and amplifies horizontal limb velocity—1.9× for the two designated feet/legs and 1.4× for all other limbs.
- **Direct tissue damage:** None in its own script; injury can arise from the resulting collision/fall physics.
- **Combat meaning:** A speed-dependent trip surface. It matters to navigation even when no enemy is present and becomes more dangerous during pursuit.

## Exclusions

- **Player character:** living, but it is the controlled subject rather than an encountered entity.
- **Corpses and animal corpses:** spawned, but not living.
- **Sidestabber:** locale/lore material exists, but the ordinary generator uses the inorganic `spikestabber` (“old machinery”) instead; no ordinary accessible spawn path for a living Sidestabber was found.
- **Spikestabber, Skullcrusher, drill pods, turrets, traps, crates, and other machines:** accessible hazards, but not living entities.
- **Ordinary plants and fungi with no autonomous or contact behaviour:** Glowplant, Stoneplant, Ceiling Rye, Geotree, Hydreed, Sand Rose, Drybush, Brown Shroom, Browncap, and similar harvestable scenery are accessible organisms, but not actors or hazards. They are omitted from the combat-facing roster because their only relevant entity behaviour is being placed/harvested/destroyed.
- **Debug, tutorial, course-only, unused, or orphaned prefabs:** excluded regardless of whether a resource can be loaded manually.

## Implications for head interfaces

The game's accessible living threats do not reduce to “enemy health and DPS.” They impose at least five distinct blocker/medical classes:

1. **Ordinary wound pressure:** Sand Stalker and Young Thornback.
2. **Venom/bleed attrition:** Shade Crawler, Cave Tick swarms, and Overgrown Tick.
3. **Catastrophic irreversible injury:** Elder Thornback and Trader 3 dismemberment.
4. **Forced ragdoll/restraint:** Manifestations, Grabber Plants, Cacti, and Banana Plants.
5. **Terrain denial or terrain destruction:** Young/Elder Thornbacks versus stationary environmental hazards.

That gives combat and medical a concrete interface: combat reports the likely injury class and time-to-next-hit; medical reports whether the projected wound is tolerable, immediately recoverable, or run-ending. Navigation remains continuously active because terrain, knockback, restraint, and the enemy's own excavation ability all change escape feasibility while combat is underway.
