"""Short live-game validation run for the CB1 protocol and agency contract."""

import numpy as np


def Run(env, steps=1_500, seed=0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    positions = set()
    completions = 0
    peak_ragdoll_time = 0.0
    recovered_from_ragdoll = False
    observed_block_damage = False
    previous_tile = None
    previous_health = None

    try:
        for step in range(1, int(steps) + 1):
            # Deliberately clumsy and excavation-heavy: enough motion to
            # exercise collisions, falling, attack edges, and recovery while
            # the Env's fail-fast checks validate every protocol macrostep.
            action = np.asarray([
                rng.integers(0, 3),
                int(rng.random() < 0.18),
                rng.integers(0, 3),
                int(rng.random() < 0.12),
                rng.integers(0, 9),
                rng.integers(0, 11),
                int(rng.random() < 0.65),
            ], dtype=np.int64)

            obs, reward, terminated, truncated, info = env.step(action)
            raw = env.latest_obs
            tile = (
                int(raw["PlayerTilePosition"]["X"]),
                int(raw["PlayerTilePosition"]["Y"]),
            )
            positions.add(tile)

            ragdoll_time = float(raw["TimeRagdolled"])
            peak_ragdoll_time = max(peak_ragdoll_time, ragdoll_time)
            if peak_ragdoll_time > 0.05 and ragdoll_time == 0.0:
                recovered_from_ragdoll = True

            health = np.asarray(
                raw["RelativeBlockMap"]["Health"], dtype=np.float32
            )
            if previous_tile == tile and previous_health is not None:
                observed_block_damage |= bool(np.any(health < previous_health - 0.01))
            previous_tile = tile
            previous_health = health.copy()

            if step % 100 == 0:
                print(
                    f"validation {step}/{steps}: unique_tiles={len(positions)}, "
                    f"progress={float(raw['LayerProgress']):.4f}, "
                    f"peak_ragdoll={peak_ragdoll_time:.2f}, "
                    f"block_damage={observed_block_damage}"
                )

            if terminated or truncated:
                if terminated:
                    completions += 1
                obs, _ = env.reset()
                previous_tile = None
                previous_health = None

        print("CB1 live validation completed without an invariant failure.")
        print(
            {
                "steps": int(steps),
                "unique_absolute_tiles": len(positions),
                "completions": completions,
                "peak_ragdoll_seconds": peak_ragdoll_time,
                "recovered_from_ragdoll": recovered_from_ragdoll,
                "observed_block_damage": observed_block_damage,
            }
        )
    finally:
        env.close()
