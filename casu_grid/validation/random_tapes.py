"""Reproducible random action tapes for CasU recording and Isaac replay (pure standard library).

Each decision is [move_x, jump, vertical, crouch, look_dx, look_dy, attack] with the neutral values used by every earlier
recording (vertical 1, look 4/5, attack 0). move_x is uniform over {0, 1, 2}, jump and crouch are fair coins, i.e. the same
marginal distribution as the ``random`` tape of ``perf/collider_stress.py``.

    python casu_grid/validation/random_tapes.py --seed 2001 --length 200
"""
import argparse
import json
import random


def tape(seed: int, length: int):
    rng = random.Random(seed)
    return [[rng.randrange(3), rng.randrange(2), 1, rng.randrange(2), 4, 5, 0] for _ in range(length)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--length", type=int, default=200)
    a = p.parse_args()
    print(json.dumps(tape(a.seed, a.length), separators=(",", ":")))
