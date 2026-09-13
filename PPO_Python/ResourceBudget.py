"""PPO_6 externally supplied excavation-resource budget."""

import numpy as np


MIN_ATTACK_BUDGET = 1_500
MAX_ATTACK_BUDGET = 3_000
FULL_BUDGET_COST = 10.0


def sample_budget(rng):
    return int(rng.integers(MIN_ATTACK_BUDGET, MAX_ATTACK_BUDGET + 1))


def normalize_remaining(remaining):
    return float(np.clip(remaining / MAX_ATTACK_BUDGET, 0.0, 1.0))


def exceeded(used, budget):
    return int(used) > int(budget)


def immediate_cost(swings, budget):
    budget = int(budget)
    if budget <= 0:
        raise ValueError("attack budget must be positive")
    return -FULL_BUDGET_COST * int(swings) / budget


def condition(processed, remaining):
    return {
        **processed,
        "general": np.concatenate([
            processed["general"],
            np.asarray([normalize_remaining(remaining)], dtype=np.float32),
        ]),
    }


def contract():
    return {
        "resource": "game-confirmed executed attack swings",
        "minimum_budget": MIN_ATTACK_BUDGET,
        "maximum_budget": MAX_ATTACK_BUDGET,
        "sampling": "uniform integer per episode",
        "observation": "remaining swings divided by maximum budget",
        "exceeded_when": "cumulative executed swings > episode budget",
        "termination": True,
        "failure_penalty": -10.0,
        "unused_budget_reward": 0.0,
        "immediate_swing_cost": "-10 / assigned episode budget",
        "full_budget_cumulative_cost": -FULL_BUDGET_COST,
    }
