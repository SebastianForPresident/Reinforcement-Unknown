"""Frozen PPO_5 target-feature isolation and single-cell response audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import test_clever_hans as hans


FEATURE_CONDITIONS = (
    "empty",
    "health_1",
    "health_100",
    "health_1000",
    "health_15000",
    "extreme_at_max_health",
    "toxicity_at_health_100",
    "sleep_1_at_health_100",
    "sleep_2_at_health_100",
    "sleep_3_at_health_100",
)


def _eligible_cursor(row, radius=4):
    try:
        x = hans.CENTER_X + int(float(row["obs_relativelookpos_x"]))
        y = hans.CENTER_Y - int(float(row["obs_relativelookpos_y"]))
    except (KeyError, TypeError, ValueError):
        return False
    return radius <= x < hans.CENTER_X * 2 + 1 - radius and (
        radius <= y < hans.CENTER_Y * 2 + 1 - radius
    )


def _evaluate(model, observations):
    batch = {
        key: np.stack([observation[key] for observation in observations])
        for key in ("general", "spatial")
    }
    obs_tensor, _ = model.policy.obs_to_tensor(batch)
    with torch.inference_mode():
        features = model.policy.extract_features(obs_tensor)
        latent = model.policy.mlp_extractor.forward_actor(features)
        flat_logits = model.policy.action_net(latent)
    cardinalities = model.action_space.nvec.tolist()
    split_logits = torch.split(flat_logits, cardinalities, dim=1)
    output = []
    for row_index in range(len(observations)):
        logits = {
            name: split_logits[index][row_index].cpu().tolist()
            for index, name in enumerate(hans.BRANCH_NAMES)
        }
        probabilities = {
            name: torch.softmax(split_logits[index][row_index], dim=0).cpu().tolist()
            for index, name in enumerate(hans.BRANCH_NAMES)
        }
        output.append({
            "branch_logits": logits,
            "branches": probabilities,
            "attack_logit_gap": logits["attack"][1] - logits["attack"][0],
        })
    return output


def _clear_block_features(observation, x, y):
    changed = {key: value.copy() for key, value in observation.items()}
    changed["spatial"][0, x, y] = 0.0
    changed["spatial"][1, x, y] = 0.0
    changed["spatial"][2, x, y] = 0.0
    changed["spatial"][3:7, x, y] = 0.0
    changed["spatial"][3, x, y] = 1.0
    return changed


def _feature_variants(original, x, y):
    empty = _clear_block_features(original, x, y)
    variants = []
    for condition in FEATURE_CONDITIONS:
        changed = {key: value.copy() for key, value in empty.items()}
        if condition.startswith("health_"):
            raw_health = float(condition.split("_")[1])
            changed["spatial"][0, x, y] = float(hans.ObsNorm._block_health(raw_health))
        elif condition == "extreme_at_max_health":
            changed["spatial"][0, x, y] = 1.0
            changed["spatial"][1, x, y] = 1.0
        elif condition == "toxicity_at_health_100":
            changed["spatial"][0, x, y] = float(hans.ObsNorm._block_health(100.0))
            changed["spatial"][2, x, y] = 1.0
        elif condition.startswith("sleep_"):
            category = int(condition.split("_")[1])
            changed["spatial"][0, x, y] = float(hans.ObsNorm._block_health(100.0))
            changed["spatial"][3:7, x, y] = 0.0
            changed["spatial"][3 + category, x, y] = 1.0
        variants.append(changed)
    return variants


def _sign_p(deltas):
    up = int(sum(value > 1e-12 for value in deltas))
    down = int(sum(value < -1e-12 for value in deltas))
    count = up + down
    extreme = min(up, down)
    tail = sum(math.comb(count, k) for k in range(extreme + 1)) / (2 ** count)
    return up, down, min(1.0, 2.0 * tail)


def _add_holm_adjustment(records):
    """Add Holm family-wise adjusted p-values to a mapping of tests."""
    ordered = sorted(
        records.items(), key=lambda item: item[1]["two_sided_exact_sign_test_p"]
    )
    running = 0.0
    for rank, (name, record) in enumerate(ordered):
        adjusted = min(
            1.0,
            (len(ordered) - rank) * record["two_sided_exact_sign_test_p"],
        )
        running = max(running, adjusted)
        records[name]["holm_adjusted_p"] = running


def run(checkpoint, trace_csv, sidecar, sample_count=50, radius=4):
    torch.set_num_threads(1)
    hans.CasualtiesEnv.Init_General()
    model = PPO.load(str(Path(checkpoint).resolve()), device="cpu")
    model.policy.set_training_mode(False)
    samples = hans.trace_observations(
        trace_csv,
        sidecar,
        sample_count=sample_count,
        row_filter=lambda row: _eligible_cursor(row, radius),
    )
    if len(samples) != sample_count:
        raise ValueError(f"Requested {sample_count} samples, found {len(samples)}")

    feature_records = []
    heatmap_records = []
    solid = float(hans.ObsNorm._block_health(100.0))
    for sample_index, (observation_id, original) in enumerate(samples):
        cursor_cells = np.argwhere(original["spatial"][15] > 0.5)
        if len(cursor_cells) != 1:
            raise ValueError(f"Observation {observation_id} cursor count={len(cursor_cells)}")
        cursor_x, cursor_y = (int(value) for value in cursor_cells[0])

        feature_outputs = _evaluate(
            model, _feature_variants(original, cursor_x, cursor_y)
        )
        for condition, output in zip(FEATURE_CONDITIONS, feature_outputs):
            feature_records.append({
                "sample_index": sample_index,
                "observation_id": observation_id,
                "condition": condition,
                **output,
            })

        common = {key: value.copy() for key, value in original.items()}
        region = (
            slice(cursor_x - radius, cursor_x + radius + 1),
            slice(cursor_y - radius, cursor_y + radius + 1),
        )
        common["spatial"][0, region[0], region[1]] = 0.0
        common["spatial"][1, region[0], region[1]] = 0.0
        sweep_observations = [common]
        offsets = [(None, None)]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                changed = {key: value.copy() for key, value in common.items()}
                changed["spatial"][0, cursor_x + dx, cursor_y + dy] = solid
                sweep_observations.append(changed)
                offsets.append((dx, dy))
        for (dx, dy), output in zip(offsets, _evaluate(model, sweep_observations)):
            heatmap_records.append({
                "sample_index": sample_index,
                "observation_id": observation_id,
                "dx_from_cursor": dx,
                "dy_from_cursor": dy,
                **output,
            })

    grouped_features = {
        condition: [record for record in feature_records if record["condition"] == condition]
        for condition in FEATURE_CONDITIONS
    }
    feature_comparisons = {}
    comparison_baselines = {
        "health_1": "empty",
        "health_100": "empty",
        "health_1000": "empty",
        "health_15000": "empty",
        "extreme_at_max_health": "health_15000",
        "toxicity_at_health_100": "health_100",
        "sleep_1_at_health_100": "health_100",
        "sleep_2_at_health_100": "health_100",
        "sleep_3_at_health_100": "health_100",
    }
    for condition, baseline in comparison_baselines.items():
        deltas = np.asarray([
            changed["attack_logit_gap"] - base["attack_logit_gap"]
            for base, changed in zip(grouped_features[baseline], grouped_features[condition])
        ])
        up, down, p_value = _sign_p(deltas)
        feature_comparisons[condition + "_minus_" + baseline] = {
            "mean_attack_gap_delta": float(np.mean(deltas)),
            "median_attack_gap_delta": float(np.median(deltas)),
            "mean_abs_attack_gap_delta": float(np.mean(np.abs(deltas))),
            "increased_count": up,
            "decreased_count": down,
            "two_sided_exact_sign_test_p": p_value,
        }
    _add_holm_adjustment(feature_comparisons)

    no_block = {
        record["sample_index"]: record
        for record in heatmap_records if record["dx_from_cursor"] is None
    }
    heatmap = {}
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            records = [
                record for record in heatmap_records
                if record["dx_from_cursor"] == dx and record["dy_from_cursor"] == dy
            ]
            deltas = np.asarray([
                record["attack_logit_gap"]
                - no_block[record["sample_index"]]["attack_logit_gap"]
                for record in records
            ])
            up, down, p_value = _sign_p(deltas)
            branch_tv = {
                branch: float(np.mean([
                    0.5 * np.abs(
                        np.asarray(record["branches"][branch])
                        - np.asarray(no_block[record["sample_index"]]["branches"][branch])
                    ).sum()
                    for record in records
                ]))
                for branch in hans.BRANCH_NAMES
            }
            heatmap[f"{dx},{dy}"] = {
                "mean_attack_gap_delta": float(np.mean(deltas)),
                "median_attack_gap_delta": float(np.median(deltas)),
                "increased_count": up,
                "decreased_count": down,
                "two_sided_exact_sign_test_p": p_value,
                "mean_branch_tv": branch_tv,
            }
    _add_holm_adjustment(heatmap)

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": "cpu",
        "sample_count": sample_count,
        "selection_seed": 20260826,
        "protocol_correction": {
            "explicit_occupancy_or_solidity_channel": False,
            "explicit_material_identity_channel": False,
            "block_channels": [
                "log-normalized health", "extreme-health flag", "toxicity",
                "four-way sleep-quality one-hot",
            ],
            "note": "presence is represented indirectly by nonzero block features",
        },
        "feature_isolation": {
            "target": "single encoded cursor cell",
            "comparisons": feature_comparisons,
            "records": feature_records,
        },
        "single_cell_sweep": {
            "radius": radius,
            "block": "ordinary raw-health-100 cell; extreme flag zero",
            "common_baseline": "health/extreme cleared throughout 9x9 cursor region",
            "coordinate_note": "positive dy is downward in encoded array",
            "aggregate_heatmap": heatmap,
            "records": heatmap_records,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--grid-sidecar", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.checkpoint, args.trace_csv, args.grid_sidecar, args.samples
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
