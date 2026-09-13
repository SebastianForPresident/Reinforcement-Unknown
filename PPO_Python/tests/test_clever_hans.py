"""Offline counterfactual probes for the trained CB1 spatial policy.

This is intentionally a diagnostic test rather than a gameplay evaluation.  It
never constructs CasulatiesEnv.Env, opens a socket, or writes a checkpoint.  A
frozen PPO checkpoint is loaded on CPU and queried with paired observations in
which only selected spatial cells differ.

Run explicitly with:

    .venv/bin/python PPO_Python/tests/test_clever_hans.py CHECKPOINT.zip

The test reports policy KL divergence, total-variation distance, branch-wise
probability changes, feature distance, and value change.  Sensitivity to a
counterfactual is evidence that the policy uses the edited input; it is not by
itself evidence that the response is behaviorally correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import CasualtiesEnv
import EpisodeTrace
import ObservationNormalization as ObsNorm
import Types


BRANCH_NAMES = ("move", "jump", "vertical", "crouch", "look_x", "look_y", "attack")
CENTER_X = Types.SIGHT_RANGE_X
CENTER_Y = Types.SIGHT_RANGE_Y


def _rankdata(values):
    """Average ranks for ties, equivalent to the usual Spearman convention."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _permutation_correlation(x, y, permutation_indices):
    """Pearson/Spearman correlations with deterministic two-sided permutation p."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def standardized(values):
        centered = values - np.mean(values)
        return centered / np.linalg.norm(centered)

    permutations = len(permutation_indices)
    result = {}
    for name, left, right in (
        ("pearson", x, y),
        ("spearman", _rankdata(x), _rankdata(y)),
    ):
        left_z = standardized(left)
        right_z = standardized(right)
        observed = float(np.dot(left_z, right_z))
        permuted = right_z[permutation_indices] @ left_z
        p_value = float(
            (1 + np.sum(np.abs(permuted) >= abs(observed)))
            / (permutations + 1)
        )
        result[name] = {"r": observed, "permutation_p_two_sided": p_value}
    return result


def base_observation() -> dict[str, np.ndarray]:
    """Make one bounded CB1 input with ordinary solid terrain around the player."""
    CasualtiesEnv.Init_General()
    spatial = np.zeros(
        (ObsNorm.SPATIAL_CHANNELS, CENTER_X * 2 + 1, CENTER_Y * 2 + 1),
        dtype=np.float32,
    )
    # An ordinary 100-health solid tile maps to this continuous health value.
    # Leave a small chamber around the player, with solid terrain elsewhere.
    spatial[0] = float(ObsNorm._block_health(100.0))
    spatial[0, CENTER_X - 2:CENTER_X + 3, CENTER_Y - 2:CENTER_Y + 3] = 0.0
    spatial[14, CENTER_X, CENTER_Y] = 1.0
    spatial[15, CENTER_X, CENTER_Y + 3] = 1.0
    return {
        "general": np.zeros(CasualtiesEnv._general_input_dim, dtype=np.float32),
        "spatial": spatial,
    }


def counterfactuals() -> tuple[list[str], dict[str, np.ndarray]]:
    """Return paired geometry edits with local and distant area-matched controls."""
    original = base_observation()
    observations = {"original": original}

    def edited(name, x_slice, y_slice, value):
        item = {key: array.copy() for key, array in original.items()}
        item["spatial"][0, x_slice, y_slice] = value
        observations[name] = item

    solid = float(ObsNorm._block_health(100.0))
    # Sixteen cells directly below the player versus the same edit far away.
    edited("open_below", slice(CENTER_X - 2, CENTER_X + 2),
           slice(CENTER_Y + 2, CENTER_Y + 6), 0.0)
    edited("open_distant", slice(1, 5), slice(1, 5), 0.0)
    # Close the existing local chamber on one side versus an area-matched edge edit.
    edited("close_below", slice(CENTER_X - 2, CENTER_X + 2),
           slice(CENTER_Y, CENTER_Y + 2), solid)
    edited("close_distant", slice(-5, -1), slice(-3, -1), solid)

    names = list(observations)
    batch = {
        key: np.stack([observations[name][key] for name in names])
        for key in ("general", "spatial")
    }
    return names, batch


def _leaf_paths(dtype, prefix=()):
    for name in dtype.names or ():
        child = dtype.fields[name][0]
        if child.subdtype is not None:
            continue
        path = (*prefix, name)
        if child.names:
            yield from _leaf_paths(child, path)
        else:
            yield path, child


def _assign_text(record, path, dtype, text):
    target = record
    for name in path[:-1]:
        target = target[name]
    if dtype.kind == "b":
        value = str(text).strip().lower() in {"1", "true", "yes"}
    elif dtype.kind in "iu":
        value = int(float(text))
    else:
        value = float(text)
    target[path[-1]] = value


def _raw_observation_from_trace(row, frame):
    """Reconstruct every general channel and retained grid channel from a trace."""
    raw = np.zeros((), dtype=Types.OBSERVATION_DTYPE)
    columns = {
        "obs_" + "_".join(name.lower() for name in path): (path, dtype)
        for path, dtype in _leaf_paths(Types.OBSERVATION_DTYPE)
    }
    for column, (path, dtype) in columns.items():
        text = row.get(column, "")
        if text != "":
            _assign_text(raw, path, dtype, text)

    raw["RelativeBlockMap"]["Health"] = frame["health"]
    raw["RelativeFluidMap"]["Type"] = frame["fluids"]
    # The v1 sidecar intentionally omitted per-cell toxicity and sleep quality.
    # Hold them at the same neutral values in every member of the pair.
    raw["RelativeBlockMap"]["Toxicity"] = 0.0
    raw["RelativeBlockMap"]["SleepQuality"] = 0
    return CasualtiesEnv.PreprocessObservation(raw)


def trace_observations(
    csv_path: Path,
    sidecar_path: Path,
    sample_count=32,
    row_filter=None,
):
    """Deterministically reservoir-sample aligned real observations."""
    rng = random.Random(20260826)
    reservoir = []
    seen = 0
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("observation_id"):
                continue
            if row_filter is not None and not row_filter(row):
                continue
            seen += 1
            if len(reservoir) < sample_count:
                reservoir.append(row)
            else:
                index = rng.randrange(seen)
                if index < sample_count:
                    reservoir[index] = row
    if len(reservoir) < sample_count:
        raise ValueError(f"Trace has only {len(reservoir)} sampleable rows")

    by_id = {int(row["observation_id"]): row for row in reservoir}
    frames = {}
    for frame in EpisodeTrace.iter_grid_history(sidecar_path):
        observation_id = int(frame["observation_id"])
        if observation_id in by_id:
            frames[observation_id] = frame
            if len(frames) == len(by_id):
                break
    missing = sorted(set(by_id) - set(frames))
    if missing:
        raise ValueError(f"Sidecar is missing sampled observation IDs: {missing}")
    return [
        (observation_id, _raw_observation_from_trace(row, frames[observation_id]))
        for observation_id, row in sorted(by_id.items())
    ]


SEMANTIC_CONDITIONS = ("open", "breakable", "damaged", "infinirock")
LOCATION_CONDITIONS = ("none", "cursor", "below", "opposite", "distant")


def semantic_target_counterfactuals(
    csv_path: Path, sidecar_path: Path, sample_count=50
):
    """Make four terrain states at the encoded cursor target in real frames."""
    margin = 1

    def cursor_patch_is_in_bounds(row):
        try:
            cursor_x = CENTER_X + int(float(row["obs_relativelookpos_x"]))
            cursor_y = CENTER_Y - int(float(row["obs_relativelookpos_y"]))
        except (KeyError, TypeError, ValueError):
            return False
        return (
            margin <= cursor_x < CENTER_X * 2 + 1 - margin
            and margin <= cursor_y < CENTER_Y * 2 + 1 - margin
            and (cursor_x != CENTER_X or cursor_y != CENTER_Y)
        )

    samples = trace_observations(
        csv_path,
        sidecar_path,
        sample_count=sample_count,
        row_filter=cursor_patch_is_in_bounds,
    )
    variants = []
    metadata = []
    health_values = {
        "open": 0.0,
        "breakable": float(ObsNorm._block_health(100.0)),
        "damaged": float(ObsNorm._block_health(1.0)),
        "infinirock": 1.0,
    }
    for sample_index, (observation_id, original) in enumerate(samples):
        cursor_cells = np.argwhere(original["spatial"][15] > 0.5)
        if len(cursor_cells) != 1:
            raise ValueError(
                f"Observation {observation_id} has {len(cursor_cells)} cursors"
            )
        cursor_x, cursor_y = (int(value) for value in cursor_cells[0])
        patch = (
            slice(cursor_x - margin, cursor_x + margin + 1),
            slice(cursor_y - margin, cursor_y + margin + 1),
        )
        for condition in SEMANTIC_CONDITIONS:
            changed = {key: value.copy() for key, value in original.items()}
            changed["spatial"][0, patch[0], patch[1]] = health_values[condition]
            changed["spatial"][1, patch[0], patch[1]] = float(
                condition == "infinirock"
            )
            variants.append(changed)
            metadata.append({
                "sample_index": sample_index,
                "observation_id": observation_id,
                "condition": condition,
                "cursor_x": cursor_x,
                "cursor_y": cursor_y,
                "patch_cells": 9,
            })
    batch = {
        key: np.stack([observation[key] for observation in variants])
        for key in ("general", "spatial")
    }
    return metadata, batch


def location_counterfactuals(csv_path: Path, sidecar_path: Path, sample_count=50):
    """Move one identical breakable patch among four controlled locations."""
    margin = 1
    below = (CENTER_X, CENTER_Y + 3)
    distant = (2, 2)

    def candidate_centers(row):
        cursor = (
            CENTER_X + int(float(row["obs_relativelookpos_x"])),
            CENTER_Y - int(float(row["obs_relativelookpos_y"])),
        )
        opposite = (2 * CENTER_X - cursor[0], 2 * CENTER_Y - cursor[1])
        return (cursor, below, opposite, distant)

    def patches_are_valid(row):
        try:
            centers = candidate_centers(row)
        except (KeyError, TypeError, ValueError):
            return False
        width = CENTER_X * 2 + 1
        height = CENTER_Y * 2 + 1
        if not all(
            margin <= x < width - margin and margin <= y < height - margin
            for x, y in centers
        ):
            return False
        return all(
            max(abs(ax - bx), abs(ay - by)) > 2
            for index, (ax, ay) in enumerate(centers)
            for bx, by in centers[index + 1:]
        )

    samples = trace_observations(
        csv_path,
        sidecar_path,
        sample_count=sample_count,
        row_filter=patches_are_valid,
    )
    variants = []
    metadata = []
    block_health = float(ObsNorm._block_health(100.0))
    for sample_index, (observation_id, original) in enumerate(samples):
        cursor_cells = np.argwhere(original["spatial"][15] > 0.5)
        if len(cursor_cells) != 1:
            raise ValueError(
                f"Observation {observation_id} has {len(cursor_cells)} cursors"
            )
        cursor = tuple(int(value) for value in cursor_cells[0])
        centers = {
            "cursor": cursor,
            "below": below,
            "opposite": (2 * CENTER_X - cursor[0], 2 * CENTER_Y - cursor[1]),
            "distant": distant,
        }
        common = {key: value.copy() for key, value in original.items()}
        for x, y in centers.values():
            patch = (slice(x - 1, x + 2), slice(y - 1, y + 2))
            common["spatial"][0, patch[0], patch[1]] = 0.0
            common["spatial"][1, patch[0], patch[1]] = 0.0
        for condition in LOCATION_CONDITIONS:
            changed = {key: value.copy() for key, value in common.items()}
            if condition == "none":
                x, y = -1, -1
            else:
                x, y = centers[condition]
                patch = (slice(x - 1, x + 2), slice(y - 1, y + 2))
                changed["spatial"][0, patch[0], patch[1]] = block_health
            variants.append(changed)
            metadata.append({
                "sample_index": sample_index,
                "observation_id": observation_id,
                "condition": condition,
                "block_x": x,
                "block_y": y,
                "cursor_x": cursor[0],
                "cursor_y": cursor[1],
                "patch_cells": 9,
            })
    if len(samples) != sample_count:
        raise ValueError(
            f"Requested {sample_count} eligible observations, found {len(samples)}"
        )
    batch = {
        key: np.stack([observation[key] for observation in variants])
        for key in ("general", "spatial")
    }
    return metadata, batch


def probe_attack_semantics(
    checkpoint: Path,
    trace_csv: Path,
    grid_sidecar: Path,
    sample_count: int = 50,
    location_audit: bool = False,
) -> dict:
    """Measure every policy branch under four matched cursor-target materials."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    torch.set_num_threads(1)
    CasualtiesEnv.Init_General()
    model = PPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)
    if location_audit:
        conditions = LOCATION_CONDITIONS
        baseline_condition = "none"
        metadata, observations = location_counterfactuals(
            trace_csv, grid_sidecar, sample_count
        )
    else:
        conditions = SEMANTIC_CONDITIONS
        baseline_condition = "open"
        metadata, observations = semantic_target_counterfactuals(
            trace_csv, grid_sidecar, sample_count
        )
    obs_tensor, _ = model.policy.obs_to_tensor(observations)
    with torch.inference_mode():
        features = model.policy.extract_features(obs_tensor)
        distribution = model.policy.get_distribution(obs_tensor)
        values = model.policy.predict_values(obs_tensor).flatten()
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        flat_logits = model.policy.action_net(latent_pi)
    probabilities = _categorical_probabilities(distribution)
    cardinalities = [len(branch[0]) for branch in probabilities]
    logits = [part.cpu().numpy() for part in torch.split(
        flat_logits, cardinalities, dim=1
    )]
    attack_output_start = sum(cardinalities[:-1])
    attack_weight = model.policy.action_net.weight[
        attack_output_start:attack_output_start + 2
    ].detach()
    attack_bias = model.policy.action_net.bias[
        attack_output_start:attack_output_start + 2
    ].detach()
    explicit_attack_bias_gap = float(attack_bias[1] - attack_bias[0])
    attack_weight_difference = attack_weight[1] - attack_weight[0]
    latent_attack_contribution = (
        latent_pi @ attack_weight_difference
    ).cpu().numpy()
    reconstructed = [
        torch.softmax(torch.as_tensor(branch), dim=1).numpy()
        for branch in logits
    ]
    reconstruction_error = max(
        float(np.max(np.abs(probability - rebuilt)))
        for probability, rebuilt in zip(probabilities, reconstructed)
    )

    samples = []
    grouped = {condition: [] for condition in conditions}
    for index, item in enumerate(metadata):
        branches = {
            name: probabilities[branch_index][index].tolist()
            for branch_index, name in enumerate(BRANCH_NAMES)
        }
        branch_logits = {
            name: logits[branch_index][index].tolist()
            for branch_index, name in enumerate(BRANCH_NAMES)
        }
        record = {
            **item,
            "value": float(values[index]),
            "feature_norm": float(torch.linalg.vector_norm(features[index])),
            "branches": branches,
            "branch_logits": branch_logits,
            "attack_logit_gap": float(
                branch_logits["attack"][1] - branch_logits["attack"][0]
            ),
            "attack_latent_contribution": float(
                latent_attack_contribution[index]
            ),
        }
        samples.append(record)
        grouped[item["condition"]].append(record)

    aggregates = {}
    for condition, records in grouped.items():
        attack = np.asarray([r["branches"]["attack"][1] for r in records])
        attack_gap = np.asarray([r["attack_logit_gap"] for r in records])
        aggregates[condition] = {
            "attack_probability_mean": float(np.mean(attack)),
            "attack_probability_median": float(np.median(attack)),
            "attack_probability_p10": float(np.quantile(attack, 0.1)),
            "attack_probability_p90": float(np.quantile(attack, 0.9)),
            "attack_logit_gap_mean": float(np.mean(attack_gap)),
            "attack_logit_gap_median": float(np.median(attack_gap)),
            "attack_logit_gap_p10": float(np.quantile(attack_gap, 0.1)),
            "attack_logit_gap_p90": float(np.quantile(attack_gap, 0.9)),
            "mean_branch_probabilities": {
                branch: np.mean(
                    np.asarray([r["branches"][branch] for r in records]), axis=0
                ).tolist()
                for branch in BRANCH_NAMES
            },
            "mean_branch_logits": {
                branch: np.mean(
                    np.asarray([r["branch_logits"][branch] for r in records]),
                    axis=0,
                ).tolist()
                for branch in BRANCH_NAMES
            },
            "value_mean": float(np.mean([r["value"] for r in records])),
        }

    paired = {}
    baseline_records = grouped[baseline_condition]
    epsilon = 1e-12
    for condition in conditions[1:]:
        records = grouped[condition]
        attack_delta = np.asarray([
            changed["branches"]["attack"][1]
            - baseline["branches"]["attack"][1]
            for baseline, changed in zip(baseline_records, records)
        ])
        attack_gap_delta = np.asarray([
            changed["attack_logit_gap"] - baseline["attack_logit_gap"]
            for baseline, changed in zip(baseline_records, records)
        ])
        branch_tv = {}
        for branch in BRANCH_NAMES:
            branch_tv[branch] = float(np.mean([
                0.5 * np.abs(
                    np.asarray(changed["branches"][branch])
                    - np.asarray(baseline["branches"][branch])
                ).sum()
                for baseline, changed in zip(baseline_records, records)
            ]))
        gap_increased = int(np.sum(attack_gap_delta > epsilon))
        gap_decreased = int(np.sum(attack_gap_delta < -epsilon))
        gap_non_ties = gap_increased + gap_decreased
        gap_sign_extreme = min(gap_increased, gap_decreased)
        gap_sign_tail = (
            sum(math.comb(gap_non_ties, k) for k in range(gap_sign_extreme + 1))
            / (2 ** gap_non_ties)
            if gap_non_ties else 0.5
        )
        paired[condition + "_minus_" + baseline_condition] = {
            "attack_probability_delta_mean": float(np.mean(attack_delta)),
            "attack_probability_delta_median": float(np.median(attack_delta)),
            "attack_probability_delta_mean_abs": float(np.mean(np.abs(attack_delta))),
            "attack_increased_count": int(np.sum(attack_delta > epsilon)),
            "attack_decreased_count": int(np.sum(attack_delta < -epsilon)),
            "attack_logit_gap_delta_mean": float(np.mean(attack_gap_delta)),
            "attack_logit_gap_delta_median": float(np.median(attack_gap_delta)),
            "attack_logit_gap_delta_mean_abs": float(
                np.mean(np.abs(attack_gap_delta))
            ),
            "attack_logit_gap_delta_min": float(np.min(attack_gap_delta)),
            "attack_logit_gap_delta_p10": float(
                np.quantile(attack_gap_delta, 0.1)
            ),
            "attack_logit_gap_delta_p90": float(
                np.quantile(attack_gap_delta, 0.9)
            ),
            "attack_logit_gap_delta_max": float(np.max(attack_gap_delta)),
            "attack_logit_gap_increased_count": gap_increased,
            "attack_logit_gap_decreased_count": gap_decreased,
            "attack_logit_gap_two_sided_exact_sign_test_p": float(
                min(1.0, 2.0 * gap_sign_tail)
            ),
            "mean_branch_tv": branch_tv,
        }

    offset_sweep = {}
    for offset in (0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0):
        by_condition = {}
        for condition, records in grouped.items():
            shifted_gaps = np.asarray([
                record["attack_logit_gap"] - offset for record in records
            ])
            shifted_probabilities = 1.0 / (1.0 + np.exp(-shifted_gaps))
            by_condition[condition] = {
                "mean_attack_probability": float(np.mean(shifted_probabilities)),
                "median_attack_probability": float(np.median(shifted_probabilities)),
                "deterministic_attack_count": int(np.sum(shifted_gaps > 0.0)),
                "deterministic_no_attack_count": int(np.sum(shifted_gaps <= 0.0)),
            }
        offset_sweep[str(int(offset))] = by_condition

    recomposed_gaps = latent_attack_contribution + explicit_attack_bias_gap
    observed_gaps = np.asarray([
        record["attack_logit_gap"] for record in samples
    ])

    if location_audit:
        none_records = grouped["none"]
        cursor_records = grouped["cursor"]
        attack_suppression = np.asarray([
            baseline["attack_logit_gap"] - occupied["attack_logit_gap"]
            for baseline, occupied in zip(none_records, cursor_records)
        ])
        per_state_tv = {
            branch: np.asarray([
                0.5 * np.abs(
                    np.asarray(occupied["branches"][branch])
                    - np.asarray(baseline["branches"][branch])
                ).sum()
                for baseline, occupied in zip(none_records, cursor_records)
            ])
            for branch in ("move", "look_x", "look_y")
        }
        per_state_tv["motor_mean"] = np.mean(
            np.stack(list(per_state_tv.values())), axis=0
        )
        permutation_count = 100_000
        correlation_seed = 20260913
        correlation_rng = np.random.default_rng(correlation_seed)
        permutation_indices = np.asarray([
            correlation_rng.permutation(sample_count)
            for _ in range(permutation_count)
        ])
        correlations = {
            branch: _permutation_correlation(
                attack_suppression, changes, permutation_indices
            )
            for branch, changes in per_state_tv.items()
        }
        raw_tests = [
            (correlations[branch][kind]["permutation_p_two_sided"], branch, kind)
            for branch in correlations
            for kind in ("pearson", "spearman")
        ]
        running_adjusted = 0.0
        for rank, (p_value, branch, kind) in enumerate(sorted(raw_tests)):
            adjusted = min(1.0, (len(raw_tests) - rank) * p_value)
            running_adjusted = max(running_adjusted, adjusted)
            correlations[branch][kind]["holm_adjusted_p"] = running_adjusted
        motor_mode_correlation = {
            "attack_measure": (
                "positive means cursor occupancy suppresses attack logit gap"
            ),
            "branch_change_measure": "total-variation distance versus no-block",
            "motor_mean_definition": "mean TV of move, look_x, and look_y",
            "sample_count": sample_count,
            "permutation_count": permutation_count,
            "random_seed": correlation_seed,
            "multiple_test_adjustment": "Holm across 8 Pearson/Spearman tests",
            "correlations": correlations,
            "per_state": [
                {
                    "sample_index": index,
                    "observation_id": none_records[index]["observation_id"],
                    "attack_gap_suppression": float(attack_suppression[index]),
                    **{
                        branch + "_tv": float(values[index])
                        for branch, values in per_state_tv.items()
                    },
                }
                for index in range(sample_count)
            ],
        }
        source_name = "trace_breakable_patch_relocation"
        eligibility = "four in-bounds, pairwise non-overlapping 3x3 patches"
        intervention = {
            "patch": "one 3x3 ordinary 100-health breakable block",
            "common_substrate": (
                "all four candidate patches cleared in health/extreme channels"
            ),
            "general_observation_held_identical": True,
            "all_non_candidate_spatial_cells_held_identical": True,
            "conditions": list(LOCATION_CONDITIONS),
            "none_condition": "shared cleared-patch baseline only",
        }
        preregistered_hypotheses = {
            "cursor_dominant": "target occupancy representation",
            "below_dominant": "travel-path obstruction representation",
            "opposite_directional_effect": "attack coupled to locomotion or aiming",
            "distant_substantial": "broader density or contextual response",
        }
    else:
        motor_mode_correlation = None
        source_name = "trace_semantic_cursor_patch"
        eligibility = "one in-bounds non-center encoded cursor with 3x3 margin"
        intervention = {
            "patch": "3x3 centered on encoded cursor",
            "general_observation_held_identical": True,
            "all_other_spatial_cells_held_identical": True,
            "conditions": {
                "open": {"health": 0.0, "extreme": 0.0},
                "breakable": {"raw_health": 100.0, "extreme": 0.0},
                "damaged": {"raw_health": 1.0, "extreme": 0.0},
                "infinirock": {"health_channel": 1.0, "extreme": 1.0},
            },
        }
        preregistered_hypotheses = None

    return {
        "checkpoint": str(checkpoint),
        "device": str(model.device),
        "source": source_name,
        "source_files": {
            "trace_csv": str(Path(trace_csv).expanduser().resolve()),
            "grid_sidecar": str(Path(grid_sidecar).expanduser().resolve()),
        },
        "branch_schema": {
            "order": list(BRANCH_NAMES),
            "cardinalities": cardinalities,
            "vector_entry_semantics": (
                "categorical probabilities in the model action-space index order"
            ),
            "logit_entry_semantics": (
                "pre-softmax action-net outputs in the same index order"
            ),
            "probability_reconstruction_max_abs_error": reconstruction_error,
        },
        "selection": {
            "random_seed": 20260826,
            "reservoir_sample_count": sample_count,
            "eligibility": eligibility,
        },
        "intervention": intervention,
        "preregistered_hypotheses": preregistered_hypotheses,
        "motor_mode_correlation": motor_mode_correlation,
        "attack_gap_decomposition": {
            "formula": (
                "gap=(attack_weight-no_attack_weight) dot actor_latent "
                "+ (attack_bias-no_attack_bias)"
            ),
            "explicit_final_layer_bias_gap": explicit_attack_bias_gap,
            "attack_weight_difference_l2_norm": float(
                torch.linalg.vector_norm(attack_weight_difference)
            ),
            "latent_contribution_mean": float(
                np.mean(latent_attack_contribution)
            ),
            "latent_contribution_median": float(
                np.median(latent_attack_contribution)
            ),
            "recomposition_max_abs_error": float(
                np.max(np.abs(recomposed_gaps - observed_gaps))
            ),
        },
        "frozen_attack_offset_sweep": {
            "definition": "shifted attack gap = original attack gap - offset",
            "note": "diagnostic only; checkpoint weights were not mutated",
            "offsets": offset_sweep,
        },
        "aggregates": aggregates,
        "paired_against_" + baseline_condition: paired,
        "samples": samples,
    }


def traced_counterfactuals(csv_path: Path, sidecar_path: Path, sample_count=32):
    """Apply local and distant patch permutations to sampled real states."""
    variants = []
    names = []
    local = (slice(CENTER_X - 4, CENTER_X + 5),
             slice(CENTER_Y - 1, CENTER_Y + 8))
    for sample_index, (observation_id, original) in enumerate(
        trace_observations(csv_path, sidecar_path, sample_count)
    ):
        sample_name = f"sample_{sample_index}_obs_{observation_id}"
        names.append(f"{sample_name}/original")
        variants.append(original)

        source = original["spatial"][:14]
        local_before = source[:, local[0], local[1]]
        local_change = float(np.abs(
            local_before - np.flip(local_before, axis=(1, 2))
        ).sum())
        candidates = []
        for x in (1, 10, 19, 28, 55, 64, 73):
            for y in (1, 10, 15, 33, 39):
                patch = (slice(x, x + 9), slice(y, y + 9))
                before = source[:, patch[0], patch[1]]
                change = float(np.abs(
                    before - np.flip(before, axis=(1, 2))
                ).sum())
                candidates.append((abs(change - local_change), -change, patch))
        distant = min(candidates, key=lambda item: (item[0], item[1]))[2]

        for label, patch in (("local_flip", local), ("distant_flip", distant)):
            changed = {key: value.copy() for key, value in original.items()}
            # Flip every retained terrain/fluid channel together. Center/cursor
            # markers remain fixed, so the intervention is geometry-only.
            changed["spatial"][:14, patch[0], patch[1]] = np.flip(
                changed["spatial"][:14, patch[0], patch[1]], axis=(1, 2)
            )
            names.append(f"{sample_name}/{label}")
            variants.append(changed)
    batch = {
        key: np.stack([observation[key] for observation in variants])
        for key in ("general", "spatial")
    }
    return names, batch


def _categorical_probabilities(distribution):
    branches = getattr(distribution, "distribution", None)
    if branches is None:
        raise TypeError("Expected SB3 MultiCategoricalDistribution")
    return [branch.probs.detach().cpu().numpy() for branch in branches]


def probe_checkpoint(
    checkpoint: Path,
    trace_csv: Path | None = None,
    grid_sidecar: Path | None = None,
    sample_count: int = 32,
) -> dict:
    """Run all probes in one CPU batch and return JSON-serializable measurements."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    torch.set_num_threads(1)
    CasualtiesEnv.Init_General()
    model = PPO.load(str(checkpoint), device="cpu")
    model.policy.set_training_mode(False)
    if (trace_csv is None) != (grid_sidecar is None):
        raise ValueError("trace CSV and grid sidecar must be supplied together")
    if trace_csv is None:
        names, observations = counterfactuals()
        pairs = [(0, index) for index in range(1, len(names))]
        source = "synthetic"
    else:
        names, observations = traced_counterfactuals(
            trace_csv, grid_sidecar, sample_count
        )
        pairs = []
        for baseline in range(0, len(names), 3):
            pairs.extend(((baseline, baseline + 1), (baseline, baseline + 2)))
        source = "trace"
    obs_tensor, _ = model.policy.obs_to_tensor(observations)

    with torch.inference_mode():
        features = model.policy.extract_features(obs_tensor)
        distribution = model.policy.get_distribution(obs_tensor)
        values = model.policy.predict_values(obs_tensor).flatten()
    probabilities = _categorical_probabilities(distribution)

    result = {
        "checkpoint": str(checkpoint),
        "device": str(model.device),
        "source": source,
        "sample_count": sample_count if source == "trace" else 1,
        "probes": {},
    }
    epsilon = 1e-12
    for baseline_index, index in pairs:
        name = names[index]
        branch_stats = {}
        total_kl = 0.0
        total_tv = 0.0
        for branch_name, branch in zip(BRANCH_NAMES, probabilities):
            baseline = branch[baseline_index].astype(np.float64)
            changed = branch[index].astype(np.float64)
            kl = float(np.sum(baseline * np.log((baseline + epsilon) / (changed + epsilon))))
            tv = float(0.5 * np.abs(baseline - changed).sum())
            max_delta = float(np.abs(baseline - changed).max())
            branch_stats[branch_name] = {
                "kl": kl,
                "tv": tv,
                "max_probability_delta": max_delta,
            }
            total_kl += kl
            total_tv += tv
        result["probes"][name] = {
            "joint_policy_kl": total_kl,
            "summed_branch_tv": total_tv,
            "spatial_input_l1": float(np.abs(
                observations["spatial"][index]
                - observations["spatial"][baseline_index]
            ).sum()),
            "feature_l2": float(torch.linalg.vector_norm(
                features[index] - features[baseline_index]
            )),
            "value_delta": float(values[index] - values[baseline_index]),
            "branches": branch_stats,
        }

    if source == "trace":
        local = [value["joint_policy_kl"] for name, value in result["probes"].items()
                 if name.endswith("/local_flip")]
        distant = [value["joint_policy_kl"] for name, value in result["probes"].items()
                   if name.endswith("/distant_flip")]
        local_normalized = [
            value["joint_policy_kl"] / max(value["spatial_input_l1"], epsilon)
            for name, value in result["probes"].items()
            if name.endswith("/local_flip")
        ]
        distant_normalized = [
            value["joint_policy_kl"] / max(value["spatial_input_l1"], epsilon)
            for name, value in result["probes"].items()
            if name.endswith("/distant_flip")
        ]
        local_wins = sum(a > b for a, b in zip(local, distant))
        non_ties = sum(a != b for a, b in zip(local, distant))
        sign_extreme = min(local_wins, non_ties - local_wins)
        sign_tail = sum(
            math.comb(non_ties, k) for k in range(sign_extreme + 1)
        ) / (2 ** non_ties) if non_ties else 0.5
        sign_p_two_sided = min(1.0, 2.0 * sign_tail)
        result["comparisons"] = {
            "mean_local_policy_kl": float(np.mean(local)),
            "mean_distant_policy_kl": float(np.mean(distant)),
            "median_local_policy_kl": float(np.median(local)),
            "median_distant_policy_kl": float(np.median(distant)),
            "median_paired_policy_kl_difference": float(np.median(
                np.asarray(local) - np.asarray(distant)
            )),
            "mean_local_to_distant_kl_ratio": (
                float(np.mean(local)) / max(float(np.mean(distant)), epsilon)
            ),
            "local_kl_greater_count": int(local_wins),
            "non_tied_pair_count": int(non_ties),
            "two_sided_exact_sign_test_p": float(sign_p_two_sided),
            "mean_local_input_normalized_kl": float(np.mean(local_normalized)),
            "mean_distant_input_normalized_kl": float(np.mean(distant_normalized)),
            "median_local_input_normalized_kl": float(np.median(local_normalized)),
            "median_distant_input_normalized_kl": float(np.median(distant_normalized)),
            "input_normalized_local_to_distant_ratio": (
                float(np.mean(local_normalized))
                / max(float(np.mean(distant_normalized)), epsilon)
            ),
        }
    else:
        result["comparisons"] = {
            "opening_local_to_distant_kl_ratio": (
                result["probes"]["open_below"]["joint_policy_kl"]
                / max(result["probes"]["open_distant"]["joint_policy_kl"], epsilon)
            ),
            "closing_local_to_distant_kl_ratio": (
                result["probes"]["close_below"]["joint_policy_kl"]
                / max(result["probes"]["close_distant"]["joint_policy_kl"], epsilon)
            ),
        }
    return result


class CounterfactualConstructionTests(unittest.TestCase):
    def test_edits_are_paired_and_leave_general_state_identical(self):
        names, batch = counterfactuals()
        self.assertEqual(names, [
            "original", "open_below", "open_distant",
            "close_below", "close_distant",
        ])
        self.assertEqual(batch["spatial"].shape, (5, 16, 85, 49))
        self.assertTrue(np.all(batch["general"] == batch["general"][0]))
        expected_changed_cells = (12, 16, 8, 0)
        for changed_index in range(1, 5):
            changed = batch["spatial"][changed_index] != batch["spatial"][0]
            self.assertEqual(
                int(changed.sum()), expected_changed_cells[changed_index - 1]
            )
            self.assertTrue(np.all(changed[1:] == 0))

    def test_inputs_remain_within_observation_contract(self):
        _, batch = counterfactuals()
        self.assertTrue(np.isfinite(batch["general"]).all())
        self.assertTrue(np.isfinite(batch["spatial"]).all())
        self.assertGreaterEqual(float(batch["general"].min()), -1.0)
        self.assertLessEqual(float(batch["general"].max()), 1.0)
        self.assertGreaterEqual(float(batch["spatial"].min()), 0.0)
        self.assertLessEqual(float(batch["spatial"].max()), 1.0)

    def test_semantic_material_values_are_distinct_and_bounded(self):
        values = (
            0.0,
            float(ObsNorm._block_health(100.0)),
            float(ObsNorm._block_health(1.0)),
            1.0,
        )
        self.assertEqual(len(set(values)), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertLess(values[2], values[1])
        self.assertLess(values[1], values[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-csv", type=Path)
    parser.add_argument("--grid-sidecar", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--semantic-target",
        action="store_true",
        help="run open/breakable/damaged/infinirock cursor-patch audit",
    )
    parser.add_argument(
        "--location-target",
        action="store_true",
        help="relocate one identical breakable patch among controlled locations",
    )
    args = parser.parse_args()
    if args.semantic_target and args.location_target:
        parser.error("choose only one of --semantic-target and --location-target")
    if args.semantic_target or args.location_target:
        if args.trace_csv is None or args.grid_sidecar is None:
            parser.error("counterfactual audits require trace CSV and grid sidecar")
        result = probe_attack_semantics(
            args.checkpoint,
            args.trace_csv,
            args.grid_sidecar,
            sample_count=args.samples,
            location_audit=args.location_target,
        )
    else:
        result = probe_checkpoint(
            args.checkpoint,
            trace_csv=args.trace_csv,
            grid_sidecar=args.grid_sidecar,
            sample_count=args.samples,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        unittest.main()
    else:
        main()
