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


def trace_observations(csv_path: Path, sidecar_path: Path, sample_count=32):
    """Deterministically reservoir-sample aligned real observations."""
    rng = random.Random(20260826)
    reservoir = []
    seen = 0
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("observation_id"):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-csv", type=Path)
    parser.add_argument("--grid-sidecar", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
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
