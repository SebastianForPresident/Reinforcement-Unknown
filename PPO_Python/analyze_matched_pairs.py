"""Analyze the 2x2 CB1 checkpoint/simulation-speed matched-pairs test."""

import argparse
import csv
import math
import statistics
from collections import defaultdict


def exact_sign_p_value(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", nargs="+")
    args = parser.parse_args()
    rows = []
    for results_csv in args.results_csv:
        with open(results_csv, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["seed"] = int(row["seed"])
                row["steps"] = int(row["steps"])
                row["completed"] = row["completed"].lower() == "true"
                rows.append(row)

    cells = defaultdict(dict)
    for row in rows:
        cells[(row["checkpoint"], row["speed"])][row["seed"]] = row

    print("Cell summaries")
    for cell, by_seed in sorted(cells.items()):
        values = list(by_seed.values())
        completed = sum(row["completed"] for row in values)
        durations = [row["steps"] * 0.2 / 60 for row in values]
        print(
            f"  {cell}: n={len(values)}, completed={completed}, "
            f"median_capped_sim_minutes={statistics.median(durations):.2f}"
        )

    def report_pair(label, left, right, left_name, right_name):
        seeds = sorted(set(left) & set(right))
        deltas = [(left[s]["steps"] - right[s]["steps"]) * 0.2 / 60 for s in seeds]
        wins = sum(delta < 0 for delta in deltas)
        losses = sum(delta > 0 for delta in deltas)
        ratios = [left[s]["steps"] / right[s]["steps"] for s in seeds]
        completion_advantage = sum(
            int(left[s]["completed"]) - int(right[s]["completed"]) for s in seeds
        )
        print(
            f"  {label}: pairs={len(seeds)}, {left_name}_wins={wins}, "
            f"{right_name}_wins={losses}, ties={len(seeds)-wins-losses}, "
            f"completion_advantage={completion_advantage:+d}, "
            f"median_delta_minutes={statistics.median(deltas) if deltas else float('nan'):.2f}, "
            f"median_time_ratio={statistics.median(ratios) if ratios else float('nan'):.3f}, "
            f"exact_sign_p={exact_sign_p_value(wins, losses):.4f}"
        )

    print("\nPaired 1x versus accelerated comparisons")
    checkpoints = sorted({row["checkpoint"] for row in rows})
    for checkpoint in checkpoints:
        one = cells.get((checkpoint, "1x"), {})
        accelerated_speeds = sorted(
            speed for name, speed in cells if name == checkpoint and speed != "1x"
        )
        for speed in accelerated_speeds:
            accelerated = cells.get((checkpoint, speed), {})
            report_pair(
                f"{checkpoint} 1x vs {speed}",
                one,
                accelerated,
                "1x",
                speed,
            )

    if len(checkpoints) == 2:
        print("\nPaired checkpoint comparisons within each speed")
        left_name, right_name = checkpoints
        for speed in sorted({row["speed"] for row in rows}):
            report_pair(
                speed,
                cells.get((left_name, speed), {}),
                cells.get((right_name, speed), {}),
                left_name,
                right_name,
            )


if __name__ == "__main__":
    main()
