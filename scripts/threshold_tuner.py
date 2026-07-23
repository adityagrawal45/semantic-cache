"""
scripts/threshold_tuner.py — Offline tool to analyze near-miss logs and
suggest an optimal similarity threshold.

Reads the JSONL near-miss log (app/near_miss_analyzer.py's output) and
sweeps a range of candidate thresholds, reporting how many additional
"hits" each would have produced. Pick the threshold where the marginal
hit-rate gain flattens out relative to your tolerance for false-positive
matches (i.e., returning a cached answer for a question that wasn't quite
the same).

Usage:
    python scripts/threshold_tuner.py --log-path logs/near_misses.jsonl
"""

import argparse
import json
import os
from collections import Counter


def load_records(log_path: str):
    if not os.path.exists(log_path):
        print(f"No log file found at {log_path}. Run some traffic through the "
              f"service first so near-miss data accumulates.")
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sweep_thresholds(records, thresholds):
    results = []
    for t in thresholds:
        would_hit = sum(1 for r in records if r["best_score"] >= t)
        results.append((t, would_hit))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="logs/near_misses.jsonl")
    parser.add_argument("--start", type=float, default=0.80)
    parser.add_argument("--end", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    records = load_records(args.log_path)
    if not records:
        return

    thresholds = []
    t = args.start
    while t <= args.end + 1e-9:
        thresholds.append(round(t, 3))
        t += args.step

    results = sweep_thresholds(records, thresholds)

    print(f"Analyzed {len(records)} near-miss records from {args.log_path}\n")
    print(f"{'threshold':>10} | {'would-be hits':>13} | {'cumulative %':>12}")
    print("-" * 42)
    for t, hits in results:
        pct = (hits / len(records)) * 100
        print(f"{t:>10.3f} | {hits:>13} | {pct:>11.2f}%")

    # Suggest the threshold where marginal gain per 0.01 step drops below 1%
    # of total records — i.e., lowering further isn't buying much more
    # recall, so further lowering mostly just adds false-positive risk.
    suggestion = results[0][0]
    for i in range(1, len(results)):
        prev_hits, cur_hits = results[i - 1][1], results[i][1]
        marginal = (prev_hits - cur_hits) / len(records)
        if marginal < 0.01:
            suggestion = results[i][0]
        else:
            break

    print(f"\nSuggested threshold: {suggestion:.3f}")
    print("(This is a heuristic based on marginal hit-rate gain from near-miss "
          "logs only — validate against real accuracy/precision before "
          "changing SIMILARITY_THRESHOLD in production.)")


if __name__ == "__main__":
    main()
