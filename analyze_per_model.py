"""Per-model classification distribution from a classification file.
No API calls. Reads existing JSON only."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analysis_shared import title_key
from analyze_classification import derive_moral_present, dedup_classification_results


def per_model_analysis(classification_path):
    data = json.load(open(classification_path))
    results = data.get("results", [])
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r.get("baseline_agent", "unknown")].append(r)

    raw_rows = []
    for agent, items in sorted(by_agent.items()):
        counts = Counter()
        moral_present = 0
        for r in items:
            scoring = r.get("scoring", {})
            counts[scoring.get("classification", "Invalid")] += 1
            mp = scoring.get("moral_present")
            if mp is None:
                mp = derive_moral_present(scoring.get("answers", {}))
            if mp:
                moral_present += 1
        total = sum(counts.values())
        raw_rows.append({
            "agent": agent, "n_raw": total,
            "PrimMoral": round(100 * counts.get("Moral", 0) / total, 1) if total else 0.0,
            "MoralPres": round(100 * moral_present / total, 1) if total else 0.0,
            "Proc": round(100 * counts.get("Procedural", 0) / total, 1) if total else 0.0,
            "Tech": round(100 * counts.get("Technical", 0) / total, 1) if total else 0.0,
            "None": round(100 * counts.get("None", 0) / total, 1) if total else 0.0,
        })

    dedup_rows = []
    for agent, items in sorted(by_agent.items()):
        groups = dedup_classification_results(items)
        counts = Counter(g["classification"] for g in groups)
        moral_present = sum(1 for g in groups if g["moral_present"])
        total = len(groups)
        dedup_rows.append({
            "agent": agent, "n_dedup": total,
            "PrimMoral": round(100 * counts.get("Moral", 0) / total, 1) if total else 0.0,
            "MoralPres": round(100 * moral_present / total, 1) if total else 0.0,
            "Proc": round(100 * counts.get("Procedural", 0) / total, 1) if total else 0.0,
            "Tech": round(100 * counts.get("Technical", 0) / total, 1) if total else 0.0,
            "None": round(100 * counts.get("None", 0) / total, 1) if total else 0.0,
        })

    return raw_rows, dedup_rows


def print_table(title, rows, n_col_label):
    print()
    print(title)
    header = f"{'Agent':<35} {n_col_label:>6} {'PrimMor':>8} {'MorPres':>8} {'Proc':>7} {'Tech':>7} {'None':>7}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        label = r["agent"][:33]
        n = r.get("n_raw") or r.get("n_dedup") or 0
        print(f"{label:<35} {n:>6} {r['PrimMoral']:>8.1f} {r['MoralPres']:>8.1f} "
              f"{r['Proc']:>7.1f} {r['Tech']:>7.1f} {r['None']:>7.1f}")
    if not rows:
        return
    avg = {k: round(sum(r[k] for r in rows) / len(rows), 1)
           for k in ("PrimMoral", "MoralPres", "Proc", "Tech", "None")}
    total_n = sum((r.get("n_raw") or r.get("n_dedup") or 0) for r in rows)
    print("-" * len(header))
    print(f"{'BALANCED AVG (equal weight per model)':<35} {total_n:>6} "
          f"{avg['PrimMoral']:>8.1f} {avg['MoralPres']:>8.1f} "
          f"{avg['Proc']:>7.1f} {avg['Tech']:>7.1f} {avg['None']:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", required=True)
    args = ap.parse_args()
    raw_rows, dedup_rows = per_model_analysis(args.classification)
    if not raw_rows:
        print("No results found.")
        sys.exit(1)
    print(f"\nSource: {args.classification}")
    print_table("PER-MODEL DISTRIBUTION (raw norms)", raw_rows, "Nraw")
    print_table("PER-MODEL DISTRIBUTION (within-model dedup)", dedup_rows, "Ndup")
    print("\nPrimMor = Primary Moral (strict). MorPres = Moral Present (M1-M4 all Y).")
    print("Balanced avg = each model equally weighted regardless of verbosity.\n")


if __name__ == "__main__":
    main()
