#!/usr/bin/env python3
"""Quick validation: score 6 sample A norms (2 solo, 2 6xClaude, 2 Mixed)."""
import glob, json
from pathlib import Path
from score_principled_moral import score_norm_pm
from analysis_shared import (
    extract_final_round_norms, flatten_norms, parse_numbered_list,
)


def get_solo_samples(prompt, n=2):
    samples = []
    for bf in sorted(glob.glob(f"baselines_capped/baseline_{prompt}_agent_1_*.json")):
        bdata = json.load(open(bf))
        r = bdata.get("result", bdata)
        text = r.get("text", "")
        if not text:
            continue
        norms = parse_numbered_list(text)
        for norm in norms:
            t = norm.strip() if isinstance(norm, str) else (norm.get("text") or norm.get("title") or "").strip()
            if len(t) >= 50:
                samples.append((Path(bf).name, t))
                if len(samples) >= n:
                    return samples
    return samples


def get_panel_samples(prompt, pattern, n=2):
    samples = []
    for tp in sorted(glob.glob(pattern)):
        d = json.load(open(tp))
        for g in flatten_norms(extract_final_round_norms(d)):
            rep = g["norm"].strip()
            if len(rep) >= 50:
                samples.append((Path(tp).name, rep))
                if len(samples) >= n:
                    return samples
    return samples


def main():
    print("=== PM RUBRIC VALIDATION ON PROMPT A ===\n")
    solo = get_solo_samples("A", n=2)
    six = get_panel_samples("A", "transcripts/deliberation_A_normgen_samemodel_rotleadoff_*.json", n=2)
    mixed = get_panel_samples("A", "transcripts/deliberation_A_normgen_mixed_rotleadoff_*.json", n=2)
    print(f"Got: {len(solo)} solo, {len(six)} 6xClaude, {len(mixed)} mixed")

    if not solo:
        print("Still no solo. Aborting.")
        return

    samples = ([("Claude solo", s) for s in solo] +
               [("6xClaude panel", s) for s in six] +
               [("Mixed panel", s) for s in mixed])

    by_cell = {}
    for cell, (source, text) in samples:
        scoring, _ = score_norm_pm(text)
        pm_count = scoring.get("pm_count", 0)
        ans = scoring.get("answers", {})
        by_cell.setdefault(cell, []).append(pm_count)
        print(f"\n[{cell}] from {source}")
        print(f"  Norm: {text[:150]}...")
        for q, desc in [("PM1","primary-thesis"),("PM2","load-bearing"),("PM3","stand-alone"),("PM4","universality")]:
            a = ans.get(q,{})
            print(f"  {q} ({desc}): {a.get('answer','?')} -- {a.get('reasoning','')[:120]}")
        print(f"  ==> PM count: {pm_count}/4")

    print("\n" + "="*80)
    print("SUMMARY (avg PM count per cell, n=2 each):")
    for cell, counts in by_cell.items():
        avg = sum(counts)/len(counts) if counts else 0
        print(f"  {cell:<20}: avg = {avg:.1f}/4, individual = {counts}")


if __name__ == "__main__":
    main()
