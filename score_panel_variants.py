#!/usr/bin/env python3
"""Score ALL variants of each panel norm, not just the representative.

Motivation: flatten_norms() deduplicates panel outcome norms by title and keeps
only the first occurrence as the representative. score_transcript() then scores
only that representative. If a deduped-away variant names a moral concept that
the representative omits, the panel's explicit-moral rate is biased downward by
representative selection.

This script removes that bias. For each title-group it scores every variant and
applies an "any-variant" rule: the norm counts as explicit if ANY variant scores
explicit, and likewise for implicit / procedural / technical. This yields the
upper-bound panel rate.

Already-scored representatives hit the cache, so only the extra variants cost
API calls (roughly 30-40 norms across A and B).

Usage:
    python3 score_panel_variants.py --transcript transcripts/deliberation_B_....json
    python3 score_panel_variants.py --all          # runs A, B, D consensus panels

Writes analysis/classification/panel_variants_<prompt>_<timestamp>.json and prints
a before/after comparison table.
"""
import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis_shared import extract_final_round_norms, flatten_norms
from analyze_classification import (
    classify_with_cache,
    load_scored_cache,
    derive_explicit_moral_label,
    derive_implicit_moral_label,
    derive_procedural_label,
    derive_technical_label,
)


def score_variants_for_transcript(transcript_path):
    """Score every variant of every title-group in a panel transcript."""
    transcript = json.load(open(transcript_path))
    per_agent = extract_final_round_norms(transcript)
    groups = flatten_norms(per_agent)  # each has norm (representative), variants, source_agents

    if not groups:
        print(f"No outcome norms found in {transcript_path}")
        return None

    cache = load_scored_cache()
    hits = misses = 0
    results = []

    for gi, g in enumerate(groups, 1):
        rep = g["norm"].strip()
        variants = g.get("variants", [rep])
        if rep not in variants:
            variants = [rep] + variants

        variant_scores = []
        for v in variants:
            scoring, hit = classify_with_cache(v, cache)
            if hit:
                hits += 1
            else:
                misses += 1
            ans = scoring.get("answers", {}) if isinstance(scoring, dict) else {}
            variant_scores.append({
                "variant": v,
                "explicit": bool(derive_explicit_moral_label(ans)),
                "implicit": bool(derive_implicit_moral_label(ans)),
                "procedural": bool(derive_procedural_label(ans)),
                "technical": bool(derive_technical_label(ans)),
            })

        # Representative-only labels (what the old pipeline reported)
        rep_score = variant_scores[0]
        # Any-variant labels (the bias-corrected upper bound)
        any_expl = any(v["explicit"] for v in variant_scores)
        any_impl = any(v["implicit"] for v in variant_scores)
        any_proc = any(v["procedural"] for v in variant_scores)
        any_tech = any(v["technical"] for v in variant_scores)

        results.append({
            "group_index": gi,
            "representative": rep,
            "n_variants": len(variants),
            "n_source_agents": len(g.get("source_agents", [])),
            "rep_explicit": rep_score["explicit"],
            "any_explicit": any_expl,
            "rep_implicit": rep_score["implicit"],
            "any_implicit": any_impl,
            "rep_procedural": rep_score["procedural"],
            "any_procedural": any_proc,
            "rep_technical": rep_score["technical"],
            "any_technical": any_tech,
            "flipped_explicit": (not rep_score["explicit"]) and any_expl,
            "variant_scores": variant_scores,
        })

    print(f"Cache: {hits} hits, {misses} misses")
    return {
        "source": str(transcript_path),
        "n_groups": len(results),
        "results": results,
    }


def summarize(report, label):
    n = report["n_groups"]
    rep_e = sum(1 for r in report["results"] if r["rep_explicit"])
    any_e = sum(1 for r in report["results"] if r["any_explicit"])
    flipped = [r for r in report["results"] if r["flipped_explicit"]]
    print(f"\n{label}: {n} norms")
    print(f"  Explicit (representative only): {rep_e}/{n} = {100*rep_e/n:.1f}%")
    print(f"  Explicit (any variant):         {any_e}/{n} = {100*any_e/n:.1f}%")
    if flipped:
        print(f"  Norms that flip to explicit when variants counted: {len(flipped)}")
        for r in flipped:
            print(f"    - {r['representative'][:60]} (variants={r['n_variants']})")
    else:
        print(f"  No norms flip. Representative selection was NOT biasing the rate.")
    return n, rep_e, any_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--all", action="store_true", help="Run A, B, D consensus panels")
    args = ap.parse_args()

    targets = []
    if args.all:
        for pid, pat in [("A", "A_normgen_mixed_rotleadoff_2"),
                         ("B", "B_normgen_mixed_rotleadoff_2"),
                         ("D", "D_normgen_mixed_rotleadoff_2")]:
            files = [p for p in sorted(glob.glob(f"transcripts/deliberation_{pat}*.json"))
                     if 'noconsensus' not in p]
            if files:
                targets.append((pid, files[-1]))
    elif args.transcript:
        targets.append(("?", args.transcript))
    else:
        ap.error("provide --transcript PATH or --all")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows = []
    for pid, tpath in targets:
        print("=" * 70)
        print(f"Prompt {pid}: {tpath}")
        report = score_variants_for_transcript(tpath)
        if not report:
            continue
        out_path = f"analysis/classification/panel_variants_{pid}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        n, rep_e, any_e = summarize(report, f"Prompt {pid}")
        summary_rows.append((pid, n, rep_e, any_e))
        print(f"  Wrote {out_path}")

    if summary_rows:
        print("\n" + "=" * 70)
        print("SUMMARY: panel explicit rate, representative-only vs any-variant")
        print(f"  {'Prompt':<8} {'N':>4} {'RepOnly':>9} {'AnyVariant':>12}")
        for pid, n, rep_e, any_e in summary_rows:
            print(f"  {pid:<8} {n:>4} {100*rep_e/n:>8.1f}% {100*any_e/n:>11.1f}%")


if __name__ == "__main__":
    main()
