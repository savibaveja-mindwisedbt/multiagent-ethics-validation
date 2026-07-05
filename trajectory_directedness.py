#!/usr/bin/env python3
"""Directedness trajectory across an iterative process.

Measures ED% (directedness) of the norms PROPOSED at each panel deliberation
round (r0..r4 plus the outcome round) and at each solo-iter pass (p1..p5), for
one scenario. The point is the panel-vs-solo-iter contrast: does directedness
fall round-over-round in the panel while staying flat across solo-iter passes?

Instrument reuse (no reimplementation):
  - classification (is_norm + de-fusion)  -> classify_norms.judge / derive / persist_class
  - de-fuse + score + ED count            -> classify_norms.recompute
  - directedness scorer                   -> score_v2.v2_classify (sonnet-4.6, temp 0)
This is the SAME instrument as the atomic pipeline. Scoring here is per-item
de-fused ED (no compound splitting), i.e. the pre-split --rescore stage. Numbers
are therefore comparable to the non-split --report cells, not to --use-splits.

Terminal-step identity (the built-in bug check):
  - the panel 'outcome' step is built with the same extractor classify_norms uses
    for the panel cell, so its text list is IDENTICAL to cell_texts(p,'panel').
  - the solo-iter 'p5' step uses the last pass; result['text'] is that pass.
  So under --dry (cache only, no API) the terminal steps are fully scored and
  reproduce the committed cells, while intermediate steps show PENDING until --run.

Usage:
  python3 trajectory_directedness.py --prompts H            # dry: counts, cost, cache-only ED
  python3 trajectory_directedness.py --prompts H --run      # classify + score misses (spends API)
"""
import sys, glob, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
from analysis_shared import parse_numbered_list, flatten_norms, extract_final_round_norms


def _dedup(seq):
    seen, out = set(), []
    for n in seq:
        n = (n or "").strip()
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out


def panel_steps(p):
    """Ordered [(label, [norm texts])] for rounds 0..4 then the outcome round.
    Intermediate rounds parse each deliberation turn; outcome uses the SAME
    extractor classify_norms.cell_texts uses for the panel cell."""
    files = sorted(glob.glob(
        f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_{C.TAG}*.json"))
    rounds = {r: [] for r in range(5)}
    outcome = []
    for f in files:
        d = json.load(open(f))
        for x in d.get("transcript", []):
            if x.get("is_outcome_round"):
                continue
            r = x.get("round_index")
            if r in rounds:
                rounds[r] += parse_numbered_list(x.get("text", ""))
        outcome += [z["norm"] for z in flatten_norms(extract_final_round_norms(d))]
    steps = [(f"r{r}", _dedup(rounds[r])) for r in range(5)]
    steps.append(("outcome", _dedup(outcome)))
    return steps


def iter_steps(p):
    """Ordered [(label, [norm texts])] for passes 1..5. Each pass parsed the same
    way cell_texts parses the final solo-iter text."""
    files = sorted(glob.glob(f"baselines_solo_iter/baseline_{p}_iter_run*.json"))
    passes = {i: [] for i in range(1, 6)}
    for f in files:
        for pp in json.load(open(f)).get("result", {}).get("passes", []):
            i = pp.get("pass")
            if i in passes:
                passes[i] += parse_numbered_list(pp.get("text", ""))
    return [(f"p{i}", _dedup(passes[i])) for i in range(1, 6)]


def ed_for(texts, class_cache, v2cache, rescore, model):
    """ED% for one step, computed by reusing recompute unchanged (monkeypatch
    cell_texts to hand it exactly this step's texts). No aggregation is
    reimplemented, so de-fuse / drop / score / count logic is identical."""
    orig = C.cell_texts
    C.cell_texts = lambda p, c, _t=texts: _t
    try:
        rows, _ = C.recompute(["_"], ["_"], class_cache, v2cache,
                              rescore=rescore, rescore_model=model, split_cache=None)
    finally:
        C.cell_texts = orig
    row = rows["_/_"]
    denom = row["atoms"] - row["pending"]
    ed = round(100 * row["ed_clean"] / denom) if denom else None
    return ed, row


def classify_missing(all_texts, class_cache, model):
    todo = [t for t in all_texts if t not in class_cache]
    if not todo:
        print("  classification: all step-norms already cached.")
        return
    m = C.preflight(model)
    print(f"  classification: {len(todo)} new judge calls with {m} (checkpoint every 10)...")
    new = []
    for i, t in enumerate(todo, 1):
        try:
            cl = C.judge(t, m)
        except C.JudgeParseError as e:
            C._log_parse_fail(t, e.raw, e)
            print(f"   [{i}/{len(todo)}] parse-fail, SKIPPED (logged)")
            continue
        except Exception as e:
            print(f"   [{i}/{len(todo)}] API error, saving and stopping: {e}")
            break
        class_cache[t] = cl
        new.append({"norm": t, "cells": ["trajectory"], "classification": cl})
        if len(new) % 10 == 0:
            C.persist_class(new); new = []; print(f"   [{i}/{len(todo)}] checkpoint saved")
    if new:
        C.persist_class(new)
    print("  classification pass complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="single scenario letter, e.g. H")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8",
                    help="classifier judge model (same default as classify_norms)")
    ap.add_argument("--run", action="store_true",
                    help="spend API: classify and score the missing step-norms. "
                         "default is a dry pass (cache only, no API).")
    args = ap.parse_args()
    p = args.prompts.strip()

    panel = panel_steps(p)
    solo = iter_steps(p)
    class_cache = C.load_class_cache()
    v2cache = C.V2.load_v2_cache()

    all_texts = _dedup([t for _, ts in (panel + solo) for t in ts])
    unclassified = [t for t in all_texts if t not in class_cache]
    print(f"scenario {p}: {len(all_texts)} distinct step-norms across all steps; "
          f"{len(unclassified)} need classification.")

    rescore_model = None
    if args.run:
        classify_missing(all_texts, class_cache, args.model)
        class_cache = C.load_class_cache()
        rescore_model = C.A_SHARED.ANALYSIS_MODEL   # sonnet-4.6, the v2 instrument

    print("\nDIRECTEDNESS TRAJECTORY  (ED% of de-fused norms; per-item, no split)")
    print("  step      kept  atoms  pend   ED%")
    for label, steps in [("PANEL", panel), ("SOLO-ITER", solo)]:
        print(f"  --- {label} ---")
        for name, texts in steps:
            ed, row = ed_for(texts, class_cache, v2cache,
                             rescore=args.run, model=rescore_model)
            eds = "-" if ed is None else f"{ed:3d}"
            note = "" if row["pending"] == 0 else "  <-PENDING, needs --run"
            print(f"  {name:8s}{row['kept']:6d}{row['atoms']:7d}{row['pending']:6d}  {eds}{note}")

    if not args.run:
        print("\n[dry] No API used. Terminal steps (outcome, p5) are scored from cache and")
        print("      equal the committed non-split cells; intermediate steps are PENDING")
        print("      until you run with --run. --run spends: 1 judge call per unclassified")
        print("      step-norm above, plus 1 score_v2 call per de-fused norm not yet cached.")


if __name__ == "__main__":
    main()
