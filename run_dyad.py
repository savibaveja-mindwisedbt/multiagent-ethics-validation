#!/usr/bin/env python3
"""Dyad experiment: a 2-agent panel that takes 3 turns each per round.

Motivation. The 6-agent panel de-directs norms relative to a solo iterating agent.
A 2-agent dyad discriminates between our two candidate mechanisms:
  - rejectability-under-stakes predicts the dyad already de-directs (near the panel),
    because any bearer-naming norm can be refused as soon as adoption needs >1 agent;
  - diffusion / many-hands (and Tomasello's dyad-is-directed view) predicts the dyad
    stays directed (near the solo agent), because dilution scales with group size.

Comparability. Each agent speaks 3 times per round, so a round has 2x3 = 6 turns,
identical to the 6-agent panel's 6 turns/round, and 5 rounds give 30 deliberation
turns exactly as the panel. Same model (claude-sonnet-4.6, the same-model panel model),
same rounds, same scenario prompt, same outcome instruction. The ONLY thing that
differs from the CORE panel is the number of distinct agents, 2 vs 6.

Files are written under a distinct tag (DYAD3) so they never collide with the CORE
panel transcripts, and scoring reuses the classify_norms pipeline pointed at that tag,
so the instrument is identical to the six committed cells.

  python3 run_dyad.py --prompt H                 # dry: plan + rough cost, no spend
  python3 run_dyad.py --prompt H --generate       # generate missing dyad runs (spends, Sonnet)
  python3 run_dyad.py --prompt H --score          # classify+score dyad norms (spends, Opus+Sonnet)
"""
import json, glob, re, argparse, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SCENARIOS

TAG = "DYAD3"                 # distinct from CORE; do not change without updating scoring
TURNS_PER_AGENT = 3          # 2 agents x 3 = 6 turns/round == CORE panel's 6/round
N_AGENTS = 2
ROUNDS = 5
TDIR = Path("transcripts")
MARKERS = ["[API error", "[empty response]", "[no visible answer"]
def bad(s): return any(m in (s or "") for m in MARKERS)


def valid_dyad(f):
    """Same validity rule run_scenario uses for panels: complete, has an outcome
    turn, and no API-error markers anywhere in the transcript."""
    try:
        d = json.load(open(f))
    except Exception:
        return False
    if not d.get("run_metadata", {}).get("complete", False):
        return False
    ts = d.get("transcript", [])
    return any(t.get("turn_type") == "outcome" for t in ts) and not any(bad(t.get("text", "")) for t in ts)


def existing_valid_indices(p):
    s = set()
    for f in glob.glob(f"{TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json"):
        m = re.search(rf"{TAG}(\d+)_", f.split("/")[-1])
        if m and valid_dyad(f):
            s.add(int(m.group(1)))
    return s


def gen_dyad(api_key, p, i):
    from agents import build_panel
    from orchestrator import run_deliberation
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = TDIR / f"deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}{i}_{st}.json"
    run_deliberation(api_key=api_key, agents=build_panel(N_AGENTS), scenario_key=p,
                     normgen=True, rounds=ROUNDS, out_path=out, verbose=True,
                     enable_ready_check=False, turns_per_agent_per_round=TURNS_PER_AGENT)


def score_dyad(p):
    """Classify (Opus) and score (Sonnet v2) the dyad's final norms, reusing the
    classify_norms pipeline pointed at the DYAD tag. Prints the dyad cell next to the
    committed solo-iter and CORE panel cells."""
    import classify_norms as C
    core_tag = C.TAG
    C.TAG = TAG                                  # cell_texts(panel) now globs the dyad files
    try:
        cc = C.load_class_cache(); vc = C.V2.load_v2_cache()
        texts = C.cell_texts(p, "panel")         # dyad final (outcome) norms, deduped
        todo = [t for t in texts if t not in cc]
        print(f"dyad {p}: {len(texts)} final norms, {len(todo)} need classification (Opus judge calls).")
        if todo:
            model = C.preflight("anthropic/claude-opus-4.8")
            print(f"classifying {len(todo)} dyad norms with {model} (checkpoint every 10)...")
            new = []
            for j, t in enumerate(todo, 1):
                try:
                    cl = C.judge(t, model)
                except C.JudgeParseError as e:
                    C._log_parse_fail(t, e.raw, e); print(f"  [{j}/{len(todo)}] parse-fail skipped"); continue
                except Exception as e:
                    print(f"  [{j}/{len(todo)}] API error, saving and stopping: {e}"); break
                cc[t] = cl; new.append({"norm": t, "cells": [f"dyad:{p}"], "classification": cl})
                if len(new) % 10 == 0:
                    C.persist_class(new); new = []; print(f"  [{j}/{len(todo)}] checkpoint saved")
            if new: C.persist_class(new)
            cc = C.load_class_cache()
        # score dyad + baselines (baselines are already cached; only the dyad spends)
        rows, _ = C.recompute([p], ["solo", "solo-iter", "panel"], cc, vc,
                              rescore=True, rescore_model=C.A_SHARED.ANALYSIS_MODEL, split_cache=None)
    finally:
        C.TAG = core_tag
    def edpct(row):
        d = row["atoms"] - row["pending"]
        return round(100 * row["ed_clean"] / d) if d else None
    print("\nDYAD RESULT (per-item de-fused ED; dyad shown in place of 'panel'):")
    for cell, label in [("solo", "solo        "), ("solo-iter", "solo-iter   "), ("panel", "DYAD (2 agents)")]:
        r = rows[f"{p}/{cell}"]
        print(f"  {label}: kept={r['kept']:3d} ED%={edpct(r)}  pend={r['pending']}")
    print("\n  for reference, committed CORE 6-agent panel (non-split): H=59, J=57")
    print("  solo-iter committed: H=95, J=52")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="H")
    ap.add_argument("--n", type=int, default=10, help="number of dyad runs")
    ap.add_argument("--generate", action="store_true", help="create missing dyad runs (spends Sonnet)")
    ap.add_argument("--score", action="store_true", help="classify+score dyad norms (spends Opus+Sonnet)")
    args = ap.parse_args()
    p = args.prompt
    if p not in SCENARIOS:
        print(f"unknown scenario {p}; known: {list(SCENARIOS)}"); return

    valid = existing_valid_indices(p)
    missing = [i for i in range(args.n) if i not in valid]
    print(f"dyad {p} ({TAG}, {N_AGENTS} agents x {TURNS_PER_AGENT} turns/round, {ROUNDS} rounds): "
          f"valid={len(valid)} missing={len(missing)} -> {missing}")

    if args.generate:
        from keychain import get_openrouter_key
        key = get_openrouter_key()
        for i in missing:
            print(f"[gen] dyad {p} run {i}")
            gen_dyad(key, p, i)
        print("done generating. now score with: python3 run_dyad.py --prompt %s --score" % p)
        return

    if args.score:
        score_dyad(p)
        return

    # dry
    n_new = len(missing)
    gen_turns = n_new * (ROUNDS * N_AGENTS * TURNS_PER_AGENT + N_AGENTS)   # delib + outcome
    print("\n[dry] no spend. Plan:")
    print(f"  --generate: {n_new} runs x {ROUNDS*N_AGENTS*TURNS_PER_AGENT}+{N_AGENTS} turns "
          f"= ~{gen_turns} Sonnet generation calls (<=2500 tokens each).")
    print(f"  --score: 1 Opus judge call per distinct dyad final norm (unknown until generated; "
          f"the 6-panel produced ~6-12 final norms/run, so expect roughly {n_new*6}-{n_new*12} "
          f"for {n_new} runs), plus 1 Sonnet v2 score per de-fused norm.")
    print("  Opus classification must stay on Opus for comparability with the committed cells.")


if __name__ == "__main__":
    main()
