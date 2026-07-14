#!/usr/bin/env python3
"""
Core experiment generation: Claude Solo vs 6xClaude, one model, reasoning off,
matched protocol. SAVI RUNS THIS LOCALLY. It makes OpenRouter calls. The
assistant cannot run it.

Design it encodes:
  - One model (Claude), so the contrast is collaboration itself, not composition.
  - Reasoning off on every call (call_openrouter sends reasoning:{enabled:False}
    by default; build_panel() agents are default Claude with omit_reasoning=False,
    so the flag is sent and reasoning is off).
  - Matched protocol: solo baselines use capped=True, which appends the same
    "between three and seven norms" instruction the panel's outcome round uses,
    so solo and panel are scored on equal footing. This is the protocol that
    produced the 89 matched-confirm Partition Solo cell, not the older 32 set.
  - A fresh, single-tagged set (TAG below). Nothing is pooled with prior runs.
    The pooling of old passes is exactly what produced the Mixed-high artifact,
    so the core set is generated clean and selected by tag only.

You set three things: PROMPTS, N (10-15), and READY_CHECK.

READY_CHECK:
  False (recommended for the core) forces all five rounds. Removes two things:
        the solo-vs-panel readiness asymmetry, and the endogeneity where agents
        choose when to stop. With the gate on, a drifting run keeps going and a
        run that held framing stops early, so rounds-run is self-selected and the
        rounds/framing relationship is partly selection rather than causation.
        Off, rounds are exogenous and fixed, so round-0-vs-outcome and a designed
        rounds dose (1/3/5) are clean.
  True  reproduces the paper's described emergent-stopping protocol. Use only if
        you specifically want that protocol; it reintroduces the selection above.

NOTE the self_reflection variant (CHARACTERIZE/ASSESS/CORRECT in the ready-check)
is OFF here and stays off. run_deliberation defaults self_reflection=False and this
script does not set it. That heavy reflection is the Appendix H intervention, not
part of the core, and its transcripts are tagged and excluded elsewhere.

Cost note: solo runs are one call each and cheap. Panel runs are roughly
6 agents x 5 rounds plus ready-checks and the outcome round, so the panel cells
are where the spend is. Budget accordingly before scaling N.

After generation, score and recompute:
  python3 analyze_classification.py --baselines-dir baselines_core --prompt-id <P>
  python3 analyze_classification.py --transcript <each core transcript>
  python3 score_core.py --tag CORE --prompts <PROMPTS>
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

# ----- decisions you set -----
PROMPTS = ["A", "E"]      # A=Cooperation (purest peer cooperation), E=Conflicting
                          # (cooperation under genuine conflict). C dropped: it is a
                          # division-of-labor task, procedural by construction, so its
                          # erosion is confounded with the task rather than collaboration.
N = 15                    # runs per cell
READY_CHECK = False       # CHANGED to False. With ready-check ON the agents choose
                          # when to stop, which makes rounds-run endogenous: a drifting
                          # run keeps going, a run that held framing stops early, so the
                          # rounds/framing correlation is partly selection, not causation.
                          # OFF forces fixed five rounds, removes that selection and the
                          # solo-vs-panel readiness asymmetry, and gives a clean
                          # rounds-as-treatment base. Study rounds as a designed dose
                          # (1/3/5 fixed) rather than via emergent early-stopping.
                          # Set True only to reproduce the paper's emergent-stopping protocol.
TAG = "CORE"
SOLO_DIR = Path("baselines_core")
SOLO_ITER_DIR = Path("baselines_solo_iter")
TDIR = Path("transcripts")

# Solo-iteration control: one Claude, neutral self-revision passes, no other
# agents. Isolates iteration from collaboration. N_PASSES matches the panel's
# five deliberation rounds (the panel agent also does an outcome turn, so 6 is
# the strict match; 5 is the conservative match and the default).
SOLO_ITER = True
N_PASSES = 5
# The revision instruction MUST be framing-neutral or the control is invalid.
# It must not nudge toward moral or procedural language. Review this wording.
REVISE_INSTRUCTION = (
    "Review your current list of norms above. If you can make it better, revise it. "
    "If not, keep it as is. Give your current best version as a numbered list of "
    "between three and seven norms, and nothing else."
)
# -----------------------------

def run_solo_iteration(api_key, scenario_key, n_runs, n_passes, out_dir):
    """One Claude, n_passes neutral self-revision passes, no other agents.
    Writes one file per run with every pass plus the final list."""
    from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
    from prompts import SCENARIOS
    out_dir.mkdir(parents=True, exist_ok=True)
    system_msg = ("You are answering a question about how a group of AI agents "
                  "should make a collective decision. Give your considered answer.")
    initial = SCENARIOS[scenario_key]["baseline_prompt"].rstrip() + (
        "\n\nProvide your norms as a numbered list of between three and seven norms.")
    written = []
    for run_index in range(n_runs):
        print(f"  [solo-iter {scenario_key}] run {run_index+1}/{n_runs}")
        messages = [{"role": "system", "content": system_msg},
                    {"role": "user", "content": initial}]
        passes = []
        for k in range(n_passes):
            resp = call_openrouter(api_key=api_key, model=MODEL, messages=messages,
                                   temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            text = extract_text(resp)
            passes.append({"pass": k + 1, "text": text, "usage": extract_usage(resp)})
            messages.append({"role": "assistant", "content": text})
            if k < n_passes - 1:
                messages.append({"role": "user", "content": REVISE_INSTRUCTION})
        result = {"scenario_key": scenario_key, "run_index": run_index, "model": MODEL,
                  "temperature": TEMPERATURE, "n_passes": n_passes,
                  "passes": passes, "text": passes[-1]["text"],
                  "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        st = stamp()
        out_path = out_dir / f"baseline_{scenario_key}_iter_run{run_index}_{st}.json"
        out_path.write_text(json.dumps({"result": result}, indent=2))
        written.append(out_path)
    return written

def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def plan(prompts, n):
    rows = []
    for p in prompts:
        rows.append(("solo", p, n, f"{SOLO_DIR}/baseline_{p}_run<0..{n-1}>_<stamp>.json"))
        if SOLO_ITER:
            rows.append(("solo-iter", p, n,
                         f"{SOLO_ITER_DIR}/baseline_{p}_iter_run<0..{n-1}>_<stamp>.json  ({N_PASSES} passes)"))
        rows.append(("6xClaude", p, n,
                     f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}<0..{n-1}>_<stamp>.json"))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, make no API calls")
    ap.add_argument("--prompts", default="".join(PROMPTS))
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()
    prompts = list(args.prompts)
    n = args.n

    print(f"Core experiment | prompts={prompts} | N={n} per cell | ready_check={READY_CHECK} | tag={TAG}")
    print(f"solo -> {SOLO_DIR}/ (single Claude, reasoning off, capped/matched)")
    print(f"6xClaude -> transcripts/ (build_panel, reasoning off, rounds=5, rotleadoff)\n")
    for cond, p, k, pat in plan(prompts, n):
        print(f"  {cond:9} prompt {p}  x{k}  -> {pat}")

    if args.dry_run:
        print("\n[dry-run] no API calls made. Remove --dry-run to generate.")
        print("\nAfter generation, run:")
        for p in prompts:
            print(f"  python3 analyze_classification.py --baselines-dir {SOLO_DIR} --prompt-id {p}")
            if SOLO_ITER:
                print(f"  python3 analyze_classification.py --baselines-dir {SOLO_ITER_DIR} --prompt-id {p}")
        print(f"  python3 score_core.py --tag {TAG} --prompts {''.join(prompts)} --score   # fills per-round proposal misses")
        print("  git add -A && git commit -m 'core solo-vs-6x runs' && git push")
        return

    # ---- live generation (Savi, local) ----
    from keychain import get_openrouter_key
    from agents import build_panel
    from baselines import run_baselines
    from orchestrator import run_deliberation
    api_key = get_openrouter_key()
    SOLO_DIR.mkdir(exist_ok=True); TDIR.mkdir(exist_ok=True)

    for p in prompts:
        print(f"\n=== SOLO (single Claude, matched) prompt {p} x{n} ===")
        for path in run_baselines(api_key=api_key, scenario_key=p, runs_per_scenario=n,
                                  out_dir=SOLO_DIR, mixed_model=False, capped=True):
            print(f"  {path}")
        if SOLO_ITER:
            print(f"\n=== SOLO-ITER (single Claude, {N_PASSES} neutral passes) prompt {p} x{n} ===")
            for path in run_solo_iteration(api_key, p, n, N_PASSES, SOLO_ITER_DIR):
                print(f"  {path}")
        print(f"\n=== 6xClaude panels prompt {p} x{n} ===")
        for i in range(n):
            out = TDIR / f"deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}{i}_{stamp()}.json"
            run_deliberation(api_key=api_key, agents=build_panel(), scenario_key=p,
                             normgen=True, rounds=5, out_path=out, verbose=True,
                             enable_ready_check=READY_CHECK)
            print(f"  DONE {out}")

    print("\nNext: score baselines, solo-iter, and transcripts, then python3 score_core.py --score, then commit+push.")

if __name__ == "__main__":
    main()
