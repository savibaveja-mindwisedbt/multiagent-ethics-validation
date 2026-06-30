#!/usr/bin/env python3
"""
Temperature-sensitivity probe for the solo directedness baseline.

QUESTION
The whole dataset was generated at temperature 0.0. Solo baselines therefore
collapse to near-identical outputs (H solo: 2-6 distinct of 15). The panels are
fine (15/15 distinct, via lead rotation), so the panel finding stands; the open
question is whether the SOLO directedness number (~97) is a property of the
model or an artifact of deterministic decoding. This probe answers ONE thing:

   When you raise temperature, does solo directedness (ED%) stay put, or move?

   - stays ~97 with low spread -> temperature only adds surface diversity; the
     temp-0 solo baseline is representative of modal behavior; the metric is not
     temperature-sensitive. Supports keeping the dataset (modal claim) or a cheap
     temperature robustness arm.
   - scatters across runs -> ED is temperature-sensitive; the temp-0 point
     estimate is not representative; the experiment needs full regeneration at a
     fixed nonzero temperature with proper sampling (all arms, for comparability).

ISOLATION
Writes ONLY to baselines_temp_probe/ with temperature-tagged filenames. It never
touches baselines_core/ or transcripts/, so the temp-0 dataset stays clean and
nothing here can pool into the main results.

METRIC
Reuses score_v2.v2_classify + typology directly, so ED here is the identical
instrument as the headline table. Same judge model, same rubric.

PHASES (separate, so you can inspect before paying to score)
  python3 temp_probe.py                         # report current state, NO API
  python3 temp_probe.py --generate --temp 1.0 --n 10   # SPENDS: 10 H-solo gens
  python3 temp_probe.py --inspect               # NO API: unique-output count + ED-free look
  python3 temp_probe.py --score --temp 1.0      # SPENDS: score the probe norms
  python3 temp_probe.py --selftest              # NO API: aggregation math check

Defaults: scenario H, temp 1.0, n 10. Matches the capped solo protocol exactly
(same system message and the "between three and seven norms" instruction).
"""
import json, glob, sys, argparse, time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list
from score_v2 import v2_classify, typology   # identical validated directedness metric

PROBE_DIR = Path("baselines_temp_probe")
SYSTEM_MSG = ("You are answering a question about how a group of AI agents "
              "should make a collective decision. Give your considered answer.")
CAP_INSTRUCTION = "\n\nProvide your norms as a numbered list of between three and seven norms."

def temp_tag(t):
    return f"temp{str(t).replace('.', '')}"   # 1.0 -> temp10, 0.7 -> temp07

def gen_files(scenario, t):
    return sorted(glob.glob(str(PROBE_DIR / f"baseline_{scenario}_{temp_tag(t)}_run*.json")))

def score_cache_path(scenario, t):
    return PROBE_DIR / f"scores_{scenario}_{temp_tag(t)}.json"

def load_scores(scenario, t):
    p = score_cache_path(scenario, t)
    if p.exists():
        try: return json.load(open(p))
        except Exception: return {}
    return {}

def save_scores(scenario, t, d):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    score_cache_path(scenario, t).write_text(json.dumps(d, indent=2))

# ---------- generate (spends) ----------
def generate(scenario, t, n):
    from keychain import get_openrouter_key
    from agents import MODEL, MAX_TOKENS, call_openrouter, extract_text, extract_usage
    from prompts import SCENARIOS
    api_key = get_openrouter_key()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    prompt = SCENARIOS[scenario]["baseline_prompt"].rstrip() + CAP_INSTRUCTION
    print(f"generating {n} runs: scenario={scenario} temp={t} model={MODEL}")
    print(f"-> {PROBE_DIR}/baseline_{scenario}_{temp_tag(t)}_run*.json  (NOT pooled with baselines_core)")
    written = []
    for i in range(n):
        t0 = time.time()
        try:
            resp = call_openrouter(api_key=api_key, model=MODEL,
                                   messages=[{"role": "system", "content": SYSTEM_MSG},
                                             {"role": "user", "content": prompt}],
                                   temperature=t, max_tokens=MAX_TOKENS)
            text = extract_text(resp); usage = extract_usage(resp)
        except Exception as e:
            print(f"  [{i+1}/{n}] ERROR: {e}; stopping."); break
        rec = {"result": {"scenario_key": scenario, "run_index": i, "model": MODEL,
                          "temperature": t, "max_tokens": MAX_TOKENS, "text": text,
                          "usage": usage, "elapsed_seconds": round(time.time()-t0, 2),
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
        st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = PROBE_DIR / f"baseline_{scenario}_{temp_tag(t)}_run{i}_{st}.json"
        out.write_text(json.dumps(rec, indent=2))
        written.append(out); print(f"  [{i+1}/{n}] {out.name}")
    print(f"done: {len(written)} files. Inspect with --inspect before scoring.")

# ---------- inspect (no API) ----------
def inspect(scenario, t):
    files = gen_files(scenario, t)
    if not files:
        print(f"no probe files for scenario={scenario} temp={t}. Run --generate first."); return
    texts, norm_counts = [], []
    for f in files:
        txt = json.load(open(f)).get("result", {}).get("text", "")
        texts.append(txt.strip())
        norm_counts.append(len(parse_numbered_list(txt)))
    uniq = len(set(texts))
    print(f"\n=== inspect: scenario={scenario} temp={t} ===")
    print(f"  files: {len(files)}")
    print(f"  DISTINCT outputs: {uniq} of {len(files)}   "
          f"({'temperature broke determinism' if uniq > 1 else 'STILL IDENTICAL - temp may not have taken effect'})")
    print(f"  norms per run: {norm_counts}  (mean {sum(norm_counts)/len(norm_counts):.1f})")
    print("  -> if distinct>1 you can score; if still identical, check that temp was passed.")

# ---------- score (spends) + report ----------
def score_and_report(scenario, t):
    files = gen_files(scenario, t)
    if not files:
        print(f"no probe files for scenario={scenario} temp={t}. Run --generate first."); return
    scores = load_scores(scenario, t)
    per_run = []
    n_scored_now = 0
    for f in files:
        run_key = Path(f).stem
        txt = json.load(open(f)).get("result", {}).get("text", "")
        norms = [n.strip() for n in parse_numbered_list(txt) if n.strip()]
        directed = total = 0
        for norm in norms:
            key = f"{run_key}::{norm[:80]}"
            if key in scores:
                tp = scores[key]
            else:
                try:
                    ans = v2_classify(norm)
                except Exception as e:
                    print(f"  scoring error ({run_key}): {e}; saving progress, stopping.")
                    save_scores(scenario, t, scores)
                    return
                tp = ans.get("typology") or typology(ans.get("answers", ans))
                scores[key] = tp; n_scored_now += 1
                if n_scored_now % 10 == 0: save_scores(scenario, t, scores)
            total += 1
            if tp == "ED": directed += 1
        ed = round(100 * directed / total) if total else None
        per_run.append((run_key, total, ed))
    save_scores(scenario, t, scores)

    eds = [ed for _, _, ed in per_run if ed is not None]
    print(f"\n=== temperature probe result: scenario={scenario} temp={t} ===")
    print(f"{'run':42} {'n':>3} {'ED%':>5}")
    for rk, n, ed in per_run:
        print(f"{rk[:42]:42} {n:>3} {str(ed):>5}")
    if eds:
        mean = sum(eds)/len(eds)
        print(f"\n  runs scored: {len(eds)}   ED% mean={mean:.0f}  min={min(eds)}  max={max(eds)}  spread={max(eds)-min(eds)}")
        print(f"  temp-0 H solo baseline for reference: 97")
        print("\n  reading:")
        print("   - spread small AND mean near 97  -> ED is temperature-insensitive at solo;")
        print("       temp-0 baseline is representative. Modal claim generalizes; no full regen needed.")
        print("   - spread large OR mean far from 97 -> ED is temperature-sensitive;")
        print("       temp-0 point estimate not representative -> regenerate ALL arms at fixed nonzero temp.")

# ---------- selftest (no API) ----------
def selftest():
    print("SELFTEST (no API): aggregation + distinct-count logic on synthetic probe files.\n")
    import tempfile, shutil
    global PROBE_DIR, v2_classify, typology
    save_dir, save_cls, save_typ = PROBE_DIR, v2_classify, typology
    PROBE_DIR = Path(tempfile.mkdtemp())
    # Stub scoring so no API: a norm is ED if it contains 'honest' or 'owe', else IM.
    v2_classify = lambda norm: norm
    typology = lambda x: "ED" if ("honest" in x.lower() or "owe" in x.lower()) else "IM"
    # 3 runs, 2 distinct texts; norms all >=20 chars so the real parser keeps them.
    body0 = ("1. Agents must share reasoning honestly with each other.\n"
             "2. Each agent should listen to the others.\n"
             "3. The group documents its decisions clearly.")
    body2 = ("1. Agents owe each other a clear account of dissent.\n"
             "2. The consortium records its rationale transparently.")
    runs = [("baseline_H_temp10_run0_x.json", body0),
            ("baseline_H_temp10_run1_y.json", body0),    # dup of run0
            ("baseline_H_temp10_run2_z.json", body2)]
    for name, txt in runs:
        (PROBE_DIR / name).write_text(json.dumps({"result": {"text": txt}}))
    inspect("H", 1.0)              # expect DISTINCT outputs: 2 of 3
    score_and_report("H", 1.0)     # uses stubbed scorer
    print("\n  expected inspect: 2 of 3 distinct, norms per run [3, 3, 2].")
    print("  expected ED%: run0=33, run1=33, run2=50.")
    shutil.rmtree(PROBE_DIR, ignore_errors=True)
    PROBE_DIR, v2_classify, typology = save_dir, save_cls, save_typ
    print("SELFTEST done (verify expected vs printed match).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="generate probe runs (SPENDS)")
    ap.add_argument("--score", action="store_true", help="score probe norms and report (SPENDS)")
    ap.add_argument("--inspect", action="store_true", help="distinct-output count, NO API")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scenario", default="H")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    if args.selftest: selftest(); return
    if args.generate: generate(args.scenario, args.temp, args.n); return
    if args.inspect: inspect(args.scenario, args.temp); return
    if args.score: score_and_report(args.scenario, args.temp); return
    # default: state of the probe dir, no API
    files = gen_files(args.scenario, args.temp)
    print(f"probe dir: {PROBE_DIR}/  | scenario={args.scenario} temp={args.temp}")
    print(f"generated runs found: {len(files)}")
    if files: inspect(args.scenario, args.temp)
    else: print("nothing yet. Start with: python3 temp_probe.py --generate --temp 1.0 --n 10")

if __name__ == "__main__":
    main()
