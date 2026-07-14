#!/usr/bin/env python3
"""
Resume an interrupted core run (e.g. after OpenRouter credits ran out).

What it does, in order:
  1. DIAGNOSE every core file and classify it VALID / BAD / and find MISSING indices.
     - solo / solo-iter: a failed call raised and wrote nothing, so these are only
       VALID or absent. We still scan for empty/error markers to be safe.
     - panel: a failed turn was caught and written as "[API error during this turn: ...]"
       then the file was marked complete. So an outage panel looks complete but is
       poisoned. We flag complete==False (interrupted) and any "[API error" turn.
  2. --clean removes only the BAD files, so their indices become MISSING.
  3. --generate regenerates only MISSING indices (live, OpenRouter). Existing VALID
     files are skipped, so good runs are never redone and never duplicated.
  4. cache check: reports poisoned scoring entries (no "answers" / error) and, for
     the core_rescore files this project writes, can drop them with --clean.

Re-scoring after regeneration is just: python3 score_core.py --tag CORE --prompts AE --score
(the scorer is cache-aware and re-attempts only what is missing).

NO API calls happen without --generate. Diagnose and --clean are local-only.
"""
import json, glob, re, argparse, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list

PROMPTS = ["A", "E"]
N = 15
N_PASSES = 5
TAG = "CORE"
SOLO_DIR = Path("baselines_core")
SOLO_ITER_DIR = Path("baselines_solo_iter")
TDIR = Path("transcripts")
ERROR_MARKERS = ["[API error", "[empty response]", "[no visible answer"]

def has_marker(s):
    return any(m in (s or "") for m in ERROR_MARKERS)

# ---- validity checks (return (ok, reason)) ----
def check_solo(path):
    d = json.load(open(path)); r = d.get("result", d)
    t = r.get("text", "")
    if has_marker(t):
        return False, "error/empty marker in text"
    if len(parse_numbered_list(t)) < 1:
        return False, "no parseable norms"
    return True, ""

def check_solo_iter(path):
    d = json.load(open(path)); r = d.get("result", d)
    passes = r.get("passes")
    if not passes:
        return False, "no passes"
    if any(has_marker(p.get("text", "")) for p in passes):
        return False, "error/empty marker in a pass"
    if len(parse_numbered_list(r.get("text", ""))) < 1:
        return False, "final pass has no parseable norms"
    return True, ""

def check_panel(path):
    d = json.load(open(path))
    if not d.get("run_metadata", {}).get("complete", False):
        return False, "incomplete (complete=False)"
    turns = d.get("transcript", [])
    if not any(t.get("turn_type") == "outcome" for t in turns):
        return False, "no outcome turn"
    if any(has_marker(t.get("text", "")) for t in turns):
        return False, "[API error] turn present"
    return True, ""

# ---- index extraction ----
def idx(path, pat):
    m = re.search(pat, path.split("/")[-1])
    return int(m.group(1)) if m else None

def files_for(prompt, kind):
    if kind == "solo":
        g = glob.glob(f"{SOLO_DIR}/baseline_{prompt}_run*.json")
        return [(f, idx(f, r"_run(\d+)_")) for f in g]
    if kind == "solo-iter":
        g = glob.glob(f"{SOLO_ITER_DIR}/baseline_{prompt}_iter_run*.json")
        return [(f, idx(f, r"_iter_run(\d+)_")) for f in g]
    g = glob.glob(f"transcripts/deliberation_{prompt}_normgen_samemodel_rotleadoff_{TAG}*.json")
    return [(f, idx(f, rf"{TAG}(\d+)_")) for f in g]

CHECK = {"solo": check_solo, "solo-iter": check_solo_iter, "panel": check_panel}

def diagnose():
    report = {}
    for p in PROMPTS:
        for kind in ("solo", "solo-iter", "panel"):
            valid_idx, bad = set(), []
            for f, i in files_for(p, kind):
                try:
                    ok, reason = CHECK[kind](f)
                except Exception as e:
                    ok, reason = False, f"unreadable: {e}"
                if ok and i is not None:
                    valid_idx.add(i)
                else:
                    bad.append((f, reason))
            missing = [i for i in range(N) if i not in valid_idx]
            report[(p, kind)] = {"valid": sorted(valid_idx), "bad": bad, "missing": missing}
    return report

def cache_poison():
    bad = []
    for f in glob.glob("analysis/classification/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for res in d.get("results", []):
            sc = res.get("scoring", {})
            if not isinstance(sc, dict) or "answers" not in sc or sc.get("error"):
                bad.append((f, res.get("norm", "")[:60]))
    return bad

def print_report(rep):
    print(f"{'cell':22} {'valid':>5} {'bad':>4} {'missing':>8}")
    print("-" * 44)
    total_bad = total_missing = 0
    for (p, kind), r in rep.items():
        total_bad += len(r["bad"]); total_missing += len(r["missing"])
        print(f"{p+'/'+kind:22} {len(r['valid']):5} {len(r['bad']):4} {len(r['missing']):8}  missing idx={r['missing']}")
        for f, reason in r["bad"]:
            print(f"        BAD: {f.split('/')[-1]}  <- {reason}")
    print(f"\ntotals: {total_bad} bad files, {total_missing} missing runs")
    return total_bad, total_missing

# ---- regeneration (live) ----
def gen_solo(api_key, p, i):
    from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
    from prompts import SCENARIOS
    SOLO_DIR.mkdir(parents=True, exist_ok=True)
    bp = SCENARIOS[p]["baseline_prompt"].rstrip() + SCENARIOS[p].get("list_instruction", "\n\nProvide your norms as a numbered list of between three and seven norms.")
    sysm = ("You are answering a question about how a group of AI agents should make a collective decision. Give your considered answer.")
    resp = call_openrouter(api_key=api_key, model=MODEL,
                           messages=[{"role": "system", "content": sysm}, {"role": "user", "content": bp}],
                           temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SOLO_DIR / f"baseline_{p}_run{i}_{st}.json"
    out.write_text(json.dumps({"result": {"scenario_key": p, "run_index": i, "model": MODEL,
        "temperature": TEMPERATURE, "text": extract_text(resp), "usage": extract_usage(resp),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()}}, indent=2))
    return out

def gen_solo_iter(api_key, p, i):
    from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
    from prompts import SCENARIOS
    SOLO_ITER_DIR.mkdir(parents=True, exist_ok=True)
    sysm = ("You are answering a question about how a group of AI agents should make a collective decision. Give your considered answer.")
    initial = SCENARIOS[p]["baseline_prompt"].rstrip() + SCENARIOS[p].get("list_instruction", "\n\nProvide your norms as a numbered list of between three and seven norms.")
    revise = ("Review your current list of norms above. If you can make it better, revise it. "
              "If not, keep it as is. Give your current best version as a numbered list of "
              "between three and seven norms, and nothing else.")
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": initial}]
    passes = []
    for k in range(N_PASSES):
        resp = call_openrouter(api_key=api_key, model=MODEL, messages=msgs, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        text = extract_text(resp)
        passes.append({"pass": k + 1, "text": text, "usage": extract_usage(resp)})
        msgs.append({"role": "assistant", "content": text})
        if k < N_PASSES - 1:
            msgs.append({"role": "user", "content": revise})
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SOLO_ITER_DIR / f"baseline_{p}_iter_run{i}_{st}.json"
    out.write_text(json.dumps({"result": {"scenario_key": p, "run_index": i, "model": MODEL,
        "n_passes": N_PASSES, "passes": passes, "text": passes[-1]["text"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat()}}, indent=2))
    return out

def gen_panel(api_key, p, i):
    from agents import build_panel
    from orchestrator import run_deliberation
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = TDIR / f"deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}{i}_{st}.json"
    run_deliberation(api_key=api_key, agents=build_panel(), scenario_key=p, normgen=True,
                     rounds=5, out_path=out, verbose=True, enable_ready_check=False)
    return out

GEN = {"solo": gen_solo, "solo-iter": gen_solo_iter, "panel": gen_panel}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="delete BAD files and poisoned core cache entries")
    ap.add_argument("--generate", action="store_true", help="regenerate MISSING indices (live API)")
    args = ap.parse_args()

    rep = diagnose()
    tb, tm = print_report(rep)
    poison = cache_poison()
    if poison:
        print(f"\ncache: {len(poison)} poisoned scoring entries (no answers / error). Sample:")
        for f, n in poison[:8]:
            print(f"   {f.split('/')[-1]}: {n}")

    if args.clean:
        removed = 0
        for (p, kind), r in rep.items():
            for f, _ in r["bad"]:
                Path(f).unlink(); removed += 1
        for f in glob.glob("analysis/classification/classification_core_*.json"):
            d = json.load(open(f))
            good = [r for r in d.get("results", []) if isinstance(r.get("scoring"), dict)
                    and "answers" in r["scoring"] and not r["scoring"].get("error")]
            if len(good) != len(d.get("results", [])):
                d["results"] = good; Path(f).write_text(json.dumps(d, indent=2))
        print(f"\n[clean] removed {removed} bad files. Re-run without --clean to see the updated plan.")
        return

    if args.generate:
        from keychain import get_openrouter_key
        api_key = get_openrouter_key()
        for (p, kind), r in rep.items():
            for i in r["missing"]:
                print(f"[generate] {p}/{kind} run {i}")
                GEN[kind](api_key, p, i)
        print("\n[generate] done. Now: python3 score_core.py --tag CORE --prompts AE --score")
        return

    print("\nDry diagnosis only. Next: --clean to remove bad files, then --generate to fill gaps.")

if __name__ == "__main__":
    main()
