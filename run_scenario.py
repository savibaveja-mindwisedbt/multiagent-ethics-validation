#!/usr/bin/env python3
"""Generate solo / solo-iter / panel runs for ONE scenario, same pipeline as the
A/E core data. Resumable: skips run indices that already have a valid file."""
import json, glob, re, argparse, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list
from prompts import SCENARIOS

N = 15
N_PASSES = 5
TAG = "CORE"
SOLO_DIR = Path("baselines_core")
ITER_DIR = Path("baselines_solo_iter")
TDIR = Path("transcripts")
MARKERS = ["[API error", "[empty response]", "[no visible answer"]
def bad(s): return any(m in (s or "") for m in MARKERS)

def valid_solo(f):
    try: t = json.load(open(f)).get("result", {}).get("text", "")
    except Exception: return False
    return (not bad(t)) and len(parse_numbered_list(t)) >= 1
def valid_iter(f):
    try: r = json.load(open(f)).get("result", {})
    except Exception: return False
    ps = r.get("passes") or []
    return bool(ps) and not any(bad(p.get("text","")) for p in ps) and len(parse_numbered_list(r.get("text","")))>=1
def valid_panel(f):
    try: d = json.load(open(f))
    except Exception: return False
    if not d.get("run_metadata",{}).get("complete",False): return False
    ts = d.get("transcript",[])
    return any(t.get("turn_type")=="outcome" for t in ts) and not any(bad(t.get("text","")) for t in ts)

def idxset(pattern, pat, checker):
    s=set()
    for f in glob.glob(pattern):
        m=re.search(pat, f.split("/")[-1])
        if m and checker(f): s.add(int(m.group(1)))
    return s

def gen_solo(api_key, p, i):
    from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
    SOLO_DIR.mkdir(parents=True, exist_ok=True)
    bp = SCENARIOS[p]["baseline_prompt"].rstrip() + SCENARIOS[p].get("list_instruction", "\n\nProvide your norms as a numbered list of between three and seven norms.")
    sysm = "You are answering a question about how a group of AI agents should make a collective decision. Give your considered answer."
    resp = call_openrouter(api_key=api_key, model=MODEL,
        messages=[{"role":"system","content":sysm},{"role":"user","content":bp}],
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (SOLO_DIR / f"baseline_{p}_run{i}_{st}.json").write_text(json.dumps({"result":{
        "scenario_key":p,"run_index":i,"model":MODEL,"temperature":TEMPERATURE,
        "text":extract_text(resp),"usage":extract_usage(resp),
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}}, indent=2))

def gen_iter(api_key, p, i):
    from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    sysm = "You are answering a question about how a group of AI agents should make a collective decision. Give your considered answer."
    initial = SCENARIOS[p]["baseline_prompt"].rstrip() + SCENARIOS[p].get("list_instruction", "\n\nProvide your norms as a numbered list of between three and seven norms.")
    revise = ("Review your current list of norms above. If you can make it better, revise it. If not, keep it as is. "
              "Give your current best version as a numbered list of between three and seven norms, and nothing else.")
    msgs=[{"role":"system","content":sysm},{"role":"user","content":initial}]; passes=[]
    for k in range(N_PASSES):
        resp=call_openrouter(api_key=api_key, model=MODEL, messages=msgs, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        text=extract_text(resp); passes.append({"pass":k+1,"text":text,"usage":extract_usage(resp)})
        msgs.append({"role":"assistant","content":text})
        if k<N_PASSES-1: msgs.append({"role":"user","content":revise})
    st=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ITER_DIR / f"baseline_{p}_iter_run{i}_{st}.json").write_text(json.dumps({"result":{
        "scenario_key":p,"run_index":i,"model":MODEL,"n_passes":N_PASSES,"passes":passes,
        "text":passes[-1]["text"],"timestamp_utc":datetime.now(timezone.utc).isoformat()}}, indent=2))

def gen_panel(api_key, p, i):
    from agents import build_panel
    from orchestrator import run_deliberation
    st=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=TDIR / f"deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}{i}_{st}.json"
    run_deliberation(api_key=api_key, agents=build_panel(), scenario_key=p, normgen=True,
                     rounds=5, out_path=out, verbose=True, enable_ready_check=False)

ARMS = [
    ("solo", SOLO_DIR, f"baseline_{{p}}_run*.json", r"_run(\d+)_", valid_solo, gen_solo),
    ("solo-iter", ITER_DIR, f"baseline_{{p}}_iter_run*.json", r"_iter_run(\d+)_", valid_iter, gen_iter),
    ("panel", TDIR, f"deliberation_{{p}}_normgen_samemodel_rotleadoff_{TAG}*.json", rf"{TAG}(\d+)_", valid_panel, gen_panel),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prompt", default="G")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--n", type=int, default=N)
    args=ap.parse_args()
    p=args.prompt
    if p not in SCENARIOS:
        print(f"unknown scenario {p}; known: {list(SCENARIOS)}"); return
    plan={}
    for name, d, pat, rx, chk, _ in ARMS:
        valid = idxset(f"{d}/{pat.format(p=p)}", rx, chk)
        missing=[i for i in range(args.n) if i not in valid]
        plan[name]=(sorted(valid), missing)
        print(f"{p}/{name:9} valid={len(valid):3} missing={len(missing):3} -> {missing}")
    total=sum(len(m) for _,m in plan.values())
    print(f"total runs to generate: {total}")
    if not args.generate:
        print("\nDRY. add --generate to create the missing runs (spends)."); return
    from keychain import get_openrouter_key
    key=get_openrouter_key()
    genmap={n:g for n,_,_,_,_,g in ARMS}
    for name,(valid,missing) in plan.items():
        for i in missing:
            print(f"[gen] {p}/{name} run {i}")
            genmap[name](key, p, i)
    print("\ndone. score with: python3 score_v2.py --score --prompts G")

if __name__=="__main__":
    main()
