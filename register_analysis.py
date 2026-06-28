#!/usr/bin/env python3
"""
Register analysis for the directedness-erosion mechanism (no API / no rescoring).

(1) Validates a grammatical-person proxy for directedness against the rubric's E2.
(2) Measures grammatical register per cell: % second-personal (I-thou, names an
    addressee) and % first-person-plural / impersonal, on cleaned norm units.
(3) Round dynamics: register over deliberation rounds in the panel transcripts.

Tests the register-shift account: if collaboration (not iteration) depersonalizes
norm grammar, panels should drop second-personal form while solo-iter stays ~solo.
Uses norm_filter.is_structural to exclude outline fragments from panel cells.
"""
import json, glob, re, sys, importlib.util
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from analysis_shared import parse_numbered_list, extract_final_round_norms, flatten_norms
spec = importlib.util.spec_from_file_location("nf", HERE / "norm_filter.py")
nf = importlib.util.module_from_spec(spec); spec.loader.exec_module(nf)

SECONDP = [re.compile(p, re.I) for p in [r'\beach other\b', r'\bone another\b', r'\banother agent\b',
    r'\bother agents?\b', r'\bto the others?\b', r'\bowes?\b', r'\bowed to\b', r'\baccountable to\b',
    r'\banswerable to\b', r'\bentitled\b', r'\bto whom\b', r'\bfellow agents?\b', r'\byour?\b']]
FPP = [re.compile(p, re.I) for p in [r'\bwe\b', r'\bour\b', r'\bus\b', r'\bthe consortium\b',
    r'\bthe group\b', r'\bthe panel\b', r'\bcollectively\b']]
def has(ps, t): return any(p.search(t) for p in ps)
def second_personal(n): return has(SECONDP, n)

def load_e2():
    rows = []
    for f in glob.glob("analysis/classification_v2/*.json"):
        for r in json.load(open(f)).get("results", []):
            n = (r.get("norm") or "").strip(); a = r.get("scoring", {}).get("answers", {})
            if n and "E2" in a:
                rows.append((n, str(a["E2"].get("answer", "N")).strip().upper().startswith("Y")))
    return rows

def validate():
    rows = load_e2(); N = len(rows) or 1
    tp = sum(1 for n, e2 in rows if e2 and second_personal(n))
    fp = sum(1 for n, e2 in rows if (not e2) and second_personal(n))
    fn = sum(1 for n, e2 in rows if e2 and not second_personal(n))
    tn = N - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    print(f"(1) proxy vs rubric E2, N={N}: agreement={(tp+tn)/N:.0%}  "
          f"precision={prec:.0%}  recall={rec:.0%}  (E2-directed base rate={sum(e2 for _,e2 in rows)/N:.0%})")
    print("    reading: explicit second-personal grammar reliably implies directed (high precision),")
    print("    but most rubric-directed norms are not second-personal in form (low recall) -> grammar != meaning\n")

def cell_norms(p, c):
    if c == "solo": groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in glob.glob(f"baselines_core/baseline_{p}_run*.json")]
    elif c == "solo-iter": groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in glob.glob(f"baselines_solo_iter/baseline_{p}_iter_run*.json")]
    else: groups = [[x["norm"] for x in flatten_norms(extract_final_round_norms(json.load(open(tp))))] for tp in glob.glob(f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_CORE*.json")]
    seen, out = set(), []
    for g in groups:
        for n in g:
            n = n.strip()
            if n in seen: continue
            if c == "panel" and nf.is_structural(n): continue
            seen.add(n); out.append(n)
    return out

def register_by_cell(prompts):
    print("(2) grammatical register on cleaned norm units: %second-personal | %first-person-plural")
    print(f"    {'cell':14}{'n':>5}{'2nd-pers%':>11}{'1st-pl%':>9}")
    for p in prompts:
        for c in ["solo", "solo-iter", "panel"]:
            ns = cell_norms(p, c); N = len(ns) or 1
            sp = round(100 * sum(second_personal(n) for n in ns) / N)
            fp = round(100 * sum(has(FPP, n) for n in ns) / N)
            print(f"    {p+'/'+c:14}{len(ns):>5}{sp:>11}{fp:>9}")
        print()

def round_dynamics(p):
    print(f"(3) {p} panel round dynamics (raw turn text): %turns 2nd-personal | 1st-plural")
    sp = defaultdict(list); fp = defaultdict(list)
    for tp in glob.glob(f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_CORE*.json"):
        for t in json.load(open(tp)).get("transcript", []):
            r = t.get("round_index"); txt = t.get("text", "")
            if r is None or not txt: continue
            sp[r].append(1 if has(SECONDP, txt) else 0); fp[r].append(1 if has(FPP, txt) else 0)
    for r in sorted(sp):
        print(f"    round {r}: turns={len(sp[r]):>3}  2nd-pers={100*sum(sp[r])/len(sp[r]):>3.0f}%  1st-plural={100*sum(fp[r])/len(fp[r]):>3.0f}%")

if __name__ == "__main__":
    prompts = [c for c in (sys.argv[1] if len(sys.argv) > 1 else "AEGH") if c in "AEGH"]
    validate(); register_by_cell(prompts); print(); round_dynamics("H")
