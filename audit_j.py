#!/usr/bin/env python3
"""
Audit parsing and scoring for one scenario (default J) against a comparison
scenario (default H), to check whether a surprising ED result is real or an
artifact. Reuses score_v2's own extraction + cache so the audit sees EXACTLY
what the scorer saw (same globs, same parse_numbered_list, same [:7] cap, same
dedup, same norm-text cache key).

It answers four questions, in order:
  1. INDEPENDENCE: are the cells actually diverse, or collapsed (temp-0)?
  2. COVERAGE: are all extracted norms actually scored (in the v2 cache)?
  3. PARSING: did parse_numbered_list extract whole norms, or shred fragments?
  4. SCORING: are the non-ED calls correct? Dumps every non-ED solo-iter norm
     with its E1/E2 answers + E2 reasoning so you can read the directedness
     judgment, plus a contrast sample of ED norms from the comparison scenario.

Usage:
  python3 audit_j.py                  # scenario J vs H, solo-iter cell
  python3 audit_j.py --scenario J --compare H --cell solo-iter
  python3 audit_j.py --cell panel
No API. Reads only existing files and the existing v2 cache.
"""
import json, glob, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_v2 as S
from analysis_shared import parse_numbered_list

CELL_FN = {"solo": S.solo_norms, "solo-iter": S.iter_norms, "panel": S.panel_norms}
CELL_GLOB = {
    "solo": lambda p: sorted(glob.glob(f"{S.SOLO_DIR}/baseline_{p}_run*.json")),
    "solo-iter": lambda p: sorted(glob.glob(f"{S.ITER_DIR}/baseline_{p}_iter_run*.json")),
    "panel": lambda p: sorted(glob.glob(f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_{S.TAG}*.json")),
}

def raw_text(f, cell):
    d = json.load(open(f))
    if cell == "panel":
        return None  # panels are multi-turn; parsing handled by flatten_norms, not raw text
    return d.get("result", {}).get("text", "")

def get_typology(sc):
    return sc.get("typology") or S.typology(sc.get("answers", {}))

def independence(scenario, cell):
    files = CELL_GLOB[cell](scenario)
    if cell == "panel":
        texts = set()
        for f in files:
            texts.add(json.dumps(json.load(open(f)).get("transcript", "?"))[:5000])
        print(f"  {cell:10} files={len(files)}  distinct_transcripts={len(texts)}")
    else:
        texts = set()
        for f in files:
            texts.add((raw_text(f, cell) or "").strip())
        flag = "" if len(texts) >= max(2, 0.5*len(files)) else "  <-- COLLAPSED (low independence, temp-0 artifact)"
        print(f"  {cell:10} files={len(files)}  distinct_outputs={len(texts)}{flag}")
    return files

def audit(scenario, compare, cell):
    cache = S.load_v2_cache()
    print(f"=== AUDIT scenario={scenario} (vs {compare}), cell={cell} ===")
    print(f"v2 cache size: {len(cache)} scored norms\n")

    print("1) INDEPENDENCE (are cells diverse or collapsed?)")
    for c in ["solo", "solo-iter", "panel"]:
        independence(scenario, c)
    print()

    # reconstruct the audited cell exactly as the scorer does
    raw_norms = CELL_FN[cell](scenario)
    seen, norms = set(), []
    for n in raw_norms:
        n = n.strip()
        if n and n not in seen:
            seen.add(n); norms.append(n)
    in_cache = [n for n in norms if n in cache]
    missing = [n for n in norms if n not in cache]
    print(f"2) COVERAGE for {scenario}/{cell}")
    print(f"   extracted (deduped): {len(norms)}   scored (in cache): {len(in_cache)}   MISSING from cache: {len(missing)}")
    if missing:
        print("   missing norms would be silently absent from the ED% -> investigate:")
        for n in missing[:5]:
            print("     -", n[:90])
    print()

    # typology distribution + the E1/E2 split (ED = E1 AND E2; report each separately)
    from collections import Counter
    dist = Counter(get_typology(cache[n]) for n in in_cache)
    def yy(n, k): return str(cache[n].get("answers", {}).get(k, {}).get("answer", "N")).upper().startswith("Y")
    e1y = sum(1 for n in in_cache if yy(n, "E1"))
    e2y = sum(1 for n in in_cache if yy(n, "E2"))
    N = len(in_cache)
    edpct = round(100*dist.get("ED", 0)/N) if N else 0
    print(f"3) TYPOLOGY for {scenario}/{cell}: {dict(dist)}")
    print(f"   ED% (E1 AND E2)        = {dist.get('ED',0)}/{N} = {edpct}%   <- the headline number")
    print(f"   E1=Y (states a moral requirement)     = {round(100*e1y/N) if N else 0}%")
    print(f"   E2=Y (says who owes what to whom)      = {round(100*e2y/N) if N else 0}%")
    print(f"   -> if ED falls, check WHICH of E1/E2 fell: directedness (E2) or moral-framing (E1)\n")

    print(f"4) PARSING spot-check: first 2 {scenario}/{cell} files, raw -> parsed")
    if cell != "panel":
        for f in CELL_GLOB[cell](scenario)[:2]:
            txt = raw_text(f, cell)
            parsed = parse_numbered_list(txt)
            print(f"   FILE {Path(f).name}: parsed {len(parsed)} norms")
            shortish = [pn for pn in parsed if len(pn) < 25]
            if shortish:
                print(f"     WARNING {len(shortish)} suspiciously short (<25 chars) -> possible fragment shredding:")
                for s in shortish: print("       *", repr(s))
            for pn in parsed[:3]:
                print("       -", pn[:100])
    else:
        print("   (panel parsing uses flatten_norms; inspect a transcript's final-round norms separately)")
    print()

    print(f"5) SCORING CHECK: every NON-ED norm in {scenario}/{cell} with its directedness judgment")
    print("   (read whether E2=N is correct: does the norm really fail to say who owes what to whom?)")
    nonED = [n for n in in_cache if get_typology(cache[n]) != "ED"]
    print(f"   {len(nonED)} non-ED norms:\n")
    for n in nonED:
        ans = cache[n].get("answers", {})
        e1 = ans.get("E1", {}).get("answer", "?"); e2 = ans.get("E2", {}).get("answer", "?")
        e2r = ans.get("E2", {}).get("reasoning", "")
        print(f"   [{get_typology(cache[n])}] E1={e1} E2={e2}")
        print(f"     NORM: {n[:160]}")
        print(f"     E2 reasoning: {e2r[:240]}\n")

    print(f"6) CONTRAST: sample of ED norms from {compare}/{cell} (scored directed) for calibration")
    craw = CELL_FN[cell](compare)
    cseen, cnorms = set(), []
    for n in craw:
        n = n.strip()
        if n and n not in cseen and n in cache:
            cseen.add(n); cnorms.append(n)
    ced = [n for n in cnorms if get_typology(cache[n]) == "ED"]
    for n in ced[:4]:
        ans = cache[n].get("answers", {})
        print(f"   [ED] E2={ans.get('E2',{}).get('answer','?')}  {n[:140]}")
        print(f"     E2 reasoning: {ans.get('E2',{}).get('reasoning','')[:200]}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="J")
    ap.add_argument("--compare", default="H")
    ap.add_argument("--cell", default="solo-iter", choices=["solo", "solo-iter", "panel"])
    a = ap.parse_args()
    audit(a.scenario, a.compare, a.cell)

if __name__ == "__main__":
    main()
