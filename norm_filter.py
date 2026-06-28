#!/usr/bin/env python3
"""
Reproducible structural-fragment filter for panel norms, plus cleaned typology.

The long panels (esp. H, G) emit tiered governance documents; the numbered-list
parser shreds outline labels (G4., O3 —, CC-3:, **Bold.**) into separate items.
Bare labels carry no principle/addressee, so they auto-score non-directed and
inflate erosion. This drops ONLY headers and bare title-labels, KEEPS any
label-bearing item that still contains a normative clause, and recomputes the
ED typology from the existing v2 cache (no rescoring).

  python3 norm_filter.py            # cleaned decomposition for A E G H
  python3 norm_filter.py --show H   # also print the dropped items for one scenario
"""
import json, glob, re, sys, argparse
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list, extract_final_round_norms, flatten_norms

CACHE_DIR = "analysis/classification_v2"
SOLO_DIR, ITER_DIR, TDIR = "baselines_core", "baselines_solo_iter", "transcripts"
TAG = "CORE"

VERB = re.compile(r'\b(must|shall|should|may|are|is|be|count|require|prohibit|expect|oblig|'
    r'entitl|responsib|acknowledg|disclos|state|record|raise|credit|penaliz|surfac|flag|declar|'
    r'defensib|accept|valu|provisional|revisable|available|appl|treat|weigh|verif|justif|preserve|'
    r'cite|propose|address|withhold|represent|engage|note|earn|assess)\b', re.I)

def strip_label(s):
    s = s.strip(); prev = None
    while s != prev:
        prev = s
        s = re.sub(r'^\*{1,2}[^*]{1,40}?\*{1,2}[.:)\s\u2014\u2013-]*', '', s).strip()        # **Bold.** / *Italic:*
        s = re.sub(r'^(?:[A-Z]{1,4}-?\d+|Col[A-Z]-?\d+|C[A-Z]I-?\d+)[.:)\s\u2014\u2013-]+', '', s).strip()  # G4. O3 — CC-3: CTI-1:
        s = re.sub(r'^(Tier|Phase|Pillar|Section)\s+\w+[:.\s\u2014\u2013-]*', '', s, flags=re.I).strip()
    return s

def is_structural(s):
    """True = drop (header or bare title-label with no normative clause)."""
    body = strip_label(s)
    wc = len(re.findall(r"[A-Za-z]+", body))
    if wc < 3: return True
    if wc < 6 and not VERB.search(body): return True
    return False

def load_cache():
    sc = {}
    for f in glob.glob(f"{CACHE_DIR}/*.json"):
        for r in json.load(open(f)).get("results", []):
            n = (r.get("norm") or "").strip()
            if n and "answers" in r.get("scoring", {}): sc[n] = r["scoring"]["answers"]
    return sc

def yy(a, k): return str(a.get(k, {}).get("answer", "N")).strip().upper().startswith("Y")
def typ(a):
    e1, e2, i2 = yy(a, "E1"), yy(a, "E2"), yy(a, "I2")
    return "ED" if (e1 and e2) else "EU" if (e1 and not e2) else "IM" if ((not e1) and i2) else "PR"

def collect(p, c, sc):
    if c == "solo":  groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in glob.glob(f"{SOLO_DIR}/baseline_{p}_run*.json")]
    elif c == "solo-iter": groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in glob.glob(f"{ITER_DIR}/baseline_{p}_iter_run*.json")]
    else:             groups = [[x["norm"] for x in flatten_norms(extract_final_round_norms(json.load(open(tp))))] for tp in glob.glob(f"{TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json")]
    seen, kept, dropped = set(), [], []
    for g in groups:
        for n in g:
            n = n.strip()
            if n in seen or n not in sc: continue
            seen.add(n)
            (dropped if (c == "panel" and is_structural(n)) else kept).append(n)
    return kept, dropped

def dist(ns, sc):
    c = Counter(typ(sc[n]) for n in ns); N = len(ns) or 1
    return {k: round(100 * c.get(k, 0) / N) for k in ["ED", "EU", "IM", "PR"]}, len(ns)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--show", default=""); ap.add_argument("--prompts", default="AEGH")
    args = ap.parse_args(); sc = load_cache()
    print(f"cache: {len(sc)} scored norms\n")
    print(f"{'cell':14}{'n_raw':>7}{'n_kept':>8}{'dropped':>9}{'ED_raw':>8}{'ED_clean':>10}   dist(clean)")
    for p in [c for c in args.prompts if c in "AEGH"]:
        eds = {}
        for c in ["solo", "solo-iter", "panel"]:
            kept, dropped = collect(p, c, sc)
            raw = kept + dropped
            d_clean, n_clean = dist(kept, sc); d_raw, n_raw = dist(raw, sc)
            eds[c] = d_clean["ED"]
            print(f"{p+'/'+c:14}{n_raw:>7}{n_clean:>8}{len(dropped):>9}{d_raw['ED']:>8}{d_clean['ED']:>10}   {d_clean}")
        print(f"   -> {p}: collaboration {eds['panel']-eds['solo-iter']:+d}  (solo-iter {eds['solo-iter']} -> panel {eds['panel']});  iteration {eds['solo-iter']-eds['solo']:+d}\n")
        if args.show == p:
            _, dropped = collect(p, "panel", sc)
            print(f"   dropped from {p} panel ({len(dropped)}):")
            for n in dropped: print("     x", n[:88])
            print()

if __name__ == "__main__":
    main()
