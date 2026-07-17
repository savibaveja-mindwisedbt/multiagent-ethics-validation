#!/usr/bin/env python3
"""
Two parallel scoring approaches for the core A/E experiment.

APPROACH 1 (LLM, costs API): the revised rubric.
  - E1 = is a moral principle from any of the seven traditions present? Returns the
    matched principle(s) and tradition(s), so the vocabulary is auditable.
  - E2 = directedness, decoupled. Read from the FULL norm whether it states what is
    owed and to whom, independent of which principle E1 found (or whether E1 fired).
  - E3/E4, I1-I4, P1-P4, T1-T4 retained so the typology and the non-moral contrast work.
  Derives the S3 typology per norm: ED (E1 & E2), EU (E1 & not E2), IM (not E1 & I2),
  PR (not E1 & not I2).

APPROACH 2 (dictionary, NO API): MFD 2.0 and the original MFD as the external
  convergence check. Pure membership, deterministic, local. Cross-tabulated against
  approach 1's E1 so you get the agreement table in the same run.

ISOLATION: approach 1 writes to analysis/classification_v2/ and reads ONLY from there.
It never touches the old analysis/classification/ cache, because the new E1 (tradition
vocabulary) is a different question than the old E1 (73-concept list); mixing them
would be a silent rubric error.

RESUMABLE: every newly scored norm is persisted immediately. A crash (e.g. credits)
loses nothing; rerun and it skips what is already scored.

USAGE
  python3 score_v2.py                  # dry: coverage report + approach 2 only, NO API
  python3 score_v2.py --limit 8        # score only 8 norms (cheap preview of the new rubric)
  python3 score_v2.py --score          # full approach-1 scoring of all A/E misses, then both
  python3 score_v2.py --score --prompts A
Dictionaries expected at moral_dicts/MFD_original.csv and moral_dicts/mfd2.txt
"""
import json, glob, csv, re, sys, argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import (parse_numbered_list, extract_final_round_norms,
                             flatten_norms, call_analysis_model, parse_json_response)

PROMPTS = ["A", "E"]
TAG = "CORE"
V2_DIR = Path("analysis/classification_v2")
SOLO_DIR = Path("baselines_core")
ITER_DIR = Path("baselines_solo_iter")
DICTS = Path("moral_dicts")

MORAL_TRADITIONS = """1. Principlism (Beauchamp & Childress): autonomy, non-maleficence, beneficence, justice. Derived: veracity, fidelity, privacy.
2. Care ethics (Gilligan; Held; Noddings): care, the caring relation, attentiveness, responsiveness to need, relational compassion and empathy.
3. Deontology (Kant; Ross): human dignity, humanity as an end (never merely as means), universalizability, respect for persons; fidelity, reparation, gratitude.
4. Contractualism / second-personal (Scanlon; Darwall): justifiability to others, reasonable rejectability, what we owe to each other, accountability, answerability.
5. Consequentialism (Mill; Singer): utility, the greatest good, aggregate welfare and well-being, harm-minimization, equal consideration of interests.
6. Virtue ethics (Aristotle; Hursthouse): flourishing, courage, temperance, practical wisdom, honesty, generosity, magnanimity.
7. Republican non-domination (Pettit): non-domination, freedom from arbitrary power, contestability."""

CLASSIFY_PROMPT_V2 = """You are scoring a norm using a strict rubric. The norm may have several kinds of content at once: it may name a moral principle (explicit moral), it may express a moral commitment through procedure without naming it (implicit moral), it may specify procedure for its own sake (procedural), and it may reference technical infrastructure (technical). Answer every question independently. Do not let one answer bias another.

The norm is:
"{norm}"

The recognized moral principles, by tradition:
{traditions}

STEP 1. EXPLICIT MORAL
E1. Does the norm name, invoke, or directly rely on one or more of the moral principles above (or a clear, recognized specification of one)? Direct uses or clear synonyms count; distant fragments do not. List the specific principle(s) and tradition(s) in the fields provided.
E2. DIRECTEDNESS. Reading the FULL norm in context, does it state what is owed and to whom, or what would count as wronging whom? Judge this only from the norm's content and context, NOT from whether a principle word appears. A norm can name a principle yet direct it to no one (answer N), and a norm can be directed in substance. Answer independently of E1.
E3. Does the norm's stated or implied reason appeal to a moral principle, rather than only to convenience, coordination, efficiency, or convention?
E4. Would the principle carry over to other cooperative situations, not only this specific deliberation?

STEP 2. IMPLICIT MORAL
I1. Does the norm specify a procedural mechanism, structural feature, or rule of operation? (If none, this fails and the implicit check is skipped.)
I2. Does that procedural specification express or operationalize a recognizable moral commitment from the principles above, WITHOUT naming it? Be strict; the commitment must be specifically identifiable, not "some general good process." Name the commitment in your reasoning.
I3. Would a plausible alternative procedure express a different commitment, or none? (Counterfactual test of expressiveness.)
I4. Can the procedure be defended by appeal to that moral commitment, rather than only to convenience, coordination, efficiency, or convention?

STEP 3. PROCEDURAL
P1. Does the norm specify a process, rule of operation, mechanism, or structure?
P2. Does it specify who acts, what roles exist, or what timing or sequencing applies?
P3. Does it specify what triggers the procedure or when it applies?
P4. Could an outsider execute it as written, without further moral interpretation?

STEP 4. TECHNICAL
T1. Does the norm reference specific technical infrastructure (logs, ledgers, APIs, data formats, cryptographic operations, specific systems)?
T2. Does it specify implementation details requiring technical capability?
T3. Does it presuppose technical capabilities (persistent storage, cryptographic operations, network protocols, specific data structures)?
T4. Would it be unimplementable without that technical infrastructure?

For each question answer Y or N with one sentence of reasoning. Borderline cases answer N. If the norm is compound, score the dominant claim and note it in decomposition_note.

Respond in JSON only, no preamble, no code fences:
{{
  "E1": {{"answer": "Y", "reasoning": "...", "principles": [], "traditions": []}},
  "E2": {{"answer": "N", "reasoning": "..."}},
  "E3": {{"answer": "N", "reasoning": "..."}},
  "E4": {{"answer": "N", "reasoning": "..."}},
  "I1": {{"answer": "Y", "reasoning": "..."}},
  "I2": {{"answer": "N", "reasoning": "..."}},
  "I3": {{"answer": "N", "reasoning": "..."}},
  "I4": {{"answer": "N", "reasoning": "..."}},
  "P1": {{"answer": "Y", "reasoning": "..."}},
  "P2": {{"answer": "N", "reasoning": "..."}},
  "P3": {{"answer": "N", "reasoning": "..."}},
  "P4": {{"answer": "N", "reasoning": "..."}},
  "T1": {{"answer": "N", "reasoning": "..."}},
  "T2": {{"answer": "N", "reasoning": "..."}},
  "T3": {{"answer": "N", "reasoning": "..."}},
  "T4": {{"answer": "N", "reasoning": "..."}},
  "decomposition_note": ""
}}"""

def y(ans, k):
    return str(ans.get(k, {}).get("answer", "N")).strip().upper().startswith("Y")

def typology(ans):
    e1, e2, i2 = y(ans, "E1"), y(ans, "E2"), y(ans, "I2")
    if e1 and e2: return "ED"
    if e1 and not e2: return "EU"
    if (not e1) and i2: return "IM"
    return "PR"

def v2_classify(norm):
    prompt = CLASSIFY_PROMPT_V2.format(norm=norm, traditions=MORAL_TRADITIONS)
    raw = call_analysis_model(prompt)
    answers = parse_json_response(raw)   # raises on bad JSON -> caller persists nothing for this norm
    return {"answers": answers,
            "E1": y(answers, "E1"), "E2": y(answers, "E2"),
            "typology": typology(answers),
            "principles": answers.get("E1", {}).get("principles", []),
            "traditions": answers.get("E1", {}).get("traditions", [])}

def load_v2_cache():
    cache = {}
    if not V2_DIR.exists(): return cache
    for f in sorted(V2_DIR.glob("classification_v2_*.json")):
        try: data = json.load(open(f))
        except Exception: continue
        for r in data.get("results", []):
            n = (r.get("norm") or "").strip()
            sc = r.get("scoring", {})
            if n and isinstance(sc, dict) and "answers" in sc:
                cache[n] = sc
    return cache

def persist_v2(new_results):
    V2_DIR.mkdir(parents=True, exist_ok=True)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (V2_DIR / f"classification_v2_{st}.json").write_text(
        json.dumps({"results": new_results}, indent=2))

# ---- norm collection ----
def solo_norms(p):
    out = []
    for f in sorted(glob.glob(f"{SOLO_DIR}/baseline_{p}_run*.json")):
        try: out += parse_numbered_list(json.load(open(f)).get("result", {}).get("text", ""))[:7]
        except Exception: pass
    return out

def iter_norms(p):
    out = []
    for f in sorted(glob.glob(f"{ITER_DIR}/baseline_{p}_iter_run*.json")):
        try: out += parse_numbered_list(json.load(open(f)).get("result", {}).get("text", ""))[:7]
        except Exception: pass
    return out

def panel_norms(p):
    out = []
    for tp in sorted(glob.glob(f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json")):
        try: out += [it["norm"] for it in flatten_norms(extract_final_round_norms(json.load(open(tp))))]
        except Exception: pass
    return out

COND = {"solo": solo_norms, "solo-iter": iter_norms, "panel": panel_norms}

def collect():
    cells = {}
    for p in PROMPTS:
        for c, fn in COND.items():
            seen, ns = set(), []
            for n in fn(p):
                n = n.strip()
                if n and n not in seen:
                    seen.add(n); ns.append(n)
            cells[(p, c)] = ns
    return cells

# ---- MFD dictionaries (approach 2) ----
def _set_pol(pol, w, p):
    if w in pol and pol[w] != p: pol[w] = "both"
    else: pol.setdefault(w, p)

def load_dicts():
    """Each dict -> {'words': set, 'pol': {word: 'virtue'|'vice'|'both'|'neutral'}}."""
    d = {}
    mfd1 = DICTS / "MFD_original.csv"
    if mfd1.exists():
        pol = {}
        for r in csv.DictReader(open(mfd1)):
            w = r["word"].strip().lower(); s = r.get("sentiment", "").strip().lower()
            _set_pol(pol, w, s if s in ("virtue", "vice") else "neutral")
        d["MFD"] = {"words": set(pol), "pol": pol}
    mfd2 = DICTS / "mfd2.txt"
    if mfd2.exists():
        legend, pol, seen = {}, {}, 0
        for line in open(mfd2):
            line = line.rstrip("\n")
            if line.strip() == "%": seen += 1; continue
            if seen == 1:  # legend: "1\tcare.virtue"
                t = line.split("\t") if "\t" in line else line.split()
                if len(t) >= 2 and t[0].strip().isdigit():
                    legend[t[0].strip()] = "virtue" if t[1].strip().lower().endswith("virtue") else "vice"
            elif seen >= 2:  # "word\t<catnums>"
                parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
                if not parts or not parts[0].strip(): continue
                w = parts[0].strip().lower()
                nums = re.findall(r"\d+", parts[1]) if len(parts) > 1 else []
                ps = {legend.get(n, "") for n in nums}; ps.discard("")
                p = next(iter(ps)) if len(ps) == 1 else ("both" if len(ps) > 1 else "neutral")
                _set_pol(pol, w, p)
        d["MFD2.0"] = {"words": set(pol), "pol": pol}
    return d

def dict_hit(norm, wordset):
    return any(w in wordset for w in re.findall(r"[a-z']+", norm.lower()))

def valence_counts(norm, pol):
    """Virtue-word and vice-word token counts in the norm (ambiguous/neutral skipped)."""
    v = vi = 0
    for t in re.findall(r"[a-z']+", norm.lower()):
        p = pol.get(t)
        if p == "virtue": v += 1
        elif p == "vice": vi += 1
    return v, vi

# ---- aggregation ----
def pct(n, N): return round(100 * n / N) if N else 0

def report(cells, cache, dicts):
    # approach 1 aggregation (only over norms present in cache)
    print("\n=== APPROACH 1: tradition rubric, S3 typology (scored norms only) ===")
    print(f"{'cell':16} {'n':>4} {'ED%':>4} {'EU%':>4} {'IM%':>4} {'PR%':>4} {'E2dir%':>6} {'E1mor%':>6}")
    agg = {}
    for (p, c), norms in cells.items():
        sc = [cache[n] for n in norms if n in cache]
        N = len(sc)
        if not N:
            print(f"{p+'/'+c:16} {0:>4}  (no scored norms yet)"); continue
        t = {"ED": 0, "EU": 0, "IM": 0, "PR": 0}
        e2 = e1 = 0
        for s in sc:
            t[typology(s["answers"])] += 1
            e2 += int(y(s["answers"], "E2")); e1 += int(y(s["answers"], "E1"))
        agg[(p, c)] = {"N": N, "ED": pct(t["ED"], N), "E2": pct(e2, N)}
        print(f"{p+'/'+c:16} {N:>4} {pct(t['ED'],N):>4} {pct(t['EU'],N):>4} "
              f"{pct(t['IM'],N):>4} {pct(t['PR'],N):>4} {pct(e2,N):>6} {pct(e1,N):>6}")
    # decomposition
    print("\n=== decomposition (ED% headline): iteration vs collaboration ===")
    for p in PROMPTS:
        if all((p, c) in agg for c in ("solo", "solo-iter", "panel")):
            s, si, pa = agg[(p,"solo")]["ED"], agg[(p,"solo-iter")]["ED"], agg[(p,"panel")]["ED"]
            print(f"  {p}: solo {s}  solo-iter {si}  panel {pa}   "
                  f"iteration {si-s:+d}   collaboration {pa-si:+d}")
        else:
            print(f"  {p}: incomplete cells, decomposition pending")
    # approach 2 convergence
    if dicts:
        print("\n=== APPROACH 2: MFD convergence vs approach-1 E1 (scored norms only) ===")
        for dname, ws in dicts.items():
            print(f"  -- {dname} --   both / mine-only / dict-only / neither / agree%")
            for (p, c), norms in cells.items():
                sc = [(n, cache[n]) for n in norms if n in cache]
                if not sc: continue
                both = mo = do = ne = 0
                for n, s in sc:
                    mine = y(s["answers"], "E1"); dh = dict_hit(n, ws["words"])
                    if mine and dh: both += 1
                    elif mine: mo += 1
                    elif dh: do += 1
                    else: ne += 1
                N = len(sc)
                print(f"     {p+'/'+c:14} {both:4} {mo:4} {do:4} {ne:4}   {pct(both+ne,N)}%")
        # valence layer (descriptive; dictionary-only, runs over ALL norms, no API needed)
        print("\n=== VALENCE (descriptive only, orthogonal to directedness; all norms) ===")
        print("    %virtue = virtue/(virtue+vice) of polarized tokens; lean+ = norms with more virtue than vice")
        for dname, ws in dicts.items():
            print(f"  -- {dname} --   v-tok / vi-tok / %virtue / lean+ / lean- / even")
            for (p, c), norms in cells.items():
                if not norms: continue
                vt = vit = lp = ln = ev = 0
                for n in norms:
                    v, vi = valence_counts(n, ws["pol"])
                    vt += v; vit += vi
                    if v > vi: lp += 1
                    elif vi > v: ln += 1
                    else: ev += 1
                pv = pct(vt, vt + vit) if (vt + vit) else 0
                print(f"     {p+'/'+c:14} {vt:5} {vit:5} {pv:6}%  {lp:5} {ln:5} {ev:4}")
    else:
        print("\n(approach 2 skipped: no dictionaries found in moral_dicts/)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="run approach-1 LLM scoring of misses (spends)")
    ap.add_argument("--limit", type=int, default=0, help="cap norms scored this run (cheap preview)")
    ap.add_argument("--prompts", default="AEG")
    args = ap.parse_args()
    global PROMPTS
    PROMPTS = [c for c in args.prompts if c in ("A", "E", "G", "H", "J", "K", "L", "M", "N", "P", "Q")]

    cells = collect()
    cache = load_v2_cache()
    dicts = load_dicts()
    all_norms = []
    for ns in cells.values():
        for n in ns:
            if n not in all_norms: all_norms.append(n)
    todo = [n for n in all_norms if n not in cache]

    print("coverage:")
    for (p, c), ns in cells.items():
        have = sum(1 for n in ns if n in cache)
        print(f"  {p}/{c:9} norms={len(ns):4}  scored={have:4}  to-score={len(ns)-have:4}")
    print(f"total unique norms={len(all_norms)}  already scored={len(all_norms)-len(todo)}  to-score={len(todo)}")
    if dicts: print("dictionaries loaded: " + ", ".join(f"{k}({len(v['words'])}w)" for k, v in dicts.items()))
    else: print("dictionaries: NONE found (put MFD_original.csv and mfd2.txt in moral_dicts/)")

    if args.score and todo:
        batch = todo[:args.limit] if args.limit else todo
        print(f"\nscoring {len(batch)} norms with the v2 rubric (persisting each)...")
        new = []
        for i, n in enumerate(batch, 1):
            try:
                sc = v2_classify(n)
            except Exception as e:
                print(f"  [{i}/{len(batch)}] FAILED, stopping and saving progress: {e}")
                break
            cache[n] = sc; new.append({"norm": n, "scoring": sc})
            if len(new) % 10 == 0:
                persist_v2(new); new = []
                print(f"  [{i}/{len(batch)}] checkpoint saved")
        if new: persist_v2(new)
        print("scoring pass complete.")
    elif args.score:
        print("\nnothing to score, all norms already in v2 cache.")
    else:
        print("\nDRY RUN (no API). Add --score to run approach 1. Approach 2 + any cached approach-1 below.")

    report(cells, cache, dicts)

if __name__ == "__main__":
    main()
