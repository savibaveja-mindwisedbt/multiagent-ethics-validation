#!/usr/bin/env python3
"""
SVO (Social Value Orientation) rubric judge for the stakes manipulation check.

WHY THIS EXISTS
The stakes-salience signal (does self-interest language rise A -> E -> H?) is
currently a hand-built regex: lexical, inverts on negation, unsourced. This
replaces it with a rubric-based LLM judge that scores text for proself vs
prosocial orientation, parallel in design to the directedness instrument
(E1-E4 in score_v2.py): atomic Y/N markers with one-sentence reasoning, a
derived category, persisted with provenance, resumable, dry by default.

CONSTRUCT (define, do not lexicon-match)
Social Value Orientation splits motivational orientation into:
  PROSELF   = competition  (maximize relative advantage, own minus other)
            + individualism (maximize own, indifferent to other)
  PROSOCIAL = cooperation  (maximize joint, own plus other)
            + equality     (minimize the difference between outcomes)
Sources to cite in the paper (CONFIRM each line-by-line before use; these are
from the handoff / general knowledge, not yet verified against the originals):
  - Murphy, Ackermann & Handgraaf (2011), the SVO Slider Measure.
  - Van Lange (1999) triple-dominance; Messick & McClintock (1968).
H's payoff (fixed pool, scored on relative contribution) IS the SVO competition
definition, so the construct fits the manipulation by design.

WHAT IT IS NOT
- Not a directedness measure. SVO is a different construct (self-interest
  orientation), so it CANNOT be used to re-confirm the ED erosion. Same boundary
  the agency-communion check drew. Keep the two instruments separate.
- Not the behavioral SVO Slider task. That is a choice task; administering it to
  a model is a separate, heavily-caveated probe, not a text score. Out of scope.

UNITS
  --unit turn  (default)  score each deliberation turn. This is the manipulation
                          check: proself% per cell and per round (A -> E -> H).
  --unit norm             score each final-round / baseline norm. For the
                          mediation view: does cell proself% track the ED drop?

JUDGE MODEL
SVO_JUDGE_MODEL below is the "high-performance Claude" judge. Default is Opus.
The exact OpenRouter slug MUST resolve; --score runs a preflight (agents.verify_
model_slugs) and refuses to spend if the slug is missing, printing the closest
available anthropic slugs so you can correct it. Sonnet 4.6 is the validated
fallback used elsewhere in the repo.

USAGE
  python3 score_svo.py                      # DRY: coverage report, no API
  python3 score_svo.py --make-sample 60     # write a blank hand-coding sheet
  python3 score_svo.py --score --limit 8    # cheap preview: score 8 units
  python3 score_svo.py --score              # full scoring of all misses (spends)
  python3 score_svo.py --score --unit norm
  python3 score_svo.py --validate svo_handcoded.json   # agreement vs human
  python3 score_svo.py --selftest           # NO API: prove the non-API logic
  python3 score_svo.py --model anthropic/claude-sonnet-4.6   # override judge
"""
import json, glob, sys, argparse, random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import (parse_numbered_list, extract_final_round_norms,
                             flatten_norms, parse_json_response)
import analysis_shared as A_SHARED

# ----- config -----
SVO_JUDGE_MODEL = "anthropic/claude-opus-4.8"   # high-performance Claude judge
SVO_FALLBACK_MODEL = "anthropic/claude-sonnet-4.6"
JUDGE_MAX_TOKENS = 1200
PROMPTS = ["A", "E", "G", "H"]
TAG = "CORE"
SVO_DIR = Path("analysis/svo")
SOLO_DIR = Path("baselines_core")
ITER_DIR = Path("baselines_solo_iter")
TDIR = Path("transcripts")

# ----- the rubric -----
SVO_PROMPT = """You are scoring one passage for Social Value Orientation (SVO): the motivational orientation it expresses toward outcomes for the self versus others. Score ONLY the orientation expressed in this passage. Do not judge whether it is moral, well-reasoned, or directed. Answer each marker independently; do not let one bias another.

The passage is:
"{text}"

SVO defines four orientations. Two are PROSELF, two are PROSOCIAL:
- competition (proself): concern with RELATIVE standing, doing better than or gaining advantage OVER others, outcompeting, zero-sum positioning.
- individualism (proself): concern with maximizing OWN outcome or payoff, indifferent to others' outcomes.
- cooperation (prosocial): concern with the JOINT or collective outcome, mutual or total benefit, everyone gaining.
- equality (prosocial): concern with EQUAL or fair distribution, minimizing the difference between parties' outcomes.

Score these four markers. A marker is Y only if the passage actually expresses that concern as its own content (asserting, advocating, or reasoning from it). Merely naming a concept while arguing against it is N. Borderline is N.
S1 OWN-OUTCOME: does the passage express concern with maximizing its own (or its side's) outcome, payoff, share, or success?
S2 RELATIVE-ADVANTAGE: does it express concern with relative standing, advantage over others, outcompeting, or zero-sum positioning?
S3 JOINT-OUTCOME: does it express concern with the joint, collective, or total outcome, or mutual benefit?
S4 EQUALITY: does it express concern with equal or fair distribution, or minimizing the difference between parties?

Then give a holistic read (used only as a cross-check):
dominant_orientation: one of "proself", "prosocial", "mixed", "neither" — the orientation that dominates the passage overall.

Respond in JSON only, no preamble, no code fences:
{{
  "S1": {{"answer": "Y", "reasoning": "..."}},
  "S2": {{"answer": "N", "reasoning": "..."}},
  "S3": {{"answer": "N", "reasoning": "..."}},
  "S4": {{"answer": "N", "reasoning": "..."}},
  "dominant_orientation": "proself"
}}"""

def y(ans, k):
    return str(ans.get(k, {}).get("answer", "N")).strip().upper().startswith("Y")

def svo_category(ans):
    """Derived SVO category from the four atomic markers. Documented, fixed rule.
    Relative-advantage is the strongest proself signal, so it wins when present."""
    s1, s2, s3, s4 = y(ans, "S1"), y(ans, "S2"), y(ans, "S3"), y(ans, "S4")
    if s2:                              return "competition"   # proself, strongest signal
    prosocial_present = s3 or s4
    if s1 and not prosocial_present:    return "individualism" # proself: own-outcome only
    if prosocial_present and not s1:                           # prosocial only
        if s3 and s4:                   return "prosocial-mixed"
        return "cooperation" if s3 else "equality"
    if s1 and prosocial_present:        return "mixed"         # own + joint/equal -> genuinely mixed
    return "neither"

PROSELF = {"competition", "individualism"}
PROSOCIAL = {"cooperation", "equality", "prosocial-mixed"}

def is_proself(ans):
    return svo_category(ans) in PROSELF

def derived_vs_holistic_conflict(ans):
    """QA flag: derived category's proself/prosocial vs the judge's holistic read."""
    cat = svo_category(ans)
    hol = str(ans.get("dominant_orientation", "")).strip().lower()
    if cat in PROSELF and hol == "prosocial": return True
    if cat in PROSOCIAL and hol == "proself": return True
    return False

# ----- unit collection -----
def turns_for(p, rounds=None):
    """(unit_id, text) for deliberation turns in scenario p panels.
    rounds: optional set/list of round_index values to keep (None = all)."""
    out = []
    for tp in sorted(glob.glob(f"{TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json")):
        stem = Path(tp).stem
        try:
            for t in json.load(open(tp)).get("transcript", []):
                txt = (t.get("text") or "").strip()
                if not txt: continue
                ri = t.get("round_index")
                if rounds is not None and ri not in rounds: continue
                uid = f"{stem}::r{ri}::{t.get('agent_id')}::t{t.get('turn_index')}"
                out.append((uid, txt, {"cell": f"{p}/panel", "scenario": p,
                                       "round": ri, "kind": "turn"}))
        except Exception:
            pass
    return out

def norms_for(p):
    """(unit_id, norm_text, meta) for solo, solo-iter, and panel final norms."""
    out = []
    for f in sorted(glob.glob(f"{SOLO_DIR}/baseline_{p}_run*.json")):
        stem = Path(f).stem
        for i, n in enumerate(parse_numbered_list(json.load(open(f)).get("result", {}).get("text", ""))):
            out.append((f"{stem}::n{i}", n.strip(), {"cell": f"{p}/solo", "scenario": p, "round": None, "kind": "norm"}))
    for f in sorted(glob.glob(f"{ITER_DIR}/baseline_{p}_iter_run*.json")):
        stem = Path(f).stem
        for i, n in enumerate(parse_numbered_list(json.load(open(f)).get("result", {}).get("text", ""))):
            out.append((f"{stem}::n{i}", n.strip(), {"cell": f"{p}/solo-iter", "scenario": p, "round": None, "kind": "norm"}))
    for tp in sorted(glob.glob(f"{TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json")):
        stem = Path(tp).stem
        for i, it in enumerate(flatten_norms(extract_final_round_norms(json.load(open(tp))))):
            out.append((f"{stem}::pn{i}", it["norm"].strip(), {"cell": f"{p}/panel", "scenario": p, "round": None, "kind": "norm"}))
    return out

def collect(unit, rounds=None):
    if unit == "turn":
        fn = lambda p: turns_for(p, rounds=rounds)
    else:
        fn = norms_for
    units = {}
    for p in PROMPTS:
        for uid, text, meta in fn(p):
            if text and uid not in units:
                units[uid] = (text, meta)
    return units

# ----- cache -----
def cache_path(unit):
    return SVO_DIR / f"svo_{unit}"

def load_cache(unit):
    cache = {}
    d = cache_path(unit)
    if not d.exists(): return cache
    for f in sorted(d.glob("svo_*.json")):
        try: data = json.load(open(f))
        except Exception: continue
        for r in data.get("results", []):
            uid = r.get("unit_id"); sc = r.get("scoring")
            if uid and isinstance(sc, dict) and "answers" in sc:
                cache[uid] = r
    return cache

def persist(unit, new_results):
    d = cache_path(unit); d.mkdir(parents=True, exist_ok=True)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (d / f"svo_{st}.json").write_text(json.dumps({"results": new_results}, indent=2))

# ----- judge call -----
def judge(text, model):
    prompt = SVO_PROMPT.format(text=text)
    raw = A_SHARED.call_openrouter(model, [{"role": "user", "content": prompt}],
                                   max_tokens=JUDGE_MAX_TOKENS, temperature=0.0,
                                   omit_reasoning=False)
    answers = parse_json_response(raw)   # raises on bad JSON -> caller persists nothing
    return {"answers": answers, "category": svo_category(answers),
            "proself": is_proself(answers),
            "holistic": str(answers.get("dominant_orientation", "")).strip().lower(),
            "qa_conflict": derived_vs_holistic_conflict(answers)}

def preflight(model):
    """Verify the judge slug resolves on OpenRouter before spending. Returns the
    model to use, or raises with the closest anthropic slugs listed."""
    try:
        from keychain import get_openrouter_key
        import agents
        key = get_openrouter_key()
        resolved, missing = agents.verify_model_slugs(key, [model])
        if resolved:
            return model
        avail_anthropic = []
        import requests
        r = requests.get(agents.OPENROUTER_MODELS_URL,
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        avail_anthropic = sorted(m["id"] for m in r.json().get("data", [])
                                 if m["id"].startswith("anthropic/"))
        raise SystemExit(
            f"\nJudge model '{model}' does not resolve on OpenRouter.\n"
            f"Available anthropic slugs:\n  " + "\n  ".join(avail_anthropic) +
            f"\nRe-run with --model <one of the above> (validated fallback: {SVO_FALLBACK_MODEL}).\n")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Preflight could not verify the model slug: {e}\n"
                         f"Fix credentials/network, or pass --model {SVO_FALLBACK_MODEL}.")

# ----- aggregation -----
def pct(n, N): return round(100 * n / N) if N else 0

def report(units, cache, unit):
    by_cell = defaultdict(list)
    by_round = defaultdict(list)   # (scenario, round) -> proself flags, panels only
    conflicts = 0
    for uid, (text, meta) in units.items():
        rec = cache.get(uid)
        if not rec: continue
        ans = rec["scoring"]["answers"]
        ps = is_proself(ans)
        by_cell[meta["cell"]].append(ps)
        conflicts += int(derived_vs_holistic_conflict(ans))
        if meta["kind"] == "turn" and meta["round"] is not None:
            by_round[(meta["scenario"], meta["round"])].append(ps)

    print(f"\n=== SVO proself rate by cell (unit={unit}, scored only) ===")
    print(f"{'cell':16} {'n':>5} {'proself%':>9}")
    order = [f"{p}/{c}" for p in PROMPTS for c in (["panel"] if unit == "turn" else ["solo","solo-iter","panel"])]
    for cell in order:
        v = by_cell.get(cell, [])
        if v: print(f"{cell:16} {len(v):>5} {pct(sum(v),len(v)):>9}")
        else: print(f"{cell:16} {0:>5}   (none scored)")

    if unit == "turn":
        print("\n=== manipulation check: proself% of turns, A -> E -> H ===")
        for p in PROMPTS:
            v = by_cell.get(f"{p}/panel", [])
            print(f"  {p}: {pct(sum(v),len(v)) if v else 0}%  (n={len(v)})")
        print("\n=== round dynamics: proself% by round (panels) ===")
        for p in PROMPTS:
            rounds = sorted(r for (s, r) in by_round if s == p)
            if not rounds: continue
            cells = "  ".join(f"r{r}:{pct(sum(by_round[(p,r)]),len(by_round[(p,r)]))}%" for r in rounds)
            print(f"  {p}: {cells}")

    if unit == "norm":
        print("\n=== mediation view: does cell proself% track the ED drop? ===")
        print("  (compare panel proself% across A<E<G<H against ED erosion -8/-2/-17/-41)")
        for p in PROMPTS:
            v = by_cell.get(f"{p}/panel", [])
            print(f"  {p}/panel proself%: {pct(sum(v),len(v)) if v else 0}  (n={len(v)})")

    scored = sum(len(v) for v in by_cell.values())
    if scored:
        print(f"\nQA: derived-vs-holistic conflicts: {conflicts}/{scored} "
              f"({pct(conflicts,scored)}%). High conflict => rubric needs review.")

# ----- hand-coding sample + validation -----
def make_sample(units, unit, n, seed=7):
    """Write a blank hand-coding sheet, stratified across cells, for human IRR."""
    by_cell = defaultdict(list)
    for uid, (text, meta) in units.items():
        by_cell[meta["cell"]].append((uid, text))
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_cell)))
    sample = []
    for cell, items in sorted(by_cell.items()):
        rng.shuffle(items)
        for uid, text in items[:per]:
            sample.append({"unit_id": uid, "cell": cell, "text": text,
                           "human_proself": "",   # fill "Y" or "N"
                           "human_category": "",   # optional: competition/individualism/cooperation/equality/neither
                           "note": ""})
    out = Path(f"svo_handcoded_{unit}_BLANK.json")
    out.write_text(json.dumps({"unit": unit, "instructions":
        "For each item set human_proself to Y (proself: competition or individualism) "
        "or N (prosocial or neither). Optionally set human_category. Save and pass to "
        "--validate.", "items": sample}, indent=2))
    print(f"wrote {len(sample)} items across {len(by_cell)} cells to {out}")
    print("fill human_proself (Y/N) for each, then: python3 score_svo.py --validate " + out.name)

def validate(path, unit):
    """Agreement of the judge vs human labels on the hand-coded sample."""
    data = json.load(open(path))
    items = data.get("items", data if isinstance(data, list) else [])
    cache = load_cache(unit)
    tp = tn = fp = fn = skipped = 0
    for it in items:
        hp = str(it.get("human_proself", "")).strip().upper()
        if hp not in ("Y", "N"): skipped += 1; continue
        rec = cache.get(it.get("unit_id"))
        if not rec: skipped += 1; continue
        model_ps = is_proself(rec["scoring"]["answers"])
        human_ps = hp == "Y"
        if human_ps and model_ps: tp += 1
        elif human_ps and not model_ps: fn += 1
        elif (not human_ps) and model_ps: fp += 1
        else: tn += 1
    N = tp + tn + fp + fn
    if not N:
        print(f"no overlap between hand-coded labels and scored cache "
              f"(skipped {skipped}). Score the sample first with --score."); return
    agree = (tp + tn) / N
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0
    po = agree
    p_yes_h = (tp + fn) / N; p_yes_m = (tp + fp) / N
    pe = p_yes_h * p_yes_m + (1 - p_yes_h) * (1 - p_yes_m)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    print(f"\n=== SVO judge vs human (n={N}, skipped {skipped}) ===")
    print(f"  agreement={agree:.0%}  precision(proself)={prec:.0%}  recall(proself)={rec_:.0%}")
    print(f"  Cohen's kappa={kappa:.2f}   (TP {tp}  TN {tn}  FP {fp}  FN {fn})")
    print("  rule of thumb: kappa >=0.6 substantial, >=0.8 near-human. Below 0.6, revise the rubric.")

# ----- no-API self-test -----
def selftest():
    print("SELFTEST (no API): category derivation, proself mapping, kappa math.\n")
    def ans(s1,s2,s3,s4,hol="proself"):
        mk=lambda v:{"answer":"Y" if v else "N","reasoning":""}
        return {"S1":mk(s1),"S2":mk(s2),"S3":mk(s3),"S4":mk(s4),"dominant_orientation":hol}
    cases = [
        (ans(1,1,0,0), "competition", True),
        (ans(1,0,0,0), "individualism", True),
        (ans(0,0,1,0), "cooperation", False),
        (ans(0,0,0,1), "equality", False),
        (ans(0,0,1,1), "prosocial-mixed", False),
        (ans(1,0,1,0), "mixed", False),
        (ans(0,0,0,0), "neither", False),
        (ans(1,1,1,1), "competition", True),   # S2 dominates
    ]
    ok = True
    for a, exp_cat, exp_ps in cases:
        gc, gp = svo_category(a), is_proself(a)
        flag = "ok" if (gc==exp_cat and gp==exp_ps) else "FAIL"
        if flag=="FAIL": ok=False
        print(f"  S1234={[y(a,k) for k in ['S1','S2','S3','S4']]} -> {gc:16} proself={gp}  [{flag}]")
    # conflict flag
    c = derived_vs_holistic_conflict(ans(1,1,0,0,hol="prosocial"))
    print(f"  conflict flag (competition vs holistic prosocial) = {c}  [{'ok' if c else 'FAIL'}]")
    if not c: ok=False
    # kappa: construct a sample with known agreement and check the math
    import tempfile, os
    fake_cache_units = {}
    items=[]
    # 8 human Y, judge Y (tp); 2 human N judge N (tn); 1 human Y judge N (fn); 1 human N judge Y (fp)
    plan = [("Y",1)]*8 + [("N",0)]*2 + [("Y",0)]*1 + [("N",1)]*1
    rows=[]
    for i,(h,m) in enumerate(plan):
        uid=f"u{i}"
        rows.append({"unit_id":uid,"human_proself":h})
        # build a scoring that yields proself==bool(m): competition if m else cooperation
        a = ans(1,1,0,0) if m else ans(0,0,1,0)
        fake_cache_units[uid]={"unit_id":uid,"scoring":{"answers":a}}
    # monkey-load: write a temp cache dir
    tmp = Path(tempfile.mkdtemp())
    (tmp).mkdir(exist_ok=True)
    global SVO_DIR
    saveroot = SVO_DIR
    SVO_DIR = tmp
    (cache_path("turn")).mkdir(parents=True, exist_ok=True)
    (cache_path("turn")/"svo_test.json").write_text(json.dumps({"results":list(fake_cache_units.values())}))
    sample = tmp/"hand.json"
    sample.write_text(json.dumps({"unit":"turn","items":rows}))
    print()
    validate(str(sample), "turn")
    SVO_DIR = saveroot
    # expected: n=12, tp8 tn2 fp1 fn1, agree=10/12=83%, kappa computed
    print("\n  expected: n=12, TP8 TN2 FP1 FN1, agreement 83%.")
    print("SELFTEST PASSED" if ok else "SELFTEST HAD FAILURES (see above)")

# ----- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="run LLM scoring of misses (spends)")
    ap.add_argument("--limit", type=int, default=0, help="cap units scored this run")
    ap.add_argument("--unit", choices=["turn", "norm"], default="turn")
    ap.add_argument("--rounds", default="", metavar="LIST",
                    help="turn unit only: comma-separated round indices to score, e.g. 0 "
                         "(round-0 openings are the strongest/cheapest manipulation signal). "
                         "Empty = all rounds.")
    ap.add_argument("--prompts", default="AEGH")
    ap.add_argument("--model", default=SVO_JUDGE_MODEL, help="judge model slug")
    ap.add_argument("--make-sample", type=int, default=0, metavar="N",
                    help="write a blank hand-coding sheet of N items and exit")
    ap.add_argument("--validate", default="", metavar="FILE",
                    help="compare judge vs human labels in FILE and exit")
    ap.add_argument("--selftest", action="store_true", help="no-API logic check and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    global PROMPTS
    PROMPTS = [c for c in args.prompts if c in ("A", "E", "G", "H", "J", "K")]

    rounds = None
    if args.rounds.strip():
        rounds = {int(x) for x in args.rounds.split(",") if x.strip().lstrip("-").isdigit()}

    if args.validate:
        validate(args.validate, args.unit); return

    units = collect(args.unit, rounds=rounds)
    if args.make_sample:
        make_sample(units, args.unit, args.make_sample); return

    cache = load_cache(args.unit)
    todo = [uid for uid in units if uid not in cache]

    print(f"unit={args.unit}  judge={args.model}")
    print("coverage by cell:")
    bycell = defaultdict(lambda: [0, 0])
    for uid, (text, meta) in units.items():
        bycell[meta["cell"]][0] += 1
        if uid in cache: bycell[meta["cell"]][1] += 1
    for cell in sorted(bycell):
        tot, have = bycell[cell]
        print(f"  {cell:16} units={tot:5}  scored={have:5}  to-score={tot-have:5}")
    print(f"total units={len(units)}  scored={len(units)-len(todo)}  to-score={len(todo)}")

    if args.score and todo:
        model = preflight(args.model)
        batch = todo[:args.limit] if args.limit else todo
        print(f"\nscoring {len(batch)} units with {model} (persisting every 10)...")
        new = []
        for i, uid in enumerate(batch, 1):
            text, meta = units[uid]
            try:
                sc = judge(text, model)
            except Exception as e:
                print(f"  [{i}/{len(batch)}] FAILED, saving progress and stopping: {e}")
                break
            rec = {"unit_id": uid, "cell": meta["cell"], "scenario": meta["scenario"],
                   "round": meta["round"], "kind": meta["kind"], "text": text, "scoring": sc}
            cache[uid] = rec; new.append(rec)
            if len(new) % 10 == 0:
                persist(args.unit, new); new = []
                print(f"  [{i}/{len(batch)}] checkpoint saved")
        if new: persist(args.unit, new)
        print("scoring pass complete.")
    elif args.score:
        print("\nnothing to score; all units already in cache.")
    else:
        print("\nDRY RUN (no API). Add --score to spend. Report below uses cached scores only.")

    report(units, cache, args.unit)

if __name__ == "__main__":
    main()
