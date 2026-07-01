#!/usr/bin/env python3
"""
Norm / non-norm classifier + de-fusion rescorer for the directedness pipeline.

WHY THIS EXISTS
The directedness table (ED% per cell in score_v2 / norm_filter) is built from
parse_numbered_list + flatten_norms over panel transcripts and baselines. That
parse pipeline lets three kinds of NON-norm text through, where they get scored
as if they were norms and (mostly) land non-directed, deflating panel ED:
  1. bare labels / cross-references  ("O8 - Violation-naming", "CC-3: ...")
  2. endorsement / ratification / meta-commentary  ("I endorse these without
     reservation", "the framework emerged through genuine deliberation")
  3. compressed list-summaries naming several norms without stating any in full.
norm_filter.py's is_structural() (regex + word-count + verb list) catches only
the shortest labels; the 2026-06-29 handoff calls for a real norm/non-norm
classifier instead of "parse_numbered_list + keyword matching". This is it.

WHAT IT DOES
  - CLASSIFY each distinct norm-text unit as norm vs non-norm via an LLM rubric
    (atomic Y/N markers + a documented derived rule), parallel in design to the
    E1-E4 directedness instrument and the SVO S1-S4 instrument. Default judge is
    the high-performance Claude (Opus 4.8); sonnet-4.6 is the validated fallback.
  - For a genuine norm fused with scaffolding, the judge returns clean_norm: the
    norm sentence(s) copied VERBATIM from the passage with scaffolding removed.
    We only trust clean_norm if it is a (normalized) substring of the original;
    otherwise we do NOT score invented text - we keep the original and flag it.
  - RECOMPUTE the ED table: drop non-norms; for unchanged norms reuse the v2
    cache; for de-fused norms rescore the CLEAN text through score_v2.v2_classify
    (sonnet-4.6, same instrument) so the number stays comparable to the rest of
    the v2 table. Anything still unscored is reported as pending, never guessed.

BOUNDARY
  - This classifier does not re-judge directedness. Directedness stays with the
    v2 instrument (E1/E2). Keep the two separate, same as the SVO boundary.
  - The classifier model (Opus) and the directedness model (Sonnet-4.6) are
    deliberately different: classification is a distinct construct, but rescored
    norms MUST use the original directedness model to stay comparable.

USAGE
  python3 classify_norms.py                         # DRY: coverage + cost, no API
  python3 classify_norms.py --prompts H             # one scenario
  python3 classify_norms.py --classify --limit 8    # cheap preview (spends)
  python3 classify_norms.py --classify              # classify all uncached units
  python3 classify_norms.py --classify --rescore    # then de-fuse + rescore clean text
  python3 classify_norms.py --report                # rebuild ED table from caches (no API)
  python3 classify_norms.py --show H                # dump this scenario's non-norm calls
  python3 classify_norms.py --make-sample 60        # blank hand-coding sheet for IRR
  python3 classify_norms.py --validate FILE         # judge vs human agreement + kappa
  python3 classify_norms.py --selftest              # NO API: prove the non-API logic
  python3 classify_norms.py --model anthropic/claude-sonnet-4.6   # override judge
"""
import json, glob, re, sys, argparse, random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import (parse_numbered_list, extract_final_round_norms,
                             flatten_norms, parse_json_response)
import analysis_shared as A_SHARED
import score_v2 as V2

# ----- config -----
CLASS_JUDGE_MODEL = "anthropic/claude-opus-4.8"     # high-performance Claude judge
CLASS_FALLBACK_MODEL = "anthropic/claude-sonnet-4.6"
JUDGE_MAX_TOKENS = 1100
TAG = "CORE"
CLASS_DIR = Path("analysis/norm_class")
SPLIT_DIR = Path("analysis/norm_split")
SOLO_DIR = Path("baselines_core")
ITER_DIR = Path("baselines_solo_iter")
TDIR = Path("transcripts")

# ----- the rubric -----
CLASS_PROMPT = """You are classifying one passage extracted from a multi-agent deliberation. A list parser pulled it out as if it were a single behavioral norm, but the parser also pulls out labels, headings, cross-references, endorsements, and summaries. Your job is to decide whether this passage actually STATES a norm, and if it is a norm fused with non-norm text, to extract just the norm.

A NORM here = a general standard of conduct for the agents: a prescription, prohibition, permission, or constitutive standard (e.g. "agents must flag uncertainty", "padding is not a contribution", "X counts as a contribution"). It governs how agents act, reason, contribute, disagree, or are evaluated.

The passage is:
"{text}"

Answer each marker independently with Y or N and one sentence of reasoning. Do not let one answer bias another. Borderline is N.

N1 STATES_NORM: Does the passage state at least one general norm IN FULL (a complete prescription/prohibition/permission/constitutive standard), such that a reader could follow or apply it without needing the rest of a document? Merely naming a norm by title is N.
N2 BARE_LABEL: Is the passage merely a label, heading, section title, identifier, or cross-reference (e.g. "O8 - Violation-naming", "CC-3: Integrative work is a contribution category", "Group 2 - Evaluation Architecture", "G3. Norm revisability subject to G1"), i.e. a pointer to a norm rather than the norm stated in full?
N3 META_ENDORSEMENT: Does the passage consist of, or is it fused with, ratification or meta-commentary about the deliberation or document itself rather than a norm (e.g. "I endorse these without reservation", "the framework emerged through genuine deliberation", "Binding for the next operational period", "No reservations")?
N4 LIST_SUMMARY: Is the passage a compressed enumeration that names multiple distinct norms without stating any single one in full (e.g. "Core behavioral norms: epistemic honesty; transparent attribution; no strategic silence; ...")?

Then:
clean_norm: If N1 is Y AND the passage is fused with label/endorsement/summary scaffolding (any of N2/N3/N4 is Y), copy out ONLY the norm sentence(s), VERBATIM from the passage, with the scaffolding removed. Do not paraphrase, do not invent, do not add words. If N1 is Y and there is no scaffolding to remove, return the passage unchanged. If N1 is N, return "".

Respond in JSON only, no preamble, no code fences:
{{
  "N1": {{"answer": "Y", "reasoning": "..."}},
  "N2": {{"answer": "N", "reasoning": "..."}},
  "N3": {{"answer": "N", "reasoning": "..."}},
  "N4": {{"answer": "N", "reasoning": "..."}},
  "clean_norm": "..."
}}"""

SPLIT_PROMPT = """You are given one passage extracted as a single norm. Usually it IS a single norm. Occasionally it bundles several genuinely independent norms. Your job is to separate ONLY genuinely independent norms, and otherwise leave the passage as one. Err strongly toward NOT splitting.

An ATOMIC norm is one complete standard of conduct on one subject. A single norm frequently spans multiple clauses or sentences. The following all belong to ONE atomic norm and must NOT be separated from it:
- its procedure or mechanism ("Name violations factually in the record. Flag, allow clarification, then record." is ONE norm: the second sentence is how the first is carried out)
- its rationale or justification ("because...", "this protects...", "...so that...")
- its scope, conditions, exceptions, or qualifications
- its consequence or enforcement ("...; violating this is itself a violation")
- clauses joined by semicolons that elaborate the SAME mechanism or subject (e.g. "flags must cite a specific principle; disputes preserved in record; unfounded flag patterns are discountable" is ONE norm about the flagging mechanism)

Split into separate atomic norms ONLY when the passage contains clearly independent standards on DIFFERENT subjects, such as an explicit enumerated list (1. ... 2. ... 3. ...) where each item is a self-contained norm on its own topic.

Hard rules:
- When in doubt, do NOT split. A single returned norm is the correct default.
- Never split a norm from its own procedure, rationale, scope, exception, or consequence.
- Never split on a semicolon or sentence boundary alone.
- Drop fragments that are NOT norms (cross-references like "Norms 1, 2, and 8 are lexically prior", headings, endorsements). Do not emit them as atoms.
- Copy each atomic norm VERBATIM from the passage. Do not paraphrase, invent, or add words.

The passage is:
"{text}"

Respond in JSON only, no preamble, no code fences:
{{"atomic_norms": ["...", "..."]}}
Return a single-element list if the passage is one norm (the common case)."""

def y(ans, k):
    return str(ans.get(k, {}).get("answer", "N")).strip().upper().startswith("Y")

def _norm_ws(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _alnum(s):
    """Alphanumeric-only, lowercased: ignores punctuation, whitespace, markdown,
    and smart quotes so a correctly-extracted norm that was lightly reformatted
    still verifies as drawn from the passage."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _strip_markup(s):
    """Deterministic cosmetic cleanup of a de-fused norm BEFORE rescoring: removes
    markdown emphasis, inline bold/italic LABEL fragments (e.g. **Scope:**), and
    em/en-dash separators. Removes only markup and label tokens, never sentence
    content. Applied to fused scoring_text only (clean_norm originals are left as
    scored). Does NOT split compound norms - that is a separate step."""
    if not s:
        return s
    t = re.sub(r'\*{1,2}[^*\n]{1,40}?:\*{1,2}', ' ', s)      # **Label:** / *Label:* -> drop
    t = re.sub(r'\*{1,2}([^*\n]+?)\*{1,2}', r'\1', t)        # unwrap remaining emphasis
    t = t.replace('*', ' ')
    t = t.replace('\u2014', ' ').replace('\u2013', ' ')      # em/en dash -> space
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t

def _covered(clean, original):
    """True if `clean` is a faithful de-fusion of `original`: every word in clean
    occurs in original (multiset coverage -> nothing invented) AND clean is strictly
    shorter (real removal happened). Accepts de-fusions that removed an interruption
    in the MIDDLE of a norm, which a contiguous-substring test wrongly rejects."""
    from collections import Counter
    if not clean:
        return False
    ac, ao = _alnum(clean), _alnum(original)
    if ac and ac in ao and ac != ao:
        return True                      # contiguous substring: definitely faithful
    ct = re.findall(r"[a-z0-9]+", clean.lower())
    ot = re.findall(r"[a-z0-9]+", original.lower())
    if not ct or len(ct) >= len(ot):
        return False                     # nothing, or no real removal
    cc, oc = Counter(ct), Counter(ot)
    return all(oc[w] >= n for w, n in cc.items())   # no invented words

def derive(ans, original):
    """Documented, fixed derivation from the four markers + clean_norm.
    Returns (is_norm, category, scoring_text, needs_rescore, flag).
    - is_norm: N1 (states a norm in full).
    - category: clean_norm / fused / non_norm:<subtype>.
    - scoring_text: the text whose directedness should count for this unit.
    - needs_rescore: scoring_text differs from original (a verified de-fusion).
    - flag: 'fusion_unverified' when the judge's clean_norm is not a substring
      of the original (we then refuse to score invented text and keep original).
    """
    n1, n2, n3, n4 = y(ans, "N1"), y(ans, "N2"), y(ans, "N3"), y(ans, "N4")
    if not n1:
        sub = "label" if n2 else "endorsement" if n3 else "summary" if n4 else "other"
        return (False, f"non_norm:{sub}", "", False, "")
    clean = (ans.get("clean_norm") or "").strip()
    fused = n2 or n3 or n4
    if fused:
        # usable only if clean is a faithful de-fusion: no invented content and real
        # removal. _covered tolerates removing an interruption in the middle of a norm.
        if _covered(clean, original):
            cleaned = _strip_markup(clean)
            # markup stripping only removes markup/labels, so coverage must still hold
            if _covered(cleaned, original):
                return (True, "fused", cleaned, True, "")
            return (True, "fused", clean, True, "")   # fallback: keep pre-strip if guard trips
        return (False, "non_norm:fused_undefused", "", False, "fusion_unverified")
    return (True, "clean_norm", original, False, "")

# ----- unit collection (mirrors norm_filter / score_v2 exactly) -----
def cell_texts(p, cell):
    if cell == "solo":
        files = sorted(glob.glob(f"{SOLO_DIR}/baseline_{p}_run*.json"))
        groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in files]
    elif cell == "solo-iter":
        files = sorted(glob.glob(f"{ITER_DIR}/baseline_{p}_iter_run*.json"))
        groups = [parse_numbered_list(json.load(open(f)).get("result", {}).get("text", "")) for f in files]
    else:
        files = sorted(glob.glob(f"{TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{TAG}*.json"))
        groups = [[x["norm"] for x in flatten_norms(extract_final_round_norms(json.load(open(f))))] for f in files]
    seen, out = set(), []
    for g in groups:
        for n in g:
            n = n.strip()
            if n and n not in seen:
                seen.add(n); out.append(n)
    return out

def all_units(prompts, cells):
    """distinct norm text -> set of cells it appears in (classification is per text)."""
    units = {}
    for p in prompts:
        for c in cells:
            for n in cell_texts(p, c):
                units.setdefault(n, set()).add(f"{p}/{c}")
    return units

# ----- classification cache -----
def load_class_cache():
    cache = {}
    if not CLASS_DIR.exists(): return cache
    for f in sorted(CLASS_DIR.glob("norm_class_*.json")):
        try: data = json.load(open(f))
        except Exception: continue
        for r in data.get("results", []):
            n = (r.get("norm") or "").strip()
            if n and "answers" in r.get("classification", {}):
                cache[n] = r["classification"]
    return cache

def _log_parse_fail(text, raw, err):
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    p = CLASS_DIR / "parse_failures.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"norm": text, "error": str(err), "raw": raw}) + "\n")

def persist_class(new_results):
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (CLASS_DIR / f"norm_class_{st}.json").write_text(json.dumps({"results": new_results}, indent=2))

# ----- judge call -----
class JudgeParseError(Exception):
    """Carries the raw model text so a failed unit can be inspected, not just counted."""
    def __init__(self, msg, raw):
        super().__init__(msg); self.raw = raw

def _ask(text, model, strict=False):
    prompt = CLASS_PROMPT.format(text=text)
    if strict:
        prompt += ('\n\nIMPORTANT: return STRICT JSON only. Inside clean_norm, escape every '
                   'double-quote as \\" and replace any newline with a space. Do not include '
                   'markdown, code fences, or any text outside the JSON object.')
    return A_SHARED.call_openrouter(model, [{"role": "user", "content": prompt}],
                                    max_tokens=JUDGE_MAX_TOKENS, temperature=0.0,
                                    omit_reasoning=False)

def _looks_truncated(raw):
    """Heuristic: a complete response ends with a closing brace. If the last
    non-space char is not '}', the JSON was very likely cut off mid-stream."""
    s = (raw or "").strip()
    return (not s.endswith("}")) or (s.count("{") > s.count("}"))

def _ask_big(text, model):
    # same prompt, much larger budget, for long verbatim clean_norm extractions
    prompt = CLASS_PROMPT.format(text=text)
    return A_SHARED.call_openrouter(model, [{"role": "user", "content": prompt}],
                                    max_tokens=max(JUDGE_MAX_TOKENS * 4, 4096),
                                    temperature=0.0, omit_reasoning=False)

def judge(text, model):
    raw = _ask(text, model, strict=False)
    try:
        return _finish(parse_json_response(raw), text)
    except Exception:
        pass
    # truncation path first: if the response was cut off, retry with a bigger budget
    if _looks_truncated(raw):
        raw_big = _ask_big(text, model)
        try:
            return _finish(parse_json_response(raw_big), text)
        except Exception:
            raw = raw_big  # carry the fullest response into the error if all fail
    # formatting path: strict-JSON reminder
    raw2 = _ask(text, model, strict=True)
    try:
        return _finish(parse_json_response(raw2), text)
    except Exception as e:
        raise JudgeParseError(str(e), raw2 or raw)

def _finish(answers, text):
    is_norm, cat, stext, rescore, flag = derive(answers, text)
    return {"answers": answers, "is_norm": is_norm, "category": cat,
            "scoring_text": stext, "needs_rescore": rescore, "flag": flag}

def preflight(model):
    try:
        from keychain import get_openrouter_key
        import agents, requests
        key = get_openrouter_key()
        resolved, _ = agents.verify_model_slugs(key, [model])
        if resolved: return model
        r = requests.get(agents.OPENROUTER_MODELS_URL,
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        avail = sorted(m["id"] for m in r.json().get("data", []) if m["id"].startswith("anthropic/"))
        raise SystemExit(f"\nJudge model '{model}' does not resolve on OpenRouter.\n"
                         f"Available anthropic slugs:\n  " + "\n  ".join(avail) +
                         f"\nRe-run with --model <one above> (validated fallback: {CLASS_FALLBACK_MODEL}).\n")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Preflight could not verify the model slug: {e}\n"
                         f"Fix credentials/network, or pass --model {CLASS_FALLBACK_MODEL}.")

# ----- norm splitting (compound -> atomic) -----
def load_split_cache():
    cache = {}
    if not SPLIT_DIR.exists():
        return cache
    for f in sorted(SPLIT_DIR.glob("norm_split_*.json")):
        try: data = json.load(open(f))
        except Exception: continue
        for r in data.get("results", []):
            n = (r.get("norm") or "").strip()
            atoms = r.get("atomic_norms")
            if n and atoms:
                cache[n] = atoms
    return cache

def persist_split(new_results):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    st = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (SPLIT_DIR / f"norm_split_{st}.json").write_text(json.dumps({"results": new_results}, indent=2))

def split_judge(text, model):
    """Decompose `text` into atomic norms. Each returned atom must be faithfully
    drawn from `text` (coverage guard) and is markup-stripped. Invented atoms are
    dropped; if none survive, returns [text] (treat as atomic)."""
    prompt = SPLIT_PROMPT.format(text=text)
    raw = A_SHARED.call_openrouter(model, [{"role": "user", "content": prompt}],
                                   max_tokens=max(JUDGE_MAX_TOKENS * 2, 2048),
                                   temperature=0.0, omit_reasoning=False)
    try:
        data = parse_json_response(raw)
    except Exception:
        if _looks_truncated(raw):
            raw = A_SHARED.call_openrouter(model, [{"role": "user", "content": prompt}],
                                           max_tokens=max(JUDGE_MAX_TOKENS * 6, 6144),
                                           temperature=0.0, omit_reasoning=False)
        data = parse_json_response(raw)   # may raise -> caller skips/logs
    atoms_raw = data.get("atomic_norms") or []
    atoms = []
    for a in atoms_raw:
        a = _strip_markup((a or "").strip())
        if a and _covered(a, text):       # never keep invented/uncovered atoms
            atoms.append(a)
    if not atoms:
        atoms = [_strip_markup(text)]     # fall back to whole text as one atom
    return atoms

# ----- ED recompute -----
def pct(n, N): return round(100 * n / N) if N else 0

def ed_of(text, v2cache):
    sc = v2cache.get(text.strip())
    if not sc: return None
    return V2.typology(sc["answers"]) == "ED"

def atoms_for(stext, split_cache):
    """Atomic norms to score for a kept item. Without splits (split_cache None or
    no entry) the item is its own single atom, so numbers match the unsplit path."""
    if not split_cache:
        return [stext]
    return split_cache.get(stext.strip(), [stext])

def recompute(prompts, cells, class_cache, v2cache, rescore=False, rescore_model=None,
              split_cache=None):
    """Per-cell stats. Unit of scoring is the atomic norm: without split_cache each
    kept item is one atom (identical to the pre-split behavior); with split_cache a
    kept item expands into its atomic norms, each scored separately."""
    rows = {}
    pending_rescore = []
    new_v2 = []

    # Pre-count atoms that will need a fresh score, so a long rescore is not silent.
    if rescore and rescore_model:
        need = 0
        for p in prompts:
            for c in cells:
                for t in cell_texts(p, c):
                    cl = class_cache.get(t)
                    if not cl or not cl.get("is_norm"): continue
                    st = cl.get("scoring_text") or t
                    for atom in atoms_for(st, split_cache):
                        if ed_of(atom, v2cache) is None:
                            need += 1
        print(f"  rescore: {need} atoms need a v2 score (checkpoint every 25)...")

    scored = 0
    def _checkpoint():
        nonlocal new_v2
        if new_v2:
            V2.persist_v2(new_v2); new_v2 = []

    for p in prompts:
        for c in cells:
            texts = cell_texts(p, c)
            n_raw = len(texts)
            ed_raw = sum(1 for t in texts if ed_of(t, v2cache) is True)
            kept = dropped = fused = unverified = 0
            atoms = pending = ed_clean = 0
            for t in texts:
                cl = class_cache.get(t)
                if cl is None:
                    pending += 1; atoms += 1; continue   # unclassified item ~ one pending atom
                if not cl.get("is_norm"):
                    dropped += 1; continue
                kept += 1
                stext = cl.get("scoring_text") or t
                if cl.get("flag") == "fusion_unverified": unverified += 1
                if cl.get("needs_rescore"): fused += 1
                for atom in atoms_for(stext, split_cache):
                    atoms += 1
                    ed = ed_of(atom, v2cache)
                    if ed is None:
                        if rescore and rescore_model:
                            try:
                                sc = V2.v2_classify(atom)   # sonnet-4.6, v2 instrument
                            except Exception as e:
                                print(f"  rescore FAILED on a norm, stopping rescore: {e}")
                                _checkpoint(); rescore = False
                                pending += 1; continue
                            new_v2.append({"norm": atom, "scoring": {"answers": sc["answers"],
                                           "typology": sc["typology"]},
                                           "source": "classify_norms rescore (atomic)"})
                            v2cache[atom.strip()] = {"answers": sc["answers"]}
                            ed = (sc["typology"] == "ED")
                            scored += 1
                            if scored % 25 == 0:
                                _checkpoint(); print(f"    scored {scored} atoms...")
                        else:
                            pending_rescore.append(atom); pending += 1; continue
                    ed_clean += int(ed)
            rows[f"{p}/{c}"] = dict(n_raw=n_raw, ed_raw=ed_raw, dropped=dropped,
                                    kept=kept, fused=fused, unverified=unverified,
                                    atoms=atoms, pending=pending, ed_clean=ed_clean)
    if new_v2:
        V2.persist_v2(new_v2)
    if scored:
        print(f"  persisted {scored} norm scores to the v2 cache.")
    return rows, pending_rescore

def print_table(rows, prompts, cells, split=False):
    acol = "atoms" if split else "kept"
    print(f"\n{'cell':14}{'n_raw':>6}{'ED%raw':>7}{'drop':>6}{'kept':>6}{acol:>7}{'fused':>6}{'pend':>6}{'ED%clean':>9}")
    for p in prompts:
        eds = {}
        for c in cells:
            r = rows.get(f"{p}/{c}")
            if not r: continue
            unit = r["atoms"]                      # scoring unit = atomic norms
            denom = unit - r["pending"] if (unit - r["pending"]) > 0 else unit
            edc = pct(r["ed_clean"], denom) if denom else 0
            eds[c] = (edc, r["pending"])
            print(f"{p+'/'+c:14}{r['n_raw']:>6}{pct(r['ed_raw'],r['n_raw']):>7}{r['dropped']:>6}"
                  f"{r['kept']:>6}{r['atoms']:>7}{r['fused']:>6}{r['pending']:>6}{edc:>9}"
                  f"{'  <-pending, partial' if r['pending'] else ''}")
        if "solo-iter" in eds and "panel" in eds:
            d = eds["panel"][0] - eds["solo-iter"][0]
            note = "  (PARTIAL: items pending)" if (eds['panel'][1] or eds['solo-iter'][1]) else ""
            print(f"   -> {p}: collaboration {d:+d}  (solo-iter {eds['solo-iter'][0]} -> panel {eds['panel'][0]}){note}\n")

# ----- show / sample / validate -----
def show(prompt, cells, class_cache):
    print(f"=== {prompt}: items classified NON-NORM (eyeball for false drops) ===")
    texts = set()
    for c in cells:
        texts.update(cell_texts(prompt, c))
    shown = 0
    for t in sorted(texts):
        cl = class_cache.get(t)
        if cl and not cl.get("is_norm"):
            print(f"  [{cl.get('category',''):16}] {t[:120]}"); shown += 1
    print(f"({shown} non-norm; {len([t for t in texts if class_cache.get(t)])} of {len(texts)} classified)")
    print(f"\n=== {prompt}: items DE-FUSED (verified clean_norm extracted) ===")
    for t in sorted(texts):
        cl = class_cache.get(t)
        if cl and cl.get("needs_rescore"):
            print(f"  ORIG : {t[:90]}")
            print(f"  CLEAN: {cl.get('scoring_text','')[:90]}\n")

def make_sample(units, n, seed=7):
    rng = random.Random(seed)
    items = list(units.keys()); rng.shuffle(items)
    sample = [{"norm": t, "human_is_norm": "", "note": ""} for t in items[:n]]
    out = Path("norm_class_handcoded_BLANK.json")
    out.write_text(json.dumps({"instructions":
        "Set human_is_norm to Y (passage states a norm in full) or N (label, "
        "endorsement/meta, list-summary, or other non-norm). Save and pass to --validate.",
        "items": sample}, indent=2))
    print(f"wrote {len(sample)} items to {out}; fill human_is_norm then --validate {out.name}")

def validate(path):
    data = json.load(open(path))
    items = data.get("items", data if isinstance(data, list) else [])
    cache = load_class_cache()
    tp = tn = fp = fn = skipped = 0
    for it in items:
        h = str(it.get("human_is_norm", "")).strip().upper()
        cl = cache.get((it.get("norm") or "").strip())
        if h not in ("Y", "N") or not cl: skipped += 1; continue
        mh, mm = h == "Y", bool(cl.get("is_norm"))
        if mh and mm: tp += 1
        elif mh and not mm: fn += 1
        elif (not mh) and mm: fp += 1
        else: tn += 1
    N = tp + tn + fp + fn
    if not N:
        print(f"no overlap between hand labels and classification cache (skipped {skipped}). Run --classify first."); return
    po = (tp + tn) / N
    pyh = (tp + fn) / N; pym = (tp + fp) / N
    pe = pyh * pym + (1 - pyh) * (1 - pym)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    print(f"\n=== classifier vs human (n={N}, skipped {skipped}) ===")
    print(f"  agreement={po:.0%}  precision(norm)={tp/(tp+fp) if tp+fp else 0:.0%}  "
          f"recall(norm)={tp/(tp+fn) if tp+fn else 0:.0%}  kappa={kappa:.2f}")
    print(f"  (TP {tp}  TN {tn}  FP {fp}  FN {fn})  kappa>=0.6 substantial, >=0.8 near-human.")

# ----- no-API selftest -----
def selftest():
    print("SELFTEST (no API): derivation rule, substring guard, recompute math.\n")
    ok = True
    def A(n1,n2,n3,n4,clean=""):
        mk=lambda v:{"answer":"Y" if v else "N","reasoning":""}
        return {"N1":mk(n1),"N2":mk(n2),"N3":mk(n3),"N4":mk(n4),"clean_norm":clean}
    cases = [
        # (answers, original, exp_is_norm, exp_cat, exp_rescore, exp_flag)
        (A(0,1,0,0), "O8 - Violation-naming", False, "non_norm:label", False, ""),
        (A(0,0,1,0), "I endorse these without reservation.", False, "non_norm:endorsement", False, ""),
        (A(0,0,0,1), "Core norms: honesty; attribution; calibration.", False, "non_norm:summary", False, ""),
        (A(1,0,0,0), "Agents must flag uncertainty when it bears on a claim.", True, "clean_norm", False, ""),
        # fused, verified substring de-fusion:
        (A(1,0,1,0, clean="Agents must flag uncertainty."),
         "Agents must flag uncertainty. I endorse these without reservation.",
         True, "fused", True, ""),
        # reformatted extraction (punctuation/markdown differs) still verifies:
        (A(1,0,1,0, clean="agents must flag uncertainty"),
         "**Agents must flag uncertainty.** I endorse these.",
         True, "fused", True, ""),
        # fused, but clean_norm NOT drawn from passage -> DROP (never keep the
        # contaminated original, never score invented text):
        (A(1,0,1,0, clean="Agents shall always be honest in all matters."),
         "Agents must flag uncertainty. I endorse these.",
         False, "non_norm:fused_undefused", False, "fusion_unverified"),
        # fused, clean_norm EMPTY (e.g. truncated at max_tokens) -> DROP:
        (A(1,0,1,0, clean=""),
         "Agents must flag uncertainty. I endorse all ten norms without reservation.",
         False, "non_norm:fused_undefused", False, "fusion_unverified"),
        # fused, clean_norm == whole passage (no de-fusion happened) -> DROP:
        (A(1,0,1,0, clean="Agents must flag uncertainty. I endorse these."),
         "Agents must flag uncertainty. I endorse these.",
         False, "non_norm:fused_undefused", False, "fusion_unverified"),
        # fused, de-fusion removed an interruption MID-norm (words non-contiguous) -> ACCEPT:
        (A(1,0,1,0, clean="Allocation is earned by positive contributions; absence of violations is not a contribution."),
         "Allocation is earned by positive contributions; I endorse without reservation; absence of violations is not a contribution.",
         True, "fused", True, ""),
        # fused with markdown label + em-dash -> ACCEPT, markup stripped from scoring_text:
        (A(1,0,1,0, clean="**Scope:** Agents must disclose interests\u2014without fabrication."),
         "**Scope:** Agents must disclose interests\u2014without fabrication. I endorse this.",
         True, "fused", True, ""),
    ]
    for ans, orig, e_norm, e_cat, e_res, e_flag in cases:
        isn, cat, stext, res, flag = derive(ans, orig)
        good = (isn==e_norm and cat==e_cat and res==e_res and flag==e_flag)
        if not good: ok=False
        print(f"  [{'ok' if good else 'FAIL'}] is_norm={isn} cat={cat} rescore={res} flag={flag or '-'}")
        if res and not _covered(stext, orig):
            print("    FAIL: scoring_text not faithfully covered by original"); ok=False

    # recompute math on synthetic caches (no API): one scenario, panel+solo-iter
    import tempfile
    global CLASS_DIR, TDIR, SOLO_DIR, ITER_DIR
    save = (CLASS_DIR, TDIR, SOLO_DIR, ITER_DIR, V2.V2_DIR, V2.TAG)
    tmp = Path(tempfile.mkdtemp())
    TDIR = tmp/"transcripts"; SOLO_DIR=tmp/"solo"; ITER_DIR=tmp/"iter"
    CLASS_DIR = tmp/"class"; V2.V2_DIR = tmp/"v2"
    for d in (TDIR,SOLO_DIR,ITER_DIR,CLASS_DIR,V2.V2_DIR): d.mkdir(parents=True)
    V2.TAG = "CORE"
    # synthetic Z panel: 3 norms (1 directed norm, 1 impersonal norm, 1 label non-norm)
    panel = {"transcript":[{"round_index":9,"agent_id":"a1","turn_index":0,
        "phase":"outcome","norms":["You must credit others' contributions.",
                                   "Contributions count by work done.",
                                   "O8 - Violation-naming"]}]}
    # extract_final_round_norms needs the project's real format; instead write
    # directly what cell_texts(panel) would parse. Simplest: stub via solo cell.
    (SOLO_DIR/"baseline_Z_run0_x.json").write_text(json.dumps({"result":{"text":
        "1. You must credit others' contributions.\n"
        "2. Contributions count by work done.\n"
        "3. O8 - Violation-naming"}}))
    (ITER_DIR/"baseline_Z_iter_run0_x.json").write_text(json.dumps({"result":{"text":
        "1. You must credit others' contributions.\n"
        "2. O8 - Violation-naming"}}))
    # classification cache: norm1 directed-clean, norm2 impersonal-clean, label dropped
    cls = [
      {"norm":"You must credit others' contributions.","classification":{"answers":A(1,0,0,0),
         "is_norm":True,"category":"clean_norm","scoring_text":"You must credit others' contributions.",
         "needs_rescore":False,"flag":""}},
      {"norm":"Contributions count by work done.","classification":{"answers":A(1,0,0,0),
         "is_norm":True,"category":"clean_norm","scoring_text":"Contributions count by work done.",
         "needs_rescore":False,"flag":""}},
      {"norm":"O8 - Violation-naming","classification":{"answers":A(0,1,0,0),
         "is_norm":False,"category":"non_norm:label","scoring_text":"","needs_rescore":False,"flag":""}},
    ]
    (CLASS_DIR/"norm_class_t.json").write_text(json.dumps({"results":cls}))
    # v2 cache: norm1 ED (E1Y E2Y), norm2 non-ED (E1N E2N)
    def v2ans(e1,e2):
        mk=lambda v:{"answer":"Y" if v else "N","reasoning":""}
        return {"E1":mk(e1),"E2":mk(e2),"I2":mk(0)}
    v2 = [
      {"norm":"You must credit others' contributions.","scoring":{"answers":v2ans(1,1)}},
      {"norm":"Contributions count by work done.","scoring":{"answers":v2ans(0,0)}},
      {"norm":"O8 - Violation-naming","scoring":{"answers":v2ans(0,0)}},
    ]
    (V2.V2_DIR/"classification_v2_t.json").write_text(json.dumps({"results":v2}))
    cc = load_class_cache(); vc = V2.load_v2_cache()
    rows,_ = recompute(["Z"], ["solo","solo-iter"], cc, vc, rescore=False)
    r = rows["Z/solo"]
    # solo raw: 3 norms, ED_raw=1 (33%). drop label. kept 2. ED_clean: 1 of 2 = 50%.
    exp = (r["n_raw"]==3 and r["ed_raw"]==1 and r["dropped"]==1 and r["kept"]==2 and r["ed_clean"]==1)
    print(f"\n  recompute Z/solo: n_raw={r['n_raw']} ED_raw={r['ed_raw']} drop={r['dropped']} "
          f"kept={r['kept']} ED_clean_count={r['ed_clean']}  [{'ok' if exp else 'FAIL'}]")
    print("  expected: n_raw=3 ED_raw=1 drop=1 kept=2 ED_clean_count=1 (50%)")
    if not exp: ok=False

    # --- split-aware recompute: split the impersonal kept norm into 2 atoms ---
    # "Contributions count by work done." -> two atoms; give one ED, one non-ED.
    src_norm = "Contributions count by work done."
    atom_a = "Contributions count by work done."          # non-ED (impersonal)
    atom_b = "You must log each contribution."             # ED (directed)
    split_cache = {src_norm: [atom_a, atom_b]}
    def v2ans2(e1,e2):
        mk=lambda v:{"answer":"Y" if v else "N","reasoning":""}
        return {"E1":mk(e1),"E2":mk(e2),"I2":mk(0)}
    vc[atom_a.strip()] = {"answers": v2ans2(0,0)}          # non-ED
    vc[atom_b.strip()] = {"answers": v2ans2(1,1)}          # ED
    rows2,_ = recompute(["Z"], ["solo"], cc, vc, rescore=False, split_cache=split_cache)
    r2 = rows2["Z/solo"]
    # kept items still 2 ("You must credit..." + "Contributions count..."); the latter
    # expands to 2 atoms, the former stays 1 atom -> atoms = 3.
    # ED atoms: "You must credit..." (ED from earlier vc) + atom_b (ED) = 2 of 3.
    exp2 = (r2["kept"]==2 and r2["atoms"]==3 and r2["ed_clean"]==2 and r2["pending"]==0)
    print(f"  split recompute Z/solo: kept={r2['kept']} atoms={r2['atoms']} ED_clean={r2['ed_clean']} pend={r2['pending']}  [{'ok' if exp2 else 'FAIL'}]")
    print("  expected: kept=2 atoms=3 ED_clean=2 pend=0 (2 of 3 atoms ED = 67%)")
    if not exp2: ok=False

    CLASS_DIR, TDIR, SOLO_DIR, ITER_DIR, V2.V2_DIR, V2.TAG = save
    print("\nSELFTEST PASSED" if ok else "\nSELFTEST HAD FAILURES (see above)")

def rederive():
    """Recompute derived fields from each cached record's raw answers, in place.
    Use after changing derive(); no API. Rewrites cache files atomically."""
    if not CLASS_DIR.exists():
        print("no classification cache."); return
    changed = files = 0
    for f in sorted(CLASS_DIR.glob("norm_class_*.json")):
        data = json.load(open(f)); dirty = False
        for r in data.get("results", []):
            cl = r.get("classification", {})
            ans = cl.get("answers")
            if not ans: continue
            isn, cat, stext, res, flag = derive(ans, (r.get("norm") or "").strip())
            newvals = {"is_norm": isn, "category": cat, "scoring_text": stext,
                       "needs_rescore": res, "flag": flag}
            if any(cl.get(k) != v for k, v in newvals.items()):
                cl.update(newvals); dirty = True; changed += 1
        if dirty:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2)); tmp.replace(f); files += 1
    print(f"re-derived {changed} records across {files} cache files (no API).")

def make_defusion_sample(n, seed=7):
    """Sample fused items (a norm was de-fused from scaffolding) and write a sheet
    pairing the ORIGINAL passage with the extracted CLEAN norm, for human rating.
    Validates de-fusion QUALITY (is the extraction the right text?), which is a
    separate question from is_norm. Draws from the classification cache."""
    cache = load_class_cache()
    fused = [(t, cl) for t, cl in cache.items()
             if cl.get("category") == "fused" and cl.get("needs_rescore")
             and (cl.get("scoring_text") or "").strip()]
    if not fused:
        print("no fused items in cache; classify something first."); return
    rng = random.Random(seed); rng.shuffle(fused)
    pick = fused[:n]
    sample = [{"original": t, "extracted_clean_norm": cl.get("scoring_text", ""),
               "defusion_ok": "", "note": ""} for t, cl in pick]
    out = Path("defusion_handcoded_BLANK.json")
    out.write_text(json.dumps({"instructions":
        "For each item, compare extracted_clean_norm against original. Set defusion_ok "
        "to Y if the extraction faithfully captures the norm with scaffolding (labels, "
        "endorsement/meta, list furniture) removed and NO norm content lost. Set N if it "
        "is over-stripped (lost part of the norm), under-stripped (scaffolding remains), "
        "or otherwise wrong. Use note to say which. Save and pass to --validate-defusion.",
        "available_fused": len(fused), "items": sample}, indent=2))
    print(f"wrote {len(sample)} of {len(fused)} fused items to {out}; "
          f"rate defusion_ok then --validate-defusion {out.name}")

def validate_defusion(path):
    data = json.load(open(path))
    items = data.get("items", data if isinstance(data, list) else [])
    good = bad = skipped = 0; bads = []
    for it in items:
        v = str(it.get("defusion_ok", "")).strip().upper()
        if v == "Y": good += 1
        elif v == "N": bad += 1; bads.append(it)
        else: skipped += 1
    rated = good + bad
    if not rated:
        print(f"no rated items (skipped {skipped}). Fill defusion_ok with Y/N."); return
    print(f"\n=== de-fusion quality (n={rated} rated, {skipped} unrated) ===")
    print(f"  faithful: {good}/{rated} = {good/rated:.0%}   bad: {bad}/{rated} = {bad/rated:.0%}")
    if bads:
        print("  --- items rated BAD (inspect; these contaminate or lose norm content) ---")
        for it in bads:
            print(f"    note: {it.get('note','') or '-'}")
            print(f"    ORIG : {(it.get('original') or '')[:140]}")
            print(f"    CLEAN: {(it.get('extracted_clean_norm') or '')[:140]}\n")
    print("  faithful>=90% is a clean second-stage validation; lower means the de-fusion")
    print("  prompt/rule needs work before trusting cleaned panel numbers.")

def make_split_sample(prompts, cells, class_cache, n, seed=7):
    """Sample kept norms that were split into >1 atom and write a sheet pairing the
    SOURCE norm with the atomic norms it was split into, for human rating."""
    scache = load_split_cache()
    if not scache:
        print("no split cache; run --split first."); return
    # restrict to norms that appear in the requested cells, and that actually split
    relevant = set()
    for p in prompts:
        for c in cells:
            for t in cell_texts(p, c):
                cl = class_cache.get(t)
                if cl and cl.get("is_norm"):
                    relevant.add((cl.get("scoring_text") or t).strip())
    multi = [(t, a) for t, a in scache.items() if t in relevant and len(a) > 1]
    if not multi:
        print("no compound (>1 atom) splits to validate in these cells."); return
    rng = random.Random(seed); rng.shuffle(multi)
    sample = [{"source_norm": t, "atomic_norms": a, "split_ok": "", "note": ""} for t, a in multi[:n]]
    out = Path("split_handcoded_BLANK.json")
    out.write_text(json.dumps({"instructions":
        "Set split_ok=Y if atomic_norms correctly decomposes source_norm into distinct "
        "standalone norms with NO single norm fragmented and NO two distinct norms merged. "
        "Set N if over-split (one norm cut into pieces) or under-split (distinct norms still "
        "joined). Note which. Save and pass to --validate-split.",
        "compound_available": len(multi), "items": sample}, indent=2))
    print(f"wrote {len(sample)} of {len(multi)} compound splits to {out}; rate then --validate-split {out.name}")

def validate_split(path):
    data = json.load(open(path))
    items = data.get("items", data if isinstance(data, list) else [])
    good = bad = skipped = 0; bads = []
    for it in items:
        v = str(it.get("split_ok", "")).strip().upper()
        if v == "Y": good += 1
        elif v == "N": bad += 1; bads.append(it)
        else: skipped += 1
    rated = good + bad
    if not rated:
        print(f"no rated items (skipped {skipped})."); return
    print(f"\n=== split quality (n={rated} rated, {skipped} unrated) ===")
    print(f"  correct: {good}/{rated} = {good/rated:.0%}   bad: {bad}/{rated} = {bad/rated:.0%}")
    for it in bads:
        print(f"  BAD note: {it.get('note','') or '-'}")
        print(f"    SOURCE: {(it.get('source_norm') or '')[:130]}")
        for a in it.get("atomic_norms", []):
            print(f"      atom: {a[:110]}")

# ----- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", action="store_true", help="LLM-classify uncached units (spends)")
    ap.add_argument("--rescore", action="store_true", help="also rescore de-fused clean norms via score_v2 (spends)")
    ap.add_argument("--report", action="store_true", help="rebuild ED table from caches, no API")
    ap.add_argument("--show", default="", metavar="P", help="dump non-norm + de-fused calls for scenario P")
    ap.add_argument("--limit", type=int, default=0, help="cap units classified this run")
    ap.add_argument("--prompts", default="AEGHJK")
    ap.add_argument("--cells", default="solo,solo-iter,panel")
    ap.add_argument("--model", default=CLASS_JUDGE_MODEL, help="classifier judge slug")
    ap.add_argument("--make-sample", type=int, default=0, metavar="N")
    ap.add_argument("--validate", default="", metavar="FILE")
    ap.add_argument("--make-defusion-sample", type=int, default=0, metavar="N",
                    help="sample fused items into a sheet for de-fusion quality rating")
    ap.add_argument("--validate-defusion", default="", metavar="FILE",
                    help="report de-fusion faithfulness from a rated sheet")
    ap.add_argument("--split", action="store_true",
                    help="decompose kept norms into atomic norms via LLM (spends)")
    ap.add_argument("--use-splits", action="store_true",
                    help="report ED with atomic norms as the unit (needs a split cache)")
    ap.add_argument("--make-split-sample", type=int, default=0, metavar="N")
    ap.add_argument("--validate-split", default="", metavar="FILE")
    ap.add_argument("--classify-sample", default="", metavar="FILE",
                    help="classify exactly the norms listed in a hand-coding sheet (cheap, for --validate)")
    ap.add_argument("--rederive", action="store_true",
                    help="recompute derived fields (is_norm/category/scoring_text) from cached answers, no API")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest: selftest(); return
    if args.rederive: rederive(); return
    if args.validate: validate(args.validate); return
    if getattr(args, "validate_defusion", ""): validate_defusion(args.validate_defusion); return
    if getattr(args, "make_defusion_sample", 0):
        make_defusion_sample(args.make_defusion_sample); return
    if getattr(args, "validate_split", ""): validate_split(args.validate_split); return

    prompts = [c for c in args.prompts if c in ("A","E","G","H","J","K")]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    units = all_units(prompts, cells)
    class_cache = load_class_cache()

    if args.make_sample: make_sample(units, args.make_sample); return
    if args.show:
        show(args.show, cells, class_cache); return

    if getattr(args, "classify_sample", ""):
        sheet = json.load(open(args.classify_sample))
        sheet_items = sheet.get("items", sheet if isinstance(sheet, list) else [])
        sample_norms = [(it.get("norm") or "").strip() for it in sheet_items if (it.get("norm") or "").strip()]
        todo_s = [t for t in sample_norms if t not in class_cache]
        print(f"sample units={len(sample_norms)}  already classified={len(sample_norms)-len(todo_s)}  to-classify={len(todo_s)}")
        if not args.classify:
            print("add --classify to spend and classify these sample units."); return
        model = preflight(args.model)
        print(f"\nclassifying {len(todo_s)} sample units with {model} (checkpoint every 10)...")
        new = []
        for i, t in enumerate(todo_s, 1):
            try:
                cl = judge(t, model)
            except JudgeParseError as e:
                _log_parse_fail(t, e.raw, e)
                print(f"  [{i}/{len(todo_s)}] parse-fail, SKIPPED (logged): {e}"); continue
            except Exception as e:
                print(f"  [{i}/{len(todo_s)}] API error, saving progress and stopping: {e}"); break
            rec = {"norm": t, "cells": sorted(units.get(t, [])), "classification": cl}
            class_cache[t] = cl; new.append(rec)
            if len(new) % 10 == 0:
                persist_class(new); new = []; print(f"  [{i}/{len(todo_s)}] checkpoint saved")
        if new: persist_class(new)
        print("sample classification complete. now run: python3 classify_norms.py --validate " + args.classify_sample)
        return

    todo = [t for t in units if t not in class_cache]
    print(f"classifier judge={args.model}   prompts={''.join(prompts)}  cells={cells}")
    print(f"distinct norm-text units={len(units)}  classified={len(units)-len(todo)}  to-classify={len(todo)}")
    # cost hint
    print(f"(each to-classify unit = 1 judge call; --rescore adds 1 score_v2 call per de-fused norm)")

    if args.classify and todo:
        model = preflight(args.model)
        batch = todo[:args.limit] if args.limit else todo
        print(f"\nclassifying {len(batch)} units with {model} (checkpoint every 10)...")
        new = []
        for i, t in enumerate(batch, 1):
            try:
                cl = judge(t, model)
            except JudgeParseError as e:
                _log_parse_fail(t, e.raw, e)
                print(f"  [{i}/{len(batch)}] parse-fail, SKIPPED (logged): {e}"); continue
            except Exception as e:
                print(f"  [{i}/{len(batch)}] API error, saving progress and stopping: {e}"); break
            rec = {"norm": t, "cells": sorted(units[t]), "classification": cl}
            class_cache[t] = cl; new.append(rec)
            if len(new) % 10 == 0:
                persist_class(new); new = []; print(f"  [{i}/{len(batch)}] checkpoint saved")
        if new: persist_class(new)
        print("classification pass complete.")
    elif args.classify:
        print("\nnothing to classify; all units cached.")

    if getattr(args, "make_split_sample", 0):
        make_split_sample(prompts, cells, class_cache, args.make_split_sample); return

    if args.split:
        targets = []; seen_t = set()
        for p in prompts:
            for c in cells:
                for t in cell_texts(p, c):
                    cl = class_cache.get(t)
                    if not cl or not cl.get("is_norm"): continue
                    st = (cl.get("scoring_text") or t).strip()
                    if st and st not in seen_t:
                        seen_t.add(st); targets.append(st)
        scache = load_split_cache()
        todo = [t for t in targets if t not in scache]
        print(f"split: {len(targets)} kept norms, {len(targets)-len(todo)} already split, {len(todo)} to split")
        if todo:
            model = preflight(args.model)
            print(f"splitting {len(todo)} norms with {model} (checkpoint every 10)...")
            new = []; multi = 0
            for i, t in enumerate(todo, 1):
                try:
                    atoms = split_judge(t, model)
                except JudgeParseError as e:
                    _log_parse_fail(t, getattr(e, "raw", ""), e)
                    print(f"  [{i}/{len(todo)}] split parse-fail, SKIPPED (logged): {e}"); continue
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] API error, saving progress and stopping: {e}"); break
                if len(atoms) > 1: multi += 1
                scache[t] = atoms; new.append({"norm": t, "atomic_norms": atoms, "n_atoms": len(atoms)})
                if len(new) % 10 == 0:
                    persist_split(new); new = []; print(f"  [{i}/{len(todo)}] checkpoint saved")
            if new: persist_split(new)
            print(f"split pass complete. {multi} of {len(todo)} norms were compound (>1 atom).")

    if args.report or args.classify or args.rescore:
        v2cache = V2.load_v2_cache()
        scache = load_split_cache() if args.use_splits else None
        if args.use_splits:
            ncomp = sum(1 for v in (scache or {}).values() if len(v) > 1)
            print(f"using split cache: {len(scache)} norms, {ncomp} compound -> atomic-norm scoring unit.")
        rows, pend = recompute(prompts, cells, class_cache, v2cache,
                               rescore=args.rescore,
                               rescore_model=(A_SHARED.ANALYSIS_MODEL if args.rescore else None),
                               split_cache=scache)
        print_table(rows, prompts, cells, split=bool(args.use_splits))
        if pend:
            print(f"\n{len(pend)} de-fused clean norms still need a v2 directedness score.")
            print("run with --rescore to score them through score_v2 (sonnet-4.6, same instrument).")
        print("\nED%raw = current pipeline (contaminated). ED%clean = non-norms dropped,")
        print("de-fused norms rescored on clean text. 'pend' rows are partial until --rescore.")

if __name__ == "__main__":
    main()
