"""Moral/Procedural/Technical/None classification using the rubric scoring sheet.

For each norm, asks the analysis model to answer the twelve rubric questions
(T1-T4, P1-P4, M1-M4) independently with reasoning. The final classification
is derived programmatically from the answers using the rule:

  any T -> Technical
  else any P -> Procedural
  else all M -> Moral
  else None

Operates on either a deliberation transcript (scores panel norms from the
outcome round) or a baseline directory (scores each baseline's norms).

Usage:
    python3 analyze_classification.py --transcript transcripts/<file>.json
    python3 analyze_classification.py --baselines-dir baselines/ --prompt-id A
    python3 analyze_classification.py --summarize analysis/classification/

Output: analysis/classification/classification_<id>_<timestamp>.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from analysis_shared import (
    title_key,
    ANALYSIS_DIR,
    ANALYSIS_MODEL,
    call_analysis_model,
    extract_final_round_norms,
    flatten_norms,
    load_baselines,
    load_transcript,
    parse_json_response,
    transcript_id_from_path,
    write_output,
)


MORAL_CONCEPTS = (
    # Foundational principles (8)
    "dignity, autonomy-as-value, respect, recognition, liberty, rights, "
    "freedom, privacy, "
    # Reciprocity / relational (10)
    "reciprocity, mutuality, solidarity, trust, trustworthiness, fidelity, "
    "promise-keeping, loyalty, gratitude, sincerity, "
    # Justice family (10)
    "justice, fairness, equity, fair burden-sharing, equal consideration, "
    "proportionality, deserving, impartiality, due process, voice, "
    # Harm / welfare family (10)
    "harm-avoidance, non-violence, beneficence, welfare, wellbeing, "
    "vulnerability, need, sufficiency, precaution, accessibility, "
    # Care / virtue family (16)
    "care, compassion, empathy, integrity, humility, courage, patience, "
    "tolerance, open-mindedness, temperance, prudence, magnanimity, "
    "charity, loving-kindness, equanimity, hospitality, "
    # Truthfulness family (5)
    "honesty, truthfulness, non-deception, transparency, publicity, "
    # Accountability family (5)
    "accountability, responsibility, consent, repair, reconciliation, "
    # Civic / political (3)
    "non-domination, inclusion, common good, "
    # Restorative (2)
    "forgiveness, mercy, "
    # Stewardship, authenticity, love, AI-specific (4)
    "stewardship, authenticity, love, corrigibility"
)


CLASSIFY_PROMPT = """You are scoring a norm using a strict rubric. The norm may have multiple kinds of content simultaneously: it may name moral concepts directly (explicit moral), it may use procedural mechanism to express a moral commitment (implicit moral), it may specify procedure for its own sake (procedural), and it may reference technical infrastructure (technical). Answer all sixteen questions below independently. Do not let your answer to one question bias your answer to another.

The norm is:
"{norm}"

STEP 1. EXPLICIT MORAL CHECKS
The norm names a moral concept directly and treats it as foundational.
E1. Does the norm name, refer to, or directly invoke one or more concepts from this list: {moral_concepts}? Count direct uses of the concept name or a clear synonym. Distant relatives or partial fragments do not count.
E2. Does the norm articulate what is owed to whom, or what counts as wronging whom, in terms that connect to a concept on the list?
E3. Does the norm's stated or implied reason appeal to a concept on the list (rather than only to convenience, coordination, efficiency, or convention)?
E4. Would the named moral concept carry over meaningfully to other cooperative situations, not only to this specific deliberation context?

STEP 2. IMPLICIT MORAL CHECKS
The norm specifies a procedure or structure whose form expresses a moral commitment, without naming the concept directly.
I1. Does the norm specify a procedural mechanism, structural feature, or rule of operation? (If no procedural specification exists, skip the implicit check; this question fails.)
I2. Does the procedural specification express or operationalize a recognizable moral commitment from this same list: {moral_concepts}? Be strict. The commitment must be specifically identifiable, not just "some general commitment to good process." Name the commitment in your reasoning.
I3. Would a plausible alternative procedural choice express a different commitment, or none? This is the counterfactual test for whether the procedural choice is meaningfully expressive.
I4. Can the procedural choice be defended by appeal to the identified moral commitment, rather than only by appeal to convenience, coordination, efficiency, or convention? If the natural defense is "we use this format because it's the standard" or "we meet this often because more would be wasteful," answer N.

STEP 3. PROCEDURAL CHECKS
The norm specifies procedural mechanism, regardless of whether moral commitments are expressed through it.
P1. Does the norm specify a process, rule of operation, mechanism, or structure?
P2. Does the norm specify who acts, what roles exist, or what timing/sequencing applies to the procedural elements?
P3. Does the norm specify what triggers the procedure or when it applies?
P4. Could an outsider execute the procedure as written, without further moral interpretation, by following the specified steps?

STEP 4. TECHNICAL CHECKS
The norm references technical infrastructure or implementation.
T1. Does the norm reference specific technical infrastructure (logs, ledgers, APIs, data formats, cryptographic operations, specific systems)?
T2. Does the norm specify implementation details that require technical capability to execute?
T3. Does the norm presuppose technical capabilities (persistent storage, cryptographic operations, network protocols, specific data structures)?
T4. Would the norm be unimplementable without the technical infrastructure it references?

For each question, answer Y or N and give one sentence of reasoning. Be honest about borderline cases. If a question is borderline, mark N rather than Y. If the norm is compound (contains two or more separable claims), score the dominant claim and note this in a "decomposition_note" field.

Respond in JSON only, no preamble, no code fences:
{{
  "E1": {{"answer": "Y", "reasoning": "..."}},
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
  "T1": {{"answer": "Y", "reasoning": "..."}},
  "T2": {{"answer": "N", "reasoning": "..."}},
  "T3": {{"answer": "N", "reasoning": "..."}},
  "T4": {{"answer": "N", "reasoning": "..."}},
  "decomposition_note": ""
}}"""


def _y(answers, key):
    """Check if a criterion answer starts with Y."""
    a = answers.get(key, {}).get("answer", "")
    return str(a).strip().upper().startswith("Y")


def _fires_3of4_with_essential(answers, criteria, essential):
    """Returns True iff at least 3 of 4 criteria fire AND the essential one fires."""
    try:
        fires = sum(1 for k in criteria if _y(answers, k))
        return fires >= 3 and _y(answers, essential)
    except Exception:
        return False


def derive_explicit_moral_label(answers):
    """True iff explicit moral content is present.
    Threshold: 3 of E1-E4 fire AND E1 (names a concept) fires.
    """
    return _fires_3of4_with_essential(answers, ("E1", "E2", "E3", "E4"), "E1")


def derive_implicit_moral_label(answers):
    """True iff implicit moral content is present.
    Threshold: 3 of I1-I4 fire AND I2 (can specifically name the commitment) fires.
    """
    return _fires_3of4_with_essential(answers, ("I1", "I2", "I3", "I4"), "I2")


def derive_moral_label(answers):
    """True iff either explicit or implicit moral content is present."""
    return derive_explicit_moral_label(answers) or derive_implicit_moral_label(answers)


def derive_procedural_label(answers):
    """True iff procedural specification is present.
    Threshold: 3 of P1-P4 fire AND P1 (specifies process/mechanism) fires.
    """
    return _fires_3of4_with_essential(answers, ("P1", "P2", "P3", "P4"), "P1")


def derive_technical_label(answers):
    """True iff technical infrastructure is referenced.
    Threshold: 3 of T1-T4 fire AND T1 (references infrastructure) fires.
    """
    return _fires_3of4_with_essential(answers, ("T1", "T2", "T3", "T4"), "T1")


def derive_moral_present(answers):
    """Back-compat alias for derive_moral_label."""
    return derive_moral_label(answers)


def derive_classification(answers):
    """Priority-rule classification for back-compat: Technical > Procedural > Moral > None.
    Use the multi-label flags (explicit_moral, implicit_moral, etc.) for new analyses.
    """
    try:
        if derive_technical_label(answers):
            return "Technical"
        if derive_procedural_label(answers):
            return "Procedural"
        if derive_moral_label(answers):
            return "Moral"
        return "None"
    except Exception:
        return "Invalid"


def classify_norm(norm):
    """Single LLM call. Returns dict with scoring + derived classification."""
    prompt = CLASSIFY_PROMPT.format(norm=norm, moral_concepts=MORAL_CONCEPTS)
    raw = call_analysis_model(prompt)
    try:
        answers = parse_json_response(raw)
    except Exception as e:
        return {"error": f"parse failed: {e}", "raw": raw,
                "classification": "Invalid", "moral_present": False}
    classification = derive_classification(answers)
    explicit_moral_label = derive_explicit_moral_label(answers)
    implicit_moral_label = derive_implicit_moral_label(answers)
    moral_label = derive_moral_label(answers)
    procedural_label = derive_procedural_label(answers)
    technical_label = derive_technical_label(answers)
    return {
        "answers": answers,
        "classification": classification,
        "explicit_moral_label": explicit_moral_label,
        "implicit_moral_label": implicit_moral_label,
        "moral_label": moral_label,
        "moral_present": moral_label,  # back-compat alias
        "procedural_label": procedural_label,
        "technical_label": technical_label,
        "decomposition_note": answers.get("decomposition_note", ""),
    }


def summarize_results(results):
    """Compute counts by classification AND multi-label flags.

    Under the new methodology:
    - explicit_moral: % of norms with E1-E4 fires (names moral concept directly)
    - implicit_moral: % of norms with I1-I4 fires (procedure expresses moral commitment)
    - moral (any): % with either explicit or implicit moral content
    - procedural: % with P1-P4 fires (procedural specification)
    - technical: % with T1-T4 fires (technical infrastructure)
    Labels are independent; a norm can have multiple labels.
    """
    counts = Counter()
    explicit_moral_count = 0
    implicit_moral_count = 0
    both_moral_count = 0
    moral_label_count = 0
    procedural_label_count = 0
    technical_label_count = 0
    total = 0
    for r in results:
        scoring = r.get("scoring")
        if not isinstance(scoring, dict):
            continue
        counts[scoring.get("classification", "Invalid")] += 1
        answers = scoring.get("answers", {})
        em = bool(scoring.get("explicit_moral_label", derive_explicit_moral_label(answers)))
        im = bool(scoring.get("implicit_moral_label", derive_implicit_moral_label(answers)))
        ml = em or im
        pl = bool(scoring.get("procedural_label", derive_procedural_label(answers)))
        tl = bool(scoring.get("technical_label", derive_technical_label(answers)))
        if em: explicit_moral_count += 1
        if im: implicit_moral_count += 1
        if em and im: both_moral_count += 1
        if ml: moral_label_count += 1
        if pl: procedural_label_count += 1
        if tl: technical_label_count += 1
        total += 1
    pct = {k: round(100 * v / total, 1) if total else 0.0 for k, v in counts.items()}
    def p(n):
        return round(100 * n / total, 1) if total else 0.0
    return {
        "counts": dict(counts),
        "percentages": pct,
        "explicit_moral_count": explicit_moral_count,
        "explicit_moral_percentage": p(explicit_moral_count),
        "implicit_moral_count": implicit_moral_count,
        "implicit_moral_percentage": p(implicit_moral_count),
        "both_moral_count": both_moral_count,
        "both_moral_percentage": p(both_moral_count),
        "moral_label_count": moral_label_count,
        "moral_label_percentage": p(moral_label_count),
        "moral_present_count": moral_label_count,  # back-compat
        "moral_present_percentage": p(moral_label_count),  # back-compat
        "procedural_label_count": procedural_label_count,
        "procedural_label_percentage": p(procedural_label_count),
        "technical_label_count": technical_label_count,
        "technical_label_percentage": p(technical_label_count),
        "total": total,
    }
def load_scored_cache(cache_dir="analysis/classification"):
    """Scan prior classification files and build a dict mapping norm text -> scoring.
    Used to skip re-classifying norms whose exact text has been scored before.
    """
    cache = {}
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return cache
    for f in sorted(cache_path.glob("classification_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data.get("results", []):
            norm = (r.get("norm") or "").strip()
            if not norm or norm in cache:
                continue
            scoring = r.get("scoring", {})
            if isinstance(scoring, dict) and "answers" in scoring and "classification" in scoring:
                cache[norm] = scoring
    return cache


def classify_with_cache(norm, cache):
    """Return scoring for norm, using cache if available. Mutates cache on miss."""
    stripped = norm.strip()
    if stripped in cache:
        return cache[stripped], True
    scoring = classify_norm(norm)
    if isinstance(scoring, dict) and "answers" in scoring:
        cache[stripped] = scoring
    return scoring, False


def score_transcript(transcript_path):
    """Score panel norms from a deliberation transcript's outcome round."""
    transcript = load_transcript(transcript_path)
    per_agent = extract_final_round_norms(transcript)
    norms = flatten_norms(per_agent)

    if not norms:
        return None

    cache = load_scored_cache()
    results = []
    hits = 0
    misses = 0
    for i, item in enumerate(norms, 1):
        scoring, hit = classify_with_cache(item["norm"], cache)
        if hit:
            hits += 1
            print(f"[{i}/{len(norms)}] [cache] {item['norm'][:80]}...")
        else:
            misses += 1
            print(f"[{i}/{len(norms)}] {item['norm'][:80]}...")
        results.append({
            "norm_index": i,
            "norm": item["norm"],
            "source_agents": item["source_agents"],
            "scoring": scoring,
        })
    print(f"Cache: {hits} hits, {misses} misses")

    return {
        "source_type": "transcript",
        "source": str(transcript_path),
        "analysis_model": ANALYSIS_MODEL,
        "norms_scored": len(results),
        "summary": summarize_results(results),
        "results": results,
    }


def score_baselines(baselines_dir, prompt_id):
    """Score every norm in every baseline for the given prompt."""
    baselines = load_baselines(baselines_dir, prompt_id)
    if not baselines:
        return None

    results = []
    norm_index = 0
    for b in baselines:
        for n in b["norms"]:
            norm_index += 1
            print(f"[{norm_index}] [{b['agent']} run {b['run']}] {n[:70]}...")
            scoring = classify_norm(n)
            results.append({
                "norm_index": norm_index,
                "norm": n,
                "baseline_agent": b["agent"],
                "baseline_run": b["run"],
                "baseline_source_file": b["source_file"],
                "scoring": scoring,
            })

    return {
        "source_type": "baselines",
        "baselines_dir": str(baselines_dir),
        "prompt_id": prompt_id,
        "analysis_model": ANALYSIS_MODEL,
        "norms_scored": len(results),
        "baselines_used": [b["source_file"] for b in baselines],
        "summary": summarize_results(results),
        "results": results,
    }


def summarize_directory(directory):
    """Aggregate every classification_*.json file. Shows multi-label flags
    (MoralL, ProcL, TechL) alongside the primary classification."""
    directory = Path(directory)
    files = sorted(directory.glob("classification_*.json"))
    if not files:
        print(f"No classification files found in {directory}")
        return

    print(f"\nFound {len(files)} classification files\n")
    rows = []
    for f in files:
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
            continue
        results = data.get("results", [])
        groups = dedup_classification_results(results) if results else []

        from collections import Counter as _C
        deduped_classes = _C(g["classification"] for g in groups)
        deduped_total = len(groups)
        explicit_moral_n = sum(1 for g in groups if g.get("explicit_moral_label", False))
        implicit_moral_n = sum(1 for g in groups if g.get("implicit_moral_label", False))
        moral_label_n = sum(1 for g in groups if g.get("moral_label", g.get("moral_present", False)))
        procedural_label_n = sum(1 for g in groups if g.get("procedural_label", False))
        technical_label_n = sum(1 for g in groups if g.get("technical_label", False))

        s = data.get("summary", {})
        total = s.get("total", 0)

        dedup_pct = {k: round(100 * v / deduped_total, 1) if deduped_total else 0.0
                     for k, v in deduped_classes.items()}
        rows.append({
            "source_type": data.get("source_type", "?"),
            "source": data.get("source") or data.get("prompt_id", "?"),
            "n_raw": total,
            "n_dedup": deduped_total,
            "PrimMoral": dedup_pct.get("Moral", 0.0),
            "ExplM": round(100 * explicit_moral_n / deduped_total, 1) if deduped_total else 0.0,
            "ImplM": round(100 * implicit_moral_n / deduped_total, 1) if deduped_total else 0.0,
            "MoralL": round(100 * moral_label_n / deduped_total, 1) if deduped_total else 0.0,
            "ProcL": round(100 * procedural_label_n / deduped_total, 1) if deduped_total else 0.0,
            "TechL": round(100 * technical_label_n / deduped_total, 1) if deduped_total else 0.0,
            "Non": dedup_pct.get("None", 0.0),
        })

    header = f"{'Source':<50} {'Nraw':>5} {'Ndup':>5} {'PrimMor':>8} {'ExplM':>6} {'ImplM':>6} {'MoralL':>7} {'ProcL':>6} {'TechL':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        label = f"{r['source_type']}: {Path(r['source']).name if r['source_type']=='transcript' else r['source']}"
        if len(label) > 50:
            label = label[:47] + "..."
        print(f"{label:<50} {r['n_raw']:>5} {r['n_dedup']:>5} {r['PrimMoral']:>8.1f} "
              f"{r['ExplM']:>6.1f} {r['ImplM']:>6.1f} {r['MoralL']:>7.1f} {r['ProcL']:>6.1f} {r['TechL']:>6.1f}")
    print()
    print("Nraw   = raw norm count (parsed numbered items)")
    print("Ndup   = deduped norm groups")
    print("PrimMor = Priority classification (back-compat)")
    print("ExplM  = Explicit moral: % with E1-E4 fires (norm NAMES a moral concept)")
    print("ImplM  = Implicit moral: % with I1-I4 fires (procedure EXPRESSES a moral commitment)")
    print("MoralL = Either explicit OR implicit moral content present")
    print("ProcL  = Procedural: % with P1-P4 fires (3 of 4, P1 essential)")
    print("TechL  = Technical: % with T1-T4 fires (3 of 4, T1 essential)")
    print("Labels are independent; a norm can have multiple labels.")

def dedup_classification_results(results):
    """Group classification results by title_key. Returns list of groups with
    representative norm text, source indices, majority classification, and
    'any-variant' multi-label flags (moral, procedural, technical)."""
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for r in results:
        idx = r.get("norm_index")
        norm = r.get("norm", "")
        scoring = r.get("scoring", {})
        cls = scoring.get("classification", "Invalid")
        answers = scoring.get("answers", {})
        em = bool(scoring.get("explicit_moral_label", derive_explicit_moral_label(answers)))
        im = bool(scoring.get("implicit_moral_label", derive_implicit_moral_label(answers)))
        ml = em or im
        pl = bool(scoring.get("procedural_label", derive_procedural_label(answers)))
        tl = bool(scoring.get("technical_label", derive_technical_label(answers)))
        key = title_key(norm)
        if not key or len(key) < 4:
            key = norm.lower()[:60]
        groups[key].append({
            "norm_index": idx,
            "norm": norm,
            "classification": cls,
            "explicit_moral_label": em,
            "implicit_moral_label": im,
            "moral_label": ml,
            "moral_present": ml,
            "procedural_label": pl,
            "technical_label": tl,
        })

    out = []
    for key, items in groups.items():
        cls_counter = Counter(i["classification"] for i in items)
        majority_cls = cls_counter.most_common(1)[0][0]
        any_em = any(i["explicit_moral_label"] for i in items)
        any_im = any(i["implicit_moral_label"] for i in items)
        any_ml = any(i["moral_label"] for i in items)
        any_pl = any(i["procedural_label"] for i in items)
        any_tl = any(i["technical_label"] for i in items)
        out.append({
            "dedup_key": key,
            "representative_norm": items[0]["norm"],
            "source_indices": [i["norm_index"] for i in items],
            "classification": majority_cls,
            "classification_dispersion": dict(cls_counter),
            "explicit_moral_label": any_em,
            "implicit_moral_label": any_im,
            "moral_label": any_ml,
            "moral_present": any_ml,
            "procedural_label": any_pl,
            "technical_label": any_tl,
            "variants": items,
        })
    return out


def cross_reference(classification_path, overlap_path):
    """Join classification + baseline-overlap on the same transcript."""
    c = json.load(open(classification_path))
    o = json.load(open(overlap_path))

    # Build overlap maps by both norm_index AND title_key. Overlap analyses
    # may run on deduped groups (different indexing than the classification file),
    # so we need title_key as a fallback match.
    ovl_by_idx = {}
    ovl_by_key = {}
    for r in o.get("results", []):
        idx = r.get("norm_index")
        norm_text = r.get("norm", "")
        count = r.get("overlap", {}).get("overlap_count", None)
        ovl_by_idx[idx] = count
        if norm_text:
            key = title_key(norm_text)
            if key:
                prev = ovl_by_key.get(key)
                if prev is None or (count is not None and (prev is None or count > prev)):
                    ovl_by_key[key] = count

    raw_results = c.get("results", [])
    groups = dedup_classification_results(raw_results)

    cls = {}
    for g in groups:
        max_ovl = None
        for idx in g["source_indices"]:
            v = ovl_by_idx.get(idx)
            if v is None:
                continue
            if max_ovl is None or v > max_ovl:
                max_ovl = v
        if max_ovl is None:
            v = ovl_by_key.get(g["dedup_key"])
            if v is not None:
                max_ovl = v
        cls[g["dedup_key"]] = {
            "classification": g["classification"],
            "moral_present": g["moral_present"],
            "norm": g["representative_norm"],
            "variant_count": len(g["variants"]),
            "source_indices": g["source_indices"],
            "max_overlap": max_ovl,
        }

    buckets = {"zero (0)": [], "low (1-2)": [], "high (3+)": []}
    for key, info in cls.items():
        count = info["max_overlap"]
        if count is None:
            continue
        if count == 0:
            buckets["zero (0)"].append((key, info))
        elif count <= 2:
            buckets["low (1-2)"].append((key, info))
        else:
            buckets["high (3+)"].append((key, info))

    print(f"\nRaw norms in classification: {len(raw_results)}")
    print(f"Deduped norm groups:         {len(groups)}")

    print(f"\nClassification: {Path(classification_path).name}")
    print(f"Overlap:        {Path(overlap_path).name}\n")
    print(f"{'Bucket':<14} {'N':>4} {'PrimMor':>8} {'MorPres':>8} {'Proc':>6} {'Tech':>6} {'None':>6}")
    print("-" * 60)
    for name, items in buckets.items():
        n = len(items)
        if n == 0:
            print(f"{name:<14} {n:>4}")
            continue
        prim_moral = sum(1 for _, i in items if i["classification"] == "Moral")
        mor_pres = sum(1 for _, i in items if i["moral_present"])
        proc = sum(1 for _, i in items if i["classification"] == "Procedural")
        tech = sum(1 for _, i in items if i["classification"] == "Technical")
        none = sum(1 for _, i in items if i["classification"] == "None")
        print(f"{name:<14} {n:>4} {prim_moral:>8} {mor_pres:>8} {proc:>6} {tech:>6} {none:>6}")
    print()
    print("Bucket = # of single-agent baselines that also contain this norm")
    print("PrimMor = Primary Moral (rubric strict)")
    print("MorPres = Moral Present (all four M fire regardless of T/P)")

    print("\n=== Zero-overlap (deduped) norm groups with moral content present ===")
    found = False
    for key, info in buckets["zero (0)"]:
        if info["moral_present"]:
            found = True
            v = info["variant_count"]
            v_label = f" ({v} variants)" if v > 1 else ""
            print(f"  [{info['classification']}]{v_label} {info['norm'][:140]}")
    if not found:
        print("  (none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--baselines-dir", default=None)
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--summarize", default=None,
                    help="Path to a directory of classification_*.json files to summarize.")
    ap.add_argument("--cross-reference", default=None,
                    help="Path to a classification file to join with overlap data.")
    ap.add_argument("--overlap", default=None,
                    help="Path to the overlap file matching --cross-reference.")
    args = ap.parse_args()

    if args.cross_reference and args.overlap:
        cross_reference(args.cross_reference, args.overlap)
        return

    if args.summarize:
        summarize_directory(args.summarize)
        return

    if args.transcript:
        payload = score_transcript(args.transcript)
        if payload is None:
            print(f"No norms extracted from {args.transcript}", file=sys.stderr)
            sys.exit(1)
        out_id = transcript_id_from_path(args.transcript)
    elif args.baselines_dir and args.prompt_id:
        payload = score_baselines(args.baselines_dir, args.prompt_id)
        if payload is None:
            print(f"No baselines found in {args.baselines_dir} for prompt {args.prompt_id}",
                  file=sys.stderr)
            sys.exit(1)
        out_id = f"baselines_{args.prompt_id}"
    else:
        ap.error("Provide --transcript OR (--baselines-dir AND --prompt-id) OR --summarize.")

    out = write_output("classification", out_id, payload)
    print(f"\nWrote {out}")
    print(f"Summary: {payload['summary']}")


if __name__ == "__main__":
    main()
