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
    "reciprocity, proportionality, sufficiency, harm-avoidance, equal "
    "consideration, fair burden-sharing, fairness, vulnerability, consent, "
    "recognition, dignity, autonomy-as-value, liberty, care, compassion, "
    "empathy, solidarity, gratitude, deserving, need, honesty, truthfulness, "
    "non-deception, humility, courage, patience, tolerance, open-mindedness, "
    "loyalty, corrigibility, accountability, responsibility, beneficence, "
    "fidelity, promise-keeping, integrity, justice, rights, trust, "
    "trustworthiness, welfare, wellbeing, respect"
)


CLASSIFY_PROMPT = """You are scoring a norm using a strict rubric that classifies it as Moral, Procedural, Technical, or None. Answer all twelve questions below independently. Do not let your answer to one question bias your answer to another. The final classification is derived from your answers by a rule that you do not need to apply yourself.

The norm is:
"{norm}"

STEP 1. TECHNICAL CHECKS
T1. Does the norm name specific engineering or infrastructure (cryptography, ledgers, atomic operations, classification schemes drawn from technology)?
T2. Does it specify how a procedure is built or executed in code or infrastructure (phrases like "executed via," "enforced by," "logged in")?
T3. Does it justify itself by engineering properties (verifiability, auditability, tamper-resistance, atomicity, automation)?
T4. If you removed the named technology, would nothing meaningful remain?

STEP 2. PROCEDURAL CHECKS
P1. Does it specify who decides (supermajority, consensus, elected allocator, rotating panel)?
P2. Does it specify how a decision is made (voting thresholds, ratification, renewal cycles, deliberation steps)?
P3. Does it specify how disagreement is resolved (arbitration, veto, tiebreaker, appeal)?
P4. Is the only justification agreement, coordination, or process legitimacy, rather than value, harm, or relationship?

STEP 3. MORAL CHECKS
M1. Does it invoke at least one moral concept from this list: {moral_concepts}?
M2. Does it say what is owed to whom, or what counts as harming whom (not merely "the group decides")?
M3. Does it give a reason that connects to value, harm, or relationship (a fact, relational claim, or principle, rather than "because we agreed")?
M4. Would it still mean something between any two parties in any cooperative situation?

For each question, answer Y or N and give one sentence of reasoning. Be honest about borderline cases; mark N if you are not sure. If the norm is compound (contains two or more separable claims), score the dominant claim and note this in a "decomposition_note" field.

Respond in JSON only, no preamble, no code fences:
{{
  "T1": {{"answer": "Y", "reasoning": "..."}},
  "T2": {{"answer": "N", "reasoning": "..."}},
  "T3": {{"answer": "N", "reasoning": "..."}},
  "T4": {{"answer": "N", "reasoning": "..."}},
  "P1": {{"answer": "Y", "reasoning": "..."}},
  "P2": {{"answer": "N", "reasoning": "..."}},
  "P3": {{"answer": "N", "reasoning": "..."}},
  "P4": {{"answer": "N", "reasoning": "..."}},
  "M1": {{"answer": "N", "reasoning": "..."}},
  "M2": {{"answer": "N", "reasoning": "..."}},
  "M3": {{"answer": "N", "reasoning": "..."}},
  "M4": {{"answer": "N", "reasoning": "..."}},
  "decomposition_note": ""
}}"""


def derive_classification(answers):
    """Apply the rubric rule to derive the final classification."""
    def y(key):
        a = answers.get(key, {}).get("answer", "")
        return str(a).strip().upper().startswith("Y")

    try:
        if any(y(k) for k in ("T1", "T2", "T3", "T4")):
            return "Technical"
        if any(y(k) for k in ("P1", "P2", "P3", "P4")):
            return "Procedural"
        if all(y(k) for k in ("M1", "M2", "M3", "M4")):
            return "Moral"
        return "None"
    except Exception:
        return "Invalid"



def derive_moral_present(answers):
    """True iff all four Moral criteria fire, regardless of T/P. This is the
    'moral content is present' flag, broader than the rubric's strict Moral
    classification (which requires T and P to be all-No first)."""
    def y(key):
        a = answers.get(key, {}).get("answer", "")
        return str(a).strip().upper().startswith("Y")
    try:
        return all(y(k) for k in ("M1", "M2", "M3", "M4"))
    except Exception:
        return False



def derive_moral_label(answers):
    """True iff all four Moral criteria fire. Same as derive_moral_present
    but named for clarity in the multi-label scheme."""
    def y(key):
        a = answers.get(key, {}).get("answer", "")
        return str(a).strip().upper().startswith("Y")
    try:
        return all(y(k) for k in ("M1", "M2", "M3", "M4"))
    except Exception:
        return False


def derive_procedural_label(answers):
    """True iff any Procedural criterion fires. Does NOT subordinate to Moral
    or Technical. A norm with both moral and procedural content has both labels."""
    def y(key):
        a = answers.get(key, {}).get("answer", "")
        return str(a).strip().upper().startswith("Y")
    try:
        return any(y(k) for k in ("P1", "P2", "P3", "P4"))
    except Exception:
        return False


def derive_technical_label(answers):
    """True iff any Technical criterion fires. Does NOT subordinate to other labels."""
    def y(key):
        a = answers.get(key, {}).get("answer", "")
        return str(a).strip().upper().startswith("Y")
    try:
        return any(y(k) for k in ("T1", "T2", "T3", "T4"))
    except Exception:
        return False


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
    moral_present = derive_moral_present(answers)
    moral_label = derive_moral_label(answers)
    procedural_label = derive_procedural_label(answers)
    technical_label = derive_technical_label(answers)
    return {
        "answers": answers,
        "classification": classification,
        "moral_present": moral_present,
        "moral_label": moral_label,
        "procedural_label": procedural_label,
        "technical_label": technical_label,
        "decomposition_note": answers.get("decomposition_note", ""),
    }


def summarize_results(results):
    """Compute counts by primary classification AND multi-label flags.

    Multi-label labels are independent: a norm can be moral AND procedural AND
    technical simultaneously. Primary classification is kept for back-compat but
    is biased by the priority rule (any-T-or-P-first, then Moral).
    """
    counts = Counter()
    moral_present_count = 0
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
        mp = bool(scoring.get("moral_present", derive_moral_present(answers)))
        ml = bool(scoring.get("moral_label", derive_moral_label(answers)))
        pl = bool(scoring.get("procedural_label", derive_procedural_label(answers)))
        tl = bool(scoring.get("technical_label", derive_technical_label(answers)))
        if mp:
            moral_present_count += 1
        if ml:
            moral_label_count += 1
        if pl:
            procedural_label_count += 1
        if tl:
            technical_label_count += 1
        total += 1
    pct = {k: round(100 * v / total, 1) if total else 0.0 for k, v in counts.items()}
    return {
        "counts": dict(counts),
        "percentages": pct,
        "moral_present_count": moral_present_count,
        "moral_present_percentage": round(100 * moral_present_count / total, 1) if total else 0.0,
        "moral_label_count": moral_label_count,
        "moral_label_percentage": round(100 * moral_label_count / total, 1) if total else 0.0,
        "procedural_label_count": procedural_label_count,
        "procedural_label_percentage": round(100 * procedural_label_count / total, 1) if total else 0.0,
        "technical_label_count": technical_label_count,
        "technical_label_percentage": round(100 * technical_label_count / total, 1) if total else 0.0,
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
        moral_present_n = sum(1 for g in groups if g["moral_present"])
        moral_label_n = sum(1 for g in groups if g.get("moral_label", g["moral_present"]))
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
            "MoralL": round(100 * moral_label_n / deduped_total, 1) if deduped_total else 0.0,
            "ProcL": round(100 * procedural_label_n / deduped_total, 1) if deduped_total else 0.0,
            "TechL": round(100 * technical_label_n / deduped_total, 1) if deduped_total else 0.0,
            "Non": dedup_pct.get("None", 0.0),
        })

    header = f"{'Source':<50} {'Nraw':>5} {'Ndup':>5} {'PrimMor':>8} {'MoralL':>7} {'ProcL':>6} {'TechL':>6} {'None':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        label = f"{r['source_type']}: {Path(r['source']).name if r['source_type']=='transcript' else r['source']}"
        if len(label) > 50:
            label = label[:47] + "..."
        print(f"{label:<50} {r['n_raw']:>5} {r['n_dedup']:>5} {r['PrimMoral']:>8.1f} "
              f"{r['MoralL']:>7.1f} {r['ProcL']:>6.1f} {r['TechL']:>6.1f} {r['Non']:>6.1f}")
    print()
    print("Nraw   = raw norm count (parsed numbered items)")
    print("Ndup   = deduped norm groups")
    print("PrimMor = Primary classification (biased: any T/P fires first)")
    print("MoralL = Multi-label: % with moral content present (M1-M4 all fire)")
    print("ProcL  = Multi-label: % with procedural content present (any P fires)")
    print("TechL  = Multi-label: % with technical content present (any T fires)")
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
        mp = bool(scoring.get("moral_present", derive_moral_present(answers)))
        ml = bool(scoring.get("moral_label", derive_moral_label(answers)))
        pl = bool(scoring.get("procedural_label", derive_procedural_label(answers)))
        tl = bool(scoring.get("technical_label", derive_technical_label(answers)))
        key = title_key(norm)
        if not key or len(key) < 4:
            key = norm.lower()[:60]
        groups[key].append({
            "norm_index": idx,
            "norm": norm,
            "classification": cls,
            "moral_present": mp,
            "moral_label": ml,
            "procedural_label": pl,
            "technical_label": tl,
        })

    out = []
    for key, items in groups.items():
        cls_counter = Counter(i["classification"] for i in items)
        majority_cls = cls_counter.most_common(1)[0][0]
        any_mp = any(i["moral_present"] for i in items)
        any_ml = any(i["moral_label"] for i in items)
        any_pl = any(i["procedural_label"] for i in items)
        any_tl = any(i["technical_label"] for i in items)
        out.append({
            "dedup_key": key,
            "representative_norm": items[0]["norm"],
            "source_indices": [i["norm_index"] for i in items],
            "classification": majority_cls,
            "classification_dispersion": dict(cls_counter),
            "moral_present": any_mp,
            "moral_label": any_ml,
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
