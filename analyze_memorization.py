"""Memorization analysis. Combines public-web relatedness and training-zone
fingerprinting into a single LLM call per norm.

Usage:
    python analyze_memorization.py --transcript transcripts/<file>.json
    python analyze_memorization.py --transcript <path> --prompt-context "..."

Output: analysis/memorization/memorization_<id>_<timestamp>.json
"""

import argparse
import sys

from analysis_shared import (
    ANALYSIS_MODEL,
    call_analysis_model,
    extract_final_round_norms,
    flatten_norms,
    load_transcript,
    parse_json_response,
    transcript_id_from_path,
    write_output,
)


ZONE_LIST = [
    "Moral philosophy (general)",
    "Deontological ethics",
    "Virtue ethics",
    "Consequentialism / utilitarianism",
    "Care ethics / relational ethics",
    "Cloud computing economics",
    "Distributed systems",
    "Cryptographic governance / blockchain",
    "Legal theory",
    "Contract law",
    "Constitutional / public law",
    "Social-contract tradition",
    "Deliberative democracy theory",
    "Mechanism design",
    "Principal-agent theory",
    "Organizational behavior / management",
    "Decision theory / game theory",
    "AI safety / AI ethics discourse",
    "Professional ethics codes",
    "Other (specify)",
]


PROMPT_TEMPLATE = """You are analyzing a norm produced by a deliberation among AI agents. Assess two things about this norm.

The norm is:
"{norm}"

Context for the deliberation: {context}

Question 1. Public-web relatedness.
Rate how strongly this norm relates to content commonly found on the public web (documentation, articles, papers, blogs, code repositories, public discourse).

Scale:
0 = No specific framework, document, or widely-used phrasing on the public web resembles this norm.
1 = The topic is broadly present in public discourse but no specific named source closely resembles the norm.
2 = A specific identifiable framework, document, theory, or widely-used pattern closely resembles the underlying concept of this norm.
3 = The norm appears to be a near-direct match to a widely-documented framework or phrasing that you can name.

If you rate 2 or 3, you must name the specific source(s) or framework(s). Honesty matters: if you cannot name a specific source, do not rate 2 or 3.

Question 2. Training-zone identification.
Which training zone is the underlying concept most likely drawn from? Pick the closest match from this list:

{zone_list}

You may name up to 2 secondary zones if the norm draws from multiple. Provide 3 pieces of evidence for your primary zone: specific terms or phrases used, structural moves made, or conceptual frames invoked.

Respond in JSON only, no preamble, no code fences, with this structure:
{{
  "public_web_rating": 0,
  "public_web_evidence": "name specific sources or frameworks if rating >= 2, describe the discourse area if rating == 1, empty string if 0",
  "primary_zone": "exact label from list",
  "secondary_zones": [],
  "zone_evidence": ["evidence 1", "evidence 2", "evidence 3"]
}}"""


def analyze_norm(norm, context):
    prompt = PROMPT_TEMPLATE.format(
        norm=norm,
        context=context,
        zone_list="\n".join(f"- {z}" for z in ZONE_LIST),
    )
    raw = call_analysis_model(prompt)
    try:
        return parse_json_response(raw)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": raw}


def build_context(transcript, override):
    if override:
        return override
    parts = []
    for key in ("scenario", "condition", "mode"):
        v = transcript.get(key)
        if v:
            parts.append(f"{key}={v}")
    return ", ".join(parts) if parts else "unspecified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--prompt-context", default=None)
    args = ap.parse_args()

    transcript = load_transcript(args.transcript)
    per_agent = extract_final_round_norms(transcript)
    norms = flatten_norms(per_agent)

    if not norms:
        print(f"No norms extracted from {args.transcript}", file=sys.stderr)
        sys.exit(1)

    context = build_context(transcript, args.prompt_context)
    results = []
    for i, item in enumerate(norms, 1):
        print(f"[{i}/{len(norms)}] analyzing: {item['norm'][:80]}...")
        analysis = analyze_norm(item["norm"], context)
        results.append({
            "norm_index": i,
            "norm": item["norm"],
            "source_agents": item["source_agents"],
            "analysis": analysis,
        })

    payload = {
        "transcript": args.transcript,
        "context": context,
        "analysis_model": ANALYSIS_MODEL,
        "norms_analyzed": len(results),
        "results": results,
    }
    out = write_output("memorization", transcript_id_from_path(args.transcript), payload)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
