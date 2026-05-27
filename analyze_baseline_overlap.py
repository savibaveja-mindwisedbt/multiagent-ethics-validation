"""Baseline overlap analysis. For each norm in the panel deliberation, checks
how many single-agent baselines contain a substantially similar concept.

This is the scrambled-agent control from the original proposal in cheaper form.
High overlap means the panel norm is largely retrieved from training. Low
overlap means the norm is a candidate for emergence (subject to the other
probes).

Usage:
    python analyze_baseline_overlap.py --transcript <path> \
        --baselines-dir baselines/ --prompt-id A

Output: analysis/baseline_overlap/baseline_overlap_<id>_<timestamp>.json
"""

import argparse
import sys

from analysis_shared import (
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


OVERLAP_PROMPT = """You are checking whether a norm produced by a multi-agent deliberation also appears in any of the single-agent baseline responses for the same prompt. Each baseline is one agent answering the same prompt alone, without seeing the others.

The deliberation norm is:
"{norm}"

The single-agent baselines are listed below. Each item shows the agent and a numbered list of the norms that agent produced alone.

{baselines_block}

For each baseline, determine whether it contains a norm or concept that is substantially similar to the deliberation norm. "Substantially similar" means the same underlying claim about what is required, permitted, owed, or forbidden, even if phrased differently or framed in different vocabulary. A baseline that mentions the topic but does not assert the same underlying claim is not a match.

For each match, quote the closest matching baseline text.

Respond in JSON only, no preamble, no code fences:
{{
  "matches": [
    {{"baseline_id": "...", "match": true, "matching_text": "...", "similarity_reasoning": "brief"}},
    {{"baseline_id": "...", "match": false, "matching_text": "", "similarity_reasoning": "brief"}}
  ],
  "overlap_count": 0,
  "total_baselines": 0,
  "overlap_summary": "brief summary of how widely this norm appears in baselines"
}}"""


def format_baselines_block(baselines):
    lines = []
    for b in baselines:
        baseline_id = f"{b['agent']} run {b['run']}"
        norms_text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(b["norms"])) or "  (no parseable numbered norms)"
        lines.append(f"[{baseline_id}]\n{norms_text}")
    return "\n\n".join(lines)


def check_overlap(norm, baselines_block):
    raw = call_analysis_model(OVERLAP_PROMPT.format(norm=norm, baselines_block=baselines_block))
    try:
        return parse_json_response(raw)
    except Exception as e:
        return {"error": f"parse failed: {e}", "raw": raw}



def check_overlap_chunked(norm, baselines, chunk_size=6):
    """Chunk baselines and run separate overlap calls per chunk.
    Avoids JSON-output breakage on large contexts."""
    all_matches = []
    overlap_count = 0
    chunk_errors = []

    for chunk_start in range(0, len(baselines), chunk_size):
        chunk = baselines[chunk_start:chunk_start + chunk_size]
        chunk_block = format_baselines_block(chunk)
        result = check_overlap(norm, chunk_block)

        if isinstance(result, dict) and "error" in result and "matches" not in result:
            chunk_errors.append({
                "chunk_start_index": chunk_start,
                "chunk_baseline_ids": [f"{b['agent']} run {b['run']}" for b in chunk],
                "error": result.get("error", ""),
            })
            for b in chunk:
                all_matches.append({
                    "baseline_id": f"{b['agent']} run {b['run']}",
                    "match": None,
                    "matching_text": "",
                    "similarity_reasoning": "chunk parse failed",
                })
            continue

        chunk_matches = result.get("matches", []) if isinstance(result, dict) else []
        if len(chunk_matches) != len(chunk):
            id_to_match = {m.get("baseline_id"): m for m in chunk_matches}
            for b in chunk:
                bid = f"{b['agent']} run {b['run']}"
                m = id_to_match.get(bid)
                if m is None:
                    all_matches.append({
                        "baseline_id": bid,
                        "match": None,
                        "matching_text": "",
                        "similarity_reasoning": "no entry from LLM",
                    })
                else:
                    all_matches.append(m)
                    if m.get("match"):
                        overlap_count += 1
        else:
            for m in chunk_matches:
                all_matches.append(m)
                if m.get("match"):
                    overlap_count += 1

    return {
        "matches": all_matches,
        "overlap_count": overlap_count,
        "total_baselines": len(baselines),
        "overlap_summary": f"{overlap_count} of {len(baselines)} baselines match"
                           + (f" ({len(chunk_errors)} chunk(s) failed to parse)" if chunk_errors else ""),
        "chunk_errors": chunk_errors if chunk_errors else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--baselines-dir", required=True)
    ap.add_argument("--prompt-id", required=True,
                    help="Prompt identifier matching baseline filenames (e.g. 'A', 'B', 'C').")
    args = ap.parse_args()

    transcript = load_transcript(args.transcript)
    per_agent = extract_final_round_norms(transcript)
    norms = flatten_norms(per_agent)

    if not norms:
        print(f"No norms extracted from {args.transcript}", file=sys.stderr)
        sys.exit(1)

    baselines = load_baselines(args.baselines_dir, args.prompt_id)
    if not baselines:
        print(f"No baselines found in {args.baselines_dir} for prompt {args.prompt_id}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(baselines)} baselines for prompt {args.prompt_id}")
    print(f"Chunking into groups of 6 to keep response context parseable")

    results = []
    for i, item in enumerate(norms, 1):
        print(f"[{i}/{len(norms)}] checking overlap: {item['norm'][:80]}...")
        overlap = check_overlap_chunked(item["norm"], baselines, chunk_size=6)
        results.append({
            "norm_index": i,
            "norm": item["norm"],
            "source_agents": item["source_agents"],
            "overlap": overlap,
        })

    total_baselines = len(baselines)
    summary = {
        "norms_with_zero_overlap": sum(1 for r in results
                                       if isinstance(r["overlap"], dict)
                                       and r["overlap"].get("overlap_count", -1) == 0),
        "norms_with_high_overlap": sum(1 for r in results
                                       if isinstance(r["overlap"], dict)
                                       and r["overlap"].get("overlap_count", 0) >= total_baselines / 2),
        "total_norms": len(results),
        "total_baselines": total_baselines,
    }

    payload = {
        "transcript": args.transcript,
        "baselines_dir": args.baselines_dir,
        "prompt_id": args.prompt_id,
        "baselines_used": [b["source_file"] for b in baselines],
        "analysis_model": ANALYSIS_MODEL,
        "summary": summary,
        "results": results,
    }
    out = write_output("baseline_overlap", transcript_id_from_path(args.transcript), payload)
    print(f"\nWrote {out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
