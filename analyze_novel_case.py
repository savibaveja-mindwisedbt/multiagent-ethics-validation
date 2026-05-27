"""Novel case application analysis. For each norm, constructs 3 cases (clear
application, edge, conflict) and asks each panel agent to apply the norm to
each case.

Usage:
    python analyze_novel_case.py --transcript <path> --prompt-text "..."
    python analyze_novel_case.py --transcript <path> --cases-file cases.json
    python analyze_novel_case.py --transcript <path> --skip-application

--prompt-text passes the original deliberation prompt for case construction.
--cases-file loads pre-edited cases instead of LLM-constructing them
(supporting the "you draft, I edit" workflow).
--skip-application runs only case construction.

Output: analysis/novel_case/novel_case_<id>_<timestamp>.json
"""

import argparse
import json
import sys

from analysis_shared import (
    ANALYSIS_MODEL,
    call_analysis_model,
    extract_final_round_norms,
    flatten_norms,
    get_mixed_panel,
    load_transcript,
    parse_json_response,
    query_panel_agent,
    transcript_id_from_path,
    write_output,
)


CASE_CONSTRUCT_PROMPT = """You are constructing test cases for evaluating whether an AI-produced norm holds up in practice.

The norm is:
"{norm}"

Original deliberation prompt context:
{prompt_text}

Construct exactly 3 cases for testing this norm. Each case should pose a concrete situation, share structural features with the deliberation context but differ in specific configuration, be one to three sentences, and avoid leading the answer.

The three cases must test different aspects:
- Case 1: A clear application. The norm should determine an answer in this case.
- Case 2: An edge case. The norm may or may not apply, and the determination is itself the test.
- Case 3: A conflict. The norm conflicts with another reasonable consideration. The test is how the agent handles the conflict.

Respond in JSON only, no preamble, no code fences:
{{
  "cases": [
    {{"case": "...", "type": "clear application", "aspect": "..."}},
    {{"case": "...", "type": "edge", "aspect": "..."}},
    {{"case": "...", "type": "conflict", "aspect": "..."}}
  ]
}}"""


APPLICATION_PROMPT = """You are evaluating how a norm applies in a specific case.

The norm is:
"{norm}"

The case is:
"{case}"

Question 1. Does this norm apply to this case?
Question 2. If yes, what does the norm require, permit, or forbid in this case?
Question 3. Does the norm give a determinate answer here, or does it leave the matter open?

Respond in JSON only, no preamble, no code fences:
{{
  "applies": true,
  "requires": "what the norm requires in this case; empty string if it does not apply",
  "determinate": true,
  "reasoning": "brief"
}}"""


def construct_cases(norm, prompt_text):
    raw = call_analysis_model(CASE_CONSTRUCT_PROMPT.format(
        norm=norm, prompt_text=prompt_text or "unspecified"
    ))
    try:
        return parse_json_response(raw).get("cases", [])
    except Exception as e:
        return [{"error": f"parse failed: {e}", "raw": raw}]


def apply_case(agent_spec, norm, case):
    prompt = APPLICATION_PROMPT.format(norm=norm, case=case)
    raw = query_panel_agent(agent_spec, prompt)
    try:
        return parse_json_response(raw)
    except Exception as e:
        return {"error": f"parse failed: {e}", "raw": raw}


def load_external_cases(path):
    """Format: {"<norm_index_as_string>": [{"case": "...", "type": "...", "aspect": "..."}, ...]}"""
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--prompt-text", default=None,
                    help="The original deliberation prompt, for case construction.")
    ap.add_argument("--cases-file", default=None,
                    help="Optional JSON file with pre-edited cases keyed by norm index.")
    ap.add_argument("--skip-application", action="store_true")
    args = ap.parse_args()

    transcript = load_transcript(args.transcript)
    per_agent = extract_final_round_norms(transcript)
    norms = flatten_norms(per_agent)

    if not norms:
        print(f"No norms extracted from {args.transcript}", file=sys.stderr)
        sys.exit(1)

    external_cases = load_external_cases(args.cases_file) if args.cases_file else None
    panel = get_mixed_panel() if not args.skip_application else []
    results = []

    for i, item in enumerate(norms, 1):
        print(f"\n[{i}/{len(norms)}] norm: {item['norm'][:80]}...")

        if external_cases is not None and str(i) in external_cases:
            cases = external_cases[str(i)]
            print(f"  using {len(cases)} pre-edited cases from file")
        else:
            print("  constructing cases...")
            cases = construct_cases(item["norm"], args.prompt_text)

        case_records = []
        for j, case in enumerate(cases, 1):
            case_text = case.get("case", "")
            applications = []
            if case_text and not args.skip_application:
                for agent in panel:
                    print(f"    case {j} application: {agent['agent_id']} ({agent['model']})")
                    try:
                        a = apply_case(agent, item["norm"], case_text)
                    except Exception as ex:
                        a = {"error": str(ex)}
                    applications.append({"agent_id": agent["agent_id"], "result": a})
            case_records.append({"case": case, "applications": applications})

        results.append({
            "norm_index": i,
            "norm": item["norm"],
            "source_agents": item["source_agents"],
            "cases": case_records,
        })

    payload = {
        "transcript": args.transcript,
        "prompt_text": args.prompt_text,
        "cases_source": "external_file" if external_cases is not None else "llm_constructed",
        "analysis_model": ANALYSIS_MODEL,
        "application_tested": not args.skip_application,
        "norms_analyzed": len(results),
        "results": results,
    }
    out = write_output("novel_case", transcript_id_from_path(args.transcript), payload)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
