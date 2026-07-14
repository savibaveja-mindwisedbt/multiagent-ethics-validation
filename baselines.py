"""Single-agent baselines for the validation runs.

When mixed_model=False (default), runs `runs_per_scenario` baselines using the
single default MODEL. When mixed_model=True, runs `runs_per_scenario` baselines
for EACH panel agent in MIXED_PANEL_SPEC, producing a blended baseline pool
that covers all panel models. Filenames encode the agent identifier so the
analyze_* scripts can filter or aggregate by model.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from agents import MODEL, MAX_TOKENS, TEMPERATURE, call_openrouter, extract_text, extract_usage
from prompts import SCENARIOS


def _safe_slug(s: str) -> str:
    """Sanitize a string for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def run_baselines(
    api_key: str,
    scenario_key: str,
    runs_per_scenario: int = 3,
    out_dir: Path = Path("baselines"),
    mixed_model: bool = False,
    capped: bool = False,
) -> list[Path]:
    """Run baselines for one scenario.

    mixed_model=False: runs_per_scenario runs using MODEL.
    mixed_model=True:  runs_per_scenario runs for EACH panel agent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    baseline_prompt = SCENARIOS[scenario_key]["baseline_prompt"]
    if capped:
        # Match the exact count constraint the panel receives in its outcome
        # instruction (see SCENARIO_*_OUTCOME_FORMAT: "between three and seven norms").
        baseline_prompt = baseline_prompt.rstrip() + (
            SCENARIOS[scenario_key].get("list_instruction", "\n\nProvide your norms as a numbered list of between three and seven norms.")
        )

    system_msg = (
        "You are answering a question about how a group of AI agents "
        "should make a collective decision. Give your considered answer."
    )

    if mixed_model:
        from agents import build_mixed_panel
        from orchestrator import _one_call
        panel = build_mixed_panel()
        for agent in panel:
            agent_label = getattr(agent, "display_name", None) or getattr(agent, "agent_id", "agent")
            agent_id = getattr(agent, "agent_id", _safe_slug(agent_label))
            for run_index in range(runs_per_scenario):
                print(f"  [scenario {scenario_key}] {agent_label} ({agent.model}) baseline run {run_index + 1}/{runs_per_scenario}")
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": baseline_prompt},
                ]
                try:
                    text, usage, elapsed = _one_call(
                        api_key=api_key, agent=agent, messages=messages,
                        max_tokens_override=None,
                    )
                except Exception as e:
                    print(f"    ERROR: {e}")
                    text = f"[API error: {e}]"
                    usage = {}
                    elapsed = None
                result = {
                    "scenario_key": scenario_key,
                    "run_index": run_index,
                    "agent_id": agent_id,
                    "display_name": agent_label,
                    "model": agent.model,
                    "temperature": getattr(agent, "temperature", TEMPERATURE),
                    "max_tokens": getattr(agent, "max_tokens", MAX_TOKENS),
                    "text": text if isinstance(text, str) else "[non-string response]",
                    "usage": usage or {},
                    "elapsed_seconds": elapsed,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                slug = _safe_slug(agent_id)
                out_path = out_dir / f"baseline_{scenario_key}_{slug}_run{run_index}_{stamp}.json"
                out_path.write_text(json.dumps({"result": result}, indent=2))
                written.append(out_path)
    else:
        for run_index in range(runs_per_scenario):
            print(f"  [scenario {scenario_key}] baseline run {run_index + 1}/{runs_per_scenario}")
            t0 = time.time()
            response = call_openrouter(
                api_key=api_key,
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": baseline_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.time() - t0
            result = {
                "scenario_key": scenario_key,
                "run_index": run_index,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "text": extract_text(response),
                "usage": extract_usage(response),
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = out_dir / f"baseline_{scenario_key}_run{run_index}_{stamp}.json"
            out_path.write_text(json.dumps({"result": result}, indent=2))
            written.append(out_path)

    return written
