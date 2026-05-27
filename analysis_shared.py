"""Shared utilities for memorization/emergence analysis.

Operates on existing transcripts in transcripts/ and baselines in baselines/.
Outputs to analysis/<analysis_type>/.

ANALYSIS_MODEL is the model used for rephrasing, fingerprinting, case
construction, and overlap checking. Default is anthropic/claude-sonnet-4.6
with reasoning disabled. Change here to swap.

This module makes assumptions about transcript and baseline file structure
(see load_transcript and load_baselines below). If your files differ, edit
those two functions only.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from keychain import get_openrouter_key

ANALYSIS_MODEL = "anthropic/claude-sonnet-4.6"
ANALYSIS_MAX_TOKENS = 2000
ANALYSIS_TEMPERATURE = 0.0

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)


def call_openrouter(model, messages, max_tokens=ANALYSIS_MAX_TOKENS,
                    temperature=ANALYSIS_TEMPERATURE, omit_reasoning=False,
                    retries=2):
    """Single OpenRouter call. Returns content string or raises.

    omit_reasoning=True for models with mandatory-on reasoning (Gemini, Qwen,
    DeepSeek) so the reasoning:disabled flag is NOT sent. For all other models
    omit_reasoning=False and the flag is sent to keep cost down.
    """
    api_key = get_openrouter_key()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not omit_reasoning:
        payload["reasoning"] = {"enabled": False}

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=240,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                raise ValueError(f"Empty content from {model}")
            return content
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries + 1} attempts: {last_err}")


def call_analysis_model(prompt, system=None):
    """Call the analysis model (default Sonnet). Returns content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return call_openrouter(ANALYSIS_MODEL, messages, omit_reasoning=False)


def parse_json_response(content):
    """Parse JSON from an LLM response, tolerating code fences and preamble."""
    content = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1)
    obj_match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
    if obj_match:
        content = obj_match.group(1)
    return json.loads(content)


def load_transcript(path):
    """Load a deliberation transcript. Returns dict.

    EDIT THIS if your transcript structure differs. Current assumption:
      {
        "scenario": "...",
        "mode": "...",
        "condition": "...",
        "turns": [
          {"agent": "Agent 1", "round": 1, "content": "..."},
          ...
        ]
      }
    """
    with open(path) as f:
        return json.load(f)


def load_baselines(baselines_dir, prompt_id):
    """Load single-agent baselines for a given prompt ID.

    Handles flat structure and nested-under-'result' structure. Tries multiple
    field names for response text and agent identifier.
    """
    baselines_dir = Path(baselines_dir)
    results = []
    for path in sorted(baselines_dir.glob(f"baseline_{prompt_id}_*.json")):
        with open(path) as f:
            data = json.load(f)
        inner = data["result"] if isinstance(data.get("result"), dict) else data
        response = (inner.get("text") or inner.get("response")
                    or inner.get("content") or inner.get("output") or "")
        agent = (inner.get("agent") or inner.get("display_name")
                 or inner.get("model") or inner.get("agent_id") or path.stem)
        run = inner.get("run") if inner.get("run") is not None else inner.get("run_index", 0)
        results.append({
            "agent": agent,
            "run": run,
            "response": response,
            "norms": parse_numbered_list(response),
            "source_file": path.name,
        })
    return results


def parse_numbered_list(text):
    """Extract top-level numbered items from text.

    Recognizes plain, markdown-header, and bold-numbered prefixes. Rejects
    sub-list items via two rules: (1) leading whitespace > 1 indicates an
    indented sub-item; (2) numeric reset (current num <= last accepted num)
    indicates a sub-list that resets numbering.
    """
    if not text:
        return []
    item_start_re = re.compile(
        r"^(?:#{1,6}\s+)?(?:\*\*)?(?:Norm\s+)?(\d+)[.):]\s+",
        re.IGNORECASE,
    )
    strip_prefix_re = re.compile(
        r"^(?:#{1,6}\s+)?(?:\*\*)?(?:Norm\s+)?\d+[.):]\s*",
        re.IGNORECASE,
    )
    hrule_re = re.compile(r"^[-*=_]{3,}$")
    header_re = re.compile(r"^#{1,6}\s+")
    bold_end_re = re.compile(r"\*\*$")

    lines = text.split("\n")
    item_starts = []
    last_top_num = 0

    for i, raw_line in enumerate(lines):
        leading = len(raw_line) - len(raw_line.lstrip())
        if leading > 1:
            continue
        stripped = raw_line.strip()
        m = item_start_re.match(stripped)
        if not m:
            continue
        try:
            num = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if num <= last_top_num:
            continue
        item_starts.append(i)
        last_top_num = num

    if not item_starts:
        return []

    norms = []
    for idx, start in enumerate(item_starts):
        end = item_starts[idx + 1] if idx + 1 < len(item_starts) else len(lines)
        block_lines = []
        for j, raw_line in enumerate(lines[start:end]):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if hrule_re.match(stripped):
                continue
            if j == 0:
                clean = strip_prefix_re.sub("", stripped)
                clean = bold_end_re.sub("", clean).strip()
                if clean:
                    block_lines.append(clean)
            else:
                if header_re.match(stripped) and not item_start_re.match(stripped):
                    continue
                block_lines.append(stripped)
        joined = " ".join(block_lines).strip()
        if joined and len(joined) >= 20:
            norms.append(joined)
    return norms
def extract_final_round_norms(transcript):
    """Extract per-agent final-round norm lists from outcome-round turns.

    Supports the project's actual format (top-level 'transcript' list, per-turn
    'text', 'display_name'/'agent_id', 'is_outcome_round', 'round_index') and a
    simpler fallback format ('turns' list with 'content', 'agent', 'round').
    """
    turns = transcript.get("transcript") or transcript.get("turns") or []
    if not turns:
        return {}
    outcome_turns = [t for t in turns if t.get("is_outcome_round") is True]
    if not outcome_turns:
        round_key = "round_index" if turns and "round_index" in turns[0] else "round"
        rounds_present = [t.get(round_key) for t in turns if t.get(round_key) is not None]
        if not rounds_present:
            return {}
        max_round = max(rounds_present)
        outcome_turns = [t for t in turns if t.get(round_key) == max_round]
    out = {}
    for t in outcome_turns:
        agent = t.get("display_name") or t.get("agent_id") or t.get("agent") or "unknown"
        text = t.get("text") or t.get("content") or ""
        norms = parse_numbered_list(text)
        if norms:
            out[agent] = norms
    return out



# ---- Title-based dedup helpers ----
_DEDUP_STOPWORDS = {"and", "or", "the", "a", "an", "of", "for", "to", "in", "on",
                    "with", "by", "via", "from", "as", "at", "is", "are", "be"}


def normalize_title(text):
    """Extract a normalized title from a norm. Strips markdown, takes the
    portion before the first ':' or em-dash, lowercases, removes punctuation.
    """
    if not text:
        return ""
    t = re.sub(r"^[#*\s]+", "", text)
    t = re.sub(r"\*+", "", t)
    sep_found = False
    for sep in [":", " \u2014 ", " - "]:
        if sep in t[:200]:
            t = t.split(sep, 1)[0]
            sep_found = True
            break
    if not sep_found:
        if "." in t[:120]:
            t = t.split(".", 1)[0]
        else:
            t = t[:80]
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_key(text, n_significant=2):
    """Dedup key: first N significant (non-stopword) words of normalized title."""
    norm = normalize_title(text)
    words = [w for w in norm.split() if w not in _DEDUP_STOPWORDS]
    return " ".join(words[:n_significant])


def flatten_norms(per_agent_norms):
    """Flatten per-agent norms with title-based dedup. Items whose first two
    significant title words match are grouped. Variants preserved for inspection."""
    seen = {}
    order = []
    for agent, norms in per_agent_norms.items():
        for n in norms:
            n_stripped = n.strip()
            key = title_key(n_stripped)
            if not key or len(key) < 4:
                key = n_stripped.lower()
            if key not in seen:
                seen[key] = {
                    "norm": n_stripped,
                    "source_agents": [],
                    "variants": [],
                    "dedup_key": key,
                }
                order.append(key)
            seen[key]["source_agents"].append(agent)
            if n_stripped not in seen[key]["variants"]:
                seen[key]["variants"].append(n_stripped)
    return [seen[k] for k in order]
def get_mixed_panel():
    """Load the mixed-model panel spec from agents.py, else use handoff fallback."""
    try:
        from agents import MIXED_PANEL_SPEC  # type: ignore
        return MIXED_PANEL_SPEC
    except Exception:
        return [
            {"agent_id": "Agent 1", "model": "anthropic/claude-sonnet-4.6",
             "max_tokens": 2500, "omit_reasoning": False},
            {"agent_id": "Agent 2", "model": "openai/gpt-5.5",
             "max_tokens": 2500, "omit_reasoning": False},
            {"agent_id": "Agent 3", "model": "google/gemini-3.1-pro-preview",
             "max_tokens": 4000, "omit_reasoning": True},
            {"agent_id": "Agent 4", "model": "x-ai/grok-4.3",
             "max_tokens": 4000, "omit_reasoning": False},
            {"agent_id": "Agent 5", "model": "qwen/qwen3-235b-a22b-thinking-2507",
             "max_tokens": 4000, "omit_reasoning": True},
            {"agent_id": "Agent 6", "model": "deepseek/deepseek-r1",
             "max_tokens": 4000, "omit_reasoning": True},
        ]


def query_panel_agent(agent_spec, prompt, system=None, max_tokens=None):
    """Call a single panel agent. Returns content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return call_openrouter(
        model=agent_spec["model"],
        messages=messages,
        max_tokens=max_tokens or agent_spec.get("max_tokens", 2000),
        temperature=0.0,
        omit_reasoning=agent_spec.get("omit_reasoning", False),
    )


def write_output(analysis_type, transcript_id, data):
    """Write analysis output to analysis/<type>/<type>_<id>_<timestamp>.json."""
    out_dir = ANALYSIS_DIR / analysis_type
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", transcript_id)
    out_path = out_dir / f"{analysis_type}_{safe_id}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def transcript_id_from_path(path):
    """Derive a stable transcript ID from the file path."""
    return Path(path).stem
