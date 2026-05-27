"""Agent configuration and OpenRouter client for validation runs.

Supports two panel configurations:
  build_panel()        — Six instances of the same model (same-model phase).
  build_mixed_panel()  — Six different models (mixed-model phase), using
                          the pilot lineup minus perturbations.

Temperature 0 across all calls. Reasoning behavior is per-model:
some providers (Gemini, reasoning-tier Qwen/DeepSeek) require reasoning
to be enabled; others accept reasoning:disabled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests


MODEL = "anthropic/claude-sonnet-4.6"
N_AGENTS = 6
MAX_TOKENS = 2500
TEMPERATURE = 0.0
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


MIXED_PANEL_SPEC = [
    {"agent_id": "agent_1", "display_name": "Agent 1",
     "model": "anthropic/claude-sonnet-4.6",        "max_tokens": 2500, "omit_reasoning": False},
    {"agent_id": "agent_2", "display_name": "Agent 2",
     "model": "openai/gpt-5.5",                     "max_tokens": 2500, "omit_reasoning": False},
    {"agent_id": "agent_3", "display_name": "Agent 3",
     "model": "google/gemini-3.1-pro-preview",      "max_tokens": 4000, "omit_reasoning": True},
    {"agent_id": "agent_4", "display_name": "Agent 4",
     "model": "x-ai/grok-4.3",                      "max_tokens": 4000, "omit_reasoning": False},
    {"agent_id": "agent_5", "display_name": "Agent 5",
     "model": "qwen/qwen3-235b-a22b-thinking-2507", "max_tokens": 4000, "omit_reasoning": True},
    {"agent_id": "agent_6", "display_name": "Agent 6",
     "model": "deepseek/deepseek-r1",               "max_tokens": 4000, "omit_reasoning": True},
]


@dataclass
class Agent:
    """A single agent in the panel.

    omit_reasoning: if True, do not send the reasoning:disabled field
        on API calls (required for Gemini and reasoning-tier models).
    """
    agent_id: str
    display_name: str
    model: str = MODEL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    omit_reasoning: bool = False


def build_panel(n: int = N_AGENTS) -> list[Agent]:
    return [Agent(agent_id=f"agent_{i+1}", display_name=f"Agent {i+1}") for i in range(n)]


def build_mixed_panel() -> list[Agent]:
    return [
        Agent(
            agent_id=spec["agent_id"],
            display_name=spec["display_name"],
            model=spec["model"],
            temperature=TEMPERATURE,
            max_tokens=spec["max_tokens"],
            omit_reasoning=spec["omit_reasoning"],
        )
        for spec in MIXED_PANEL_SPEC
    ]


def verify_model_slugs(api_key: str, models: list[str]) -> tuple[list[str], list[str]]:
    """Check each model slug resolves on OpenRouter. Returns (resolved, missing)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    available = {m["id"] for m in data.get("data", [])}
    resolved = [m for m in models if m in available]
    missing = [m for m in models if m not in available]
    return resolved, missing


def call_openrouter(
    api_key, model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    timeout=180, max_retries=3, omit_reasoning=False,
):
    """Send chat completion to OpenRouter.

    By default includes reasoning:disabled. Providers that require reasoning
    on their endpoint (Gemini, reasoning-tier Qwen/DeepSeek) reject this with
    HTTP 400; for those, pass omit_reasoning=True.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not omit_reasoning:
        payload["reasoning"] = {"enabled": False}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed. Last error: {last_err}")


def extract_text(response):
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response shape: {e}\n{response}")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if "reasoning_details" in message or "reasoning" in message:
        return "[no visible answer; reasoning content present but content field empty]"
    return "[empty response]"


def extract_usage(response):
    return response.get("usage", {})
