#!/usr/bin/env python3
"""Reconfirm the neutral-prompt (C = partition) emergence finding with one fresh
Claude-solo baseline and one fresh 6xClaude panel run. Writes to separate dirs."""
from pathlib import Path
from datetime import datetime, timezone

from keychain import get_openrouter_key
from agents import build_panel
from baselines import run_baselines
from orchestrator import run_deliberation

api_key = get_openrouter_key()
print(f"Key loaded (starts {api_key[:8]}..., len {len(api_key)})\n")

RECONFIRM_BASELINE_DIR = Path("baselines_C_reconfirm")
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# 1) One Claude-solo baseline on C (single model = Claude, capped to 3-7 norms)
print("="*70)
print("STEP 1: One Claude-solo baseline on C (partition / neutral)")
print("="*70)
baseline_paths = run_baselines(
    api_key=api_key,
    scenario_key="C",
    runs_per_scenario=1,
    out_dir=RECONFIRM_BASELINE_DIR,
    mixed_model=False,   # single Claude model
    capped=True,
)
for p in baseline_paths:
    print(f"  wrote {p}")

# 2) One 6xClaude panel on C (normgen, 5 rounds, rotating leadoff)
print("\n" + "="*70)
print("STEP 2: One 6xClaude panel on C (partition / neutral)")
print("="*70)
panel = build_panel()   # six identical Claude agents
print("Panel composition:")
for a in panel:
    print(f"  {a.display_name}: {a.model}")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_path = TRANSCRIPTS_DIR / f"deliberation_C_normgen_samemodel_rotleadoff_RECONFIRM_{stamp}.json"
run_deliberation(
    api_key=api_key, agents=panel, scenario_key="C",
    normgen=True, rounds=5, out_path=out_path, verbose=True,
)
print(f"\n  wrote {out_path}")

print("\n" + "="*70)
print("NEXT: score both with the current rubric")
print("="*70)
print(f"  python3 analyze_classification.py --baselines-dir {RECONFIRM_BASELINE_DIR} --prompt-id C")
print(f"  python3 analyze_classification.py --transcript {out_path}")
