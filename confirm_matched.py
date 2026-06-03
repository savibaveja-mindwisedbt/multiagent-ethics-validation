#!/usr/bin/env python3
"""Matched-protocol confirmation. Verbose ON, ready-checks OFF for speed."""
from pathlib import Path
from datetime import datetime, timezone
from keychain import get_openrouter_key
from agents import build_panel, build_mixed_panel
from baselines import run_baselines
from orchestrator import run_deliberation

api_key = get_openrouter_key()
print(f"Key loaded (len {len(api_key)})\n")
SOLO_DIR = Path("baselines_matched_confirm")
TDIR = Path("transcripts"); TDIR.mkdir(exist_ok=True)
N = 3
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

for scen in ["C", "A"]:
    print("="*70); print(f"SOLO (capped, single Claude) x{N}: scenario {scen}"); print("="*70)
    for p in run_baselines(api_key=api_key, scenario_key=scen, runs_per_scenario=N,
                           out_dir=SOLO_DIR, mixed_model=False, capped=True):
        print(f"  {p}")

six_paths, mix_paths = [], []
print("\n" + "="*70); print(f"6xClaude panels x{N}: scenario C"); print("="*70)
for i in range(N):
    out = TDIR / f"deliberation_C_normgen_samemodel_rotleadoff_MATCHED{i}_{stamp()}.json"
    run_deliberation(api_key=api_key, agents=build_panel(), scenario_key="C",
                     normgen=True, rounds=5, out_path=out, verbose=True,
                     enable_ready_check=False)
    six_paths.append(out); print(f"  DONE {out}")

print("\n" + "="*70); print(f"Mixed (6-model) panels x{N}: scenario C"); print("="*70)
for i in range(N):
    out = TDIR / f"deliberation_C_normgen_mixed_rotleadoff_MATCHED{i}_{stamp()}.json"
    run_deliberation(api_key=api_key, agents=build_mixed_panel(), scenario_key="C",
                     normgen=True, rounds=5, out_path=out, verbose=True,
                     enable_ready_check=False)
    mix_paths.append(out); print(f"  DONE {out}")

print("\n" + "="*70); print("SCORE EVERYTHING:"); print("="*70)
print(f"python3 analyze_classification.py --baselines-dir {SOLO_DIR} --prompt-id C")
print(f"python3 analyze_classification.py --baselines-dir {SOLO_DIR} --prompt-id A")
for p in six_paths + mix_paths:
    print(f"python3 analyze_classification.py --transcript {p}")
print("\nThen: git add -A && git commit -m 'Matched confirmation runs' && git push")
