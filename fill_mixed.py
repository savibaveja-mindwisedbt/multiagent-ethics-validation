#!/usr/bin/env python3
"""One additional standard 6-model mixed panel each for A, B, E, F to reach n=3.
Matched protocol: normgen, 5 rounds, rotating leadoff, ready-checks ON (as existing runs)."""
from pathlib import Path
from datetime import datetime, timezone
from keychain import get_openrouter_key
from agents import build_mixed_panel
from orchestrator import run_deliberation

api_key = get_openrouter_key()
print(f"Key loaded (len {len(api_key)})\n")
TDIR = Path("transcripts"); TDIR.mkdir(exist_ok=True)
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

paths = []
for scen in ["A", "B", "E", "F"]:
    panel = build_mixed_panel()
    models = "+".join(a.model.split("/")[-1].split("-")[0] for a in panel)
    print("="*68)
    print(f"Mixed panel (standard 6-model) for {scen}: {models}")
    print("="*68)
    out = TDIR / f"deliberation_{scen}_normgen_mixed_rotleadoff_FILL_{stamp()}.json"
    run_deliberation(api_key=api_key, agents=panel, scenario_key=scen,
                     normgen=True, rounds=5, out_path=out, verbose=True)
    paths.append(out)
    print(f"  DONE {out}\n")

print("="*68)
print("SCORE THESE:")
print("="*68)
for p in paths:
    print(f"python3 analyze_classification.py --transcript {p}")
print("\nThen: git add -A && git commit -m 'Fill mixed runs A/B/E/F to n=3' && git push")
