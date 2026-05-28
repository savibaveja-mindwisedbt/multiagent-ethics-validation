"""Top-level runner for the validation experiments.

Same-model panel by default; --mixed-model uses the Phase 2 lineup.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from agents import (
    MAX_TOKENS, MODEL, TEMPERATURE,
    build_panel, build_mixed_panel, verify_model_slugs,
)
from baselines import run_baselines
from keychain import get_openrouter_key
from orchestrator import run_deliberation
from prompts import SCENARIOS

PROJECT_ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
BASELINES_DIR = PROJECT_ROOT / "baselines"
CAPPED_BASELINES_DIR = PROJECT_ROOT / "baselines_capped"

SCENARIO_KEYS = ["A", "B", "C"]
CONDITIONS = [True, False]
BASELINE_RUNS_PER_SCENARIO = 3


def print_plan(mixed_model: bool = False, only=None, normgen_only: bool = False) -> None:
    n_agents = 6
    keys = only if only else SCENARIO_KEYS
    n_scenarios = len([k for k in keys if k in SCENARIOS])
    n_conditions = 1 if normgen_only else len(CONDITIONS)
    n_deliberations = n_scenarios * n_conditions
    n_baselines = n_scenarios * BASELINE_RUNS_PER_SCENARIO

    print("Multi-Agent Ethics Validation Runs")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    if mixed_model:
        from agents import MIXED_PANEL_SPEC
        print(f"Panel mode: MIXED (Phase 2 lineup)")
        for spec in MIXED_PANEL_SPEC:
            print(f"  {spec['display_name']}: {spec['model']} (max_tokens={spec['max_tokens']})")
    else:
        print(f"Model: {MODEL}")
        print(f"Panel mode: SAME-MODEL ({n_agents} instances)")
        print(f"Temperature: {TEMPERATURE}, max_tokens: {MAX_TOKENS}")
    print(f"Reasoning: disabled")
    print()
    print(f"Scenarios: {n_scenarios}")
    for k in SCENARIO_KEYS:
        print(f"  - {k}: {SCENARIOS[k]['name']}")
    print()
    print(f"Conditions per scenario: {n_conditions} (norm-generating, no-norms)")
    print(f"Leadoff rotation: round R led by Agent (R mod {n_agents}) + 1")
    print(f"  -> Agent 1 leads round 1, ..., Agent 6 leads outcome round")
    print(f"Round indicator prepended to every non-outcome turn.")
    print()
    print(f"Total deliberation runs: {n_deliberations}")
    print(f"Total baseline runs: {n_baselines}")


def run_all_baselines(api_key, only=None, mixed_model=False, capped=False):
    keys = only if only else SCENARIO_KEYS
    for k in keys:
        if k not in SCENARIOS:
            print(f"  Skipping unknown scenario: {k}")
            continue
        print(f"\nBaselines for scenario {k} ({SCENARIOS[k]['name']}):")
        run_baselines(api_key=api_key, scenario_key=k,
                      runs_per_scenario=BASELINE_RUNS_PER_SCENARIO,
                      out_dir=CAPPED_BASELINES_DIR if capped else BASELINES_DIR,
                      mixed_model=mixed_model, capped=capped)


def run_all_deliberations(api_key, only=None, mixed_model=False, normgen_only=False,
                            self_reflection=False, no_consensus_outcome=False):
    keys = only if only else SCENARIO_KEYS
    panel = build_mixed_panel() if mixed_model else build_panel()
    panel_tag = "mixed" if mixed_model else "samemodel"
    print(f"\nPanel ({panel_tag}):")
    for a in panel:
        print(f"  {a.display_name}: {a.model} (max_tokens={a.max_tokens})")
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    for k in keys:
        if k not in SCENARIOS:
            print(f"  Skipping unknown scenario: {k}")
            continue
        conditions = [True] if normgen_only else CONDITIONS
        for normgen in conditions:
            cond_label = "normgen" if normgen else "nonorm"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            reflect_tag = "_reflect" if self_reflection else ""
            consensus_tag = "_noconsensus" if no_consensus_outcome else ""
            out_path = (TRANSCRIPTS_DIR /
                        f"deliberation_{k}_{cond_label}_{panel_tag}_rotleadoff{reflect_tag}{consensus_tag}_{stamp}.json")
            print(f"\n=== Scenario {k}, {cond_label}, {panel_tag} panel ===")
            print(f"Saving to {out_path}")
            run_deliberation(
                api_key=api_key, agents=panel, scenario_key=k,
                normgen=normgen, rounds=5, out_path=out_path, verbose=True,
                self_reflection=self_reflection,
                no_consensus_outcome=no_consensus_outcome,
            )


def main():
    parser = argparse.ArgumentParser(description="Validation runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--deliberations", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--mixed-model", action="store_true",
                        help="Use the heterogeneous Phase 2 panel.")
    parser.add_argument("--normgen-only", action="store_true",
                        help="Skip the no-norm condition; run only the norm-generating condition.")
    parser.add_argument("--self-reflection", action="store_true",
                        help="Use the self-reflection ready-check that asks agents to "
                             "characterize and assess deliberation trajectory before voting.")
    parser.add_argument("--capped-baselines", action="store_true",
                        help="Append the panel's 3-to-7-norm count cap to the baseline "
                             "prompt, writing to baselines_capped/. Matched control to isolate "
                             "the cap effect from deliberation.")
    parser.add_argument("--no-consensus-outcome", action="store_true",
                        help="Each agent states their own final norm list rather than "
                             "the consortium's consolidated norms. Tests whether the "
                             "convergence we observe is genuine persuasion or "
                             "structural sycophancy.")
    args = parser.parse_args()

    only_keys = [k.strip().upper() for k in args.only.split(",") if k.strip()] or None
    print_plan(mixed_model=args.mixed_model, only=only_keys, normgen_only=args.normgen_only)
    print()

    if args.dry_run:
        print("Dry run requested. No API calls will be made.")
        return 0

    do_baselines = args.baselines or args.all
    do_deliberations = args.deliberations or args.all
    if not (do_baselines or do_deliberations):
        print("No action requested.")
        return 0

    api_key = get_openrouter_key()
    print(f"Key loaded from Keychain (starts with {api_key[:11]}..., length {len(api_key)}).")
    print()

    if args.mixed_model and do_deliberations:
        from agents import MIXED_PANEL_SPEC
        mixed_slugs = [spec["model"] for spec in MIXED_PANEL_SPEC]
        print(f"Verifying {len(mixed_slugs)} model slugs against OpenRouter registry...")
        try:
            resolved, missing = verify_model_slugs(api_key, mixed_slugs)
        except Exception as e:
            print(f"  ERROR: slug verification call failed: {e}")
            return 1
        for m in resolved:
            print(f"  OK  {m}")
        for m in missing:
            print(f"  MISSING  {m}")
        if missing:
            print(f"\n{len(missing)} of {len(mixed_slugs)} slugs not found on OpenRouter.")
            print("Aborting before any compute. Update MIXED_PANEL_SPEC in agents.py.")
            return 1
        print(f"\nAll {len(resolved)} slugs resolved. Proceeding.")
        print()

    only = [k.strip().upper() for k in args.only.split(",") if k.strip()] or None

    if do_baselines:
        print("Stage 1: Running baselines.")
        run_all_baselines(api_key, only=only, mixed_model=args.mixed_model,
                          capped=args.capped_baselines)

    if do_deliberations:
        print("\nStage 2: Running deliberations.")
        run_all_deliberations(api_key, only=only, mixed_model=args.mixed_model,
                              normgen_only=args.normgen_only,
                              self_reflection=args.self_reflection,
                              no_consensus_outcome=args.no_consensus_outcome)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
