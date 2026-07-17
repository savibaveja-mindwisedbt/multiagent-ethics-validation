#!/usr/bin/env python3
"""Cheap decision gate for scenario Q: generate and score the SOLO arm only.

Why. Scenario M was run in full (55 runs + ~346 classification calls) before we
learned its task was inherently procedural: its solo agent already produced 39%
implicit-moral norms, so there was little directed obligation left for a panel to
reduce. The solo baseline would have told us that for ~15 cheap calls. This script
runs that check for P first.

  python3 gate_Q.py             # dry: plan, no API
  python3 gate_Q.py --generate  # generate the 15 solo runs (cheap, 1 call each)

Then score just those:
  python3 classify_norms.py --classify --rescore --prompts Q --cells solo
  python3 gate_Q.py --report    # prints ED% / IM% for P/solo, no API

READ:
  ED ~95-100, IM ~0   -> anchoring worked; P is a real test. Run the full scenario.
  ED ~60,     IM ~40  -> it is M again; stop here, having spent ~15 calls.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N_RUNS = 15
KEY = "Q"


def plan():
    from run_scenario import SOLO_DIR, valid_solo, idxset
    valid = idxset(f"{SOLO_DIR}/baseline_{KEY}_run*.json", r"_run(\d+)_", valid_solo)
    missing = [i for i in range(N_RUNS) if i not in valid]
    return sorted(valid), missing


def do_generate():
    from run_scenario import gen_solo
    from keychain import get_openrouter_key
    valid, missing = plan()
    if not missing:
        print(f"{KEY}/solo: all {N_RUNS} runs present, nothing to generate."); return
    key = get_openrouter_key()
    for i in missing:
        print(f"[gen] {KEY}/solo run {i}")
        gen_solo(key, KEY, i)
    print("\ndone. now classify+score the solo arm:")
    print("  python3 classify_norms.py --classify --rescore --prompts Q --cells solo")
    print("  python3 gate_Q.py --report")


def do_report():
    import classify_norms as C
    import score_v2 as V2
    from collections import Counter
    cc = C.load_class_cache(); vc = C.V2.load_v2_cache()

    rows, _ = C.recompute([KEY], ["solo"], cc, vc, rescore=False,
                          rescore_model=None, split_cache=None)
    r = rows[f"{KEY}/solo"]
    denom = r["atoms"] - r["pending"]
    if denom <= 0:
        print(f"{KEY}/solo: nothing scored yet (atoms={r['atoms']}, pending={r['pending']}).")
        print("run: python3 classify_norms.py --classify --rescore --prompts Q --cells solo")
        return
    ed = round(100 * r["ed_clean"] / denom)

    # typology breakdown over the same cell
    txts = C.cell_texts(KEY, "solo")
    seen = set(); cnt = Counter(); e1 = e2 = n = 0
    for t in txts:
        cl = cc.get(t)
        if not cl or not cl.get("is_norm"):
            continue
        st = (cl.get("scoring_text") or t).strip()
        if st in seen:
            continue
        sc = vc.get(st)
        if not sc:
            continue
        seen.add(st); a = sc["answers"]
        cnt[V2.typology(a)] += 1; n += 1
        if V2.y(a, "E1"): e1 += 1
        if V2.y(a, "E2"): e2 += 1

    print(f"\n=== P / solo baseline (n={n} scored norms, pending={r['pending']}) ===")
    if n:
        pc = lambda x: round(100 * x / n)
        print(f"  ED  {ed}%   (EU {pc(cnt['EU'])}%  IM {pc(cnt['IM'])}%  PR {pc(cnt['PR'])}%)")
        print(f"  E1 explicit {pc(e1)}%   E2 directed {pc(e2)}%")
        print("\n  the comparison that matters — Q is the TASK variant of L:")
        print("    L  solo 100 / IM  0     <- Q's control. baselines must MATCH for the")
        print("                                task manipulation to be unconfounded")
        print("    M  solo  71 / IM 29     <- what a confounded task scenario looks like")
        im = pc(cnt['IM'])
        print()
        if ed >= 90 and im <= 10:
            print("  READ: baseline MATCHES L (100). The task manipulation is clean ->")
            print("        run the full arms. Prediction: Q panel ~55 vs L panel 85.")
        elif ed <= 75 or im >= 25:
            print("  READ: baseline FELL vs L (100). Adding work also cost headroom, so")
            print("        task-presence is confounded with baseline -> STOP, do not run the panel.")
        else:
            print("  READ: in between. Judgment call; inspect the norms before committing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="generate the 15 solo runs (spends)")
    ap.add_argument("--report", action="store_true", help="print the scored baseline (no API)")
    args = ap.parse_args()

    from prompts import SCENARIOS
    if KEY not in SCENARIOS:
        sys.exit("scenario Q not registered; run: python3 setup_Q.py")

    if args.report:
        do_report(); return

    valid, missing = plan()
    print(f"{KEY}/solo  valid={len(valid)}  missing={len(missing)} -> {missing}")
    if not args.generate:
        print(f"\n[dry] no API. --generate creates {len(missing)} solo runs "
              f"(1 generation call each, cheap).")
        return
    do_generate()


if __name__ == "__main__":
    main()
