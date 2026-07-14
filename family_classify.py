#!/usr/bin/env python3
"""Family classifier for the NEW impersonal panel norms (experiment 1c).

The reverse matcher split panel impersonal norms into rephrased-from-solo vs new.
This codes each NEW norm into one governance family, so the two-tier claim
(generic joint-authorship governance vs scenario-specific allocation governance)
rests on counted categories, not my reading. Families are NOT pre-assigned to
tiers here; we classify into families and read the tiering off the family-by-
scenario distribution (allocation-type families should concentrate in H, the
competitive-stakes scenario, and be near-absent in J, the no-stakes twin).

Input: the committed reverse_match caches (analysis/reverse_match). "New" = the
norms whose match_index is null. Same LLM call path as the matchers. Cached,
checkpointed, Ctrl-C safe.

Families (one per norm):
  norm_governance    - adopting/revising/retiring/prioritizing the norms themselves
  dispute_resolution - handling disagreement, impasse, closure, preserving dissent
  enforcement        - detecting/flagging/responding to violations; compliance; recusal
  epistemic_conduct  - reasoning well together: uncertainty, calibration, steelmanning,
                       guarding premature convergence, legibility
  contribution_credit- what counts as a contribution; attribution/credit tracking;
                       evaluation criteria; allocation administration; anti-gaming credit
  record_provenance  - maintaining an auditable record / output provenance
  other              - none of the above

  python3 family_classify.py --prompts H,J,A          # dry: counts + cost
  python3 family_classify.py --prompts H,J,A --run      # classify (1 call per new norm)
"""
import json, re, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
import analysis_shared as A

CACHE = Path("analysis/norm_family")
CACHE.mkdir(parents=True, exist_ok=True)
FAMILIES = ["norm_governance", "dispute_resolution", "enforcement",
            "epistemic_conduct", "contribution_credit", "record_provenance", "other"]


def new_norms(p):
    f = Path(f"analysis/reverse_match/reverse_match_{p}.json")
    if not f.exists():
        return None
    d = json.load(open(f))
    return [k for k, v in d.items() if v.get("match_index") is None]


def _cache_path(p):
    return CACHE / f"norm_family_{p}.json"


def _parse_family(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = str(json.loads(m.group(0)).get("family", "")).strip().lower()
            if v in FAMILIES:
                return v
        except Exception:
            pass
    low = text.lower()
    for fam in FAMILIES:                       # fallback: first family named in the text
        if fam in low:
            return fam
    return "other"


def _prompt(norm):
    defs = (
        "norm_governance: adopting, revising, retiring, or prioritizing the NORMS themselves "
        "(thresholds, hierarchies, resolving conflicts between norms).\n"
        "dispute_resolution: handling disagreement, impasse, reaching closure/conclusion, "
        "preserving dissent.\n"
        "enforcement: detecting, flagging, or responding to violations; compliance monitoring; recusal.\n"
        "epistemic_conduct: reasoning well together \u2014 marking uncertainty, calibration, "
        "steelmanning, guarding against premature convergence, making reasoning legible.\n"
        "contribution_credit: what counts as a contribution; attribution or credit tracking; "
        "evaluation criteria; administering allocation of reward; preventing gaming of credit.\n"
        "record_provenance: maintaining an auditable record or output provenance, not tied to credit.\n"
        "other: none of the above."
    )
    sys_msg = "You classify a governance norm into exactly one family. Answer only with JSON."
    user = ("NORM:\n" + norm.strip() + "\n\nFAMILIES:\n" + defs +
            '\n\nAssign the single best family. Respond with exactly: {"family": "<name>"}')
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def run(prompts, model, do_run):
    total_todo = 0
    per = {}
    for p in prompts:
        nn = new_norms(p)
        if nn is None:
            print(f"{p}: no reverse_match cache found, skipping."); continue
        cache = json.load(open(_cache_path(p))) if _cache_path(p).exists() else {}
        todo = [x for x in nn if x not in cache]
        per[p] = (nn, cache, todo)
        total_todo += len(todo)
        print(f"{p}: {len(nn)} new norms, {len(todo)} need a family call ({len(nn)-len(todo)} cached).")

    if not do_run:
        print(f"\n[dry] no API. --run makes {total_todo} calls total, model {model}, tiny outputs.")
        return

    m = C.preflight(model)
    for p, (nn, cache, todo) in per.items():
        if todo:
            print(f"  classifying {len(todo)} {p} norms with {m}...")
            done = 0
            for x in todo:
                try:
                    txt = A.call_openrouter(m, _prompt(x), max_tokens=60, temperature=0)
                except Exception as e:
                    print(f"   API error, saving and stopping: {e}"); break
                cache[x] = {"family": _parse_family(txt)}
                done += 1
                if done % 10 == 0:
                    json.dump(cache, open(_cache_path(p), "w"), indent=0)
                    print(f"   [{done}/{len(todo)}] checkpoint saved")
            json.dump(cache, open(_cache_path(p), "w"), indent=0)

    # family x scenario matrix
    print("\nFAMILY x SCENARIO (counts of NEW impersonal norms):")
    hdr = "  " + "family".ljust(20) + "".join(p.rjust(6) for p in per)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    tallies = {p: {fam: 0 for fam in FAMILIES} for p in per}
    for p, (nn, cache, _t) in per.items():
        for x in nn:
            fam = cache.get(x, {}).get("family")
            if fam:
                tallies[p][fam] += 1
    for fam in FAMILIES:
        row = "  " + fam.ljust(20) + "".join(str(tallies[p][fam]).rjust(6) for p in per)
        print(row)
    print("  " + "-" * (len(hdr) - 2))
    print("  " + "TOTAL".ljust(20) + "".join(str(sum(tallies[p].values())).rjust(6) for p in per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="H,J,A", help="comma-separated scenarios")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    run([s.strip() for s in args.prompts.split(",") if s.strip()], args.model, args.run)


if __name__ == "__main__":
    main()
