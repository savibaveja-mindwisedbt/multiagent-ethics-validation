#!/usr/bin/env python3
"""
Score and decompose the core experiment for defensibility.

Three arms, one model (Claude), reasoning off, matched protocol:
  Solo            single Claude, one pass            (baselines_core/)
  Solo-iter       single Claude, N neutral revision passes, no other agents
                  (baselines_solo_iter/)  -> isolates ITERATION
  6xClaude        six Claudes, fixed rounds + outcome (transcripts/ tag CORE)
                  -> adds CROSS-TALK on top of iteration

Decomposition:
  iteration effect = Solo-iter(final) - Solo(1 pass)
  collaboration    = 6xClaude(outcome) - Solo-iter(final)
The panel is scored PER ROUND so you can show it starts near Solo at round 0 and
erodes over rounds. Round-0 position-1 is the purest pre-cross-talk proposal.
Solo-iter is scored per pass for the same trajectory: if iterating alone does not
drift but the panel does, the driver is cross-talk, not iteration.

Primary measure E2; all-4 as wrapper. Deterministic from the classification cache.
--score fills misses via OpenRouter (Savi only) and persists them to a new
classification_core_*.json so the cache is complete after one pass.
"""
import json, glob, re, argparse, sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list, extract_final_round_norms, flatten_norms
from analyze_classification import load_scored_cache, classify_with_cache

CAP = 7
LABEL = {"A": "Cooperation", "B": "Human-AI", "C": "Partition",
         "D": "Vulnerable", "E": "Conflicting", "F": "Authority"}
VARIANTS = ["noconsensus", "reflect", "nonorm", "start1", "start4"]

def yy(a, q): return str(a.get(q, {}).get("answer", "N")).strip().upper().startswith("Y")
def wc(t): return len(re.findall(r"\b\w+\b", t))

class Scorer:
    def __init__(self, cache, do_score):
        self.cache, self.do_score, self.newly = cache, do_score, {}
        self.miss = 0
    def score(self, norm):
        n = norm.strip()
        s = self.cache.get(n)
        if s is None and self.do_score:
            s, _ = classify_with_cache(n, self.cache)
            if s and "answers" in s:
                self.newly[n] = s
        if s is None:
            self.miss += 1
            return None
        a = s.get("answers", {})
        return {"all4": all(yy(a, q) for q in ("E1", "E2", "E3", "E4")), "E2": yy(a, "E2"), "w": wc(n)}
    def score_list(self, norms):
        return [r for r in (self.score(x) for x in norms) if r is not None]
    def persist(self):
        if not self.newly:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        p = Path("analysis/classification") / f"classification_core_{stamp}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"source_type": "core_rescore",
            "results": [{"norm": k, "scoring": v} for k, v in self.newly.items()]}, indent=2))
        return p

def rate(rows, key):
    return round(100 * sum(r[key] for r in rows) / len(rows)) if rows else None

def solo_singlepass_norms(p, existing):
    if existing:
        files = sorted(glob.glob("baselines_matched_confirm/*.json")) if p in ("C", "A") else []
    else:
        files = sorted(glob.glob(f"baselines_core/baseline_{p}_run*.json"))
    out = []
    for f in files:
        dd = json.load(open(f)); inner = dd["result"] if isinstance(dd.get("result"), dict) else dd
        txt = inner.get("text") or inner.get("response") or inner.get("content") or ""
        out.extend(parse_numbered_list(txt)[:CAP])
    return out

def solo_iter_passes(p):
    runs = []
    for f in sorted(glob.glob(f"baselines_solo_iter/baseline_{p}_iter_run*.json")):
        dd = json.load(open(f)); inner = dd["result"] if isinstance(dd.get("result"), dict) else dd
        passes = inner.get("passes")
        if passes:
            runs.append([parse_numbered_list(ps.get("text", ""))[:CAP] for ps in passes])
        else:
            runs.append([parse_numbered_list(inner.get("text", ""))[:CAP]])
    return runs

def panel_transcripts(p, tag, existing):
    out = []
    for tp in sorted(glob.glob(f"transcripts/deliberation_{p}_normgen_samemodel_rotleadoff_*.json")):
        nm = tp.split("/")[-1]
        if any(v in nm for v in VARIANTS):
            continue
        if not existing and tag not in nm:
            continue
        out.append(tp)
    return out

def panel_by_round(tp):
    d = json.load(open(tp))
    delib = [t for t in d["transcript"] if t.get("turn_type") == "deliberation"]
    rounds = defaultdict(list)
    for t in sorted(delib, key=lambda t: t.get("turn_index", 0)):
        rounds[t.get("round_index", -1)].extend(parse_numbered_list(t.get("text", "") or ""))
    r0 = sorted([t for t in delib if t.get("round_index", -1) == 0], key=lambda t: t.get("turn_index", 0))
    r0p1 = parse_numbered_list(r0[0].get("text", "")) if r0 else []
    outcome = [it["norm"] for it in flatten_norms(extract_final_round_norms(d))][:CAP]
    return rounds, r0p1, outcome

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="CE")
    ap.add_argument("--tag", default="CORE")
    ap.add_argument("--existing", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    cache = load_scored_cache()
    sc = Scorer(cache, args.score)
    print(f"cache={len(cache)} | mode={'EXISTING' if args.existing else 'tag '+args.tag} "
          f"| score={'ON' if args.score else 'cache-only'}\n")

    for p in args.prompts:
        print(f"================ {LABEL[p]} ({p}) ================")
        solo = sc.score_list(solo_singlepass_norms(p, args.existing))
        print(f"  Solo (1 pass)        E2={rate(solo,'E2')!s:>4}  all4={rate(solo,'all4')!s:>4}  n={len(solo)}")

        iruns = solo_iter_passes(p); finals = []
        if iruns:
            maxp = max(len(r) for r in iruns); per_pass = []
            for k in range(maxp):
                rows = []
                for r in iruns:
                    if k < len(r): rows += sc.score_list(r[k])
                per_pass.append(rate(rows, "all4"))
            for r in iruns: finals += sc.score_list(r[-1])
            print(f"  Solo-iter (final)    E2={rate(finals,'E2')!s:>4}  all4={rate(finals,'all4')!s:>4}  n={len(finals)}  runs={len(iruns)}")
            print(f"     per-pass all4 trajectory: {per_pass}")
        else:
            print("  Solo-iter (final)    [no baselines_solo_iter runs yet]")

        tps = panel_transcripts(p, args.tag, args.existing); outcome_rows = []
        if tps:
            round_rows = defaultdict(list); r0p1_rows = []; per_run_out = []
            for tp in tps:
                rounds, r0p1, outcome = panel_by_round(tp)
                for r, norms in rounds.items(): round_rows[r] += sc.score_list(norms)
                r0p1_rows += sc.score_list(r0p1)
                orow = sc.score_list(outcome); outcome_rows += orow
                if orow: per_run_out.append(rate(orow, "all4"))
            print(f"  6xClaude runs={len(tps)}")
            print(f"     round-0 pos-1 (purest)   all4={rate(r0p1_rows,'all4')!s:>4}  n={len(r0p1_rows)}")
            for r in sorted(k for k in round_rows if k >= 0):
                rr = round_rows[r]
                print(f"     round {r} proposals       all4={rate(rr,'all4')!s:>4}  E2={rate(rr,'E2')!s:>4}  n={len(rr)}")
            print(f"     OUTCOME (consolidated)   all4={rate(outcome_rows,'all4')!s:>4}  E2={rate(outcome_rows,'E2')!s:>4}  n={len(outcome_rows)}")
            if p == "C":
                print(f"     per-run outcome all4 (sorted): {sorted(per_run_out)}")
        else:
            print(f"  6xClaude  [no tag={args.tag} transcripts yet]")

        s = rate(solo, "all4"); si = rate(finals, "all4") if iruns else None; po = rate(outcome_rows, "all4") if tps else None
        if None not in (s, si, po):
            print(f"  --> iteration effect {si-s:+d}  | collaboration effect {po-si:+d}  | total {po-s:+d}")
        print()

    if sc.miss:
        path = sc.persist()
        print(f"{sc.miss} cache misses" + (f" (filled and persisted to {path})" if path else " (run --score locally to fill)") + ".")
    else:
        print("No cache misses. Fully recomputed from cache.")

if __name__ == "__main__":
    main()
