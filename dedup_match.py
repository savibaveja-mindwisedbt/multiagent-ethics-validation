#!/usr/bin/env python3
"""Deduplicate a scenario's solo directed obligations, then re-aggregate the
forward substance-match result over distinct obligations.

Why. The solo cell pools norms across many runs, so the same obligation can
appear several times in slightly different wording. The forward matcher scores
each independently and can miss a real counterpart on one wording while finding
it on another, which inflates the "dropped" count. This step groups the solo
directed obligations that mean the same thing with one model call (temperature 0,
fixed prompt, so it is reproducible), then aggregates each group by its best
instance, since one genuine match proves the counterpart exists. It reuses the
committed forward-match cache and makes no new matching calls.

Aggregation per group: kept-directed if any instance matched a directed panel
norm; else de-directed if any matched an impersonal one; else dropped.

  python3 dedup_match.py --prompt J            # dry: counts + cost, no API
  python3 dedup_match.py --prompt J --run       # one clustering call, then aggregate
"""
import json, re, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
import analysis_shared as A
import substance_match as SM

OUT = Path("analysis/norm_dedup"); OUT.mkdir(parents=True, exist_ok=True)


def solo_directed(p):
    """Ordered list of the scenario's solo directed obligations (deduped by exact text)."""
    return [t for t, e in SM._norms(p, "solo", want_directed=True)]


def match_status(text, cache):
    v = cache.get(text)
    if not v or v.get("match_index") is None:
        return "drop"
    return "dir" if v.get("match_directed") else "imp"


def parse_groups(txt, n):
    """Parse {"groups": [[1,3],[2],...]} into a list of 0-based index lists.
    Repairs the partition: every index 1..n must appear exactly once. Missing
    indices become singletons; out-of-range and duplicate indices are dropped."""
    m = re.search(r"\{.*\}", txt, re.S)
    groups = []
    if m:
        try:
            raw = json.loads(m.group(0)).get("groups", [])
            seen = set()
            for g in raw:
                cur = []
                for x in g:
                    try:
                        i = int(x)
                    except Exception:
                        continue
                    if 1 <= i <= n and i not in seen:
                        seen.add(i); cur.append(i - 1)
                if cur:
                    groups.append(cur)
            for i in range(1, n + 1):        # any norm the model omitted -> its own group
                if i not in seen:
                    groups.append([i - 1])
        except Exception:
            groups = []
    if not groups:                            # total parse failure -> no dedup
        groups = [[i] for i in range(n)]
    return groups


def cluster_prompt(norms):
    listing = "\n".join(f"{i}. {t.strip()}" for i, t in enumerate(norms, 1))
    sys_msg = ("You group behavioral norms that express the same underlying obligation. "
               "Two norms belong together if they require or prohibit substantially the same "
               "thing, regardless of wording. Answer only with JSON.")
    user = ("NORMS:\n" + listing +
            "\n\nGroup the norms that express the same underlying obligation. Every norm "
            "number must appear in exactly one group; a norm with no duplicate is its own "
            'group of one. Respond with exactly: {"groups": [[1,4],[2],[3,5,6]]}')
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def aggregate(groups, norms, cache):
    buckets = {"kept_directed": 0, "de_directed": 0, "dropped": 0}
    detail = []
    for g in groups:
        sts = [match_status(norms[i], cache) for i in g]
        agg = "dir" if "dir" in sts else ("imp" if "imp" in sts else "drop")
        key = {"dir": "kept_directed", "imp": "de_directed", "drop": "dropped"}[agg]
        buckets[key] += 1
        detail.append({"members": len(g), "statuses": sts, "result": key,
                       "example": norms[g[0]].strip()[:80]})
    return buckets, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="J")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    p = args.prompt

    norms = solo_directed(p)
    cache_path = Path(f"analysis/substance_match/substance_match_{p}.json")
    if not cache_path.exists():
        print(f"no forward-match cache for {p}; run substance_match.py --prompt {p} --run first.")
        return
    cache = json.load(open(cache_path))
    print(f"{p}: {len(norms)} solo directed obligations; forward-match cache has {len(cache)} entries.")
    raw = {"kept_directed": sum(1 for t in norms if match_status(t, cache) == "dir"),
           "de_directed": sum(1 for t in norms if match_status(t, cache) == "imp"),
           "dropped": sum(1 for t in norms if match_status(t, cache) == "drop")}
    print(f"  raw (no dedup): kept_directed={raw['kept_directed']} de_directed={raw['de_directed']} dropped={raw['dropped']}")

    if not args.run:
        print(f"\n[dry] no API. --run makes ONE clustering call ({args.model}) over the "
              f"{len(norms)} obligations, then aggregates from the cache with no new match calls.")
        return

    m = C.preflight(args.model)
    print(f"  clustering {len(norms)} obligations with {m} (1 call)...")
    try:
        txt = A.call_openrouter(m, cluster_prompt(norms), max_tokens=1200, temperature=0)
    except Exception as e:
        print(f"  API error: {e}"); return
    groups = parse_groups(txt, len(norms))
    buckets, detail = aggregate(groups, norms, cache)
    json.dump({"groups": groups, "buckets": buckets, "detail": detail},
              open(OUT / f"dedup_{p}.json", "w"), indent=1)

    print(f"\n{len(norms)} obligations -> {len(groups)} distinct after dedup")
    for d in detail:
        print(f"  [{d['result'][:12]:12}] n={d['members']} {d['statuses']}  {d['example']}")
    print(f"\nDEDUPED {p} (best-of-group): kept_directed={buckets['kept_directed']} "
          f"de_directed={buckets['de_directed']} dropped={buckets['dropped']} (n={sum(buckets.values())})")


if __name__ == "__main__":
    main()
