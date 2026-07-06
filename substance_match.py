#!/usr/bin/env python3
"""Same-substance matcher (experiment 1, proper version).

Question: when the panel de-directs, is it the SAME obligations phrased less
directedly, or DIFFERENT obligations? Lexical overlap cannot answer this because
rephrasing a directed norm into an impersonal one lowers token overlap even when
the substance is identical. So we use an LLM to match by MEANING.

Method, per solo directed norm:
  - present the numbered list of panel norms and ask which single one expresses
    the SAME underlying obligation (same substance), regardless of wording or
    directedness, or none.
  - the matcher only decides SUBSTANCE. The directedness of the matched panel norm
    is read from the committed v2 cache (ed_of), NOT re-judged, so directedness
    stays on the same instrument as every other cell this project reports.

Each solo directed norm is then bucketed:
  - kept_directed   : matched a panel norm that is itself directed
  - de_directed     : matched a panel norm that is impersonal  <- the diffusion signature
  - dropped         : no same-substance panel counterpart

Decision rule:
  - mostly de_directed  -> panel softens the same obligations (diffusion / rephrasing)
  - mostly dropped      -> panel drops obligations rather than softening them (different mechanism)
  - mostly kept_directed-> panel preserves directedness (would contradict the ED drop)

Reuse: classify_norms for the norm populations and caches (same instrument as the
cells); analysis_shared.call_openrouter for the LLM call (same call path as scoring).
Matches are cached so re-runs are free and it is Ctrl-C safe.

  python3 substance_match.py --prompt H            # dry: counts + cost, no API
  python3 substance_match.py --prompt H --run       # match (spends ~1 call per solo directed norm)
"""
import json, re, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
import analysis_shared as A

CACHE = Path("analysis/substance_match")
CACHE.mkdir(parents=True, exist_ok=True)


def _norms(p, cell, tag=None, want_directed=None):
    """De-fused scoring_text of classified norms in a cell, optionally filtered by
    directedness, using the committed class + v2 caches."""
    cc, vc = C.load_class_cache(), C.V2.load_v2_cache()
    save = C.TAG
    if tag:
        C.TAG = tag
    txts = C.cell_texts(p, cell)
    C.TAG = save
    out = []
    for n in txts:
        cl = cc.get(n)
        if not cl or not cl.get("is_norm"):
            continue
        st = (cl.get("scoring_text") or n).strip()
        e = C.ed_of(st, vc)
        if e is None:
            continue
        if want_directed is None or e == want_directed:
            out.append((st, e))
    # dedup preserving order
    seen, dedup = set(), []
    for st, e in out:
        if st not in seen:
            seen.add(st); dedup.append((st, e))
    return dedup


def _cache_path(p):
    return CACHE / f"substance_match_{p}.json"


def _load_cache(p):
    f = _cache_path(p)
    if f.exists():
        return json.load(open(f))
    return {}


def _save_cache(p, d):
    json.dump(d, open(_cache_path(p), "w"), indent=0)


def _parse_index(text, n_panel):
    """Extract match index from model JSON. Returns int in 1..n_panel, or None."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = obj.get("match", obj.get("match_index"))
            if v in (None, 0, "none", "null", "None"):
                return None
            v = int(v)
            return v if 1 <= v <= n_panel else None
        except Exception:
            pass
    # fallback: a bare integer
    m2 = re.search(r"\b(\d{1,3})\b", text)
    if m2:
        v = int(m2.group(1))
        return v if 1 <= v <= n_panel else None
    return None


def _prompt(solo_norm, panel_norms):
    listing = "\n".join(f"{i}. {t}" for i, (t, _e) in enumerate(panel_norms, 1))
    sys_msg = (
        "You compare behavioral norms by their underlying substance, not their wording. "
        "Two norms match if they require or prohibit the same thing, even if one is phrased "
        "as a personal duty ('each agent must X') and the other impersonally ('X is expected'). "
        "Answer only with JSON."
    )
    user = (
        "SOLO NORM:\n" + solo_norm.strip() +
        "\n\nPANEL NORMS:\n" + listing +
        "\n\nWhich single panel norm, if any, expresses the SAME underlying obligation as the "
        "solo norm? Ignore differences in wording, tone, and how directed the phrasing is. "
        "If none of the panel norms covers the same obligation, answer null.\n"
        'Respond with exactly: {"match": <panel number or null>}'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def run(p, model, do_run):
    solo_dir = _norms(p, "solo", want_directed=True)
    panel = _norms(p, "panel", tag="CORE")
    n_panel = len(panel)
    print(f"{p}: {len(solo_dir)} solo directed norms; {n_panel} panel norms "
          f"({sum(1 for _t,e in panel if e)} directed, {sum(1 for _t,e in panel if not e)} impersonal).")

    cache = _load_cache(p)
    todo = [s for s, _e in solo_dir if s not in cache]
    print(f"  {len(todo)} need a match call ({len(solo_dir)-len(todo)} cached).")

    if not do_run:
        print("\n[dry] no API. --run makes 1 matching call per uncached solo directed norm "
              f"(={len(todo)} calls), model {model}, tiny outputs. Directedness of each match is "
              "read from the committed v2 cache, not re-judged.")
        return

    if todo:
        m = C.preflight(model)
        print(f"  matching {len(todo)} norms with {m}...")
        done = 0
        for s in todo:
            try:
                txt = A.call_openrouter(m, _prompt(s, panel), max_tokens=150, temperature=0)
            except Exception as e:
                print(f"   API error, saving and stopping: {e}"); break
            idx = _parse_index(txt, n_panel)
            cache[s] = {"match_index": idx,
                        "match_text": (panel[idx-1][0] if idx else None),
                        "match_directed": (bool(panel[idx-1][1]) if idx else None)}
            done += 1
            if done % 10 == 0:
                _save_cache(p, cache); print(f"   [{done}/{len(todo)}] checkpoint saved")
        _save_cache(p, cache)
        print("  matching complete.")

    # aggregate
    buckets = {"kept_directed": 0, "de_directed": 0, "dropped": 0}
    for s, _e in solo_dir:
        r = cache.get(s)
        if not r:
            continue
        if r["match_index"] is None:
            buckets["dropped"] += 1
        elif r["match_directed"]:
            buckets["kept_directed"] += 1
        else:
            buckets["de_directed"] += 1
    total = sum(buckets.values())
    print(f"\nSUBSTANCE MATCH RESULT ({p}), n={total} solo directed norms:")
    for k in ("kept_directed", "de_directed", "dropped"):
        pct = round(100 * buckets[k] / total) if total else 0
        print(f"  {k:13s}: {buckets[k]:3d}  ({pct}%)")
    print("\n  de_directed = same obligation, panel made it impersonal (diffusion signature)")
    print("  dropped     = no same-substance panel counterpart (obligation removed, not softened)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="H")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    run(args.prompt, args.model, args.run)


if __name__ == "__main__":
    main()
