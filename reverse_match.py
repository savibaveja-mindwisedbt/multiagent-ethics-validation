#!/usr/bin/env python3
"""Reverse matcher (experiment 1b): softened vs added.

For each IMPERSONAL panel norm, ask whether any SOLO norm expresses the same
obligation. matched -> panel softened an existing solo obligation; no match ->
impersonal material the solo agent never wrote (added scaffolding). Whether a panel
norm is impersonal is read from the committed v2 cache, not re-judged.
"""
import json, re, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
import analysis_shared as A

CACHE = Path("analysis/reverse_match")
CACHE.mkdir(parents=True, exist_ok=True)


def _norms(p, cell, tag=None, want_directed=None):
    cc, vc = C.load_class_cache(), C.V2.load_v2_cache()
    save = C.TAG
    if tag:
        C.TAG = tag
    txts = C.cell_texts(p, cell)
    C.TAG = save
    out, seen = [], set()
    for n in txts:
        cl = cc.get(n)
        if not cl or not cl.get("is_norm"):
            continue
        st = (cl.get("scoring_text") or n).strip()
        e = C.ed_of(st, vc)
        if e is None:
            continue
        if want_directed is not None and e != want_directed:
            continue
        if st not in seen:
            seen.add(st); out.append(st)
    return out


def _cache_path(p):
    return CACHE / f"reverse_match_{p}.json"


def _parse_index(text, n):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = obj.get("match", obj.get("match_index"))
            if v in (None, 0, "none", "null", "None"):
                return None
            v = int(v)
            return v if 1 <= v <= n else None
        except Exception:
            pass
    m2 = re.search(r"\b(\d{1,3})\b", text)
    if m2:
        v = int(m2.group(1))
        return v if 1 <= v <= n else None
    return None


def _prompt(panel_norm, solo_norms):
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(solo_norms, 1))
    sys_msg = (
        "You compare behavioral norms by their underlying substance, not their wording. "
        "Two norms match if they require or prohibit the same thing, even if one is phrased "
        "impersonally ('X is expected') and the other as a personal duty ('each agent must X'). "
        "Answer only with JSON."
    )
    user = (
        "PANEL NORM:\n" + panel_norm.strip() +
        "\n\nSOLO NORMS:\n" + listing +
        "\n\nWhich single solo norm, if any, expresses the SAME underlying obligation as the "
        "panel norm? Ignore differences in wording, tone, and how directed the phrasing is. "
        "If no solo norm covers the same obligation, answer null.\n"
        'Respond with exactly: {"match": <solo number or null>}'
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def run(p, model, do_run):
    panel_imp = _norms(p, "panel", tag="CORE", want_directed=False)
    solo = _norms(p, "solo")
    print(f"{p}: {len(panel_imp)} impersonal panel norms to test against {len(solo)} solo norms.")

    cache = json.load(open(_cache_path(p))) if _cache_path(p).exists() else {}
    todo = [x for x in panel_imp if x not in cache]
    print(f"  {len(todo)} need a match call ({len(panel_imp)-len(todo)} cached).")

    if not do_run:
        print(f"\n[dry] no API. --run makes 1 call per uncached impersonal panel norm (={len(todo)}), "
              f"model {model}, tiny outputs.")
        return

    if todo:
        m = C.preflight(model)
        print(f"  matching {len(todo)} impersonal panel norms against solo with {m}...")
        done = 0
        for x in todo:
            try:
                txt = A.call_openrouter(m, _prompt(x, solo), max_tokens=150, temperature=0)
            except Exception as e:
                print(f"   API error, saving and stopping: {e}"); break
            idx = _parse_index(txt, len(solo))
            cache[x] = {"match_index": idx, "match_text": (solo[idx-1] if idx else None)}
            done += 1
            if done % 10 == 0:
                json.dump(cache, open(_cache_path(p), "w"), indent=0)
                print(f"   [{done}/{len(todo)}] checkpoint saved")
        json.dump(cache, open(_cache_path(p), "w"), indent=0)
        print("  matching complete.")

    matched = sum(1 for x in panel_imp if cache.get(x, {}).get("match_index") is not None)
    newimp = sum(1 for x in panel_imp if x in cache and cache[x]["match_index"] is None)
    total = matched + newimp
    print(f"\nREVERSE MATCH RESULT ({p}), n={total} impersonal panel norms:")
    if total:
        print(f"  rephrased_from_solo: {matched:3d}  ({round(100*matched/total)}%)")
        print(f"  new_impersonal     : {newimp:3d}  ({round(100*newimp/total)}%)")
    print("\n  rephrased_from_solo = panel softened an existing solo obligation")
    print("  new_impersonal      = impersonal material the solo agent never wrote (added scaffolding)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="H")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    run(args.prompt, args.model, args.run)


if __name__ == "__main__":
    main()
