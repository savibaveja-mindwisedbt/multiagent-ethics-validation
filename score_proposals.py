#!/usr/bin/env python3
"""Score per-turn DELIBERATION proposals under the all-4 rubric, tagged by
model, round, leadoff-position, and word count. Enables role-taking, anchoring,
consensus-compression, and length-controlled tests in one pass."""
import json, glob, re, random, argparse
from pathlib import Path
from collections import defaultdict
from analysis_shared import parse_numbered_list
from analyze_classification import load_scored_cache, classify_with_cache

random.seed(13)

def compstr(tp):
    ag = json.load(open(tp)).get("run_metadata", {}).get("agents", [])
    return "+".join(a.get("model","").split("/")[-1].split("-")[0].lower() for a in ag)
def is_std6(tp):
    m = compstr(tp); return "qwen" in m and "deepseek" in m
def fam(model):
    s = (model or "").split("/")[-1].lower()
    for k in ["claude","gpt","gemini","grok","qwen","deepseek","mistral","llama"]:
        if k in s: return k
    return s or "unknown"
def wc(t): return len(re.findall(r"\b\w+\b", t))

EXCLUDE = ["noconsensus","reflect","nonorm","start1","start4","RECONFIRM"]

def select_transcripts(conditions, prompts, runs_per_cell):
    out = []
    for p in prompts:
        for cond in conditions:
            samm = "samemodel" if cond == "6x" else "mixed"
            paths = []
            for tp in sorted(glob.glob(f"transcripts/deliberation_{p}_normgen_{samm}_rotleadoff_*.json")):
                nm = tp.split("/")[-1]
                if any(x in nm for x in EXCLUDE): continue
                if samm == "mixed" and not is_std6(tp): continue
                if not json.load(open(tp)).get("run_metadata", {}).get("complete", True): continue
                paths.append(tp)
            if runs_per_cell: paths = paths[:runs_per_cell]
            for tp in paths: out.append((tp, p, cond))
    return out

def collect_items(transcripts):
    items = []
    for tp, prompt, cond in transcripts:
        d = json.load(open(tp))
        amap = {a["agent_id"]: a.get("model","") for a in d["run_metadata"]["agents"]}
        by_round = defaultdict(list)
        for turn in d["transcript"]:
            if turn.get("turn_type") != "deliberation": continue
            by_round[turn.get("round_index", -1)].append(turn)
        for rnd, turns in by_round.items():
            turns = sorted(turns, key=lambda t: t.get("turn_index", 0))
            for pos, turn in enumerate(turns, start=1):
                model = fam(amap.get(turn.get("agent_id",""), ""))
                for norm in parse_numbered_list(turn.get("text","") or ""):
                    norm = norm.strip()
                    if wc(norm) < 3: continue
                    items.append({"run": tp.split("/")[-1], "prompt": prompt,
                        "condition": cond, "model": model, "round": rnd,
                        "position": pos, "words": wc(norm), "norm": norm})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["mixed","6x"])
    ap.add_argument("--prompts", default="ABCDEF")
    ap.add_argument("--max-per-cell", type=int, default=30)
    ap.add_argument("--runs", type=int, default=0)
    args = ap.parse_args()
    transcripts = select_transcripts(args.conditions, list(args.prompts), args.runs or None)
    print(f"Selected {len(transcripts)} transcripts")
    items = collect_items(transcripts)
    print(f"Parsed {len(items)} proposal items total")
    by_cell = defaultdict(list)
    for it in items: by_cell[(it["prompt"], it["condition"], it["model"])].append(it)
    sampled = []
    for cell, lst in by_cell.items():
        random.shuffle(lst); sampled.extend(lst[:args.max_per_cell])
    print(f"Sampling <= {args.max_per_cell}/cell -> {len(sampled)} items to score")
    cache = load_scored_cache()
    out_dir = Path("analysis/proposals"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proposal_scores.jsonl"
    n_hit = n_miss = 0
    with open(out_path, "w") as fh:
        for i, it in enumerate(sampled, 1):
            scoring, hit = classify_with_cache(it["norm"], cache)
            n_hit += hit; n_miss += (not hit)
            row = dict(it); ans = scoring.get("answers", {})
            def Y(q): return ans.get(q, {}).get("answer","N").upper().startswith("Y")
            row["all4"] = bool(ans) and all(Y(q) for q in ("E1","E2","E3","E4"))
            row["E2"] = Y("E2")
            row["classification"] = scoring.get("classification")
            row["explicit_moral_label"] = scoring.get("explicit_moral_label")
            fh.write(json.dumps(row) + "\n")
            if i % 25 == 0: print(f"  {i}/{len(sampled)} (hits {n_hit}, misses {n_miss})")
    print(f"\nWrote {out_path} (items={len(sampled)}, cache hits={n_hit}, new calls={n_miss})")

if __name__ == "__main__":
    main()
