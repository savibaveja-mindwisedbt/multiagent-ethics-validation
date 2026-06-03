import json
from pathlib import Path
from score_proposals import select_transcripts, collect_items
from analyze_classification import load_scored_cache, classify_with_cache

transcripts = select_transcripts(["6x"], list("ABCDEF"), None)
items = collect_items(transcripts)
# sample 12 per prompt
import random; random.seed(13)
from collections import defaultdict
by = defaultdict(list)
for it in items: by[it["prompt"]].append(it)
sampled = []
for p, lst in by.items():
    random.shuffle(lst); sampled.extend(lst[:12])
print(f"6x items to score: {len(sampled)}")

cache = load_scored_cache()
out = Path("analysis/proposals/proposal_scores_6x.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
done = 0
with open(out, "w") as fh:
    for i, it in enumerate(sampled, 1):
        try:
            scoring, hit = classify_with_cache(it["norm"], cache)
        except Exception as e:
            print(f"  skip {i}: {str(e)[:50]}"); continue
        ans = scoring.get("answers", {})
        Y = lambda q: ans.get(q, {}).get("answer","N").upper().startswith("Y")
        row = dict(it)
        row["all4"] = bool(ans) and all(Y(q) for q in ("E1","E2","E3","E4"))
        row["E2"] = Y("E2")
        row["classification"] = scoring.get("classification")
        fh.write(json.dumps(row) + "\n"); done += 1
        if i % 10 == 0: print(f"  {i}/{len(sampled)} (written {done})")
print(f"Wrote {out}: {done} rows")
