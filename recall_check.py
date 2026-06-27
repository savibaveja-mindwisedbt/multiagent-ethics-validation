import json, glob, re, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_shared import parse_numbered_list
SOLO_DIR = "baselines_core"
STOP = set("the a an of to and or for in on with that this each their its what when how should must may not no nor but if then than as at by from into over under about between among is are be they them other others one another agent agents norm norms client clients".split())
CANON = ["first come","first-come","first served","queue","waitlist","wait list","rotation","rotate","round-robin","round robin","lottery","random selection","randomly","take turns","turn-taking","take-a-number"]
def toks(norms):
    s = set()
    for n in norms:
        for w in re.findall(r"[a-z']+", n.lower()):
            if w not in STOP and len(w) > 3: s.add(w)
    return s
def main():
    prompts = sys.argv[1] if len(sys.argv) > 1 else "AEG"
    print("recall: higher Jaccard = more templated; canonical = scheduling boilerplate")
    for p in [c for c in prompts if c.isalpha()]:
        runs = []
        for f in sorted(glob.glob(f"{SOLO_DIR}/baseline_{p}_run*.json")):
            try: ns = parse_numbered_list(json.load(open(f)).get("result", {}).get("text", ""))[:7]
            except Exception: ns = []
            if ns: runs.append(ns)
        if len(runs) < 2:
            print(f"  {p}: <2 solo runs, skip"); continue
        sets = [toks(r) for r in runs]
        sims = [len(a & b)/len(a | b) for a, b in itertools.combinations(sets, 2) if (a | b)]
        msim = sum(sims)/len(sims) if sims else 0
        cov = sum(1 for r in runs if any(c in " ".join(r).lower() for c in CANON))
        print(f"  {p}: runs={len(runs):2}  Jaccard={msim:.3f}  canonical={cov}/{len(runs)}")
if __name__ == "__main__":
    main()
