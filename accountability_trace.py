#!/usr/bin/env python3
"""Accountability-trace classifier for impersonal norms (dyad vs panel).

Codes every non-explicit (non-ED) norm produced in dyad and panel runs on
whether responsibility under it stays individually attributable, in any of
three modes: LIABILITY (bears consequences if it goes wrong), ANSWERABILITY
(conduct/reasoning stays visible and attributable on an ongoing basis,
independent of outcome), or DECISIONAL (the outcome is causally traceable to
one identifiable agent's judgment, even with no stated consequence). A norm
needs to satisfy only one of the three to count TRACEABLE; otherwise DIFFUSED.

This reuses the existing E1/E2/I2 typology already cached by classify_norms.py
and score_v2.py (no new calls needed to know which norms are non-ED). It adds
exactly one new call per unique non-ED norm text, cached by text so dyad and
panel runs sharing an identical norm string are only scored once.

"Impersonal" here means IM or PR under the existing typology (matches Table 6's
convention; EU is excluded, since EU already names a principle and is scored
separately).

Usage:
  python3 accountability_trace.py --prompts A,L,Q,M,N,H,J                 # dry: counts + cost
  python3 accountability_trace.py --prompts A,L,Q,M,N,H,J --run           # classify (1 call per unique norm)
  python3 accountability_trace.py --prompts A,L,Q,M,N,H,J --report        # dyad-vs-panel table + permutation tests
"""
import json, re, glob, random, argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_norms as C
import analysis_shared as A

CACHE_PATH = Path("analysis/accountability_trace/accountability_trace.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

MODES = ["liability", "answerability", "decisional"]

ACCOUNTABILITY_TRACE_PROMPT = """You are scoring a governance norm for whether it keeps some agent's conduct, judgment, or responsibility individually attributable.

The norm is:
"{norm}"

TRACEABILITY. A norm can make accountability clear in any of three ways. LIABILITY: if the situation this norm governs goes wrong, does the norm make clear who bears the consequences. ANSWERABILITY: independent of outcome, does the norm keep some specific agent's contribution or reasoning individually identifiable and attributable, rather than pooled. DECISIONAL: does the norm make the outcome causally traceable to one identifiable agent's judgment, even if that agent faces no stated consequence and is not expected to disclose their reasoning. A norm needs to satisfy only one of these to count as traceable.

Answer TRACEABLE if the norm ties a stated outcome, contribution, decision, or piece of reasoning to an identifiable agent or defined subset of agents, under any of the three modes above -- even if the norm is procedural, even if the group jointly reaches the outcome, and even if the party named faces no explicit consequence for their choice.

Answer DIFFUSED if the norm resolves the situation through a mechanism, vote, aggregate, or group-level standard where no single agent's judgment, contribution, or reasoning can be pointed to as the reason the outcome came out as it did, and no individual's conduct is separately attributable -- even if the norm specifies a procedure, a threshold, or a process for reaching that aggregate.

Borderline or ambiguous cases: answer DIFFUSED. Do not infer an implied bearer that the norm's text does not support.

Worked examples:
- "The agent that approves a submission is answerable for defects that reach production." -> TRACEABLE (liability)
- "Each agent remains answerable for the portions of the analysis it authored, regardless of the group's overall conclusion." -> TRACEABLE (liability, no final say needed)
- "Each agent's contribution to a joint output is logged and remains attributable to that agent." -> TRACEABLE (answerability, standing, no failure required)
- "Each agent must flag its own uncertainty before the group finalizes a conclusion." -> TRACEABLE (answerability)
- "In a disagreement, the rotating lead decides how to proceed." -> TRACEABLE (decisional: the lead in that round is identifiable, even with no stated consequence for the choice)
- "The group reaches consensus before proceeding." -> DIFFUSED (no single agent's judgment can be pointed to as decisive)
- "Decisions require agreement from a majority of agents." -> DIFFUSED (an aggregate; no one agent's vote is the reason, and dissenters are unrecorded)
- "The final recommendation is presented as the group's joint output, without individual attribution." -> DIFFUSED (deliberately pooled)

Respond in JSON only, no preamble, no code fences:
{{
  "trace": "TRACEABLE" or "DIFFUSED",
  "mode": "liability" or "answerability" or "decisional" or "",
  "bearer": "<the specific party or agent(s), or empty string if DIFFUSED>",
  "reasoning": "<one sentence>"
}}"""


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            trace = str(d.get("trace", "")).strip().upper()
            mode = str(d.get("mode", "")).strip().lower()
            if trace not in ("TRACEABLE", "DIFFUSED"):
                trace = "DIFFUSED"
            if mode not in MODES:
                mode = ""
            return {"trace": trace, "mode": mode,
                    "bearer": str(d.get("bearer", "")).strip(),
                    "reasoning": str(d.get("reasoning", "")).strip()}
        except Exception:
            pass
    return {"trace": "DIFFUSED", "mode": "", "bearer": "", "reasoning": "parse_fallback"}


def _prompt(norm):
    user = ACCOUNTABILITY_TRACE_PROMPT.format(norm=norm.strip())
    return [{"role": "system", "content": "You classify a governance norm for accountability traceability. Answer only with JSON."},
            {"role": "user", "content": user}]


def run_files(p, cell, tag):
    if cell == "dyad":
        return sorted(glob.glob(f"{C.TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_DYAD3*.json"))
    return sorted(glob.glob(f"{C.TDIR}/deliberation_{p}_normgen_samemodel_rotleadoff_{tag}*.json"))


def run_norms(f):
    g = [x["norm"] for x in C.flatten_norms(C.extract_final_round_norms(json.load(open(f))))]
    seen, out = set(), []
    for n in g:
        n = n.strip()
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out


def impersonal_atoms_per_run(p, cell, class_cache, v2cache, tag="CORE"):
    """Returns list of (run_file, [impersonal norm texts]) for a scenario/cell."""
    out = []
    for f in run_files(p, cell, tag):
        atoms = []
        for t in run_norms(f):
            cl = class_cache.get(t)
            if cl is None or not cl.get("is_norm"):
                continue
            stext = cl.get("scoring_text") or t
            sc = v2cache.get(stext.strip())
            if not sc:
                continue
            ty = C.V2.typology(sc["answers"])
            if ty in ("IM", "PR"):
                atoms.append(stext)
        out.append((f, atoms))
    return out


def collect_unique_norms(prompts):
    class_cache = C.load_class_cache()
    v2cache = C.V2.load_v2_cache()
    unique = set()
    per_scenario = {}
    for p in prompts:
        dyad = impersonal_atoms_per_run(p, "dyad", class_cache, v2cache, tag="DYAD3")
        panel = impersonal_atoms_per_run(p, "panel", class_cache, v2cache, tag="CORE")
        per_scenario[p] = {"dyad": dyad, "panel": panel}
        for _f, atoms in dyad + panel:
            unique.update(atoms)
    return per_scenario, unique


def perm_p(a, b, n=20000, seed=0):
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool = a + b
    na = len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(sum(pool[:na]) / na - sum(pool[na:]) / len(pool[na:])) >= obs - 1e-12:
            hits += 1
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="A,L,Q,M,N,H,J", help="comma-separated scenario repo keys")
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--run", action="store_true", help="make the API calls")
    ap.add_argument("--report", action="store_true", help="print dyad-vs-panel traceability table")
    args = ap.parse_args()
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]

    per_scenario, unique = collect_unique_norms(prompts)
    cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    todo = [n for n in unique if n not in cache]
    print(f"{len(unique)} unique impersonal (IM/PR) norms across {len(prompts)} scenarios, "
          f"{len(todo)} need a call ({len(unique) - len(todo)} cached).")

    if args.run and todo:
        m = C.preflight(args.model)
        print(f"classifying {len(todo)} norms with {m}...")
        done = 0
        for norm in todo:
            try:
                txt = A.call_openrouter(m, _prompt(norm), max_tokens=150, temperature=0)
            except Exception as e:
                print(f"  API error, saving and stopping: {e}"); break
            cache[norm] = _parse(txt)
            done += 1
            if done % 10 == 0:
                json.dump(cache, open(CACHE_PATH, "w"), indent=0)
                print(f"  [{done}/{len(todo)}] checkpoint saved")
        json.dump(cache, open(CACHE_PATH, "w"), indent=0)
    elif args.run:
        print("nothing to run, cache already covers all unique norms.")

    if not args.report:
        if not args.run:
            print("[dry] no API. --run makes the calls; --report prints the dyad-vs-panel table.")
        return

    # ---- report: per-run % TRACEABLE, dyad vs panel, per scenario ----
    print("\nTRACEABLE share of impersonal norms, dyad vs panel (run-level means):")
    hdr = f"  {'scenario':>10} {'dyad %':>8} {'panel %':>8} {'n(dyad)':>8} {'n(panel)':>9} {'p (perm)':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    all_dyad_pct, all_panel_pct = [], []
    for p in prompts:
        d_runs = per_scenario[p]["dyad"]
        pa_runs = per_scenario[p]["panel"]
        d_pct = [100 * sum(1 for n in atoms if cache.get(n, {}).get("trace") == "TRACEABLE") / len(atoms)
                 for _f, atoms in d_runs if atoms]
        pa_pct = [100 * sum(1 for n in atoms if cache.get(n, {}).get("trace") == "TRACEABLE") / len(atoms)
                  for _f, atoms in pa_runs if atoms]
        if not d_pct or not pa_pct:
            print(f"  {p:>10} {'--':>8} {'--':>8} {len(d_pct):>8} {len(pa_pct):>9} {'--':>9}  (no impersonal atoms in one condition)")
            continue
        pv = perm_p(d_pct, pa_pct)
        all_dyad_pct.extend(d_pct); all_panel_pct.extend(pa_pct)
        print(f"  {p:>10} {sum(d_pct)/len(d_pct):8.1f} {sum(pa_pct)/len(pa_pct):8.1f} "
              f"{len(d_pct):>8} {len(pa_pct):>9} {pv:9.3f}")
    print("  " + "-" * (len(hdr) - 2))
    if all_dyad_pct and all_panel_pct:
        pooled_p = perm_p(all_dyad_pct, all_panel_pct)
        print(f"  {'POOLED':>10} {sum(all_dyad_pct)/len(all_dyad_pct):8.1f} "
              f"{sum(all_panel_pct)/len(all_panel_pct):8.1f} "
              f"{len(all_dyad_pct):>8} {len(all_panel_pct):>9} {pooled_p:9.3f}")

    # ---- mode breakdown, DIFFUSED and TRACEABLE-by-mode, dyad vs panel pooled ----
    print("\nMode breakdown (pooled dyad vs panel, count of impersonal norms):")
    for cond, runs in (("dyad", [r for p in prompts for r in per_scenario[p]["dyad"]]),
                       ("panel", [r for p in prompts for r in per_scenario[p]["panel"]])):
        tallies = {"DIFFUSED": 0, "liability": 0, "answerability": 0, "decisional": 0}
        total = 0
        for _f, atoms in runs:
            for n in atoms:
                rec = cache.get(n, {})
                total += 1
                if rec.get("trace") == "TRACEABLE":
                    tallies[rec.get("mode") or "decisional"] += 1
                else:
                    tallies["DIFFUSED"] += 1
        print(f"  {cond:>6} (n={total:4d} norms): " +
              ", ".join(f"{k}={v}" for k, v in tallies.items()))


if __name__ == "__main__":
    main()
