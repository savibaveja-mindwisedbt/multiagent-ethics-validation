# Handoff: Multi-Agent Ethics Paper — Verification State and Publication Prep

Date: June 23, 2026
Supersedes: earlier handoff_multiagent_ethics.md (May 18) and the May handoff in the paper-prep thread.

## Who and what

Savi Baveja, Stanford DCI Fellow. Paper: "When AI Agents Co-Operate, Explicit
Moral Obligations Become Implicit and Non-Directed." Target venues, in order of
realism: NeurIPS workshop (Aug), FAR.ai alignment workshop (fall), AAAI-27 (late
July, archival, 8-9pp). The June 23 aiagentbehavior.com 4-page deadline is
effectively past.

Voice: plain declarative, no em-dashes, no colons in running prose, concise,
bolded takeaways. His voice, not the model's. Source every number. Flag
uncertainty. Do not re-break corrected findings. He has been burned by mid-project
reversals; precision and not fabricating are the priorities.

The assistant CANNOT make OpenRouter API calls. Savi runs all scoring locally
(key in Mac keychain) and pushes results to GitHub. The assistant pulls and reads.

## Repository

github.com/savibaveja-mindwisedbt/multiagent-ethics-validation (public).
Pull via codeload, NOT the GitHub API (the shared egress IP is rate-limited on
api.github.com; codeload.github.com and raw.githubusercontent.com work):

    curl -s -o repo.tar.gz https://codeload.github.com/savibaveja-mindwisedbt/multiagent-ethics-validation/tar.gz/refs/heads/main

All files in the current main branch are dated June 3, 2026. The tarball has no
.git metadata, so commit history is not visible this way. If the paper's numbers
turn out to need an earlier commit, the git log must be obtained another way
(Savi pushing, or a clone with history).

## THE CENTRAL UNRESOLVED ISSUE: Table 1 does not fully reproduce

This is the most important thing in this handoff. Do not gloss it.

The paper's locked metric (from the prior handoff's terminology block) is:
explicit and directed moral obligation = passes all four Explicit criteria,
written "all-4", with E2 (states what is owed to whom) as the binding criterion,
and the claim "all-4 reduces to E2; E4 is inert." Scored by an LLM rubric at
temperature 0, reading-based, against a 73-concept vocabulary.

The repo contains TWO different scoring rubrics:
  1. analyze_classification.py — 16 questions (E1-E4, I1-I4, P1-P4, T1-T4).
     Its derive_explicit_moral_label uses a 3-of-4-with-E1-essential threshold,
     NOT all-4, and the essential criterion is E1, not E2.
  2. score_principled_moral.py — 4 questions (PM1 primary-thesis, PM2 load-bearing,
     PM3 stand-alone, PM4 universality), strict all-4 rule. This is the script
     that reads the CAPPED baselines (baselines_capped/). Its output files
     (pm_*.json) were NOT committed, so its results cannot be recomputed from
     cached data.
  3. score_proposals.py line 91 computes a strict all-4 on E1-E4 (row["all4"]),
     used for the proposal-level tables (Tables 2, 3).

So the paper says "all-4 on E1-E4 with E2 binding." The classification code uses
3-of-4 with E1 essential. The PM code uses all-4 but on a different 4 criteria.
The proposal code uses strict all-4 on E1-E4. These do not all agree, and the
paper text matches none of them exactly (it names E1-E4 but claims E2 is the
binding criterion, which is not how either E-based threshold is coded).

### What was actually verified (June 23 session)

Recomputed strict all-4 (E1 AND E2 AND E3 AND E4) from the cached classification
JSONs, filtered to capped baseline source files and the canonical normgen
transcripts. Results vs paper Table 1 (computed / paper):

  Vulnerable   Solo 100/100 MATCH | 6xClaude 95/95 MATCH | Mixed 88/84 close
  Authority    Solo 100/100 MATCH | 6xClaude 96/96 MATCH | Mixed 91/83 off
  Cooperation  Solo  99/100 MATCH | 6xClaude 95/94 MATCH | Mixed 83/80 close
  Conflicting  Solo 100/100 MATCH | 6xClaude 91/90 MATCH | Mixed 70/70 MATCH
  Human-AI     Solo  97/100 close | 6xClaude 95/95 MATCH | Mixed 85/62 OFF BY 23
  Partition    Solo  34/81  OFF BY 47 | 6xClaude 65/56 off | Mixed 31/31 MATCH

CONCLUSION: The owes-prompt Solo and 6xClaude cells reproduce under strict all-4.
This corroborates the metric and those numbers. But two things DO NOT reproduce:
  - Partition Solo: computed 34, paper says 81. This is the anchor cell of the
    entire moral-pressure thesis. A 47-point gap.
  - The Mixed column runs systematically HIGH in recomputation (Human-AI Mixed
    85 vs 62 is the worst). Several Mixed cells off by 8-23 points.

Also: sample sizes do not match. Paper footnote says Solo n=36 for Partition and
Cooperation. Recomputation pools 209 (Partition) and 312 (Cooperation) even after
filtering to capped source files. The cache appears to pool multiple scoring
passes and/or multiple C baseline sets (there are three: baselines_C_partition,
baselines_C_mixed, baselines_C_reconfirm). The exact 36-norm selection that
produced the paper's Partition Solo cell was NOT isolated.

### Why this is not yet proof of an error

The cap step and the exact run-selection that produced each Table 1 cell are NOT
captured in the cached classification outputs. The script that reads the capped
baselines (score_principled_moral.py) uses the PM rubric and its outputs were
never committed. So the cached data alone cannot reproduce the paper's exact
pipeline. This is a reproducibility GAP, not a demonstrated mistake. The findings
may be entirely correct; they just cannot be re-derived from what is in the repo.

### The two paths forward (Savi must choose)

PATH 1 — Confirm the existing numbers. Find the exact script/notebook that
assembled Table 1 (which rubric, which cap, which run selection per cell). It is
not in the committed repo. Either it exists in Savi's local files (upload it), or
the assembly was partly manual (disclose that, and the IRR concern becomes
central). Once found, reproduce Table 1 exactly and write the methods section to
match.

PATH 2 — Re-baseline cleanly. Define ONE canonical pipeline now: strict E1-E4
all-4 with E2 binding (as the terminology locks), fixed 3 runs per cell, an
explicit stated cap, recompute every table from the transcripts in one pass,
update the paper to whatever that pipeline produces. Removes version ambiguity
permanently. RISK: numbers may shift from the locked Table 1, and Partition Solo
in particular could move a lot (recomputation suggests it may not be 81).

The assistant's recommendation: Path 1 first (cheap if the script exists), and if
the script cannot be found, Path 2 is the only defensible option given the
project's history of version churn.

## Verified findings (preserve; do NOT re-derive or fabricate)

These are from the prior handoff and are consistent with the repo. The Solo and
6xClaude owes-prompt cells were independently corroborated this session.

Codebase->paper label map: A=Cooperation, B=Human-AI, C=Partition (neutral),
D=Vulnerable, E=Conflicting, F=Authority. (The repo's prompts.py registry is
STALE and maps these differently; ignore it. The actual run prompts are in
prompt_A.txt etc and embedded in transcript scenario_prompt fields, and those
match the paper's Appendix A.)

Main results all-4 (solo/6xClaude/mixed/shift):
  Vulnerable 100/95/84/-16; Authority 100/96/83/-17; Cooperation 100/94/80/-20;
  Conflicting 100/90/70/-30; Human-AI 100/95/62/-38; Partition 81/56*/31/-50.
  *6xClaude neutral high-variance.

Per-model proposals, mixed, n=248 (overall/at 20-50 words): Claude 57/67;
GPT 59/62; Gemini 56/44; Qwen 55/33; DeepSeek 72/53; Grok 17/- (n=12 too thin).
At matched length the low-obligation framers are Qwen and Gemini, NOT DeepSeek.

Consensus passthrough (proposal mean -> outcome): Partition 21->31,
Cooperation 74->80, Human-AI 64->62, Vulnerable 85->84, Conflicting 68->70.
Consensus passes the pool through; does not compress.

Dilution/degradation (6xClaude prop->outcome | mixed all/Claude/outcome):
  Cooperation 83->94 | 74/83/80; Human-AI 80->95 | 64/67/62;
  Vulnerable 92->95 | 85/92/84; Partition 50->56 | 21/0/31.
  Claude's own framing holds on high-pressure prompts, degrades on low
  (Partition 50->0).

Length mediation (pooled outcome norms, n=530): <15w 24%, 15-30w 50%,
30-50w 78%, >50w 95%. corr 0.39.

Reasoning burn (% completion tokens on reasoning): Claude 9, GPT 2, Grok 0,
Gemini 76, Qwen 61, DeepSeek 62. OFF for Claude/GPT/Grok; mandatory ON for
Gemini/Qwen/DeepSeek (endpoints 400 on the off flag, confirmed). Confound cannot
be configured away.

6xClaude neutral drift: per-run all-4 = [7,40,56,67,70,75,80];
corr(rounds_run, all-4) = -0.63; early-stop@round-3 scored 56-80, full-5-round
scored 7-70. Leadoff near-identical at temp 0, so anchoring ruled out.
Mechanism = procedural drift over rounds, amplified by homogeneity.

Drop test: remove Qwen+DeepSeek -> 4-model mixed recovers (Cooperation 80->95,
Human-AI 62->100).

Interventions: no-consensus ≈ consensus (compression NOT supported); reflection
raises framing but confounded with length.

## Corrections already made (do NOT re-break)

1. "DeepSeek floods with terse fragments" was a PARSER ARTIFACT. Bold-marker
   regex gave 1,125 items/11-word median; parse_numbered_list (the scoring
   parser) gives 329 items/52-word median. DeepSeek's role is VOLUME, not
   terseness.
2. Mechanism is "dilution + pressure-dependent degradation," NOT "dilution not
   contagion." Claude degrades its framing on low-pressure prompts in mixed
   panels (Partition 50->0). Length stability misled the earlier claim.
3. Theory 2.3 is posed as an OPEN QUESTION (Tomasello pro-group vs Binmore/Skyrms
   pro-solo), with Discussion adjudicating pro-solo. Not a verdict up front.
4. Weber and Arendt REMOVED (untested proceduralization/diffusion framing).
   Counter-claim now sourced only from Binmore and Skyrms, with the "procedure is
   the cheaper equilibrium" step explicitly marked as the paper's inference, not
   their claim. Citations: [1] Tomasello, [2] Darwall, [3] Binmore, [4] Skyrms,
   [5] Korsgaard.

RECURRING RIGOR LESSON: length has repeatedly misled as a proxy for framing.
Every framing claim must rest on the rubric/read score, never on word counts.

## Reviewer comments to address (from Jared + inline reviewer)

INLINE REVIEWER (answers established this session where noted):
- Consolidation: per-agent in the final round, NO separate summarizer. CONFIRMED
  in both the session orchestrator and the repo orchestrator (each agent states
  its own consolidated list in the outcome round). State this explicitly in
  methods; it strengthens "consensus does not compress."
- Numbered lists: agents ARE instructed to produce a numbered list of 3-7 norms.
  CONFIRMED in prompts (Appendix A says so; baseline prompt and final-round
  instruction both say so). Surface in methods; makes the parser principled.
- Norm cap: a scoring-time normalization, agents NOT told. The 3-7 in the prompt
  is a generation instruction; the scoring cap is separate. State both, clearly
  distinguished. (NOTE: the exact cap mechanism is the crux of the repro gap above.)
- 16 criteria grounding: E1-E4 are grounded in Darwall/Tomasello (Section 2.1).
  The other 12 (I/P/T families) are NEVER enumerated or grounded in the paper.
  Either enumerate+ground them in an appendix, or drop the "16 criteria across
  four families" framing and say the primary metric uses 4 criteria. The paper
  currently writes a check it doesn't cash.
- 73-concept provenance: the paper never says where the concepts came from. The
  code groups them 8/10/10/10/16/5/5/3/2/4; the paper's Appendix B groups them
  18/22/13/9/8/3 (deont/virtue/care/rights/conseq/util). Savi confirmed the
  PAPER grouping is canonical, code is stale. MUST verify the 73 concept STRINGS
  are the same set in both (just regrouped) vs genuinely different. And MUST add
  provenance: where did the 73 come from? string-match was tried and rejected
  (under-recalls); say so.
- Table 1 measure unclear ("whether they reach consensus?"): HIGH PRIORITY.
  Restate the measure in plain language with a passing and a failing example
  IMMEDIATELY before Table 1. Reviewer could not tell what was measured.
- IRR: scorer is an LLM. The original design (scoring_rubric.docx, May 13) called
  for TWO AI judges with human adjudication on disagreement. The paper as written
  uses one scoring model validated against own reading on ~15 norms. For archival
  venues, real two-coder kappa is needed. For workshops, report the validation and
  list IRR as future work.
- Variants requested: (a) no-norm-request emergence condition (does norm/consensus
  emerge unprompted); (b) no-consensus-priming condition. Appendix H has partial
  versions. The no-norm-REQUEST emergence variant is the most interesting
  extension and the cleanest answer to "is the effect an artifact of asking."
- Prompts as one template varying one dimension: the dimension is strength of one
  party's moral claim (the "Shift" axis). Present the six prompts as a template
  with that one parameter, neutral (Partition) to strong (Vulnerable/Authority).

JARED (venue-level):
- Connect to cooperative AI literature (Dafoe et al "Open Problems in Cooperative
  AI"; LLM-debate work). The finding challenges the "deliberation improves
  outputs" assumption in that literature. Do NOT fabricate citations; Savi must
  supply or approve them.
- Foreground a deployment "cover story" in the intro (why this game matters for
  real multi-agent AI systems). Material exists in Section 7.2; move it forward.
- Add an intro game-play graphic (cited example: arxiv 2404.16698).
- Spend more on framing than on new experiments.

## Pending text edits (proposed, NOT yet applied to source)

a. Intro sentence "Procedures can be gamed, transferred, and stripped of their
   reason. Obligations carry their justification with them." is UNSUPPORTED by the
   study. Cut, or mark as theoretical motivation tied to Korsgaard.
b. Discussion 7.1 corrected Binmore/Skyrms version drafted (removes residual
   Weber/Arendt); not confirmed applied.
c. "Length is a poor proxy" limitation bullet has a clearer rewrite naming the
   Appendix E evidence; not applied.

## Possible source/Word divergence (open)

The saved .tex/.docx have Weber/Arendt removed and Binmore/Skyrms in place. A
passage Savi pasted recently still showed old Weber/Arendt wording and a "Sykes"
misspelling (should be Skyrms), suggesting his working Word copy diverged from the
saved source. RECONCILE which copy is canonical before applying further edits.

## Files in this handoff (outputs/)

- HANDOFF_verification_and_publication.md (this file)
- verify_thresholds.py — recomputes strict-all4 and 3-of-4-E1 on full cached pool
- verify_capped_filtered.py — recomputes strict-all4 on capped/filtered subset
  (the script that produced the partial-reproduction table above)
- verification_snapshot/ — the four key scoring scripts pulled from the repo:
  analyze_classification.py, score_principled_moral.py, score_proposals.py,
  analysis_shared.py
- (pre-existing) scoring_rubric.docx, scoring_sheet.docx — the original
  three-category rubric design with two-judge + human adjudication IRR plan
- (pre-existing) multiagent_ethics_paper.docx/.pdf/.tex — the paper

## Immediate next action for the next session

1. Ask Savi which path (1 confirm, or 2 re-baseline).
2. If Path 1: get the Table 1 assembly script or confirmation it was manual.
   Then reproduce exactly and write methods to match.
3. If Path 2: write the canonical pipeline spec + a recompute harness Savi runs
   locally; he pushes outputs; verify every table from pushed results.
4. Either way, resolve the rubric question definitively: is Table 1 from E1-E4
   (and which threshold) or from PM1-PM4? The paper text must describe the rubric
   that actually produced the numbers.
5. Do NOT start API scoring from the assistant side (cannot, and would burn spend
   on a possibly-wrong path).
6. The framing edits (restate-measure-before-Table-1, cooperative-AI related work,
   deployment cover story, prompt parameterization) can proceed in parallel since
   they do not depend on the repro resolution. Draft in Savi's voice.

## RESOLVED THIS SESSION: the 73-concept vocabulary

Checked the concept STRINGS in analyze_classification.py MORAL_CONCEPTS against
the paper's Appendix B. They are the SAME set of 73 concepts. The only apparent
differences are transcription noise (autonomy vs "autonomy-as-value"; "repair" vs
"recovery"). The two differ ONLY in family grouping:
  - Code groups: 8 foundational / 10 reciprocity / 10 justice / 10 harm-welfare /
    16 care-virtue / 5 truthfulness / 5 accountability / 3 civic / 2 restorative /
    4 stewardship-AI.
  - Paper Appendix B groups: 18 deontological / 22 virtue / 13 care / 9 rights /
    8 consequentialist / 3 utilitarian.
Savi confirmed the PAPER grouping is canonical. Action: keep the paper's grouping;
no concept-content change needed. The methods text still needs a provenance
sentence (where the 73 came from), which only Savi can supply.
