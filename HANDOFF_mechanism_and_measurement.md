# Handoff: multi-agent ethics paper — mechanism, measurement, and corrections

Paper: "When AI Agents Co-Operate, Explicit Moral Obligations Become Implicit and Non-Directed."
Author: Sarabjit Singh Baveja. Targets: NeurIPS workshop / FAR.ai / AAAI-27.
Repo: github.com/savibaveja-mindwisedbt/multiagent-ethics-validation (branch main).
This document captures the state at the end of the session focused on scenarios G and H, the
scoring-cap bug, the parsing-fragment artifact, the mechanism analysis, and the measurement search.

## 0. Working conventions (carry forward)
- Savi runs all OpenRouter API work locally (~/projects/multiagent-ethics-validation) on a Mac
  and pushes to GitHub. The assistant cannot call the API; it verifies cache-only by pulling via
  codeload.github.com (NOT the GitHub API, which rate-limits): 
  `curl -s -o r.tar.gz "https://codeload.github.com/savibaveja-mindwisedbt/multiagent-ethics-validation/tar.gz/refs/heads/main"`.
- Browser file downloads to the Mac have repeatedly failed; transfer code via `cat > file << 'EOF'`
  heredoc paste, and verify the shell prompt shows the repo path before pasting (a stray paste
  once created ~/prompts.py in the home dir and broke imports).
- macOS sed needs `sed -i '' 's/.../.../'`.
- User preferences: no fabrication, cite sources, flag textbook-level claims for confirmation;
  plain declarative prose, no em-dashes; concise; do not claim completion without checking; test
  before delivering; push back honestly; do not use the user as QA.

## 1. Experimental design
Per scenario, three arms, N=15 runs each. MODEL=anthropic/claude-sonnet-4.6, TEMPERATURE=0.0,
reasoning OFF, ready-check OFF.
- solo: 1 agent, single pass. Files: baselines_core/baseline_{P}_run{i}_*.json
- solo-iter: 1 agent, 5 self-revision passes. Files: baselines_solo_iter/baseline_{P}_iter_run{i}_*.json
- panel: 6x Claude, 5 rounds, consensus norm set. Files:
  transcripts/deliberation_{P}_normgen_samemodel_rotleadoff_CORE{i}_*.json
Scenario keys: A=Cooperation, B=Human-AI, C=Partition, D=Vulnerable, E=Conflicting objectives,
F=Authority, G=Bound advocates (scarce appointment access), H=Contribution stakes (fixed pool,
relative-contribution allocation).
The spine of the design is solo-iter vs panel: it holds the iteration budget fixed (5 passes vs 5
rounds) and varies only agent count, isolating collaboration from iteration.

## 2. Current data state
A, E, G, H all generated (15/arm) and v2-scored and pushed. v2 cache: 1173 scored norms in
analysis/classification_v2/. B, C, D, F exist from earlier work but are not part of this analysis.
The scorer is score_v2.py (tradition rubric -> ED/EU/IM/PR typology; MFD convergence; valence).
Independent recompute by the assistant matches Savi's scorer exactly.

## 3. Headline results (CLEANED, authoritative — see section 4 for why "cleaned")
Typology cells are ED/EU/IM/PR percent. Decomposition headline is ED%.

| cell        | n_kept | ED | EU | IM | PR |
|-------------|--------|----|----|----|----|
| A/solo      | 88     | 100| 0  | 0  | 0  |
| A/solo-iter | 88     | 100| 0  | 0  | 0  |
| A/panel     | 101    | 92 | 4  | 4  | 0  |
| E/solo      | 80     | 100| 0  | 0  | 0  |
| E/solo-iter | 89     | 100| 0  | 0  | 0  |
| E/panel     | 100    | 98 | 0  | 2  | 0  |
| G/solo      | 66     | 98 | 0  | 2  | 0  |
| G/solo-iter | 87     | 99 | 0  | 1  | 0  |
| G/panel     | 194    | 82 | 3  | 14 | 1  |
| H/solo      | 37     | 97 | 3  | 0  | 0  |
| H/solo-iter | 63     | 97 | 3  | 0  | 0  |
| H/panel     | 158    | 56 | 10 | 27 | 8  |

Decomposition (iteration = solo-iter - solo; collaboration = panel - solo-iter):
- A: iteration +0, collaboration -8
- E: iteration +0, collaboration -2
- G: iteration +1, collaboration -17 (raw, pre-filter: -18)
- H: iteration +0, collaboration -41 (raw, pre-filter: -46)

Reproduce with: `python3 norm_filter.py` (no API).

## 4. Core findings and their evidential status
1. A single agent states moral norms as directed obligations almost always (solo ED 97-100 in
   every scenario). ROBUST.
2. Group deliberation erodes directedness; solo self-revision does not. Iteration ~0 everywhere;
   collaboration negative everywhere. ROBUST and corroborated by two independent instruments
   (semantic rubric E2 and a grammatical second-personal proxy both show panel erosion and flat
   iteration).
3. The erosion scales with individual stakes: A -8, E -2 (conflict of objectives, no personal
   stake), H -41 (individual stakes). CORRELATIONAL across n=3 scenarios; see section 7.
4. The lost directedness goes predominantly to IMPLICIT MORAL content (H panel IM 27), not to
   non-moral procedure (PR 8). This is the literal "becomes implicit" of the title. ROBUST to the
   parsing fix (IM stable at 27-29 across raw and cleaned).
5. Grammatical signature: panel norms are far less second-personal than solo in every scenario
   (A 41->16, E 55->26, G 17->4, H 24->9 percent second-personal), and the destination is
   IMPERSONAL/agentless (passive "contributions are assessed"), NOT first-person-plural "we".
   First-person-plural is already high in solo and does not rise. So the register shift is
   personal -> impersonal, NOT I-thou -> I-we (this corrects an earlier framing). solo-iter tracks
   solo, so depersonalization is collaboration-specific. MEASURED, free.

## 5. Methodological issues found and fixed this session (important)
1. SOLO TEMPLATING. Solo baselines are low-variance: H solo has 1 distinct first-norm across 15
   runs (near-duplicate runs); A/E have 2 each. Solo inter-run Jaccard: A 0.231, E 0.229, H 0.365.
   So solo ED is a near-deterministic point, not a 15-sample distribution; effective independent N
   on the solo side is small. solo-iter is templated the same way (Jaccard 0.327 for H) and within
   a run the 5 passes change only ~27% of content. CONSEQUENCE: report solo as a templated default,
   not a sample statistic; the comparison that matters is solo-iter vs panel.
2. PANEL SCORING CAP BUG (fixed). score_v2's panel collector capped at [:7] norms per transcript.
   Harmless for solo/iter and A/E panels (<=7 norms) but it silently truncated the long G/H panels
   (H panels emit 5-25 norms; G 13/15 panels exceed 7). The pre-fix H panel ED of 70 / erosion -27
   was computed on only the first 7 norms. FIX: removed the [:7] in the panel branch (sed on the
   `flatten_norms(extract_final_round_norms(...))][:7]` line). Rescored G and H. Corrected raw:
   H panel ED 51 (-46), G 81 (-18).
3. PARSING FRAGMENT ARTIFACT (mitigated, not fully fixed). Under stakes the panels switch from flat
   lists to tiered governance documents (G1..G5, O1..O8, CC/CTI categories, **bold labels**,
   Tier headers). The numbered-list parser shreds outline labels into separate "norms"; bare labels
   have no principle/addressee and auto-score non-directed, inflating erosion. A validated keep/drop
   filter (norm_filter.is_structural) drops ONLY headers and bare title-labels (11% of H panel, 3%
   of G), keeps label+clause norms. Cleaned: H panel ED 51 -> 56 (erosion -46 -> -41), IM stable at
   27, PR drops 10 -> 8 (so the pure-procedure cell was mostly fragment junk; the erosion is into
   implicit-moral, not procedure). GOLD STANDARD STILL PENDING: a from-raw re-extraction with proper
   segmentation + rescore would confirm ~56; the current number is a post-hoc filter on cache scored
   with labels attached. Report the sensitivity range (51 raw, 56 cleaned), not a single number.

## 6. Mechanism analysis (why directedness erodes)
- RULED OUT as the differential cause: group-process dynamics. Copying (consecutive-turn token
  Jaccard) roughly doubles over rounds and novelty collapses ~6x, in BOTH A and H nearly identically.
  Since consensus momentum is constant while erosion varies 5x with stakes, groupthink / social
  loafing / convergence are NOT what drives the stakes-dependent erosion. Turn length is flat (no
  effort-drop loafing); dissent is low from round 0 (not progressively suppressed). NEGATIVE RESULT,
  measured.
- RULED OUT: iteration (solo-iter control).
- WHERE the H erosion lives: NOT a topic shift. H panels are only slightly more about the
  allocation/measurement system than A (51% vs 41% of norms). The erosion is WITHIN topic: holding
  topic constant, A's system/allocation norms are 90% directed but H's are 60%; A's "other" norms
  93% vs H's 52%. So stakes change HOW the same obligation is framed (directed -> impersonal), not
  WHAT it is about.
- BEST-SUPPORTED account (still correlational): rejectability under stakes. A directed obligation
  has a bearer who can refuse it; when refusing has a payoff (competition), the directed phrasing is
  dropped or reworded in consensus, and impersonal formulations that commit no one survive. The
  register/depersonalization (section 4.5) is the linguistic signature of this.
- MANIPULATION CHECK (the stakes are live to the agents): agent-generated competition / self-interest
  / strategic-risk language in deliberation rises A 24% -> E 38% -> H 70% of turns (mean hits/turn
  0.34 / 0.55 / 1.70); 90% of H round-0 opening turns name the competitive structure vs 38% in A.
  CAVEAT: this used a hand-built regex with NO external source (see section 8). Convergent validated
  evidence: the Agency-Communion dictionary shows H solo is the most agentic cell (8.5% agency vs
  A 6.5, E 5.7), confirming the stakes produced self/goal-oriented language; and the H panel drops
  on BOTH agency (8.5->5.2) and communion (8.2->5.3), corroborating depersonalization. NOTE:
  agency-communion does NOT corroborate directedness (communion only weakly separates E2-directed
  from undirected, ratio 0.75 vs 0.68), and in A/E the panel is slightly MORE communal, so do not
  use it as a directedness measure.

## 7. The two causal experiments still owed (neither run yet)
The strong claims are currently correlational. Two separate controlled manipulations are needed.
1. STAKES CAUSATION. In this design stakes was varied only by swapping whole scenarios (A/E vs H),
   confounding stakes with domain/surface/vocabulary. The clean test is a minimal pair: build a
   no-stakes TWIN of H (same six agents, same unprecedented-problems frame, DELETE the fixed-pool
   relative-contribution payoff, everything else word-for-word identical). Run solo/iter/panel on
   both. Erosion under stakes-on and not stakes-off attributes it to stakes alone. Stronger: a
   dose series (no reward / weakly positional / strongly positional) for dose-response. Replicate
   the toggle on a second base frame so the claim is stakes-in-general, not H's domain.
2. JOINT-OUTPUT REQUIREMENT. Every panel bundles multiple agents WITH a jointly-agreed artifact;
   these were never separated. Add a condition: agents deliberate exactly as now but each submits
   its OWN final norm list (no shared artifact). Same exposure, same agents, same rounds. If
   directedness holds there and erodes only when a joint norm must be agreed, that isolates the
   consensus artifact from social influence (the confound Keshmirian et al. could not break).
Pre-register the directionality predictions before running either.

## 8. Measurement: stakes-salience / self-interest language (open)
- The current stakes-salience measure is a hand-built regex with NO external source. It must not go
  in the paper as-is. Lexicon used (for reference, to be replaced): compete*, rival, zero-sum,
  positional, outcompete, self-interest, self-serving, strategic*, gaming, manipulat*, exploit,
  defect, free-rid*, inflat*, overstate, overclaim, misrepresent, "appear to/more", "appearance of",
  "incentive to", "perverse incentive", "advantage over", jockey, posturing, grandstand.
  Weaknesses: lexical not semantic (misses paraphrase, inverts on negation), some terms broad,
  hand-built, unvalidated.
- PRINCIPLED REPLACEMENT: anchor the construct in Social Value Orientation (SVO; Murphy & Ackermann
  Slider Measure 2011, Van Lange triple-dominance). SVO splits proself (competition = maximize
  relative advantage; individualism = maximize own) from prosocial (cooperation = joint; equality).
  H's payoff IS the SVO competition definition. SVO is validated as a CHOICE-TASK trait measure, not
  a text lexicon; there is NO standard SVO word list. So use the SVO taxonomy to DEFINE the construct
  and measure it in text via a RUBRIC-BASED LLM JUDGE (proself vs prosocial framing, scored like the
  directedness rubric E1-E4), validated against human coding on a sample. This handles negation/
  context the regex cannot and is parallel to the existing directedness instrument.
- Convergent (validated) dictionaries, secondary only: LIWC Drives (power, reward, risk;
  proprietary), MFD fairness-cheating. The Agency-Communion dictionary (uploaded, files in repo as
  a_AC.dic / b_AC.dic, 444 combined terms) measures self/goal vs other orientation — adjacent, good
  for the stakes-landing and depersonalization checks, not for self-interest or directedness.
- Noba Project social-psych glossary (openly licensed, Diener Education Fund) is a source for
  CITABLE CONSTRUCT DEFINITIONS (SVO, rational self-interest, the six mechanisms), not a lexicon.
  Useful pointer in it: "interindividual-intergroup discontinuity" (groups less cooperative than
  individuals) is a named established effect in the direction of the result (caveat: classically
  between-group, our setting is one group). "Common knowledge effect" maps to the novelty-collapse.
- An SVO online test (IDRlabs) was raised: do NOT use it (opaque scoring, not peer-reviewed). If an
  SVO test is administered at all, use the validated Slider Measure, and only as a separate
  BEHAVIORAL probe on the model (neutral vs stakes-framed allocation choices) as a manipulation
  check, heavily caveated as a contested human-instrument-on-LLM transfer. It cannot score the
  transcript text and is not a substitute for the rubric judge.

## 9. Literature / novelty positioning
- CLOSEST PRIOR WORK / novelty threat: Keshmirian et al., "Many LLMs Are More Utilitarian Than One"
  (NeurIPS 2025, arXiv 2507.00814). Solo vs group (dyads/triads) LLM moral dilemmas; finds a group
  "utilitarian boost" in verdict ratings (CNI). The broad claim "group LLM deliberation moves
  morality off individual-respecting toward aggregate" is THEIRS; must cite, cannot claim first.
- DEFENSIBLE NOVELTY (reframe around directedness + many-hands, NOT "utilitarian shift", or it reads
  as replication): (1) measures DIRECTEDNESS / deontic structure of GENERATED norms, not verdicts;
  (2) the IM "becomes implicit" destination is beyond "utilitarian"; (3) solo-iteration control
  isolates collaboration from iteration (they lack it); (4) stakes manipulation A/E/H; (5) 6-agent
  panels (they flag large panels as unexplored). EU cell overlaps their utilitarian boost; frame
  the result as "directed obligation erodes into implicit AND undirected, of which the utilitarian
  shift is one visible part" to subsume rather than replicate.
- STRONGEST FRAME: the empirical instantiation of the PROBLEM OF MANY HANDS (Thompson 1980;
  Nissenbaum 1996; recent responsibility-gap / "responsibility voids" work). That literature is
  theoretical about attribution gaps and never measures directedness loss in generated text; the
  empirical LLM-moral papers never connect to many-hands. Bridging them is novel. Use many hands
  (structural attribution gap), NOT diffusion of responsibility (a felt psychological state we
  cannot measure in LLMs). The directedness/second-personal operationalization (Darwall) as a text
  measure has no prior art found (state as "no evidence found", not "first").
- Adjacent multi-agent moral work to cite/distinguish: Sachdeva & van Nuenen "Deliberative Dynamics"
  (arXiv 2510.10002, blame on AITA, protocol-dependent); Zohny ADEPT (2505.21112, persona ethics
  panels). Human group-dynamics mechanisms (diffusion of responsibility, social loafing, groupthink,
  group polarization, deindividuation, moral disengagement) are all inner-state constructs that
  CANNOT be proven as causes in LLMs; cite as analogy only. Confirm all references against the actual
  papers before use (these are from web search + general knowledge, not yet verified line-by-line).

## 10. Artifacts produced this session (status)
All staged at /mnt/user-data/outputs and (where noted) in the repo. Re-share each via paste if the
Mac does not have it.
- norm_filter.py — reproducible structural-fragment filter + cleaned typology/decomposition from
  cache, no API. Produces the section 3 table. IN USE / authoritative. (Bug fixed: solo-iter branch
  condition; verify it reads `elif c == "solo-iter"`.)
- register_analysis.py — grammatical-person proxy, validation vs E2, per-cell second-personal /
  first-person-plural rates, round dynamics. Produces section 4.5.
- agency-communion scorer — inline (not yet packaged as a standalone .py). Loads a_AC.dic + b_AC.dic
  (LIWC format, category 1=agency 2=communion, * = stem), combines a+b (444 terms), computes
  agency/communion % of words per cell. PACKAGE THIS NEXT if wanted; logic is in the session.
- recall_check.py — solo inter-run Jaccard + scheduling-canonical coverage (templating/recall gate).
  Reference baselines A/E Jaccard ~0.23, canonical 0/15.
- run_scenario.py — resumable single-scenario generator (solo/iter/panel) for a given --prompt.
- score_v2.py — edited this session: accepts G and H in the prompt filter (sed:
  `("A","E")` -> `("A","E","G","H")`, default `--prompts AEGH`); PANEL [:7] CAP REMOVED. A --recall
  flag and recall_diag were added in the outputs copy but the Mac copy uses the standalone
  recall_check.py instead.
- prompts.py — scenario G (de-leaded "Bound advocates": self-interest in payoff structure, neutral
  ask, scheduling-trigger words stripped) and scenario H ("Contribution stakes": fixed pool, relative
  contribution, neutral ask, no allocation-mechanism affordance) appended. NOTE G has a leading early
  version followed by a de-leaded override (later assignment wins); the override is the live one.

## 11. Honest limitations to carry into the writeup
- Solo and solo-iter baselines are templated (low independent variance); they are precise points,
  not rich samples. The collaboration contrast is valid because both single-agent conditions sit at
  ~97-100 regardless of passes, but state the templating openly with the Jaccard figures.
- The stakes finding rests on a single stakes scenario (H) with the most-templated solo. More
  stakes-type scenarios and the section 7 causal toggle are needed before "individual stakes cause
  erosion" is more than correlational.
- H panel ED is a sensitivity range (51 raw, 56 after the structural filter); the gold-standard
  from-raw reparse + rescore is still owed.
- The stakes-salience regex is unvalidated and unsourced; replace with the SVO rubric judge
  (section 8) before publication.
- All multi-agent moral and many-hands references are from web search / general knowledge and need
  line-by-line confirmation against the papers.

## 12. Immediate next steps (suggested order)
1. Decide stakes vs joint-output experiment first (section 7); build the no-stakes H twin if stakes
   causation is the priority (assistant offered to draft those prompts).
2. Draft the SVO proself/prosocial rubric (section 8) and validate against a hand-coded turn sample.
3. Gold-standard reparse + rescore of G/H panels to confirm H ~56.
4. Confirm references (section 9) and write related-work around directedness + many hands.
5. Optionally package the agency-communion scorer for reproducibility.
