# LP→CP agent: system, capability survey, pilots

> Original design document, dated Nov 10, 2026 (Yixi Zhao). Preserved verbatim
> as the plan that the Stage 0 pilot in this repo was built to test. The pilot's
> results and how they bear on this plan are in `../HANDOFF.md`.

The defensible contribution is **budget allocation under a closure/interface conflict** — not orchestration, not constraint compilation, and not deeper evaluation. All three of those are now established in the literature, two of them positively and one as a confound that must be controlled away before the evidence-allocation hypothesis can be tested at all.

## Division of labor

The LLM never emits a residue identity. It emits a design problem; specialized solvers emit molecules under the constraints it chose. Every residue in a final design traces to ProteinMPNN, GenKIC or BoltzGen operating inside an LLM-chosen mask.

|Layer|Owner|Why|
|-|-|-|
|Parent selection, stopping rule, budget allocation|LLM|Requires reading heterogeneous evidence against explicit quality criteria|
|Diagnosis — PLIP / BSA / strain / pLDDT output into a named defect class|LLM|The load-bearing cognitive step, and the one with no prior-art evidence|
|Design-problem specification — which positions to fix, which to open, which motif to preserve, which segment to rebuild|LLM|A constraint set, not a molecule|
|Residue identities on a given mask|ProteinMPNN / BoltzGen / Rosetta packer|Calibrated against structure; the LLM has no energy model|
|Backbone coordinates|GenKIC / BoltzGen|Geometric closure is a solver problem|
|Scores|Rosetta / Boltz-2 / geometry checks|Must be LLM-independent or the loop is self-grading|
|Mask verification, budget enforcement, caching, provenance|Execution service (deterministic)|A guarantee, not a request|

Four invariants make the experiment interpretable. Break any one and a win becomes uninterpretable.

1. The LLM never emits a sequence. If it also picks residues, a win cannot be attributed to diagnosis quality rather than luck on chemistry.
2. The qualification function is frozen, hashed and versioned before the run. The agent may change what it proposes and what it accepts locally; it may never change what qualifies.
3. Every returned sequence is diffed against the parent LP **and** against the lineage's declared mask before anything is scored. Soft conditioning is a request, not a contract.
4. `defect_class` is a closed enum fixed in advance. If the agent can mint categories mid-campaign, the repair matrix becomes unanalyzable after the fact.

The diagnosis is a typed object, not prose. Free text means the LLM re-derives the mask on every call, so a failure cannot be attributed to bad diagnosis versus bad compilation — and the "aggregate scores only" arm stops being a well-defined difference in fields present on one object.

```json
DiagnosisRecord {
  candidate, defect_class, defect_locus,
  preserved_features[], evidence_refs[], confidence
}
```

## The control loop

[embedded content: one iteration · 8 steps, 3 LLM-owned]

Every iteration runs the same eight steps. Step 4 catches the case where the LLM opens three positions with only two substitutions left: the service does not ask, it enforces the predicate at the filter stage. Step 6 catches a solver that silently moved a position declared fixed — the failure mode PROTEAN identifies as scope leakage and one of its two dominant bottlenecks.

## Worked example

LP-04 = `ADVWSGRLTNKP`, a 12-mer linear binder of target T. Mutation allowance k = 4 relative to LP. Goal: a head-to-tail cyclic analog.

Frozen qualification, set before the run:

```
ipTM >= 0.80 AND >=3/5 seed mode-consensus
BSA >= 550 A^2
Rosetta ddG_bind <= -30 REU
max per-residue backbone strain <= 4.0 REU
free peptide: bound conformer >= 5% of the top-10 kcal/mol window
edit distance to LP <= 4
```

**Iteration 0 — intake.** Deterministic. Biological-assembly check on T, pocket mapping, PLIP on the LP–T complex to record the parent's contacts. Constrained generation: BoltzGen with cyclic topology, LP identities fixed at mapped hotspots, 200 candidates, Boltz-2 at 5 seeds, Rosetta interface scoring. Zero qualify; 14 near misses.

The interesting one is **C-017** = `cyclo(ADVWTGRLGNKP)`, 2 substitutions used, 2 remaining: ipTM 0.81, BSA 620 A² with H-bonds from W4 and R7, ddG -32 REU — all passing — and max strain 11.2 REU at N10-K11-P12, failing. A good interface on a bad closing turn. One-shot generation discards it.

**Iteration 1 — diagnose, specify, compile.** The LLM reads the PLIP table, per-residue strain decomposition, ring geometry and seed ensemble, and emits one typed object: `defect_class: BACKBONE_STRAIN_AT_CLOSURE`, locus 10–12, preserved features = W4 and R7 identities, both parent H-bonds, and BSA floor 527 (0.85 × parent). The design problem opens 10–12 for pivots and redesign, fixes 1–9, leaves 2 substitutions.

The compiler catches that three positions were opened against a two-substitution budget and enforces the predicate at the filter stage. GenKIC returns 41 closed solutions, 12 survive the strain filter, ProteinMPNN proposes 8 sequences, 6 are rejected on budget. Verification discards one more: an MPNN output had quietly moved position 9.

Result: **C-041** = `cyclo(ADVWTGRLGDGP)`, 4/4 substitutions. Strain 3.1 REU — defect resolved. BSA 495 A² — contract violated.

> This single row is the 2→3 ablation. Arm 2, which gets the mask but no preservation contract, counts C-041 as a repair because the ring closed. Arm 3 does not. If backbone sampling can always buy closure by spending the interface, Arm 2's repair rate measures nothing — and this makes that demonstrable rather than asserted.

**Iteration 2 — tighten.** The turn rebuild rotated the backbone enough to pull R7 off E191. Narrow pivots to 11–12, restrain N10 phi/psi, add coordinate constraints on W4 and R7 side-chain heavy atoms through both GenKIC and relax. **C-047** = `cyclo(ADVWTGRLGNGP)`, 3 substitutions: strain 2.8, BSA 601, both H-bonds, ipTM 0.83 at 4/5, ddG -34. Qualifies at search time.

**Iteration 3 — spend evidence, not mutations.** ipTM 0.83 falls in the escalation band, so the evidence plan fires: 10 seeds rather than a new edit. Mean ipTM drops to 0.79, 6/10 consensus. The LLM then spends the next block on free-peptide conformer sampling instead of another substitution: the bound-like population is 2.4%, below the 5% floor. C-047 binds well from a state it rarely occupies.

That is a non-greedy decision — the agent burned budget to disqualify its own best candidate. It is also the step with the weakest prior-art support, for reasons the survey below sets out.

**Iteration 4 — explicit relaxation.** Rigidifying the free ring wants a change at position 6, but the contract requires both H-bonds, BSA ≥ 527, W4/R7 fixed, and one substitution remains. The feasible set is empty. The agent records a `RelaxationEvent` — lineage branches, the two-H-bond clause weakens to one, with justification and evidence refs — rather than drifting silently. **C-052** = `cyclo(ADVWTPRLGNGP)`, 4/4: conformer population 11%, BSA 560, W4 retained, R7 lost, ipTM 0.81 at 7/10, strain 3.3. Qualifies under the relaxed contract, flagged.

Relaxation events are a result, not plumbing. Their rate against k is direct evidence for the closure–interface conflict the whole paper presumes exists, and is probably the cleanest quantitative support for that premise.

**Termination.** Stop on budget exhaustion, two consecutive iterations with no contract-holding improvement, or n qualifying designs. The shortlist is frozen by the qualification function with no agent input into ranking, then every method's frozen shortlist goes through the same held-out protocol: fresh folding seeds, a second predictor, chemistry-aware refinement, extended free-peptide sampling. The reportable number is the search-time-to-held-out gap per method, not the win.

## Refinement, memory, reflection

Three distinct mechanisms, not two. Refinement is the core; retrieved memory is the main addition; reflection is an ablation whose benefit must be measured, not assumed.

|Mechanism|What it does|Where it appears in the trace|
|-|-|-|
|Iterative refinement (external feedback, not Self-Refine's self-feedback)|Modifies one candidate from its own evaluation|Iterations 1–2: C-017 → C-041 → C-047|
|Episodic memory|Retrieved outcomes from other attempts condition the next decision|Would have flagged that charged substitutions at the turn's i+2 position broke closure before iteration 1 spent its budget|
|Reflection (experience distillation)|Turns outcomes into reusable lessons|"On this target, backbone repair has outperformed further side-chain substitution" — a hypothesis tagged to its supporting attempts, never a biochemical rule|

The archive stores everything for reproducibility; storing alone influences nothing. Retrieval is what conditions a decision, and it is scoped — current diagnostics plus relevant prior successes, failures and contradictory evidence, never the whole archive in every prompt.

History enters the loop as a **proposal prior** — a soft down-weight on edit classes that keep failing, scoped to where it was observed and gated on a count fixed in advance. Not a ban: a ban drawn from n=1 noise permanently deletes a region of design space and you never notice.

That gives the five arms, tools and budget matched throughout:

|Arm|Diagnosis flows into|
|-|-|
|1|Nothing — fresh proposals|
|2|Conditioning only (masks, pivots, fixed identities)|
|3|+ preservation contract|
|4|+ history-derived proposal prior|
|5|+ distilled reflections|

2→3 tests whether protecting a partial solution is what makes repair work. 3→4 tests whether broader history adds anything. 4→5 tests whether distilling experience beats retrieving the evidence itself.

## LLM capability survey

One responsibility is demonstrated, two are demonstrated-but-already-claimed, one has negative evidence, and the three that carry the paper have no prior-art answer at all.

|LLM responsibility|Status|Evidence|
|-|-|-|
|Compile a specification into backend tool arguments|**Proved**|[PROTEAN](https://arxiv.org/abs/2610.17765): 380 sub-tasks, six intervention families, five frontier LLMs × three backends. Constraint compilation is already highly accurate; the bottlenecks are backend scope leakage and refolding failure. End-to-end biophysical success starts at 7.0%|
|Select appropriate tools for a design stage|**Proved**|[BioDesignBench](https://www.biorxiv.org/content/10.64898/2026.05.06.723381v2.full): agents generally pick appropriate tools; top agents beat a deterministic hardcoded pipeline (58.6–60.6 vs 54.5). [ABLE](https://arxiv.org/abs/2609.05818): best models score highest on tool selection|
|Detect an infeasible specification|**Partly proved**|[MolDesignBench](https://arxiv.org/pdf/2609.27349): agents detect infeasible specifications far more reliably when constraints are explicit rather than implicitly embedded in natural language|
|Allocate evidence — decide when to re-evaluate rather than re-edit|**Negative evidence**|BioDesignBench: no LLM condition discarded a candidate across all 836 task–condition observations; scoring tools invoked at ~14% of expert intensity. A forced-depth prompt recovers most of the gap (+9.3 and +15.9 points under matched compute)|
|Turn structured numeric diagnostics into a named defect class|**Not tested anywhere**|Adjacent negative signal: agents lag the expert on Approach (9–11 vs 14), and ABLE finds models inconsistent in planning, strategy generation, and integrating biological knowledge with tool use|
|Allocate a scarce mutation budget across positions|**Not tested**|Every system above operates inside a pre-given region mask; none has to decide where to spend|
|Choose which objective to sacrifice when the contract is infeasible|**Not tested**|No analogue in the literature|
|Distil outcomes into reusable lessons that beat retrieval|**Not tested in this domain**|Reflexion and Self-Refine establish the mechanism generically; neither distinguishes it from retrieving the evidence directly under expensive external verifiers|

### The one that needs care

The evidence-allocation row is the problem. BioDesignBench concludes the deficit is behavioral rather than capability-limited — the capability is present but not reliably expressed under default prompting. A plain forced-depth instruction captures most of what "the agent allocates evidence well" would buy.

So every arm must be forced-depth prompted, and the escalation rule has to beat **uniform** forced depth at matched compute, not a shallow default. One useful detail: applying the same intervention to a deterministic pipeline with multi-metric evaluation already built in produced no improvement (Δ = +1.6, n.s.), so the tuned-fixed-workflow baseline is immune to the trick and is the right control.

## What this does to the framing

The contribution narrows, and sharpens.

**No longer available as claims.** That an LLM can orchestrate protein-design tools (BioDesignBench, ABLE, T-REX). That an LLM can emit constraints rather than molecules (PROTEAN — cite as foundation, not contribution). That evaluating more deeply improves designs (BioDesignBench, achieved by prompt).

**Still open, and ours.** Whether an LLM can read structured structural diagnostics and decide *where to spend a scarce edit budget* — the allocation decision that a region mask hands out for free in every prior system. That is the 2→3 arm, and it is the only one with no prior-art answer.

The honest framing: PDAgent shows orchestration works when verifiers are cheap and objectives are decoupled. We ask whether it works when verifiers are expensive and noisy and the objectives compete for the same few editable positions. Also worth saying explicitly — Agent Rosetta found that prompt engineering alone often fails to generate valid Rosetta actions, so a meaningful share of the engineering is the deterministic execution service, not the prompt. Reviewers otherwise read "LLM coordinates tools" as trivial.

The bad news is that none of this literature gives a prior that the allocation step will work.

## Which harness to mimic

Three-way split. The loop skeleton comes from PDAgent because that is what reviewers will compare against; the prompt scaffold comes from Agent Rosetta because it is the only one published at template granularity and the only one whose LLM emits constraints rather than molecules.

|Source|Take|Leave|
|-|-|-|
|[PDAgent](https://github.com/Gift-OYS/PDAgent)|ReAct five-phase loop (THINK, PLAN, ACT, OBSERVE, REFLECT); coordinator shape — `ProteinDesignCoordinator(config).design(request, name)` returning FASTA + PDB + JSON report; repo layout (`agent/`, `tools/`, `bioinformatics/`, `data/`)|Its action space — the LLM emits point mutations directly, which is the line our design forbids crossing|
|[Agent Rosetta](https://arxiv.org/html/2603.15952v1)|The whole prompt scaffold (Appendix B.2), below|Rosetta-specific action vocabulary|
|[OpenDDE-Harness](https://github.com/aurekaresearch/OpenDDE-Harness)|The candidate lineage dashboard — our relaxation events and repair outcomes need a viewer|Everything else: it is a product, with no published prompts, benchmarks, success rates or controller ablation|

### The Agent Rosetta scaffold

* A system prompt with a `{reasoning_formatting}` slot normalizing reasoning and non-reasoning models to the same output shape — needed to compare models across arms fairly
* A design brief templating `{prompt}`, `{state}` (textual environment state) and `{action_formatting_instructions}`
* Action history as a **tabular `{summary}`**, not a transcript — directly the shape the retrieved-memory arm needs
* A **two-call action pattern**: choose the next action, then re-prompt with the relevant documentation appended to generate the full call with parameters
* An explicit error-correction re-prompt on syntax or runtime failure

From [ABLE](https://arxiv.org/abs/2609.05818) instead, take the tagged terminal output convention (`<pdb_id>…</pdb_id>`) for extracting `DiagnosisRecord` fields.

### Two things this raises

**PDAgent's REFLECT phase is our arm-5 ablation, already named.** Read what it actually does. Experience distillation across candidates means the reflection arm is contested and the framing needs adjusting; per-step self-critique means we are clear. The poster does not say — this needs the `agent/` source.

**"Which prompt style" is half the wrong question.** Agent Rosetta's own headline finding is that prompt engineering alone often fails to generate valid Rosetta actions, and that tool and environment design are what make it work. The transferable asset is the action schema plus deterministic validation and retry — which is the execution service, and which no amount of prompt copying substitutes for.

## Pilot tests

Test the hypothesis in a controlled, ground-truth setting before building end-to-end, then engineer the prompt and tool descriptions against it. The construction is an offline decision benchmark built from a one-time exhaustive enumeration — which doubles as a campaign simulator.

### Stage 0 — SKEMPI allocation slice

Cost: near zero. Tests allocation only, with no closure objective.

[SKEMPI 2.0](https://academic.oup.com/bioinformatics/article-pdf/35/3/462/48964825/bioinformatics_35_3_462.pdf) holds 7,085 experimentally measured ΔΔG records on structurally resolved complexes, including single and multiple mutations, some measured in the background of a third mutation, plus hold-out metadata for avoiding binding-site contamination. For complexes with 200+ entries this is close to a budget-allocation oracle at k = 1–3, with experimental rather than predictor ground truth — which also avoids the self-grading problem in PDAgent's validation.

Limits: linear protein–protein, no closure objective, and coverage is opportunistic (alanine scans dominate), so the oracle is best-among-measured, not best-overall.

**Kill criterion.** If the LLM cannot beat a hotspot/conservation heuristic at position selection where ground truth is experimental and the objective is single, the cyclic version will not rescue it. Stop there.

### Stage 1 — enumerated cyclic parents

One large compute run, then every later test is free.

1. Fix 20–40 failed parents from one unguided generation run, stratified across the `defect_class` enum with roughly balanced cells.
2. Enumerate the action space exhaustively per parent — every locus × operator × budget-split on a coarse grid, 60–150 interventions each, run to saturation with recorded variance.
3. Label every cell under the frozen qualification function **and** that parent's preservation contract: `CONTRACT_HOLDING_REPAIR`, `DEFECT_RESOLVED_CONTRACT_BROKEN`, `NO_CHANGE`, `INFEASIBLE`.

The agent sees only the diagnostics and ranks interventions. Score top-1 and top-k regret against the empirical oracle, per defect class.

Why this predicts end-to-end performance rather than merely correlating with it: the table answers every query a controller would have made, so a whole campaign replays offline at any budget T. End-to-end success becomes a deterministic function of measured per-decision regret, and hundreds of prompt variants can run against the simulator in an afternoon.

Fix the evidence budget per decision across arms here. That isolates diagnosis quality from evaluation depth and removes the BioDesignBench confound by construction.

### Stage 2 — end-to-end

Does offline regret predict campaign success? Run the five arms with the engineered prompts, held-out protocol on every frozen shortlist.

### Three things to settle before starting

* **Held-out split, declared first.** Tuning prompts against the oracle table is fitting to it. Split by target, not by parent, and freeze before touching anything. Otherwise Stage 2 underperforms Stage 1 and the cause is unrecoverable.
* **Joint interventions in the grid.** If the oracle covers only single interventions, you cannot detect whether the LLM finds coordinated moves (open position 6 *and* rebuild 10–12) — which is a plausible place for it to actually beat a rule-based controller.
* **A tie-break rule for two simultaneous defects.** Needed before the run, or the repair matrix has rows assigned post hoc.

Worth chasing: PDAgent's `agent/` source and paper, to confirm whether memory and reflection exist there (if they do, the ablation framing needs adjusting), and whether PROTEAN's constraint-compilation accuracy holds when constraints *conflict* rather than merely being complex.

## Sources

|Work|What it supplies|
|-|-|
|[PROTEAN](https://arxiv.org/abs/2610.17765)|Constraint compilation is accurate; scope leakage and refolding are the bottlenecks|
|[BioDesignBench](https://www.biorxiv.org/content/10.64898/2026.05.06.723381v2.full)|Zero discarded candidates in 836 observations; forced-depth intervention and its compute-matched control|
|[ABLE](https://arxiv.org/abs/2609.05818)|Frontier-model performance on tool selection and use; inconsistency in planning and strategy|
|[MolDesignBench](https://arxiv.org/pdf/2609.27349)|Infeasibility detection depends on constraints being explicit|
|[Agent Rosetta](https://arxiv.org/html/2603.15952v1)|Environment design, not prompting, is what makes Rosetta integration work|
|[SKEMPI 2.0](https://academic.oup.com/bioinformatics/article-pdf/35/3/462/48964825/bioinformatics_35_3_462.pdf) · [HF mirror](https://huggingface.co/datasets/LiteFold/SKEMPI2)|Experimental ΔΔG ground truth for the Stage 0 allocation slice|
|[Cyclic peptide conformer benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC13656215)|uPA, C3c compstatin analogs, HIV integrase binders — the compstatin series is a real mutation-budget trajectory|
|[MacroPPI-Bench](https://github.com/sambhajithakar/macroppi-bench)|77 RFpeptides designs joined to synthesis and binding outcome; a calibration set for qualification thresholds|
|[HADDOCK cyclic-peptide benchmark](https://github.com/haddocking/cyclic-peptides)|30 cyclic peptide–protein complexes|
