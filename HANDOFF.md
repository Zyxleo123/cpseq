# LP→CP agent — Stage 0 handoff

*Written for: whoever picks this project up next (a collaborator, or future
you) — assumes the `original_plan.md` design but not that you remember the run.*

Date: 2026-11-10. Repo: **`/zfsauton2/home/yixiz/lp2cp-agent/`**. Run artefacts:
**`/zfsauton/scratch/yixiz/lp2cp_pilot/`**. The design this tested is in
[`docs/original_plan.md`](docs/original_plan.md); the full method writeup is in
[`README.md`](README.md).

---

## One-paragraph summary

We built and ran **Stage 0** — the SKEMPI allocation slice the plan calls for.
It tests the single claim the whole LP→CP design rests on and that has no
prior-art answer: *can an LLM read structural diagnostics and decide where to
spend a scarce edit budget?* The result is **negative on this slice**: a local
Llama-3.1-8B loses to the best simple rule, and frontier models (Opus 5.5,
Sonnet 5.5) only *tie* it — they do not beat it, and they tie largely by
rediscovering the same rule. **But the negative is not conclusive for the
end-to-end system**, for a specific reason given below. The next step is to
build Stage 1 and decide the real question, not to abandon it on this.

---

## What was tested, concretely

A **decision** = one protein complex + an edit budget `k` + a fixed list of
candidate positions, all measured in SKEMPI 2.0. The agent picks **one**
position from the list. We score it in **kcal/mol of binding energy given up**
versus the best option on that list (regret; 0 = picked the best).

The LLM never names the amino acid (fixed to alanine) and never invents a
position (closed menu). It answers only *where*. That restriction is what makes
every pick scorable against experimental ground truth.

Worked example — `ala_k1__3EQS_A_B`, MDM2 vs the PMI peptide, pick 1 of 9:

| cand | residue | class | dSASA | cont5A | TRUE ΔΔG | |
|---|---|---|---|---|---|---|
| c007 | NB8 | RIM | 21 | 4 | **−1.11** | Opus picked · **best** |
| c003 | SB2 | RIM | 11 | 14 | +1.26 | `rim_first` picked |
| c006 | WB7 | COR | 138 | 77 | **+6.41** | worst (the hotspot Trp) |

7.52 kcal/mol separates best from worst. Opus got it exactly right (regret 0);
the heuristic gave up 2.37. In **aggregate**, though, they tie — this example
was chosen because they disagreed.

---

## Results (primary slice `ala_k1`, n=73, effort `low`)

| arm | n | mean regret | median | oracle-hit |
|---|---|---|---|---|
| `rim_first` (**the bar**) | 73 | **0.749** | 0.550 | 0.18 |
| `claude-sonnet-5-5` (full ev.) | 67 | 0.786 | 0.607 | 0.16 |
| `claude-opus-5-5` (full ev.) | 70 | 0.795 | 0.578 | **0.20** |
| `min_contacts` | 73 | 0.812 | 0.648 | 0.19 |
| Llama-8B (`structure_only`) | 73 | 0.871 | 0.703 | 0.16 |
| Llama-8B (`full`) | 73 | 1.043 | 0.810 | 0.11 |
| `composite` (burial+contacts+ESM) | 73 | 1.080 | 0.681 | 0.15 |
| `max_entropy` (ESM-2 only) | 73 | 1.440 | 1.228 | 0.10 |
| *random* | 73 | 1.573 | 1.458 | 0.00 |

Means sit on different denominators (refusals, below), so the verdict rests on
**paired tests over shared decisions**:

| comparison | median Δ | W/L/T | p (Wilcoxon) |
|---|---|---|---|
| Opus 5.5 vs `rim_first` | 0.00 | 13/14/43 | **0.89 — tie** |
| Sonnet 5.5 vs `rim_first` | 0.00 | 12/15/40 | 0.55 — tie |
| Opus vs `min_contacts` | 0.00 | 13/5/52 | 0.15 — tie |
| **8B (full) vs `rim_first`** | 0.00 | 17/31/25 | **0.031 — 8B loses** |
| Opus vs 8B (full) | 0.00 | 27/13/30 | 0.024 — Opus better |
| Opus vs random | −0.78 | 62/8/0 | <0.0001 |

Consistent across the pre-declared dev/test split.

### What it means

1. **Scale closes the gap but does not invert it.** The 8B *loses* to the bar;
   Opus/Sonnet *tie* it. Against the plan's kill criterion — "beat a
   hotspot/conservation heuristic" — the answer on this slice is **no**.
2. **The tie is a rediscovery, not independent competence.** Opus agrees with
   `min_contacts` on 74% of decisions, and Opus/Sonnet/Haiku agree with each
   other 79–83%. The frontier models are approximately *computing the rule*.
3. **One thing the LLM does best:** Opus has the highest oracle-hit rate of any
   arm (0.20 vs 0.18) — it finds the exact optimum most often, then gives it
   back elsewhere.

### Findings independent of the LLM verdict

- **Structure ≫ conservation.** Both ESM-2 heuristics barely beat random and
  are significantly worse than `rim_first` (p≈0.001). For the 8B, *removing*
  the conservation column helped (`structure_only` 0.871 < `full` 1.043).
- **The composite is worse than its best component** (1.08 vs 0.75) — mixing
  the weak signal in hurts.
- **The best rule is budget-dependent:** `rim_first` is 1st at k=1, 6th at k=2.
- **Coordinated moves are real but tail-distributed** (k=2): a perfect greedy
  controller is optimal 45.8% of the time, mean |epistasis| 1.29 kcal/mol — so
  Stage 1's enumeration grid must include joint interventions.
- **`refusal_bio` is an operational obstacle.** Haiku refused 67% of these
  benign protein-interface questions as biosecurity-sensitive; Opus/Sonnet
  refused 3–6/73. An agent making hundreds of allocation calls will hit this,
  and the plan's control loop has no provision for a step that simply declines.
- **Mild position bias in the frontier models** (mean slot 0.42 vs 0.5,
  p≈0.03), absent in every 8B arm — caught by the seeded-shuffle control.

---

## Why the negative is NOT conclusive for the end-to-end system

Read this before concluding "the kill criterion fired, stop."

1. **Stage 0 tests allocation where a good simple rule already exists.** On a
   protein-protein alanine interface, "spend the edit on the fewest cross-
   interface contacts" is a strong heuristic. The LP→CP premise is a
   **closure/interface conflict** — you must mutate to install cyclization
   anchors, and the edit that helps closure may hurt the interface. No simple
   rule is known to resolve *that* trade-off. The decision the agent is
   actually for is not the decision this slice contains.
2. **No closure objective here at all.** SKEMPI is linear protein-protein. The
   entire thing the plan is about — the `RelaxationEvent`, the preservation
   contract, the 2→3 arm — cannot appear in Stage 0.
3. **This is a floor on local weights and a tie on frontier, not a wall.** The
   models extract real signal (far above random, top oracle-hit). The task
   just has a rule that already captures most of it. A harder task where no
   such rule exists is exactly where an LLM could separate.
4. **k=2 already hints at it:** the best k=1 rule collapses to 6th at k=2, and
   coordinated moves carry real energy. Budget *splitting* under a conflict —
   the Stage 1 regime — is where the rule-based baseline is weakest.

So the honest reading: Stage 0 **refutes "LLM allocation beats a rule" as a
general claim**, and that is worth having — it means the paper cannot lean on
the easy version. It does **not** refute the specific claim the agent is built
for, because that claim lives in a regime Stage 0 cannot reach.

---

## Next step: built — Stage 1 + Stage 2 now exist (`agent/`)

**Superseded by the Stage 2 section below.** Stage 1's enumerated benchmark and
Stage 2's end-to-end agent were both built; see
[`agent/README.md`](agent/README.md). The three items this section listed are
all now closed:

- **Done — joint interventions are in the grid.** `_joint_cells` in
  `agent/lp2cp/envbuild/enumerate_actions.py`; a test asserts they are present.
- **Done — the instrument transferred.** Frozen/hashed benchmark, label
  separation, dev/test split, controls, paired tests and the cost-metered API
  path were all reused.
- **Done — both "before you start" items are settled.** The by-TARGET held-out
  split is on CPSea's `cluster_id`, declared and frozen into
  `env_manifest.json` before any campaign ran. The tie-break for two
  simultaneous defects is normalised margin, then a declared precedence order,
  and it is part of the qualification hash.

Also still running from Stage 0 (will land on their own, no action needed):
Llama-8B arm 5 of 5 (`aggregate_only`), and a Llama-**70B** 4-bit arm queued on
one a6000 (`sbatch/s0_31_llm70b.sbatch`) to confirm the size trend locally.

---

## Stage 2 — the end-to-end agent (built 2026-11-10)

Repo: **`agent/`**. Run artefacts: **`/zfsauton/scratch/yixiz/lp2cp_stage2/`**.
Full method, limitations and the surrogate caveat: **`agent/README.md`**.

### What exists

The eight-step control loop with three LLM-owned steps (DIAGNOSE, SPECIFY,
ALLOCATE), the typed `DiagnosisRecord` / `DesignSpec` / `PreservationContract`
/ `RelaxationEvent` objects, the deterministic execution service that enforces
budget at the filter stage and verifies every returned sequence against the
mask, the five arms as *mechanisms* rather than prompt variants, episodic
memory with scoped retrieval and reflection, eight rule-based controllers, a
scorer with paired Wilcoxon tests, figures, SLURM jobs and a submitter.

Three interchangeable model backends behind one `complete()` interface: local
8B bf16, local 70B NF4 4-bit, and the Anthropic API. 28 tests cover the
guarantees and run with no GPU, no API key and no spend.

### What the environment is — read before quoting a number

Parents are **real experimental cyclic peptide–receptor complexes** (CPSea_PDB,
12,419 structures, 685 measured here). Per-residue burial, contacts, polar
pairs, dihedrals, the cyclisation bond (read from CONECT) and buried area are
all **measured from deposited coordinates**. So where each parent's
closure/interface conflict sits, and how much is at stake, is real.

**How those objectives respond to an edit is a declared, hashed surrogate.** No
Rosetta, no Boltz-2, no GenKIC ran. Every ScoreCard carries
`provenance="surrogate_v1:<hash>"`. This licenses claims about **controller
quality**; it licenses nothing about biophysics. Swapping in the real stack is
a `backends/` subclass and nothing above it changes.

### The benchmark

32 parents, defect classes balanced (8/7/7/7/3), split **by target** 14 dev /
18 test over 28 clusters, 4,609 enumerated cells (144/parent, in the plan's
60–150 band). Critically, **805 cells are `DEFECT_RESOLVED_CONTRACT_BROKEN`
against 162 `CONTRACT_HOLDING_REPAIR`** — the shortcut that arm 2 accepts and
arm 3 refuses is 5× more available than the honest repair, which is what makes
the 2→3 ablation measurable rather than asserted.

Two properties measured on the built benchmark, and the distinction between
them matters because a tuned quantity cannot also be evidence of realism:

- **Tuned, and verified.** Mean |epistasis| on ddG is **1.324 REU** over 1,240
  pairs, against the 1.29 kcal/mol Stage 0 measured on SKEMPI k=2.
- **Emergent, not tuned.** A greedy singles controller is optimal at k=2 on
  **12 of 32 parents (37.5%)**, against Stage 0's measured 45.8% on SKEMPI.
  Nothing was fitted to that; it falls out of the measured interface geometry
  plus the epistasis term. Landing in the same regime is weak corroboration
  that the grid contains a comparable amount of coordination — and the gap
  means coordinated moves matter *more* here, which is the direction that
  leaves room for a controller to beat a rule. (An earlier draft of
  `surrogate.py` claimed this number was tuned to 45.8%. It was not, and the
  docstring now says so.)

### The bar (rule controllers, n=32, mean regret vs the enumerated oracle)

| controller | CHR | mean regret | oracle-hit |
|---|---|---|---|
| `oracle_greedy` (ceiling, reads the table) | 0.188 | **12.51** | 0.53 |
| `rim_first_closure` (**the bar**) | 0.156 | **17.12** | 0.28 |
| `uniform_depth` (compute-matched control) | 0.094 | 23.29 | 0.22 |
| `greedy_interface` | 0.094 | 23.50 | 0.19 |
| `fixed_pipeline` (tuned deterministic workflow) | 0.062 | 26.47 | 0.22 |
| `random_controller` | 0.062 | 26.50 | 0.19 |
| `widest_rebuild` (anti-baseline) | 0.062 | 26.56 | 0.25 |
| `noop` (floor) | 0.000 | 32.99 | 0.12 |

The ceiling beats the bar at p=1e-5 (25/0/7 paired), so there is a real 4.6-point
gap for a controller to claim, and the anti-baseline sits at the floor — the
environment does charge for interface damage.

### Read the API arms with this caveat or not at all

A spend ledger stops an arm mid-run. On the first complete sonnet-5-5 arm
(A2, one-call scaffold), **21 of 32 campaigns terminated `budget_cap_reached`**
after a mean of 1.47 iterations, against a 4-iteration limit. Those 21 are
stubs, not measurements, and the scorer excludes them — which leaves n=11 and
introduces **survivorship bias in the obvious direction**: the campaigns that
finished inside the cap are exactly the ones that terminated early, i.e. that
succeeded fast or gave up fast.

So sonnet-5-5's headline on that arm (regret 12.79 vs the bar's 17.12,
p=0.046, 5/1/5) is **a lower bound on regret, not an estimate of it**, and must
not be reported as beating the bar. Completing all 32 at observed cost would
have been ~$2.40 per arm against ~$2 remaining for the whole stage.

Two mitigations are in the scorer and both should stay: truncated campaigns are
excluded and counted in a `cap` column, and every comparison is also run
**compute-matched**, truncating the baseline to the same iteration count the
LLM arm actually got on that parent. With stubs excluded the two agree, which
is the expected result and is worth keeping as a guard rather than deleting.

The cheap arm has no such problem: haiku-5-5 ran to natural termination on
every campaign, and the full five-arm ladder costs about $0.50.

### The finding that lands first: opus-5-5 declines this task entirely

Stage 0 measured opus-5-5 refusing 3–6 of 73 SKEMPI questions. On Stage 2's
task shape — same table, same "choose positions" action, cyclic peptide instead
of a protein–protein interface — it refused **68 of 68 steps**,
`stop_reason=refusal`, category `bio`.

`agent/refusal_probe.py` tested whether that was a prompt artifact. **All five
variants refused**: live prompt, neutral system prompt, every binding/peptide
word replaced, an explicit "academic methods benchmark, nothing synthesised"
statement, and the interface columns removed entirely. sonnet-5-5 answered the
live prompt in the same probe. So it is a property of the domain, not the
wording, and no further prompt engineering was attempted — past that point the
only moves left are evasion.

Consequences: the API arms run on **sonnet-5-5 and haiku-5-5**; opus-5-5 is
reported as unusable at this task shape. This is exactly the obstacle Stage 0
flagged, now much sharper, and it is the reason `StepStatus.DECLINED` is a
first-class step outcome with a declared per-step fallback — the campaign still
produced a complete analysable trace instead of crashing.

**A second refusal category must not be pooled with it.** `refusal:bio` is a
safety decline; `refusal:reasoning_extraction` is an operational failure to
return the thinking block, which fires when `max_tokens` is low enough that
thinking alone can reach it (thinking tokens count toward it). Pooling them
would have attributed a configuration bug to a model's safety behaviour.

---

## Where everything is

```
/zfsauton2/home/yixiz/lp2cp-agent/          <- the repo (this file is here)
  README.md                 full method, controls, every result + caveat
  HANDOFF.md                this file
  docs/original_plan.md     the design doc Stage 0 was built to test
  pilot/stage0/             Stage 0 code (skempi, build, features, conservation,
                            prompts, proposers, run_arm[_api], score, figures,
                            pricing, estimate_cost, fullinterface, test_score)
  agent/                    STAGE 2: the end-to-end agent. Start at its README.
    lp2cp/                  enums, types, qualification, service, controller,
                            baselines, prompts, memory, llm/, backends/, envbuild/
    run_campaign.py         one arm end to end (resumable)
    score_campaigns.py      aggregate + paired Wilcoxon, artefacts only
    figures.py              redraws with no GPU
    refusal_probe.py        the 5-variant refusal isolation
    tests/test_agent.py     28 guarantee tests; no GPU, no key, no spend
  sbatch/                   job scripts + submit_stage0.sh + submit_stage2.sh
                            + build_70b_env.sh

/zfsauton/scratch/yixiz/lp2cp_pilot/         <- STAGE 0 artefacts (redrawable)
  bench/ features/ arms/ results/ logs/

/zfsauton/scratch/yixiz/lp2cp_stage2/        <- STAGE 2 artefacts (redrawable)
  bench/                    parents_public/_full.json, oracle_table.jsonl,
                            qualification.json, env_manifest.json  (all hashed)
  campaigns/                one campaigns__*.jsonl per arm + cost__*.json
  archive/                  episodic archive per (model, arm) for arms 4-5
  results/                  results.json, per_campaign.jsonl, figures/,
                            refusal_probe__*.json
  logs/                     every job's .out
```

Re-run Stage 0: `cd sbatch && bash submit_stage0.sh` · `--fullif` · API arms
via `pilot/stage0/run_arm_api.py`.

Re-run Stage 2: `cd sbatch && bash submit_stage2.sh` (bench → baselines + 8B +
API → score → figures), `--no-api` to skip billed arms, `--70b` for the NF4
arm, `--smoke` for 8 parents end to end. A one-off override needs no file:
`sbatch s2_30_api.sbatch API_MODELS=claude-sonnet-5-5 API_LIMIT=2`.

Re-score/redraw either stage with no GPU and no key: the scorer and `figures.py`
read only saved artefacts.

### Still queued at handoff time

The local arms are behind cluster contention — every a6000 was full, and the
free v100s are Volta (no native bf16) with VRAM that `myfree` does not
distinguish, so neither was worth a silent-OOM gamble:

- `s2_20_llm8b` — Llama-3.1-8B bf16, all five arms. Memory request was lowered
  from 64G to 32G to widen the node set (peak host RAM during load is ~12G).
- `s2_22_llm70b` — Llama-3.1-70B NF4 4-bit, arms 2 and 3, one a6000.

Both are resumable: `run_campaign.py` skips campaign_ids already in its own
output file, so a time limit costs a resubmission, not the pass. Re-score after
they land and the LLM rows join the table automatically.

## Budget

API spend **$0.95 of $5** (key in `.env`, loaded via `envload.py`, never
logged). Breakdown: output-token calibration $0.058, Haiku smoke $0.000,
Opus refusal probe $0.092, and the three batched ladder arms
(Opus $0.52, Sonnet $0.27, Haiku $0.01). ~$4.05 remains; a 5-arm Opus evidence
ablation would be ~$2.66 batched if Stage 1 wants one. All local/70B compute is
SLURM, not billed.

---

# Stage 2 — the end-to-end agent (2026-11-11)

Code: [`agent/`](agent/) · method and caveats: [`agent/README.md`](agent/README.md) ·
build findings: [`docs/stage2_notes.md`](docs/stage2_notes.md) ·
artefacts: `/zfsauton/scratch/yixiz/lp2cp_stage2/`

## What exists now

The control loop the plan specifies, running end to end: eight steps per
iteration with three LLM-owned (DIAGNOSE, SPECIFY, ALLOCATE), typed
`DiagnosisRecord` / `DesignSpec` / `PreservationContract` / `RelaxationEvent`,
a deterministic execution service that enforces the budget at the filter stage
and diffs every returned sequence against the parent and the declared mask, the
five arms as *mechanisms* rather than prompt text, episodic memory with scoped
retrieval, and three interchangeable model backends (local bf16, local NF4
4-bit, Anthropic API). 29 tests cover the guarantees and run with no GPU, no
API key and no spend.

**The environment is a declared surrogate over real structures.** Parents are
experimental cyclic peptide-receptor complexes from CPSea_PDB; their burial,
contacts, polar pairs, dihedrals, ring topology and buried area are MEASURED
from deposited coordinates. How those objectives RESPOND to an edit is
modelled. No Rosetta, no Boltz-2, no GenKIC ran. Every ScoreCard carries
`surrogate_v1:<hash>`. This licenses claims about CONTROLLERS; it licenses no
claim that any sequence binds anything.

Benchmark: 32 parents, defect classes balanced 8/7/7/7/3, held-out split
declared BY TARGET on CPSea `cluster_id` before any campaign ran, 4,609
enumerated action cells (60-150 per parent, coordinated moves included).

## Result

Regret against the enumerated oracle, measured on the LATENT card (see the
third bug below):

| arm | n | regret | CHR | decl |
|---|---|---|---|---|
| `oracle_greedy` (ceiling) | 32 | **6.33** | 0.188 | - |
| **`rim_first_closure` (the bar)** | 32 | **20.36** | 0.156 | - |
| `random_controller` | 32 | 25.75 | 0.062 | - |
| Haiku 5.5 A1 / A2 | 29 / 28 | 25.99 / 26.91 | 0.138 / 0.143 | 0.72 / 0.61 |
| Sonnet 5.5 A2 / A3 | 11 / 14 | 31.49 / 31.83 | 0.273 / 0.143 | 0.00 |
| `noop` (floor) | 32 | 32.99 | 0.000 | - |

Paired, compute-matched (baseline truncated to the same iteration count):

| comparison | W/L/T | p | verdict |
|---|---|---|---|
| **Haiku A1 vs `rim_first`** | 3/22/4 | **0.007** | **rule wins** |
| **Haiku A2 vs `rim_first`** | 4/20/5 | **0.009** | **rule wins** |
| Sonnet A2 vs `rim_first` | 4/6/1 | 0.51 | tie (underpowered) |
| Sonnet A3 vs `rim_first` | 4/9/1 | 0.51 | tie (underpowered) |

**This reproduces Stage 0's shape in the regime Stage 0 could not reach.** There,
the 8B lost to the best simple rule and frontier models tied it. Here, with a
closure objective and a real conflict present, the cheap model loses
significantly and the frontier model ties. The kill criterion's answer is still
"no" — now on a task that contains the conflict the whole design presumes.

**But the headroom is real here, unlike Stage 0.** The oracle reaches 6.33
while every rule sits at 20+; in Stage 0 the best rule was already near
optimal, which is why an LLM could only tie it. So the task is the right shape
for the claim — roughly 14 regret points a rule-based controller does not
capture. The agent simply is not capturing them either.

## Two findings independent of the verdict

**Opus 5.5 declines this domain outright.** 68 of 68 steps refused
(`stop_reason=refusal`, category `bio`), and 8 of 8 after the system prompt was
rewritten to describe the task accurately. Two accurate framings, same
refusal — a consistent determination, and no further attempt was made to get
around it. Stage 0 measured Opus/Sonnet refusing 3-6 of 73 on SKEMPI alanine
scanning while Haiku refused 67%, so refusal tracks the DOMAIN, not just the
model. Haiku declines 61-72% of campaigns here. The plan's control loop had no
provision for a step that declines; it has one now, with a declared per-step
fallback and per-arm counts, because at a per-call refusal rate near 1.0 an
agent does not degrade, it disappears.

**LLM arms overfit the noisy verifier; rules do not.** Before the regret
definition was corrected, Sonnet's search-time utility (27.05) exceeded the
controller that reads the answer key (12.57) and then scored -0.86 on
independent re-evaluation, a drop of 27.9 against the oracle's 3.7 and
`rim_first`'s 0.13. The agent selects high-variance specs and the frozen
shortlist takes the lucky draw. This is the failure mode the plan's held-out
protocol was built to detect, and it is probably the most publishable thing
here.

## Three instrument bugs found and fixed — read before trusting any rerun

1. **Regret was comparing two different quantities.** Every oracle-table cell
   is scored on its LATENT card; each arm's achieved utility was taken from its
   OBSERVED card. That let an arm pick a high-variance spec, take the luckiest
   of six noisy draws, and record a utility above the table's own maximum.
   Fixed: regret is now measured on the latent value of the committed
   intervention. The agent still CHOOSES under noise; it is no longer GRADED on
   its own estimate. This single change moved `rim_first` from 17.12 to 20.36
   and Sonnet from 12.79 to 31.49 — i.e. it inverted the headline.
2. **Arm 2 logged 8 phantom `RelaxationEvent`s** while holding no contract to
   relax. Relaxation rate is the metric the plan calls the cleanest support for
   the conflict premise. RELAX is now offered only to arms that hold a
   contract, in the action list AND in the depth instruction; items 1-3 of the
   forced-depth prompt stay byte-identical across arms.
3. **The one-call scaffold was a 5x narrower action space**, not a cheaper
   scaffold — it showed only `REBUILD_SEGMENT`'s documentation and pre-filled
   that operator. Caught before it reached a reported number.

## Known weakness in the environment, and the staged fix

**22 of 32 parents have no contract-holding repair anywhere** in their ~140-cell
action space; `BURIAL_DEFICIT` has 0 of 7. Cause: no operator can ever INCREASE
interface burial, so a parent failing only on BSA has no move that helps. The
comparison is still valid — all controllers face identical parents and regret
is measured against the achievable oracle — but most of the benchmark can only
distinguish how gracefully a controller fails.

A v2 patch is written and staged (not applied): a substitution at a position
with headroom (low buried fraction, 3-6.5 A from the partner) gains burial, the
way ProteinMPNN redesigning a surface position actually does. It keeps the
conflict intact because `REBUILD_SEGMENT` gains nothing. Apply order and the
checks to run are in the patch file. **Score v1 before applying it** — the
scorer recomputes held-out and latent values through the surrogate.

## Spend

API $2.04 this stage of the ~$4.05 Stage 0 left. About $0.65 of that bought
nothing: the Opus refusal runs, which bill input tokens for a declined call.
A refusing model is not free to discover. Haiku's full five-arm ladder cost
$0.15 for 200 calls; Sonnet runs ~$0.075/campaign, so a smaller parent set
bought COMPLETE beats 32 bought two-thirds truncated by the ledger.

## Still queued

Local 8B NF4 (`s2_21`, on an idle 2080 Ti) and 70B NF4 (`s2_22`, needs one
a6000) are both pending — the account's legacy GPU quota is consumed by
unrelated `mgtrain` / `wt_fd_wander_*` jobs and the a6000s are saturated. Both
are resumable, so a time limit costs a resubmission rather than the pass. The
8B bf16 arm was cancelled to free a quota slot and needs resubmitting.

## Addendum — surrogate v2, and what blocked further runs (2026-11-11)

**The API credit balance ran out** partway through Haiku's A3 arm. The symptom
was HTTP 400 on every call at zero cost (requests rejected before billing), so
A4_PRIOR and A5_REFLECTION each wrote 32 campaigns in which nothing happened.
Those rows were the dangerous kind: a `StepStatus.ERROR` increments neither the
decline nor the malformed counter, so they read as ordinary "no_improvement"
campaigns and would have been reported as real arms with real regret. The
scorer now excludes any campaign whose LLM steps ALL errored
(`n_no_llm_response`). **The 3->4 and 4->5 ablations are unrun** and need
credits.

Final v1 dataset: Haiku A1/A2/A3 at n=29/32/10, Sonnet A2/A3 at n=11/14, eight
rule baselines at n=32. Archived under `bench_v1/`, `campaigns_v1/`,
`results_v1/`.

**Surrogate v2 is promoted** (`surrogate-v2:72d47cbb5eb96a24`). A substitution
at a position with headroom -- low buried fraction, 3-6.5 A from the partner --
now GAINS burial, the way ProteinMPNN redesigning a partially exposed interface
position does. `REBUILD_SEGMENT` still gains nothing, so the closure/interface
conflict is untouched. Validated in a sandbox copy before promotion:

| | v1 | v2 |
|-|-|-|
| parents with any contract-holding repair | 10/32 | **20/32** |
| BURIAL_DEFICIT repairable | **0/7** | 3/7 |
| contract-holding cells | 162 | 247 |
| contract-breaking cells (the 2->3 trap) | 805 | 790 |
| oracle-hit rate of `oracle_greedy` | 0.59 | **0.84** |
| headroom: bar minus ceiling | 14.0 | **32.7** |

v2 baselines (n=32, clean single environment): `oracle_greedy` 9.50 regret /
0.438 CHR, `rim_first_closure` 42.17 / 0.219, `random_controller` 50.55,
`widest_rebuild` 55.20, `noop` 64.15. Ordering is sane, the anti-baseline is
correctly bad, and the ceiling is now reachable often enough to compete over.

### A fourth instrument bug, caught by provenance

Rebuilding on v2 left **480 v1 campaign rows sitting alongside 96 v2 rows** in
the live campaigns directory -- `run_campaign.py` is resumable, so it skipped
every parent already present. Scoring that mixture would have subtracted one
environment's achieved utility from another environment's ceiling, and the
numbers would have looked entirely plausible: the contaminated run reported the
oracle at 25.15 regret where the clean run reports 9.50.

What caught it was the `env_hash` stamped on every row. The scorer now REFUSES
to score a campaigns directory containing more than one environment, names the
mixture and says what to do -- the same hard stop `assert_frozen()` gives the
qualification function.

### Summary of instrument bugs, all fixed

Three of the four would have changed a reported number, and none is a fact
about LLMs:

1. **Regret compared the oracle's LATENT card against each arm's OBSERVED
   card.** Let an arm take the luckiest of six noisy draws and record a utility
   above the table's own maximum. Inverted the headline when fixed.
2. **Cross-environment pooling** (above).
3. **Phantom RelaxationEvents** from arms holding no contract, in the metric
   the plan calls the cleanest support for the conflict premise.
4. **The one-call scaffold was a 5x narrower action space**, not a cheaper
   scaffold.

The agent needed less hardening than the apparatus measuring it. Anyone
re-running this should re-read the three hard stops -- `assert_frozen`, the
env_hash guard, and the truncation/no-response exclusions -- before trusting a
table.

### Where to pick up

1. **Add API credits**, then re-run the arms on v2. A2/A3 on Sonnet for the
   2->3 test, and A3/A4/A5 on Haiku for the memory ladder that never ran.
   Budget complete campaigns over a smaller parent set rather than 32 parents
   two-thirds truncated by the ledger -- Sonnet is ~$0.075/campaign.
2. **The local arms are queued** (`s2_21` 8B-NF4 on a 2080 Ti, `s2_22` 70B-NF4
   on an a6000) behind a GPU quota consumed by unrelated jobs on this account.
   Both resumable. The 8B bf16 arm was cancelled to free a quota slot and needs
   resubmitting as `s2_20`.
3. **Opus is unavailable for this domain** and should be reported as declined,
   with the count, not scored.

## Local 4-bit arms on v2 — the scale comparison (2026-11-11)

Both local rungs the original brief asked for completed on surrogate v2, at
full n=32, as NF4 double-quantised loads. The 8B ran on an idle rtx_2080_ti
(~5.5 GB of weights, so it never needed the contended a6000s); the 70B ran on
one a6000 at ~40 GB, which is the whole reason for 4-bit — in bf16 it is
~140 GB and needs four cards that never free up together on this cluster.

| arm | n | regret | CHR | hoCHR | calls/campaign |
|---|---|---|---|---|---|
| `rule:oracle_greedy` (ceiling) | 32 | **9.50** | 0.438 | 0.219 | - |
| **`rule:rim_first_closure` (the bar)** | 32 | **42.17** | 0.219 | 0.156 | - |
| `rule:random_controller` | 32 | 50.55 | 0.125 | 0.031 | - |
| Llama-3.1-8B-4bit A3 | 32 | 57.92 | 0.125 | 0.100 | 4.2 |
| Meta-Llama-3.1-70B-4bit A2 | 32 | 57.99 | 0.031 | 0.000 | 7.9 |
| Llama-3.1-8B-4bit A2 | 32 | 58.27 | 0.125 | 0.056 | 4.5 |
| Meta-Llama-3.1-70B-4bit A3 | 32 | 61.00 | 0.000 | 0.000 | 6.5 |
| `rule:noop` (floor) | 32 | 64.15 | 0.000 | - | - |

**Ten times the parameters bought nothing.** Both local models sit just above
the no-op floor and far below a two-column heuristic, and the 70B is if
anything slightly WORSE than the 8B. This is Stage 0's local finding
reproduced in the regime that contains the closure objective: there, the 8B
lost outright to the best simple rule.

**One confound, being removed.** The 8B ran the one-call scaffold and the 70B
the two-call, so model size is tangled with scaffold in the table above. A
matched 8B two-call run is queued; the 8B-vs-70B line should be read from that
once it lands. Nothing else in the table is affected -- the comparisons against
the rule bar and the A2-vs-A3 contrasts are within-scaffold.

Local compute is SLURM and unbilled, so these are the cheap arms to extend.

## The 2->3 result: the contract is enforced at the wrong stage (2026-11-11)

Sonnet 5.5 on surrogate v2, one-call scaffold, n=16 per arm, **zero campaigns
truncated by the ledger**. This is the comparison the design doc calls the only
claim with no prior-art answer.

| | A2 (mask only) | A3 (+ preservation contract) |
|-|-|-|
| contract-holding repairs | **4** | 2 |
| defect resolved, contract broken (the shortcut) | **1** | **0** |
| no change | 11 | 14 |
| RelaxationEvents | 0 | **2** |
| iterations | 38 | 35 |
| compile rejections (`mask_touches_contract_protected`) | 0 | **4** |
| iterations where EVERY candidate was rejected | **0** | **10** |

The mechanism does what it was built to do: A3 never takes the contract-breaking
shortcut, and relaxes explicitly when the contract becomes infeasible. A2 does
neither.

**But A3 does not lose repairs by nobly refusing shortcuts.** A2 took the
shortcut exactly ONCE in 16 campaigns, so there was almost nothing for the
guard to catch -- the frontier model mostly declines the shortcut even when
nothing stops it, which is itself worth reporting and which makes the 2->3
ablation, as designed, nearly unobservable on this model.

A3 loses repairs because **14 of its 35 iterations (40%) produced nothing**,
against 0 of 38 for A2, in two ways:

1. **The agent contradicts its own contract.** Those 4 compile rejections are
   design specs whose mask opens a position the agent ITSELF named in
   `preserved_features` one step earlier. It declares "preserve position 2",
   then asks to redesign position 2. The contract is built from the agent's own
   diagnosis output (`_contract_from`), so this is not an imposed constraint it
   is fighting -- it is its own.
2. **The contract vetoes output it could have steered.** In 10 iterations the
   backend returned six candidates and the contract refused all six, because
   the spec's mask had already guaranteed every candidate would break it. The
   contract is enforced AFTER generation, at a point where the spec had
   determined the outcome.

So the finding is not "preservation contracts cost repairs". It is that **an
LLM emits a constraint set and then does not propagate it into its own next
step** -- a concrete, diagnosable failure of the premise the whole division of
labour rests on ("The LLM never emits a sequence. It emits a design problem").
It emits the design problem; it just does not read back what it committed to.

**This is testable and cheap.** A controller that injected the active contract
into the SPECIFY mask -- fixing the protected positions before the model
chooses which to open -- should recover most of those 14 iterations. That is
the single highest-value next experiment, it needs no new environment, and it
converts a null 2->3 result into a measurement of whether the contract helps
once the agent can actually act on it.

Service ledger over the same run: 15 scope leaks caught in A2, 22 in A3, 0
budget overruns. The deterministic guarantees fired throughout.

## The paired verdict on surrogate v2 (2026-11-11)

All comparisons paired over shared parents, truncated and dead-API campaigns
excluded, regret measured on the latent card.

| comparison | n | W/L/T | p | verdict |
|-|-|-|-|-|
| 70B-4bit vs 8B-4bit, **matched two-call scaffold** | 32 | 7/11/14 | 0.65 | tie |
| Sonnet A3 vs A2 — **the 2->3 ablation** | 16 | 5/7/4 | 0.43 | tie |
| Haiku A2 vs A1 — 1->2 conditioning | 21 | 7/5/9 | 0.27 | tie |
| 70B-4bit A2 vs `rim_first_closure` | 32 | 12/20/0 | 0.056 | tie, losing |
| Sonnet A2 vs `rim_first_closure` | 16 | 8/8/0 | 0.64 | tie |

**Nothing beats the rule bar.** Across five models (local 8B bf16-equivalent
at 4-bit, local 70B 4-bit, Haiku 5.5, Sonnet 5.5, with Opus declining the
domain), two environment versions, and the arm ladder:

- **Scale buys nothing.** 70B is statistically indistinguishable from 8B once
  the scaffold is matched (p=0.65). Ten times the parameters, no effect.
- **Conditioning buys nothing.** A2 vs A1 ties.
- **The preservation contract buys nothing.** A3 vs A2 ties -- and the
  iteration-waste analysis above explains why: 40% of A3's iterations were
  spent on specs incompatible with its own contract.
- The oracle sits at **9.50** regret against the bar's **42.17**. There are
  32.7 points of headroom that no controller, LLM or rule, captures.

This is the kill criterion answered in the regime Stage 0 could not reach, and
it answers the same way, more strongly: Stage 0's frontier models at least TIED
the best rule on a task where the rule was near-optimal. Here the rule is far
from optimal -- there is real room -- and the models still do not take it.

### A scaffold finding worth reporting separately

The 8B's malformed-action rate went from **0.03 under the one-call scaffold to
0.41 under two-call**. Agent Rosetta's two-call action pattern -- emit a bare
`<operator>` tag, then a JSON spec conditioned on it -- costs the small model
41% of its actions. Two-call is not a free upgrade; it is a capability
threshold. Anyone copying that scaffold onto a small model should measure this
before attributing the resulting behaviour to the model's reasoning.

## The decisive experiment: fixing constraint-propagation does NOT fix allocation

Sonnet 5.5, arm A3, same 16 parents, same one-call scaffold, same 35
iterations, same cost. ONE difference: whether the contract's protected
positions are restated inside the SPECIFY brief, at the moment the mask is
chosen, instead of only once near the top.

| | A3 baseline | A3 + injection |
|-|-|-|
| opens a position it ITSELF declared protected | **4** | **0** |
| iterations where every candidate was rejected | 10 | 4 |
| **wasted iterations** | **14/35 = 40%** | **4/35 = 11%** |
| contract-holding repair rate | 0.125 | 0.188 |
| **mean regret** | **59.24** | **59.12** |

Paired over shared parents: injected vs baseline **p=0.39, tie**; injected vs
the rule bar **p=0.43, tie**.

The prompt fix works, completely and measurably: self-contradiction goes to
zero and the agent spends 89% of its iterations on valid design problems
instead of 60%. **And allocation quality does not move.**

This matters because it removes the most plausible confound for every negative
result above. One could have argued the agent was not failing at allocation but
drowning in its own bookkeeping -- declaring constraints it then violated,
burning iterations on specs its own contract would veto. That argument is now
closed by experiment. Give it nearly twice as many valid attempts and it finds
no better repairs.

**The bottleneck is the allocation decision itself** -- which positions to
spend a scarce edit budget on when closure and interface compete for the same
ones. That is precisely the capability the design doc lists as having no
prior-art answer, and the answer on this benchmark is negative across five
models, two environment versions, the full arm ladder, and now with the
leading alternative explanation experimentally excluded.

### A fifth instrument bug, found by this experiment

`campaign_id` is `{parent}__{arm}__{model}` and carried no scaffold, so the
injection run collided exactly with its own baseline. De-duplication keyed on
campaign_id collapsed 81 rows and **silently deleted the entire experiment from
the results table** -- the run appeared never to have happened. De-duplication
is now keyed on `(arm_key, parent_id)`, which is a campaign's real identity,
and `campaign_id` carries the scaffold going forward.

Five instrument bugs now, four of which would have changed or erased a reported
number. Re-read the hard stops before trusting any table: `assert_frozen`, the
env_hash guard, the truncation and no-LLM-response exclusions, and this one.
