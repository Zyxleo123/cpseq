# LP→CP agent — pilot tests

> **The end-to-end agent now exists.** This file documents Stage 0, the SKEMPI
> allocation pilot. The eight-step control loop, the execution service, the
> five arms and the enumerated cyclic-parent benchmark are in
> **[`agent/`](agent/)** — start at [`agent/README.md`](agent/README.md), and
> [`HANDOFF.md`](HANDOFF.md) for what each stage measured.

Offline decision benchmarks that test the one capability the LP→CP agent design
rests on, **before** anything end-to-end gets built: can an LLM read structured
structural diagnostics and decide *where to spend a scarce edit budget*?

That is the only claim in the design doc with no prior-art answer. Orchestration
(BioDesignBench, ABLE), constraint compilation (PROTEAN) and evaluation depth
(BioDesignBench, achieved by prompt alone) are all established, two positively
and one as a confound. So the pilots test allocation and nothing else.

This directory is standalone. It shares no code with `Proteina-Complexa` — only
its Python environment, and later its Boltz-2 / Rosetta scoring if Stage 1 runs.

```
pilot/stage0/     SKEMPI allocation slice   (built, running)
sbatch/           job scripts + submitter
```

---

## Result so far

**Llama-3.1-8B extracts real signal from structural diagnostics but loses to
the single best simple rule.** Primary slice `ala_k1`, n=73, full evidence:

| arm | mean | median | p90 | >2 kcal | oracle hit |
|---|---|---|---|---|---|
| `rim_first` (**the bar**) | **0.75** | 0.55 | 1.96 | 0.10 | 0.18 |
| `min_contacts` | 0.81 | 0.65 | 1.98 | 0.08 | 0.19 |
| **LLM 8B** | **1.04** | 0.81 | 2.48 | 0.15 | 0.11 |
| `composite` | 1.08 | 0.68 | 2.85 | 0.18 | 0.15 |
| *random* | *1.57* | 1.46 | 2.58 | 0.23 | 0.00 |

| comparison | all (n=73) | test (n=46) |
|---|---|---|
| LLM vs **random** | 57W/16L, **p<0.0001** | 36W/5L, **p<0.0001** |
| LLM vs **`rim_first`** | 17W/31L/25T, **p=0.031** | 10W/22L/14T, **p=0.020** |
| LLM vs `min_contacts` | 15W/26L/32T, p=0.074 | 8W/17L/21T, p=0.053 |
| LLM vs `composite` | 28W/22L/23T, p=0.63 | 21W/13L/12T, p=0.37 |

The LLM captures 34% of the random→oracle headroom; `rim_first` captures 52%.
It beats random, `max_entropy`, `max_dsasa`, `most_measured` and
`greedy_singles` decisively, and ties `composite` — so it is genuinely reading
the diagnostics rather than guessing. It simply is not as good as "prefer the
rim". 0/186 invalid actions.

**This is a floor, not a kill** (see the kill-criterion section): local weights
only, and the design doc's capability survey concerns frontier models. What it
does establish is that the benchmark has the resolution to decide the question
— the median gap of 0.29 kcal/mol sits above the 0.32 MDE, the result is
consistent across dev/test, and every control holds.

**The heavy-tail framing applies only to the wide-candidate-set regime.** On
the full-interface arm the 8B matched the best heuristic's oracle-hit rate and
lost purely in the tail. On this restricted slice it is worse on *every*
statistic — hit rate, median and tail alike. Do not generalise the tail story
from one arm to the other; it was an early over-reading here, corrected after
both arms had run.

---

## Stage 0 — SKEMPI allocation slice

### The task

A **decision** is one allocation problem: a protein–protein complex, an edit
budget `k`, and a list of candidate position-sets of size `k` that SKEMPI 2.0
actually measured. The agent picks exactly one. The oracle is the candidate with
the lowest ΔΔG, so **regret is measured in kcal/mol of binding energy thrown
away**.

```
ddG = RT ln(Kd_mut / Kd_wt)      negative = binds tighter
regret(decision, arm) = value(chosen) - value(oracle)   >= 0
```

### Why this is the right Stage 0 — and why it differs from the design doc

The doc framed SKEMPI as "close to a budget-allocation oracle". The data is
better than that, but for a different reason than assumed: **only 20.5% of
SKEMPI records improve binding** (median ΔΔG = +0.77 kcal/mol). Alanine scans
dominate. So this is not a "find the affinity-improving mutation" oracle — it is
a **forced-budget damage-minimisation** oracle:

> You must spend `k` edits. Where do you spend them so the interface survives?

That is exactly the LP→CP allocation decision minus closure. The LP→CP agent
*must* mutate — to install cyclisation anchors — and the question is where to
spend so the interface is not destroyed. SKEMPI's destabilising-mutation bias
makes it a near-perfect analog rather than a limitation.

### Measured feasibility

| slice | what it is | n decisions | dev / test | median candidates | median spread |
|---|---|---|---|---|---|
| **`ala_k1`** | alanine scan, k=1 — **primary** | **73** | 27 / 46 | 16 | 3.72 kcal/mol |
| `allsub_k1` | all substitutions, k=1, optimistic solver | 84 | 32 / 52 | 16 | 3.69 |
| `allsub_k2` | real measured doubles, k=2 — budget *splitting* | 29 | 8 / 21 | 8 | 4.65 |
| `allsub_k3` | k=3 | 4 | — | 5 | **auto-flagged NOT TESTABLE** |

"Spread" is best-to-worst within one complex: the energy an allocator can win or
lose. A median of 3.7 kcal/mol means there is real signal to capture.

### The confound that picked the primary slice

For the all-substitution slices, a candidate's value is the *minimum* over
however many substitutions happened to be assayed there. A position measured ten
times gets ten draws at that minimum. Measured within-complex correlation:

```
corr(n_records, V_opt) = -0.128      (mean over 52 complexes)
```

Real, though modest. Two consequences, both built in:

1. **`ala_k1` is the primary slice** — exactly one record per candidate by
   construction, so the value is draw-count invariant.
2. A **`most_measured` control arm** reads `n_records`, which no legitimate arm
   may see. If it wins, the benchmark is measuring assay density rather than
   allocation, and the all-substitution numbers must be discarded.

### Four safeguards

**Labels live in a separate file.** `build.py` writes
`decisions_public.json` (what arms may see — no ΔΔG anywhere) and
`decisions_oracle.json` (values, read only by `score.py`). A key-level leak
check runs on the public file at build time and fails the build. Leakage is a
structural impossibility, not a thing we remember to avoid.

**The split was declared before any prompt existed.** `dev` = 1/3 of complexes,
grouped by `pdb_id`, assigned by `sha256(salt|pdb_id)`. Prompt engineering
happens on dev; `test` is reported once. Reproducible from the salt alone.

*Which number is the headline.* The first pass did **zero** prompt iteration —
one prompt was written and run — so nothing has been fitted to either split and
the better-powered `all` figure (n=73) is the honest one to quote, with `test`
(n=46) reported beside it as the pre-registered check. The moment any prompt is
tuned against results, that stops being true and `test` becomes the only
reportable number. The scorer emits both on every pass so the distinction never
has to be reconstructed later.

**Candidate order is a seeded shuffle, and there is a second one.** Without
this, "the model picks the first option" is indistinguishable from "the model
picks the best option". `presentation_order_alt` lets the same decision be
re-asked permuted, giving an order-stability measurement.

**Invalid answers are counted, never imputed.** A model that fails to emit a
usable action gets `choice_valid: false` and no fallback pick. Scoring drops
those pairwise and reports the rate. A defaulted choice would be scored as if
the model had made it; a null that later arithmetic treats as zero would score
an abstention as a perfect answer.

### Arms

The LLM never names a residue — it picks a position. Identity is the solver's,
which is what makes a win attributable to allocation quality rather than luck on
chemistry.

**Rule-based bar** (the doc's kill criterion names a "hotspot/conservation
heuristic"): `min_dsasa`, `min_contacts`, `rim_first`, `max_entropy`,
`min_wt_prob`, and `composite` (z-scored burial + contacts − ESM tolerance).
The headline comparison is against the **strongest** of them — taking the max
over several heuristics inflates the bar, which makes an LLM win harder to
claim, not easier.

**References and control arms:** `random_expected` (computed analytically — the
mean candidate value minus the oracle, exact, no seed noise), `first_listed`
(position-bias detector), `most_measured` (data-density confound),
`greedy_singles` (what a perfect greedy per-position controller would pick at
k≥2), `max_dsasa` (deliberate anti-baseline: spend the edit at the *most*
buried position — it must come out worse than random, or the value function is
not measuring interface damage).

`most_measured` and `greedy_singles` are the only arms permitted to read the
oracle file, and only for `n_records` and the additive single-mutation
prediction — never for a candidate's own measured value.

**Six control analyses** run on every scoring pass, in `score.py`:

| control | question it answers |
|---|---|
| `value_function_sanity` | does `max_dsasa` come out worse than random? If not, nothing downstream means anything |
| `data_density_confound` | does picking a more-assayed candidate pay? |
| `position_bias` | does an arm prefer early slots in the shuffled list? |
| `order_stability` | does the same arm choose the same candidate under a permuted list? |
| `coordination_headroom` | do joint moves exist at all, or is greedy already optimal? |
| `arm_agreement` | is the LLM just reimplementing a heuristic we already have? |

The last one matters for reading any LLM result: high agreement with one
heuristic *plus* matching regret is the signature of a rediscovered rule, while
low agreement with everything is a genuinely different policy whatever it
scores.

**LLM evidence ablation**, all on Llama-3.1-8B-Instruct:

| arm | sees |
|---|---|
| `full` | structure + conservation, per candidate |
| `structure_only` | burial, contacts, polar pairs, min distance |
| `conservation_only` | ESM-2 masked entropy, wt probability, wt rank |
| `aggregate_only` | interface summary + candidate names, **no per-candidate breakdown** |
| `no_evidence` | candidate names alone — the floor |

`aggregate_only` is the Stage 0 analogue of the doc's 2→3 ablation, and
`no_evidence` says how much of any win is the prompt rather than the
diagnostics.

### Forced depth is shared, not an arm

BioDesignBench's finding is that the evidence-allocation deficit is
**behavioural, not capability-limited** — a plain "evaluate thoroughly"
instruction recovers most of the apparent gap. So the deliberation instruction
is part of the shared scaffold in `prompts.py` and every arm gets it. Arms
differ only in which *evidence* they receive. If only some arms were
forced-depth prompted, any win would be confounded with that instruction.

### Prompt scaffold

Shape from Agent Rosetta (the only harness published at template granularity):
system prompt with a `{reasoning_formatting}` slot, a design brief templating
`{state}` and `{action_formatting_instructions}`, and an error-correction
re-prompt on a malformed action. Tagged output `<choice>c014</choice>` follows
ABLE's convention. Parsing is unit-tested on 7 cases including no-tag,
unknown-id and multiple-tag.

---

## Running it

```bash
cd sbatch
bash submit_stage0.sh                 # full chain, 8B arm included
bash submit_stage0.sh --no-llm        # deterministic stages only
bash submit_stage0.sh --llm-only      # GPU arm once a card frees
bash submit_stage0.sh --smoke         # 6 decisions per slice, end to end
bash submit_stage0.sh --set LLM_MODEL_PATH=/path/to/70B --set LLM_DEVICE_MAP=auto
```

| stage | partition | runtime | notes |
|---|---|---|---|
| `s0_00_build` | cpu | seconds | freezes + hashes the benchmark; probes RCSB egress |
| `s0_10_features` | cpu | ~5 min | ~90 structures, 3 SASA passes each; resumable |
| `s0_15_conservation` | debug (2080 Ti) | ~5 min | ESM-2 650M, masked marginals |
| `s0_20_heuristics` | cpu | seconds | all 10 rule/control arms |
| `s0_30_llm` | general (a6000) | ~2.5 h | 5 evidence arms × 186 decisions |
| `s0_40_score` | cpu | seconds | runs **twice** — after heuristics, and after the LLM |

The early scorer depends only on the heuristic stage, so the rule-based bar and
the oracle spread are readable without waiting for a GPU. The LLM stage is a
*sibling* of the heuristic stage, so it queues for a card concurrently.

Configuration travels as a timestamped env **file** passed positionally.
`sbatch --export` is never used: any explicit export list makes sbatch set
`SLURM_GET_USER_ENV=1`, slurmd then fails to rebuild the login environment on
the compute node, and the job is requeued and **held**, stranding every
dependent. Gates print why and exit 0, so a missing input never becomes a
silent `DependencyNeverSatisfied`.

GPU class is pinned deliberately. Llama-3.1-8B in bf16 is ~16 GB of weights
before any KV cache, so an 11 GB rtx_2080_ti cannot hold it and a 16 GB v100 is
borderline; a CUDA OOM kills the process with no Python traceback, which reads
as a mysterious silent `FAILED`.

Figures redraw from `results.json` + `per_decision.jsonl` with no GPU, no model
and no arm rerun.

---

## The kill criterion, and what it can and cannot conclude

From the design doc:

> If the LLM cannot beat a hotspot/conservation heuristic at position selection
> where ground truth is experimental and the objective is single, the cyclic
> version will not rescue it. Stop there.

**This pilot runs on local weights only (Llama-3.1-8B, optionally 70B).** There
is no frontier-model API key in this environment. So a loss here is a **floor,
not a kill**: the design doc's capability survey is about frontier models, and
my own prior measurements on this cluster found model family and size move
results by more than the effect sizes at stake here. The proposer backend is
kept behind one interface so an API arm drops in without touching the runner or
the scorer.

A **win** on local weights, by contrast, is strong evidence — it would mean the
capability is present well below frontier scale.

### Power

Paired Wilcoxon over decisions, with the minimum detectable effect reported
alongside every comparison. At n=46 (`ala_k1` test) and an assumed paired-
difference sd of ~1.5 kcal/mol, the MDE at 80% power is ≈0.6 kcal/mol against a
median spread of 3.7 — adequate. `allsub_k2` at n=21 is marginal and flagged.
`allsub_k3` at n=4 is auto-flagged `NOT TESTABLE` so it cannot later be quoted
as a trend.

### Known limitations

- **Restricted candidate set.** The primary arms choose from measured positions
  only, which hands the agent the oracle's support. The full-interface secondary
  arm below is what proves the restriction is not doing the work.
- SKEMPI is linear protein–protein with **no closure objective**. That is the
  whole point of Stage 1, and Stage 0 cannot speak to the closure/interface
  conflict at all.
- Coverage is opportunistic (alanine scans dominate), so the oracle is
  **best-among-measured, not best-overall**.
- Structures are asymmetric units, waters and non-standard residues stripped
  before SASA, so a cofactor in the interface does not bury its neighbours and
  those dSASA values are underestimates.
- ESM-2 masked entropy substitutes for an MSA-derived conservation signal.

---

## API arms and what they cost

The proposer backend is pluggable, so a Claude arm runs the **identical**
prompt, evidence table, candidate order and `<choice>` parsing as the local
one. That is deliberate: no structured outputs, no tool calling, no
model-specific prompt tuning, even though all three would make parsing more
reliable. A scaffold change bundled with a model change produces a number that
cannot be attributed to either.

Two further deliberate choices in `AnthropicProposer`:

- **`fallbacks` is not enabled.** The server-side refusal fallback would
  silently re-run a refused request on a *different* model and return it in the
  same call — right for production, wrong for an experiment where every row
  must name the model that produced it. A refusal is recorded as an outcome.
- **Batch API, not synchronous.** A fixed prompt set with no latency
  requirement is exactly what it is for, and it bills at 50%.

### Measured cost (primary slice `ala_k1`, 73 decisions, effort `low`)

| model | in/dec | out/dec | $/decision | $/arm (sync) | $/arm (batch) |
|---|---|---|---|---|---|
| `claude-opus-5-5` | 1638 | 401 | $0.0146 | $1.06 | **$0.53** |
| `claude-sonnet-5-5` | 1638 | 304 | $0.0063 | $0.46 | **$0.23** |
| `claude-haiku-5-5` | 1638 | 227 | $0.0003 | $0.020 | **$0.010** |

Input is exact — 119,578 tokens counted with `messages.count_tokens`, which
costs nothing. **Output is measured, not estimated**: `estimate_cost.py
--calibrate N` runs N real calls per model, because thinking is always on in
this model generation and bills as output, so the output side cannot be
predicted from the prompt. That calibration cost $0.0584.

Scaling: one arm over all 186 decisions on Opus is ~$2.71 sync / $1.36 batch;
a five-arm evidence ablation on `ala_k1` is $5.32 sync / **$2.66 batch**.

### Three budget-safety properties

A fixed budget makes some ordinary bugs expensive, so they are designed out
rather than watched for:

1. **The batch id is written to disk before any polling.** A crash resumes that
   batch instead of submitting a second one. Without it, a crash loop bills the
   whole arm again on every restart.
2. **A projected cost is checked against `--max-spend-usd` before submission.**
   Over budget, nothing is sent at all.
3. **Every row carries its own `usage` and `cost_usd`**, and an arm-level
   ledger lands beside the rows, so spend is recoverable from the artefacts
   rather than from a console that has scrolled away.

Prices live in `pricing.py` with the date they were taken, so a cost recorded
next to a result is the price that was in force when the row was produced. An
unknown model raises rather than silently costing zero.

---

## The full-interface secondary arm

```bash
bash submit_stage0.sh --fullif     # after the primary chain has produced features.json
```

Same decisions, but the agent sees **every interface position on the mutated
side** (median ~40 instead of 16) and may pick anywhere. Built by
`fullinterface.py` into `bench_fullif/`.

Most of the interface was never assayed, so many picks are unscorable. That is
a gap in the ground truth, not a model failure, and the scorer keeps the two
apart: `valid` means the model emitted a usable action, `scorable` means that
action landed on a measured position. Reported metrics:

| metric | what it answers |
|---|---|
| `coverage_of_measured` | fraction of the arm's picks that are scorable |
| `blind_pick_coverage` | what a uniform pick over the interface would score — the calibration reference |
| `mean_regret` | computed on the covered subset, therefore **biased**; never the headline |

**Coverage lift is not a quality signal — measured, not assumed.** The arm with
the highest coverage is `max_dsasa`, the deliberate anti-baseline (0.904 vs a
0.569 blind reference), and the arm with the lowest is `min_dsasa` (0.370).
SKEMPI's measured subset is biased toward buried hotspot positions because
that is what experimentalists scan, so high coverage means "agrees with
hotspot-scanning priorities" — the *opposite* of damage minimisation. Read
coverage as a description of SKEMPI's sampling, not of allocation quality.

The calibration does work: `random` scores 0.548 and `first_listed` 0.562
against a 0.569 blind reference, as they must.

What this arm therefore establishes is narrower than it first appears, and
worth stating plainly: the heuristic *ranking* survives widening the candidate
set (`min_contacts` best at 0.695, `max_dsasa` worst at 2.674), so the
restriction is not manufacturing the primary result. But because each arm is
scored on a differently-biased subset, the full-interface **regret numbers are
not comparable between arms** and cannot carry a headline. That is also the
argument for keeping the restricted slice primary: there every candidate is
measured, so there is no selection bias at all.

### Result (Llama-3.1-8B, `full` evidence, n=73 answered)

| arm | coverage | n sc. | mean | median | p90 | max | >2 kcal | oracle hit |
|---|---|---|---|---|---|---|---|---|
| `min_contacts` | 0.589 | 43 | **0.69** | 0.55 | **1.94** | 2.37 | **0.05** | 0.23 |
| `composite` | 0.452 | 33 | 0.72 | 0.49 | 1.51 | 3.15 | 0.06 | 0.18 |
| `min_dsasa` | 0.370 | 27 | 0.99 | 0.84 | 2.08 | 2.37 | 0.11 | 0.15 |
| **LLM 8B** | 0.411 | 30 | **1.31** | 0.91 | **3.63** | 5.62 | **0.20** | **0.23** |
| `max_entropy` | 0.671 | 49 | 1.46 | 1.32 | 3.10 | 6.46 | 0.27 | 0.12 |
| *random* | 0.548 | 40 | 1.85 | 1.47 | 4.45 | 6.67 | 0.35 | 0.05 |

0/73 invalid actions — the action schema and retry path hold up.

The 8B does not beat the structure heuristics here. But **the shape of the loss
matters more than the ordering**, and mean regret hides it: the LLM's
oracle-hit rate *equals* `min_contacts`'s (0.23 both), its median regret is
only modestly worse (0.91 vs 0.55), and the gap is carried almost entirely by
the tail — p90 of 3.63 against 1.94, and **4× the rate of catastrophic misses**
(20% vs 5% above 2 kcal/mol).

So the 8B's failure mode is not mediocre judgement. It is occasional
high-confidence, large-magnitude error on an otherwise competitive policy.

**This has a direct consequence for the design doc.** If LLM allocation errors
are heavy-tailed, the preservation contract — the 2→3 arm, the one the doc
calls the only claim with no prior-art answer — is exactly the right
mitigation, and for a reason the doc does not currently state: it is a guard
against catastrophic *single* decisions, not an average-case improvement. An
ablation powered to detect a shift in mean regret may well miss it; the
quantity to pre-register is the tail.

Agreement with `min_contacts` is 0.247 and with everything else below 0.18, so
this is not a rediscovered rule — it is a distinct policy whose mean is
dragged down by its tail. Caveat throughout: each arm is scored on a different
30–49 decisions, so these means are not strictly comparable.

Two details that keep this honest: SKEMPI's COR/SUP/RIM class is withheld for
**all** positions here (it exists only for measured ones, so supplying it where
available would flag them), and the arm gets its own ESM-2 conservation pass
over the wider pool, because dashes on unmeasured positions would mark them
just as effectively. k=1 only — at k=2 the full interface is thousands of pairs,
which is a candidate-set-size experiment wearing this one's name.

---

## Stage 1 — enumerated cyclic parents

Not built. Gated on Stage 0, by design. Three things the doc says to settle
first, none of which are decided yet: the held-out split must be by **target**
rather than parent; the enumeration grid must include **joint** interventions or
it cannot detect coordinated moves; and a **tie-break rule for two simultaneous
defects** has to exist before the run, or the repair matrix has rows assigned
post hoc.

**The joint-intervention question is already answered, from Stage 0.** The doc
worries that an oracle covering only single interventions cannot detect whether
an LLM finds coordinated moves. `allsub_k2` carries, per candidate pair, the
additive prediction from singles and the measured epistasis, and a
`greedy_singles` control arm picks the pair a perfect greedy per-position
controller would. Measured over 24 decisions / 357 measured pairs:

| | value |
|---|---|
| perfect greedy controller finds the best pair | **45.8%** of the time |
| mean \|epistasis\| | **1.29** kcal/mol |
| pairs with \|epistasis\| > 1 kcal/mol | **39.8%** |
| coordination headroom (greedy → oracle) | mean **0.93**, median **0.03**, max **6.57** |

So coordinated moves exist and are worth real energy — but it is a **tail
phenomenon**: greedy is optimal on roughly half the decisions and near-optimal
on most of the rest, then occasionally wrong by 6+ kcal/mol. Two consequences
for Stage 1: the enumeration grid **must** include joint interventions (the doc
is right that it matters), and mean regret at k≥2 will be driven by a handful
of decisions, so a k=2 comparison needs far more than 24 of them to resolve.
