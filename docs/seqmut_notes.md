# Sequence-mutation lever on LNR — notes

*Written for: whoever continues the sequence-mutation thread (including the
parallel LLM-improvement session). This covers the SEQUENCE-EDIT layer added on
top of the real ring-closure environment. The closure / Boltz-physical / replica
machinery it sits on is documented in [`lnr_notes.md`](lnr_notes.md) — this file
does not repeat it, it builds on it.*

Date: 2026-11-16. Code: [`../agent/lnr/`](../agent/lnr/). Artefacts:
`/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/`.

---

## What this is

The proven LNR action is a held window `[a,b]` (which residues keep their bound
pose while the rest close a head-to-tail ring). This thread adds a SECOND lever:
editing the peptide SEQUENCE before closure, to recover binding lost to
cyclisation. Grounded in a literature pass (see below), not copied.

**The action set (deliberately not committed to one fork).** A mutation is a set
of substitutions `{(position, amino_acid)}` under a budget `k_mut`. v1 arms:
- `nomut` — baseline; `random_mut` — the DUMMY (random AA at interface positions);
- `conservative` — random AA at the LOWEST-binding sites (a smarter floor);
- `anti` — hotspots→Ala (anti-baseline, must be catastrophic);
- `llm` / `llm-ala` — the controller NAMES position AND identity (fork B, the
  invariant-BREAKING fork, run as a scouting arm). `llm-ala` additionally sees
  the WT alanine-scan binding column as evidence.
All go through ONE scorer (`seqmut.score_target`, `subs_override`), so the
invariant-PRESERVING fork A (controller names the position, a packer names the
identity) drops onto the same instrument later. Files: `lnr/seqmut.py`
(+ `test_seqmut.py`, 16 tests, no GPU/API).

**The one correctness point.** `score_target` NEVER reuses the WT alanine scan
for a mutant — `mutant_scan()` recomputes native_dG AND the per-residue scan on
the MUTANT bound pose (clean → substitute+repack → relax → alanine scan). Reusing
the WT table would make a beneficial mutation invisible and over-credit a held
hotspot — the project's recurring bug class. `nomut` reproduces the WT number.

**The proxy and its limit.** The proxy fitness is
`recovered = energy_weighted_retention([a,b]) × native_dG` on the native bound
pose with local repack. It is POSE-PRESERVING: it credits only mutations that
keep the native binding mode and CANNOT see a fold/binding-mode change. It is a
cheap SCOUTING oracle, not the headline.

## Literature grounding (informed, not copied)

LLM sequence-mutation action sets and search, surveyed this project:
- **Wang et al. 2501.09274** ("LLM is Secretly a Protein Sequence Optimizer") —
  iterative DE: population, LLM proposes via mutation/crossover conditioned on
  shown fitness, elitist selection, Hamming-budget cap; beats a random-operator
  EA especially on nonlinear landscapes. The direct template for iterative search.
- **StructEvo 2609.39048** — factors mutation into (position, identity) with
  structure guiding POSITION (independent evidence the LP→CP where/what split is
  the better decomposition, not just cleaner). Cheap proxy drives the loop,
  oracle scores finalists (two-tier).
- **EVOLVEpro / ALDE** — active-learning DE, large gains in ~3 rounds.
- **Reflexion / Self-Refine** — iterative self-critique (= the arm-5 reflection).
- **LMX / CAMP** — LLM/PLM as crossover operator.
Forced control: more rounds helps even random search, so an iterative LLM win
must beat an iterative random-EA at MATCHED rounds/budget (Wang's control = this
project's compute-matched principle).

## Proxy breadth scorecard (8-target pilot, proxy dG = recovered, REU)

| arm | proxy dG | note |
|-|-|-|
| NATIVE linear (upper bound) | −67.2 | retention=1, unreachable |
| nomut | −39.6 | baseline |
| random_mut | −35.0 | **DUMMY** (correctly below nomut) |
| conservative | −39.7 | ≈ nomut |
| anti | −34.3 | anti-baseline (worst, as it must be) |
| **llm:luna:k2** | **−43.1** | best; beats nomut by −3.5 (tail-driven, 4/3) |
| llm-ala:luna:k2 | −40.0 | over-declines (6/8) |
| llm:terra:k2 | −39.6 | declined all 8 |
| **llm-ala:terra:k2** | **−41.7** | beats nomut by −2.1 (4/0 in proxy) |

Breadth findings: **model × evidence interaction** — luna mutates eagerly
(aggressive, noisy win); terra refuses structure-only but is selective and
all-beneficial given the binding column. Budget: aggressive `llm` over-reaches
at k=3, selective `llm-ala` improves with k. Scale: gpt-5.4-mini loses the
effect. Scorer sanity holds (nomut < conservative < random < anti). Run:
`sbatch/s4_lnr_seqmut.sbatch` (CPU, resumable); aggregate across arm files with
`seqmut.py --report-dir`.

## Boltz validation — the "actual score"

Fold each sequence (held window) with Boltz-2 → PyRosetta dG. Pipeline + the
single→multi-fold replica denoising is in `lnr_notes.md` (the other session owns
`boltz_validate.py`; `run_held` uses replicas, `run_mut_held` was single-fold as
of this writing). Key instrument fix this thread made: the old validity gate
required held-RMSD ≤ 2 Å of the LINEAR pose — but cyclisation MUST move residues,
so that gate contradicted the premise and rejected every fold. held-RMSD is now
a cyclisation-deviation DIAGNOSTIC; a generous pair-quality guard (both folds
≤ 8 Å) drops only grossly mislocalised folds (e.g. 1d4t luna at 12 Å).

Paired ΔdG vs nomut (single-fold, 8 targets; **negative = mutation strengthens
real binding**): luna-llm **−6.50** (3/6), terra-llm-ala **−6.62** (2/8). Positive
and sign-consistent with the proxy, BUT **tail-driven** (1ssc, whose nomut fold
bound anomalously weakly at −7.7) and ~50/50 per-target. terra's proxy "4/0
all-beneficial" did NOT hold in Boltz (2/4 hurt) — the pose-preserving proxy is
an imperfect guide, as predicted.

## The real dG CEILING (linear WT, same pipeline) — `lnr/boltz_linear_ref.py`

The scorecard's NATIVE −67 is Rosetta on DEPOSITED coords — a different pipeline
from the Boltz-folded cyclic arms, so not a fair ceiling. `boltz_linear_ref.py`
folds the WT sequence LINEAR through the SAME Boltz→Rosetta pipeline (strip the
one `cyclic: true` line — mainchain adds nothing else cyclic; whole peptide
templated at native; plain two-chain interface dG). So `linear_ref − nomut_cyclic`
is the PURE cyclisation cost. Run: `sbatch/s4_lnr_linref.sbatch` (job 112047).

**Validated:** Boltz-linear ≈ deposited native (RMSD 0.2–2.9 Å; e.g. 1jrr −155 vs
deposited −159). **Mean cyclisation cost = +29.4 REU**, very target-dependent
(1bjr/1ky6 ≈ free; 1jrr +89, 1ssc +66). **Existence proof that mutation can close
the whole gap:** on 1fiw, terra's mutant reaches the linear ceiling
(−54.6 vs −54.0); 1ssc ~84% (caveat: suspect nomut fold). But inconsistent
(1ee5 terra HURT). **The bar is now the linear ceiling, not nomut.**

| target | LIN ceiling | nomut cyc | cyc cost | luna | terra |
|-|-|-|-|-|-|
| 1bjr | −24.1 | −27.3 | −3.2 | −25.0 | −27.3 |
| 1ky6 | −16.8 | −19.6 | −2.9 | −19.6 | −15.6 |
| 1fiw | −54.0 | −38.6 | +15.4 | −43.4 | **−54.6** |
| 1elr | −41.3 | −21.7 | +19.6 | −18.5 | −21.7 |
| 1d4t | −70.3 | −46.7 | +23.6 | −32.4* | −46.7 |
| 1ee5 | −71.2 | −43.0 | +28.1 | −40.7 | −30.0 |
| 1ssc | −73.5 | −7.7 | +65.8 | −35.4 | −61.7 |
| 1jrr | −155.2 | −66.2 | +89.0 | −77.3 | −66.2 |

`*` 1d4t luna is a drifted fold (12 Å), unreliable. Caveat: cyclic arms scored by
ring-aware `score_structure`, linear ref by plain InterfaceAnalyzer (same core
dG_separated, no ring declaration — no "linear" chemistry exists).

## Dead thread (negative result): no free dG from energy-ranking the window

Hypothesis: `placement_search` commits the max-CONTACT window but we score by
ENERGY, so an energy-ranked window might recover dG for free. **Refuted.** Over
the same VERIFIED clash-free closer set, energy-max and contact-max are the SAME
window on all 5 targets with clean closers (free gain +0.0). The earlier "1d4t
+11" was a **stochastic-solver artifact**: `energy_ceiling.py` independently
re-solved closure (random-restart torsion), so its clash/closure verdicts
DISAGREE with both `committed_clean_window` and the saved clash map. The real
issue uncovered is **closure-verdict inconsistency** across the three consumers
(committed window, clash map, ceiling) — a solver-determinism problem, a DEEP fix
in shared code, not a cheap win. Conclusion: the ~40%/+29-REU cyclisation loss is
closer to fundamental; the lever is mutation, not window re-ranking.

## RESOLVED: the proxy does NOT rank within a complex like the true score — `lnr/rank_proxy_vs_true.py`

The decisive question for whether the proxy is a usable SELECTION oracle. Per
complex: WT + random single mutations + the LLM pick, window fixed, each scored
by PROXY and by TRUE (Boltz `diffusion_samples=4` → `score_replicas` median),
then within-complex Spearman + Pearson(proxy, true), top-1 agreement, and regret
(true dG forfeited by trusting the proxy's pick). Self-contained (no edits to
`boltz_validate.py`). Run: `sbatch/s4_lnr_rank.sbatch` (debug/gpu22, resumable).

**Result (job 112872, 2026-11-16):**

| complex | n (≤8 Å) | Spearman | Pearson | top-1 | regret |
|-|-|-|-|-|-|
| 1ee5 | 9 | +0.32 | +0.23 | DISAGREE (proxy llm_luna, true rand_9Q) | +11.7 REU |
| 1ssc | 6 | +0.26 | +0.16 | DISAGREE (proxy rand_3R, true rand_10P) | +11.0 REU |
| 1elr | 2 | — | — | — (5/7 folds drifted >8 Å, filtered) | — |
| **mean** | | **+0.29** | **+0.20** | **0/2** | **+11.4 REU** |

**Verdict: the pose-preserving proxy is NOT a usable standalone selection
oracle.** Within-complex ordering agreement is weak (ρ ≈ +0.3, near 0), it picks
the true-best mutation in 0/2 complexes, and trusting its pick forfeits ~11 REU
of real binding. This is the "ρ ≈ 0 ⇒ proxy misleads selection" branch and
directly explains the ~50/50 real win rate seen in the Boltz paired-ΔdG test.
The iterative loop (other session) cannot select on the proxy alone — it needs
the real 4-fold Boltz→Rosetta score for selection; the proxy is at most a cheap
PRE-FILTER to shortlist candidates before the expensive oracle.

**Metric note (Pearson vs Spearman).** Reported BOTH plus top-1/regret.
Spearman is the headline because the proxy is a selection oracle (ordering is
what the loop uses), the proxy/true scales differ (different pipelines; retention
saturates at 1 → monotone-but-curved map), and at n~5-7 one drifted fold
dominates Pearson. On the full data Pearson came in BELOW Spearman on both
complexes (small-n Pearson was optimistic: an n=5 partial read gave +0.76 that
collapsed to +0.23 at n=9) — so Pearson is a corroborating secondary, not the
headline. All four metrics agree the signal is weak.

**Bug caught here (fixed):** `rank_proxy_vs_true.py` called the other session's
updated `score_replicas` with the OLD signature → every TRUE score failed with
`missing 2 required keyword-only arguments: 'refined_dir' and 'stem'`, while the
proxy half looked fine. The resume key `(target_id, variant_id)` ignores
`true_error`, so a full run would have marked all variants "done" and computed
the Spearman on ZERO true scores — a confidently wrong answer. Fix: pass
`refined_dir=work, stem=f"rank_{vid}"` (matches `run_held`). Lesson: pinning the
job to a genuinely-free node (gpu22 idle) surfaced this early; queued on a
RAM-full node it would have taught nothing.

## Files / jobs

```
agent/lnr/seqmut.py            the mutation lever + arms + proxy scorer (+ test_seqmut.py)
agent/lnr/boltz_linear_ref.py  linear-WT Boltz→Rosetta ceiling
agent/lnr/rank_proxy_vs_true.py proxy-vs-true within-complex ranking (OPEN)
agent/lnr/energy_ceiling.py    cheap energy-window ceiling (CAVEAT: stochastic-solver inconsistency)
sbatch/s4_lnr_seqmut.sbatch    proxy breadth (CPU); s4_lnr_linref.sbatch (GPU gpu22);
sbatch/s4_lnr_rank.sbatch      ranking (GPU gpu24); boltz_validate.py = other session's
artefacts: seqmut_rows.jsonl (+ per-arm files), seqmut_scans.jsonl, boltz_mut_held.jsonl,
           boltz_linear_ref.jsonl, energy_ceiling.jsonl, rank_proxy_vs_true.jsonl
```

---

## The iterative loop was selecting NOISE — diagnosis and fix (2026-11-18)

*Code: [`../agent/lnr/seqmut_iter.py`](../agent/lnr/seqmut_iter.py),
[`boltz_inloop.py`](../agent/lnr/boltz_inloop.py). The v1 real-score run's rows
and trajectory are archived as `seqmut_iter_boltz_{rows,traj}.V1_SINGLEDRAW.20261117.jsonl`
and are NOT poolable with v2 (different budget and seeding). The fold cache
`seqmut_iter_boltz_cache.jsonl` IS shared — it is keyed by (variant, draw) and
v2's draw-0 folds use v1's seed, so no fold is recomputed.*

### What the first real-score-in-loop run actually showed

The firewall this thread built (loop folds with one Boltz seed; the report
re-folds the finalist with another) fired, and it indicted the SEARCH, not the
model:

| target | arm | subs | in-loop Δ vs WT | independent redraw Δ |
|-|-|-|-|-|
| 1bjr | iter-ea | 1:F,3:T | −11.69 | −5.47 |
| 1bjr | iter-llm | 2:M | **−8.01** | **−0.57** |
| 1elr | iter-ea | — (WT won) | 0.00 | 0.00 |
| 1elr | iter-llm | 1:Q,2:N | −1.11 | −2.01 |

**61% of the apparent gain did not survive a redraw**; the LLM's 1bjr win was
93% noise. Three compounding defects, each measured:

**F1 — the loop selected the argmin of noisy draws and never re-measured.** A
fitness is a MEASUREMENT: per-variant fold sd 3.74 REU mean (0.79–11.73), i.e.
SEM ≈1.9 on the 4-fold median; the SAME variant across independent runs spreads
~4 REU (1bjr WT: −20.9 / −24.3 / −26.8 / −26.9). Taking the min of ~15 such draws
buys ~1–2 SEM of pure optimism. On 1bjr the winner was within 1.2–1.6σ of three
runners-up; on 1elr it was 1.0σ from wild-type. Worse, the incumbent was never
re-scored, so one lucky draw owned the frontier for the whole run and steered
every later generation.

**F2 — the matched-budget control was confounded at the seed.** `evolve()` keyed
its RNG on `f"{target}|{arm}|{k_mut}"`, so each arm drew a DIFFERENT random
starting pool — despite `_seed_population`'s docstring claiming the opposite.
Mean seed-pool gap between arms **7.66 REU** (1jrr: EA −149.6 vs LLM −129.9),
against a headline arm difference of ~1 REU. And 53–62% of the Boltz budget went
on that random pool, so most of the "matched budget" measured random search.

**F3 — the LLM stopped searching after generation 1.** `beat_frontier_rate` 0.09
(91% of proposals worse than what it was shown); `best_gen ≤ 1` in 9/16 runs; the
random seed pool won outright in 3/16. **34–45% of proposals re-proposed a genome
already evaluated** — it was shown only the elite, so it had no memory of
failures. And it fixated: on 1elr 6 of 7 post-seed proposals carried the elite's
`1:Q`; on 1bjr gen 2 it spent the whole generation scanning position 2
(`2:L/V/M/F/Y/W`). With F1 that is the worst case — it locks onto a measurement
artifact and then polishes it.

### The fix (v2), and why each part is there

1. **Repeated measurement.** A `Population` member keeps every observation;
   `recovered` is their running MEAN and `sem` its standard error (from repeats
   at n≥2, else the evaluator's own within-draw SEM). The incumbent is
   **re-measured once per generation** with an independent Boltz seed
   (`BoltzScorer.score(..., draw=k)`: a distinct seed AND its own output dir, so
   it cannot read back draw 0's structures). Charged to the same budget both arms
   spend. `--no-reassess` restores v1.
2. **Noise-aware selection.** Elite ranked OPTIMISTICALLY (`mean − z·sem`, so a
   noisy candidate survives to be re-measured rather than crowned or dismissed on
   one draw); the finalist committed PESSIMISTICALLY (`mean + z·sem`, so the
   genome we spend the report fold on is good because it is good). Deterministic
   proxy ⇒ sem=0 ⇒ both collapse to the old argmin, so `--evaluator proxy` is
   behaviourally unchanged.
3. **Shared seed pool.** Two RNGs: seeding keyed WITHOUT the arm (identical
   starting pool ⇒ the comparison is paired), operator keyed WITH it (arms stay
   independent and each replays deterministically). Side benefit: the second
   arm's seed folds are cache hits.
4. **Failure memory + an exploration floor in the brief.** The elite block now
   carries ±SE and n; a new "already measured, do not repeat" block lists the
   non-elite members with their scores; the brief states the noise explicitly
   ("two sets within ~2·SE are TIED, not ranked") and requires ⌈m/2⌉ proposals to
   touch a position not on the frontier.
5. **Repeats cost a re-ask, not a proposal slot.** `llm_offspring` drops
   already-known genomes and re-asks ONCE naming them before the EA fallback.
   **The same applies to the EA** (`ea_offspring(..., known=)` resamples) — caught
   in a dry run where the EA made 18 distinct evals to the LLM's 24, which would
   have handed the LLM arm extra oracle calls and voided the control.
6. **Budget re-split**: 24 evals = 5 seed + 5 re-measure + 14 offspring (was 8
   seed of 15). Operator-chosen share 58%, up from 47%; random seeding 21%, down
   from 53%.

### Does the fix work? Known-truth simulation, empirical noise

`evolve()` run against a synthetic landscape with KNOWN truth, noise drawn from
the **empirical** per-variant fold spreads (n=56, mean 3.74, max 11.73 REU),
120 replicates, same budget:

| configuration | claimed | real | bias | % of claim that is noise |
|-|-|-|-|-|
| v1: argmin, one draw, never re-measured | −8.46 | −6.05 | −2.41 | 28% |
| + pessimistic commit only | −8.33 | −6.33 | −2.00 | 24% |
| + re-measure incumbent only | −7.57 | −6.52 | −1.05 | 14% |
| **v2: re-measure + pessimistic commit** | **−7.45** | **−6.52** | **−0.93** | **12%** |

Selection bias down **61%**, and the finalist's TRUE fitness is BETTER
(−6.05 → −6.52) even though 5 of 24 evaluations now go to re-measurement rather
than new genomes. Re-measurement does the heavy lifting; the pessimistic bound
adds a little because the real noise is heteroscedastic (sd 0.79–11.73).

### What to read in the output

New per-generation trajectory fields (redrawable, no recompute):
`incumbent_shift` (how far the re-measured leader moved — the optimizer's curse,
live; POSITIVE = the leader got worse), `best_sem`, `best_n_obs`,
`n_dup_proposed`. New `iter` block fields: `n_reassess`,
`mean_incumbent_shift`, `best_sem`, `best_n_obs`, `n_dup_proposed`; the report
table gained `meanSEM`, `shift` and `dup` columns.

**Expect `best_recovered` to go UP between generations now.** That is the
correction working, not a regression — v1's monotone frontier was monotone only
because it never re-measured.

### Jobs

```
sbatch/submit_lnr_seqmut_iter_boltz.sh   v2 real-score-in-loop (writes *_v2_rows/traj)
  SEQITER_REASSESS=0 ...                 the A/B: v1 behaviour at the v2 budget
sbatch/submit_iter_boltz_report.sh       finalist fold + believed number
  SEQITER_SEQMUT_SRC=<rows file>         (now overridable, so v2 is reported without touching v1)
```

### Open lever: re-measurement should be CPU-only, not a GPU fold

`BoltzScorer.score(..., draw=k)` re-measures by shifting the **Boltz** seed, so
each repeat costs a full ~140 s GPU fold. That is more than it needs to cost.
Per [`lnr_notes.md`](lnr_notes.md) the scatter is **Rosetta-side**: 1bjr nomut
folded twice at the SAME Boltz seed 42 gave dG_refined −26.95 vs −24.28 with
byte-identical held_rmsd (5.475), i.e. Boltz diffusion is deterministic at fixed
seed and the whole ±3 REU floor lives in FastRelax.

So an independent re-measurement can re-run `boltz_validate.score_replicas` on
the **same cached PDBs** in a fresh process (`rosetta_interface._score_one` calls
`pyrosetta.init` per child with no `-run:constant_seed`, so the relax draw is
already independent) — CPU seconds instead of a GPU fold. The whole GPU budget
then goes to new genomes, and repeats can go well past 2.

Not implemented, because it changes what the matched budget COUNTS (GPU folds
vs. fitness evaluations) and that is a design decision, not a drop-in. Decide it
when v2's numbers are in.

---

## Tested: do interface-CONFIDENCE scores rank variants better than Rosetta dG? **No.** (2026-11-18)

*Code: [`../agent/lnr/confidence_metrics.py`](../agent/lnr/confidence_metrics.py)
(+ `test_confidence_metrics.py`, 9 tests, no GPU).
Job: `sbatch/s4_lnr_confidence_rank.sbatch` (CPU, minutes, resumable, safe to run
while the GPU search writes). Rows: `conf_metrics.jsonl`. Re-render with
`--report` — no recompute.*

### Why it was worth testing

Our in-loop fitness uses Boltz→Rosetta dG as an **affinity regressor**, the
hardest possible use of it: dedicated ΔΔG methods cap at Pearson ~0.63
([Flex ddG](https://www.biorxiv.org/content/10.1101/221689v1.full), and only with
a 20–30 model ensemble; ~0.45 on a SKEMPI v2 test split), and our own
proxy-vs-true check measured ρ ≈ +0.29. Meanwhile
[Overath et al. 2025](https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1)
(3,766 designs, 436 binders vs 3,330 non-binders) found **interface-focused
confidence scores beat the physics energies**, best paired with a physicochemical
descriptor. And Boltz already writes `confidence_*.json` (ipTM, *pairwise
per-chain* ipTM, ipLDDT, ipDE), the full `pae_*.npz` and `plddt_*.npz` for every
fold — which `boltz_inloop` discarded. Free to recover, retroactively.

### What the test can and cannot answer

There is **no experimental ground truth anywhere in this project**, so this is
NOT "which metric predicts affinity". It measures the three things the data can
settle: **(a) resolution** = between-variant SD / within-variant SEM (below ~1 a
metric cannot rank at all), **(b) reproducibility** across independent fold
seeds, **(c) agreement** with dG and with drift.

### Result (61 folds, 1bjr + 1elr, job 114710)

| metric | discrim 1bjr | 1elr | mean | redraw (% of variant spread) | ρ(dG) 1bjr / 1elr |
|-|-|-|-|-|-|
| **dG_refined** | 2.03 | 2.41 | **2.22** | **55%** | — |
| iptm_min | 2.12 | 1.84 | 1.98 | 195% | +0.57 / +0.14 |
| ipae_min_dir | 2.17 | 1.44 | 1.80 | 191% | +0.52 / +0.25 |
| lis | 2.11 | 1.61 | 1.86 | 243% | +0.51 / +0.13 |
| iplddt | 3.22 | 4.63 | 3.92 | 84% | −0.09 / −0.18 |
| pep_plddt | 2.89 | 4.53 | 3.71 | 132% | +0.10 / −0.07 |
| complex_plddt | 4.39 | 6.00 | 5.20 | 62% | −0.12 / −0.13 |

**1. The ipSAE family loses.** Every interface-confidence score (ipTM, ipAE, LIS,
ipDE) resolves variants *worse* than dG (1.80–1.98 vs 2.22), and on the one
re-measured variant each moves **~2× the entire between-variant spread** on an
independent redraw. The caveat stated when this was proposed turned out to be the
reality: those metrics were validated on **binder vs non-binder for de novo
designs** — a coarse contrast — not on 2-residue substitutions of a peptide that
already binds.

**2. `complex_plddt`'s "2.3× dG" is an artifact, and the first verdict rule fell
for it.** The receptor is 281 of 291 residues and *identical in every variant*,
so complex-wide metrics measure a near-constant: tiny noise, flattering
signal/noise ratio, ordering correlated with nothing (ρ ≈ −0.12 with dG, −0.15
with drift). **Precision is not relevance.** The verdict now requires resolution
AND cross-seed reproducibility, and discounts whole-complex metrics by name
(`WHOLE_COMPLEX`). Keep this trap in mind for any future fitness: a
discriminability ratio alone can be bought with an invariant.

**3. Confidence and energy disagree about drift, informatively.** ρ(ipTM,
held_rmsd) = **+0.48** but ρ(dG, held_rmsd) = **+0.41** (dG more positive = weaker
binding). So dG *penalises* deviation from the bound pose while ipTM *rewards*
it. That is coherent: a strained ring held near the linear template is
low-confidence, a relaxed properly-closed ring that has moved is high-confidence.
**ipTM answers "is this a confident cyclic structure", not "does it bind in the
native mode"** — a different question from the one this project asks.

### Caveats

- Reproducibility is **n=1** (one variant folded twice so far) — anecdotal, not a
  measurement. It sharpens automatically as the v2 run's re-measurement draws
  land; just re-run `--report`.
- Only 2 targets, both small peptides on one receptor each.
- This is the ipSAE *family*, not Dunbrack's exact ipSAE (which renormalises d0
  per residue over the PAE-passing partner set). Given the family shows no
  signal at this contrast, exact ipSAE is unlikely to rescue it — but that is an
  inference, not a measurement.

### Conclusion

**The lever stays ensemble averaging of the energy, not a new score.** Which is
what [Flex ddG's 0.54 → 0.63 curve](https://www.biorxiv.org/content/10.1101/221689v1.full)
already said, and what the re-measurement fix above implements. The open CPU-only
re-scoring lever is now the highest-value next step, not a confidence composite.

---

## Tested: which OBJECTIVE ranks variants best — physics proxy vs sequence LMs. **None do.** (2026-11-18)

*Code: [`../agent/lnr/seq_rankers.py`](../agent/lnr/seq_rankers.py) (+ `test_seq_rankers.py`,
8 tests, fake-model, no weights). Job: `sbatch/s4_lnr_seq_rankers.sbatch` (CPU,
minutes). Rows: `seq_rankers.jsonl`. Re-render: `--report`.*

### What was compared, and why these

Grounded in how the directed-evolution field actually drives sequence search —
EVOLVEpro (PLM embeddings → RF surrogate), ProtRL (ProteinMPNN log-likelihood as
the optimization target), ESM-IF (structure-conditioned likelihood, ρ up to 0.65
vs affinity on antibody DMS). ESM-IF isn't installed and is cyclic-OOD; **PepMLM-650M**
(a peptide-*binder* LM, cached) is the better available analog. Candidates, all
**fold-free (no GPU)**, judged against the Boltz×4→Rosetta oracle on the 23
variants where that oracle is already computed (`rank_proxy_vs_true.jsonl`):

- `proxy_recovered` — current pose-preserving Rosetta proxy (baseline)
- `esm2_pll` — ESM-2 650M peptide pseudo-log-likelihood (generic naturalness)
- `pepmlm_cll` — PepMLM-650M conditional LL of the peptide **given the receptor**
- z-composite — z-scored mean of the above

### Result (23 variants, 3 complexes)

| complex | n | proxy | esm2 | pepmlm | z-comp |
|-|-|-|-|-|-|
| 1ee5 | 9 | +0.32 | +0.72 | −0.05 | +0.22 |
| 1elr | 7 | −0.71 | −0.11 | +0.39 | +0.07 |
| 1ssc | 7 | −0.04 | −0.36 | −0.46 | −0.57 |
| **mean ρ** | | **−0.14** | **+0.08** | **−0.04** | **−0.09** |
| **positive on** | | **1/3** | **1/3** | **1/3** | **2/3** |

Top-1 regret (true dG lost by trusting the ranker's pick): proxy **+10.2**, esm2
+13.6, pepmlm +14.7 REU — all catastrophic; the physics proxy is the *least bad*.

### Verdict: NONE is usable — and that IS the finding

Every candidate sign-flips across complexes (positive on only 1 of 3), every mean
ρ sits inside the noise floor, and my first verdict rule wrongly crowned esm2 on a
+0.08-beats-−0.14 difference that is pure small-n noise (rule fixed: now requires
sign-consistency on a majority of complexes AND mean ρ ≥ 0.30, else "noise").

**The real answer to "what is the objective":** the true between-variant
differences (~few REU) are **at the ground truth's own measurement noise** (oracle
median fold sd 3.4 REU → SEM ~1.7). So the ranking problem is unresolved for
*every* scorer — physics, generic PLM, binder LM, composite alike. **The
bottleneck is measurement resolution, not the choice of score.** You cannot pick a
better objective when the ruler you'd grade it with cannot itself tell the
variants apart.

### What follows (in order)

1. **Denoise the oracle first** — ensemble-average the energy (Flex ddG's
   0.54→0.63 with 20–30 models; here, CPU-only re-relax of cached Boltz PDBs since
   the noise is Rosetta-side, [[lnr-dg-noise-floor]]). Until the *ground truth*
   resolves variants, no ranker can be benchmarked, and the v2 search's
   re-measurement is attacking exactly this.
2. **Or change the objective** from ranking tiny within-target ΔdG to a coarser,
   *resolvable* contrast — binder vs non-binder, or pass/fail vs the cyclization
   ceiling — which is the contrast the field actually predicts well (Overath:
   classification, not regression).
3. Only then revisit which cheap signal tracks the denoised truth.

### Caveats

- 23 variants, 3 complexes — small. But the conclusion is about the **noise floor**
  (independently measured SEM 1.7 REU vs few-REU effects), which small n only
  reinforces: detecting a ranking signal here would need far more replicates.
- LMs score **linear** sequences; the peptide is cyclic. The OOD risk was real and
  is now measured, not assumed: the binder LM does not rescue ranking here.

---

## Lever #1: CPU re-relax denoiser — shrink the oracle's noise without a GPU (2026-11-18)

*Code: [`../agent/lnr/rescore_denoise.py`](../agent/lnr/rescore_denoise.py)
(+ `test_rescore_denoise.py`, 6 tests, fake-relax, no PyRosetta). Job:
`sbatch/s4_lnr_denoise.sbatch` (CPU, parallel). Rows: `denoise_scores.jsonl`.*

### Why this is the right next move (and free)

The ranking comparison showed the bottleneck is the oracle's own resolution, not
the choice of score. The noise-floor memory pins WHERE that noise is: the SAME
Boltz structure (fixed seed, identical coords) scored −26.95 vs −24.28 on two
relaxes — so the ±3.4 REU is dominated by **Rosetta FastRelax**, not Boltz. The
scorer's `pyrosetta.init` (in `rosetta_energy._ensure_pyrosetta_init`) sets **no
`-constant_seed`**, so every `score_structure` call is an INDEPENDENT relax draw.
Therefore the dominant noise averages down on CPU alone: re-relax the N cached
diffusion structures R times each, aggregate over N×R valid draws, SEM ∝
1/√(N×R). No new fold, no GPU.

### What it measures (the payoff metric)

Per target, BEFORE (cached N-fold estimate, 1 relax each) vs AFTER (N×R relaxes):
- within-variant **SEM** (should fall ~√R),
- **resolution** = between-variant SD / within-variant SEM — the number that
  decides whether the oracle can rank at all. The whole point is to see it cross
  ~1, the threshold below which nothing is rankable.

It also prints the **within-structure relax spread** (re-relaxing the same PDB):
>0 confirms the draws are independent and quantifies the Rosetta noise directly.
If that spread were ~0, the noise would be in the FOLD (coordinate sampling) and
the lever would be GPU samples, not CPU re-relax — so the number self-diagnoses
which regime we're in.

### Design / safety

Reads cached fold PDBs read-only, writes its own refined PDBs under a dedicated
scratch dir → safe to run WHILE the GPU search writes new folds. Parallel across
worker processes (each inits PyRosetta once; its consecutive relaxes are
independent draws — the efficient pattern, no per-task re-init). Resumable per
(variant, model, repeat). First run scoped to 10 variants/target × R=6 on the
boltz_inloop cache (1bjr 45 vars, 1elr 27); scale via `DENOISE_MAX_VARIANTS` /
`DENOISE_REPEATS`.

### If it works

Re-benchmark the cheap rankers (`seq_rankers.py`, `confidence_metrics.py`) against
the DENOISED oracle: a signal that was buried under ±1.7 REM SEM may become
visible once the ruler is sharper. Then wire the chosen depth into the in-loop
scorer (`boltz_inloop.fold_score_one` → score each fold R times) so the SEARCH
selects on a denoised number, not one relax draw. Result pending (job 114779).

### RESULT (job 114779): CPU re-relax does NOT work — the noise is fold sampling, not relax

Ran 480 relax draws (20 variants × 4 folds × 6 relaxes, 1bjr+1elr). The variance
decomposition is decisive:

| source | 1bjr | 1elr | reducible by |
|-|-|-|-|
| relax noise (within one fold, re-relaxed) | 1.83 | 1.36 REU | **CPU re-relax** |
| **fold noise** (between the 4 diffusion samples) | **4.43** | **3.42 REU** | **GPU (more samples)** |

Fold sampling dominates relax by ~2.5×. The honest **clustered** SEM (sd of fold
means / √n_folds) is 2.22 / 1.71 — **identical to the cached SEM** (2.28 / 1.63),
and discrim did not improve (1.45 vs 1.60; 1.94 vs 2.09). CPU re-relax cut the SEM
by −1%.

**Caught my own bug.** The first report claimed "SEM halved, resolution doubled"
because `aggregate()` pooled all 24 correlated draws as iid and divided by √24 —
fabricating precision the clustered draws do not earn. Fixed: `aggregate_clustered`
computes the SEM as fold-mean-spread / √n_folds, and the report now prints the
relax-vs-fold decomposition and the honest SEM. The lesson is general: **any SEM
over re-scored structures must cluster by fold, or it lies.**

**The real lever is GPU, not CPU.** SEM = fold_sd / √n_folds, so the only way to
sharpen the oracle is more Boltz diffusion samples: 4 → 16 folds ~halves it
(predicted SEM ~1.0, discrim ~3–4). The pipeline already supports it
(`configure_replicas` / `--boltz-replicas`, chunkfix shadow for >4). CPU re-relax
past ~2 per fold is wasted — keep 1–2 relaxes/fold for the relax floor, spend the
budget on folds.

This also refines the noise-floor memory: the ±3 REU floor is **not** mostly
FastRelax — across diffusion samples it is ~70% fold-coordinate sampling, ~30%
relax. The earlier "same seed, folded twice → different dG" evidence measured only
the relax part (identical coords); it did not see the larger between-sample
spread.

### VERIFY (job 114828): more diffusion samples helps precision modestly — and the 4-fold std was UNDER-estimating the noise

Re-folded 6 cached variants at replicas=16 (gpu22, idle). Two findings:

| variant | std over 4 folds (cached) | std over 16 folds | 
|-|-|-|
| 1bjr WT | 0.79 | **5.05** |
| 1elr WT | 0.93 | **2.36** |
| 1bjr 1:D,9:Y | 2.96 | 4.47 |
| 1elr 1:A,4:E | 4.44 | 3.05 |

1. **The 4-fold std systematically underestimates fold variance.** std-of-4 is a
   poor estimator of a ~4–5 REU spread (it undershoots, badly for the WTs). The
   16-fold std (~4 REU mean) is the reliable number. So every per-variant SEM we
   reported from 4 folds was **too small** — the real noise is worse than the
   cached stds implied, and the project's "±3 REU floor" is optimistic for the
   fold component.
2. **SEM: 1.39 (4 folds) → 1.01 (16 folds).** The samples are ~independent
   (per-fold std first-4 3.10 vs all-16 4.03, consistent), so √n scaling roughly
   holds *on the honest fold sd* (~4 REU → SEM 2.0 at n=4, 1.0 at n=16). The
   "achieved 1.45× worse than predicted" line is an artifact of seeding the
   prediction with the too-tight cached std; using the honest sd, 4→16 folds does
   halve the SEM.

**Verdict on the lever:** more diffusion samples works, but it is a weak return —
4× the GPU for SEM ~1.4→1.0, and driving SEM below ~0.5 needs ~64 folds. More
importantly it is **still only precision.** It makes the oracle agree with itself
more tightly; it says nothing about whether the oracle is right. See the
accuracy/ground-truth thread below.

### v2 noise-defended search (job 114631): the fix works mechanically; LLM ≈ EA

All 4 cells done. Every incumbent-reassessment shift is POSITIVE (1bjr ea +0.87,
llm +1.86; 1elr ea +0.46, llm +0.27) — the re-measurement consistently caught an
optimistic leader and corrected it, exactly the optimizer's-curse fix. Finalists:
1bjr ea −29.07±0.14 vs llm −29.36±1.54; 1elr ea −23.49±0.88 vs llm −23.69±0.59.
LLM beats EA by ~0.2–0.3 REU on both — **far inside the SEM and the ~4 REU fold
floor, i.e. a wash.** The v2 machinery is correct; it cannot manufacture a signal
the noisy, unvalidated oracle does not carry.

---

## THE ACCURACY GATE result (job 115191, 2026-11-19): the scorer is only WEAKLY accurate

First-ever test of the oracle's ACCURACY (vs reality), not precision. Ran our
Rosetta interface-dG scorer on SKEMPI v2 crystal structures (no Boltz fold — best
case) and correlated predicted ΔΔG against **measured** ΔΔG. Code:
`agent/lnr/skempi_accuracy.py`, rows `skempi_ddg.jsonl`.

**Result (n=44 single interface mutations, 41 complexes):**
- **Spearman ρ = +0.22, Pearson r = +0.29**
- **sign agreement = 59%** (barely above the 50% coin flip — it can't reliably
  even call the *direction* of a mutation)
- reference: Flex ddG (35-model ensemble) reaches Pearson ~0.6 here; our
  single-relax scorer ~0.29, consistent with having no ensemble.
- the n=5 early peek (+0.80) collapsed to +0.22 at n=44 — textbook small-n
  regression; good that it wasn't banked.

**Robustness note:** 108/363 tasks failed, 102 of them PyRosetta **segfaults**
(exitcode −11) on specific SKEMPI structures, not OOM. The per-task subprocess
isolation (added after a `BrokenProcessPool` wiped the first attempt) turned each
crash into one dropped row. Segfaults are structure-specific, uncorrelated with
mutation effect, so the surviving n=44 is not biased toward easy cases.

**Verdict: WEAK, and this is the BEST case.** ρ≈0.22–0.29 on a crystal structure
with zero fold noise. Our actual pipeline folds the peptide with Boltz first
(+~4 REU fold noise) on an OOD cyclic backbone — it can only get worse from here.
So **ranking binding affinity by this score is not a viable objective**, and
neither is calling mutation direction (sign 59%).

### Decision (data-driven, ends the "which score" thread)

Stop trying to rank ΔdG. The whole chain — pose-preserving proxy (ρ0.29 vs a
noisy oracle), confidence metrics (worse than dG), sequence LMs (no signal),
denoising (precision only), and now the accuracy gate (ρ0.29 vs *truth*) — all
say the same thing: **fine affinity ranking is not achievable with the scores we
have.** The viable objective is the coarse, large-signal, field-validated one:
**native-mode recovery / self-consistency** — does the designed cyclic peptide
FOLD to and retain the native binding mode (RMSD/recovery vs the deposited pose),
a pass/fail contrast, not a ΔdG rank. That is the signal AF2-initial-guess binder
design actually succeeds on, it does not depend on the ΔΔG sign the scorer gets
wrong 41% of the time, and it is computable from what we already have.

---

## The within-target CYCLIC-PEPTIDE accuracy gate — ground truth at last (2026-11-19)

*Code: [`../agent/lnr/cpset_accuracy.py`](../agent/lnr/cpset_accuracy.py)
(+ `test_cpset_accuracy.py`, 33 tests, no PyRosetta/network). Job:
`sbatch/s4_lnr_cpset_accuracy.sbatch` via `submit_lnr_cpset_accuracy.sh` (CPU,
resumable). Rows: `cpset_dg.jsonl`. Re-render with `--report` — no recompute.*

### Where this came from

A paper proposing a better P–cp binding metric: Zhao, Huang, Weng, Jiang, Hu,
Kang & Hou, *Brief Bioinform* 2025;26(6):bbaf632,
[doi 10.1093/bib/bbaf632](https://doi.org/10.1093/bib/bbaf632), data at
[github.com/huifengzhao/MM_PBGBSA-CP](https://github.com/huifengzhao/MM_PBGBSA-CP).
Fine-tuned **minimization-only MM/PBSA** (ff03, implicit solvent, ε_in = 2,
LCPO surface term γ = 0.0072, **no entropy, no MD** — MD made it worse), two-step
with polarity-dependent pose re-ranking, ~3 s per prediction, **Rp = −0.732**.

### Why its headline must NOT be banked (and what to take instead)

**The paper contains no within-target experiment at all** — no mutation series, no
congeneric series on a fixed receptor, no per-target metric, no sensitivity
analysis to small sequence changes. Its only stratifications are interface
polarity and peptide length. The −0.732 is Pearson across **24 heterogeneous
complexes** spanning pKd 3.55–10.29, i.e. ~9 kcal/mol of dynamic range — the easy
contrast. Our task is ranking 2-residue substitutions of one peptide on one
receptor. Importing −0.732 as evidence for that would be this project's
best-documented mistake for the third time (the n=5 peek at +0.80 that collapsed
to +0.22; `complex_plddt`'s discriminability bought with an invariant).

**What the dataset gives us is far better than its metric.** Grouping their 50
complexes by receptor identity (union-find on the longest receptor chain, ≥0.95)
shows **38 of the 50 fall into 10 CONGENERIC SERIES** — same receptor, cyclic
peptide variants, measured Kd — which the authors never used. That is **70
within-series ordered pairs** of ground truth, on cyclic peptides, which this
project has never had:

| series | n | pairs | tier | ΔpKd range | peptides |
|-|-|-|-|-|-|
| **S03_keap1** (KEAP1 Kelch) | **6** | **15** | **BB** | **3.30** | GDPETGE/GEPETGE/GDEETGE/GAPETGE/GAEETGE/GDEEAGE |
| S01_strep | 7 | 21 | SS | 2.06 | HPQF / HPQGPP cores |
| S02_qfdlstrrlk | 6 | 15 | SS | 2.07 | QFDLSTRRLK point mutants |
| S04_sfti | 4 | 6 | MIX | 3.08 | SIPPI family |
| S05_6d3 | 4 | 6 | BB | 1.59 | GRxYKSxPPIxFPD mutants |
| S06–S10 | 2–3 | 7 | SS/MIX | 0.02–1.28 | |

**S03_keap1 is a drop-in replica of the LNR task**: one receptor, six
all-canonical **head-to-tail (BB)** cyclic 7-mers differing by 1–2 substitutions,
pKd 3.55–6.85 = **3.3 log units ≈ 4.5 kcal/mol**, with single substitutions worth
2.3–2.8 logs each (D2A, T5A). That is **3–4× our ~4 REU noise floor, not inside
it** — so unlike the SKEMPI gate, a null result here is informative rather than
ambiguous.

This is the real answer to "what objectively evaluates a sequence's binding once
it is folded head-to-tail": **not a new score — the instrument to grade any
candidate score on our chemistry and our task.**

### What the gate removes that the SKEMPI gate could not

The SKEMPI gate (ρ +0.22, sign 59%) carried two confounds: **wrong chemistry**
(linear protein–protein) and **effects inside the noise** (~1 kcal/mol vs ~4 REU).
This fixes both. Like SKEMPI it scores **deposited crystal coordinates — no Boltz
fold** — so it isolates the scoring function, best case.

Three numbers, deliberately separated:
1. **HEADLINE — within-series pair sign agreement.** Does the scorer put the
   measured-stronger peptide at the lower dG? Pooled over all series, stratified
   by |ΔpKd| (0 / 0.3 / 0.5 / **1.0** = "can it call a 10× affinity difference"),
   with a **Wilson 95% CI**, because the question is literally "better than a coin
   flip". Ties count as FAILURES. Reported separately for the **BB/MIX tier**
   (28 of the 70 pairs), which is the only tier a head-to-tail claim may use.
2. **Per-series Spearman ρ(dG, pKd)**, *never pooled across series* — pooling
   would let the cross-target dynamic range masquerade as within-target skill,
   the exact inflation this gate exists to prevent.
3. **Cross-complex correlation over all 50** — reported ONLY as the
   paper-comparable number, printed beside the paper's own −0.545 / −0.813
   ceiling / −0.316 ADCP, with an explicit note that those calibrate row 3 alone.

**Sign convention** (stated once, asserted in tests): higher pKd = stronger;
dG_separated more negative = stronger; so a working scorer gives **negative**
ρ(dG, pKd). Pair agreement is defined sign-free, so it cannot be fooled by a slip.

### The verdict rule is PRE-REGISTERED

Fixed in the module docstring before the first run, because the `seq_rankers`
rule had to be rewritten after it crowned esm2 on a +0.08-vs-−0.14 difference and
the `confidence_metrics` rule after `complex_plddt` fooled it:

- **PASS** — agreement at |ΔpKd| ≥ 1.0 has CI95 lower bound ≥ 0.65 **and** mean
  per-series ρ ≤ −0.40 **and** ≥60% of n≥3 series negative
- **WEAK** — CI lower bound > 0.50 but PASS not met
- **FAIL** — CI lower bound ≤ 0.50 (a coin flip)

**A PASS reopens the affinity objective**: it would mean the pivot was forced by
SKEMPI's wrong chemistry and sub-noise effects, not by the scorer. **A FAIL closes
it properly** — with ground truth that is finally the right chemistry *and* well
above the noise floor, a far stronger negative result than SKEMPI could give.

### Instrument details that are not cosmetic

- **Chains are read from the STRUCTURES, never from the CSV.** They disagree for
  **1SLE** (CSV `BD:M`, files hold `D` and `P`); a wrong chain spec makes
  InterfaceAnalyzer return a meaningless dG **without raising**.
- **The series table is FROZEN in code** and `--regroup` re-derives it from the
  structures and diffs it; the sbatch runs that check as a hard gate *before*
  spending CPU. A grouping that drifted would silently move a within-series
  headline. Verified: grouping reproduces exactly, and it is **unambiguous** —
  every same-receptor identity ≥0.95, every different-receptor ≤0.55, **nothing in
  between**. Two cases the longest-chain rule gets right: 6D3X ships a receptor
  *dimer* (492 res vs 247) yet is the same target; 1SFI's and 6O21's groups are
  both ~223 res and share the SIPPI motif but are only ~0.67 identical, so they
  stay separate — merging them would have turned a cross-target comparison into a
  "within-target" one.
- **Prep is recorded, never silent**: CYX→CYS (997 residues) so Rosetta detects
  disulfides geometrically; ACE/NHE caps **dropped** (14 complexes). All 50
  prepare cleanly. The capped complexes are **entirely in the SS tier** (all of
  S01, both of S10) — the **BB/MIX headline tier is cap-free**, which the report
  prints as a flag rather than leaving to be discovered. Also flagged: **6O21**'s
  structure has 14 peptide residues where the CSV says 11 (it sits in S08, whose
  ΔpKd is 0.02 and which drops out at every threshold ≥0.3 anyway).
- Subprocess isolation per task is kept: the SKEMPI run lost **102/363** tasks to
  uncatchable PyRosetta segfaults, and a shared pool turns one crash into a
  `BrokenProcessPool` that discards every finished result.

### MM/PBSA is GATED on this result, deliberately

AmberTools 26 installs from conda-forge, and because the paper's protocol is
**minimization-only**, free `sander` suffices (no commercial `pmemd.cuda`). But
their repo **stripped the hard part**: there is no tleap input and no `bond`
command anywhere, so head-to-tail topology building is absent and would have to
be written; the published script also has an unassigned-variable bug on its own
documented `implicit` path. So the order is: measure the incumbent on this
instrument first. If Rosetta already matches the paper's cross-complex −0.545
*and* ranks within series, MM/PBSA buys nothing. `--scorer` is the drop-in hook
and every row records which scorer produced it, so the two can never be pooled.

### RESULT (job 115806, 2026-11-19): FAIL — and the cross-complex number is a SIZE ARTIFACT

150/150 rows, **0 failures**, 50/50 complexes (contrast: the SKEMPI run lost
102/363 to segfaults — crystal P–cp complexes are far better behaved).

| metric | value | PASS needs |
|-|-|-|
| pair agreement \|ΔpKd\|≥1.0 | **0.424**, CI95 **[0.27, 0.59]** | CI lo ≥ 0.65 |
| mean per-series ρ | **+0.089** (negative = working) | ≤ −0.40 |
| n≥3 series with negative ρ | **1 of 6** (0.167) | ≥ 0.60 |
| resolution (within-series SD / SEM) | **5.32** | — |

**Verdict FAIL — and critically NOT noise-limited.** Within-complex SEM is
0.695 REU against a within-series dG SD of 3.70, i.e. resolution 5.3: the ruler
separates these variants easily. This is the project's **first clean
accuracy-limited (not precision-limited) result**. Every earlier null
([[ranking-is-noise-limited-not-score-limited]]) was ambiguous because the effects
sat inside the noise; here they do not.

Per series (negative = working): S01_strep +0.667, S06_swrglenh +0.500,
S02_qfdlstrrlk +0.174, S04_sfti +0.105, **S03_keap1 +0.086**, **S05_6d3 −1.000**.

**S03_keap1 — the drop-in LNR replica — is ordered at chance.** Six head-to-tail
7-mers, one receptor, 3.30 log units (≈4.5 kcal/mol) of measured range:

| | pKd | dG (REU) |
|-|-|-|
| 7K2H | 6.85 | −27.28 ± 0.10 |
| 7K2M | 6.82 | −26.81 ± 1.10 |
| 7K2E | 6.33 | −29.94 ± 0.58 |
| 7K2I | 4.89 | **−30.31** ± 0.26 |
| 7K2F | 4.00 | −26.07 ± 2.57 |
| 7K2G | 3.55 | −29.11 ± 1.23 |

Pair agreement 0.467 (CI [0.25, 0.70]). The scorer gives the **weakest-but-one**
peptide (7K2I, pKd 4.89) the **best** dG and the strongest (7K2H) nearly the
worst, compressing a 4.5 kcal/mol real range into 4.2 REU of scatter.

Tiers at \|ΔpKd\|≥1.0: **BB only 0.615** (n=13, CI [0.36, 0.82]), BB+MIX 0.588
(n=17), **SS only 0.250** (n=16, CI [0.10, 0.49]). The sub-chance SS number is
plausibly a prep artifact — the SS tier is where ACE/NHE caps were dropped
(all 7 of S01, both of S10, both of S07, 1 of S06) — whereas the BB tier is
cap-free. So the head-to-tail tier is the *least* bad, but at n=13 no claim is
available either way.

### The mechanism: dG measures peptide SIZE, not affinity

| correlation (cross-complex, n=38–50) | ρ |
|-|-|
| ρ(dG, **peptide length**) | **−0.718** |
| ρ(dG, pKd) | −0.225 |
| ρ(pKd, peptide length) | +0.403 |
| **partial ρ(dG, pKd \| peptide length)** | **−0.002** |

**Removing peptide length, Rosetta's correlation with measured affinity is
exactly zero.** The whole cross-complex −0.225 was the size confound: more
residues → more buried interface → more negative dG, and in this benchmark longer
cyclic peptides happen to bind better. Within a series length is near-constant,
the size signal vanishes, and ρ collapses to ~0. One mechanism explains both rows.

**Null-scorer control** (lower = predicted stronger, same instrument):

| scorer | cross-complex ρ | within-series agreement (\|ΔpKd\|≥1.0) |
|-|-|-|
| Rosetta dG | −0.225 | 0.424 |
| **NULL: −peptide_length** | **−0.376** | 0.061 |
| NULL: −dSASA_interface | −0.077 | 0.303 |
| NULL: −receptor_size | +0.100 | 0.424 |
| dG / peptide_length | +0.234 | 0.545 |

**A scorer that counts nothing but peptide residues beats Rosetta dG (−0.376 vs
−0.225) AND AutoDock CrankPep (−0.316) on the cross-complex task** — while
scoring 0.061 within-series, exactly as the mechanism predicts.

### Consequence for the paper's headline, and for the Amber decision

The paper's MM/PBSA numbers are on **this same 50-complex set**, whose peptides
span **6–24 residues** with ρ(pKd, length) = +0.403. **The paper reports no length
or size control.** So its −0.545 must be read against a trivial length baseline of
**−0.376**, not against zero; its own length-subset split (short ≤10 aa Rp −0.400
vs long >10 aa −0.587) is consistent with size carrying much of it. The honest
increment of a tuned, fine-grained MM/PBSA over "count the residues" is therefore
much smaller than −0.545 suggests, and **nothing in the paper speaks to the
within-target contrast at all**.

**So the Amber gate did NOT really open.** The cross-complex comparison that was
supposed to justify installing AmberTools is a comparison on a metric a residue
count wins. MM/PBSA would have to be shown to carry within-target signal — a
quantity for which there is currently zero evidence from any source, and on which
our instrument shows the physics dG has none once size is removed.

**This confirms the native-mode pivot with a much better argument than SKEMPI
gave**: not "the scorer is weak and the noise is large", but "the scorer's
apparent affinity signal is entirely a size artifact, and within-target it has
none, on ground truth of the right chemistry with effects 3–4× the noise floor
and a resolution of 5.3."

### Honest correction logged

A partial read at 38/50 complexes showed agreement 0.263 with CI [0.12, 0.49] and
was reported as "significantly worse than chance". **That did not survive the full
data** (0.424, CI [0.27, 0.59], which includes 0.5). Same lesson as the n=5 peek at
+0.80 that became +0.22: do not characterise a direction from a partial CI.

### Jobs

```
sbatch/submit_lnr_cpset_accuracy.sh      the gate (CPU, resumable)
  CPSET_REPLICAS=5 ...                   deeper replication if the SEM is limiting
$PY agent/lnr/cpset_accuracy.py --fetch --data-dir <dir>    provenance of the ground truth
$PY agent/lnr/cpset_accuracy.py --regroup --data-dir <dir>  verify the frozen series table
artefacts: cpset_raw/{dataset1.csv,dataset1/}, cpset_dg.jsonl, cpset_prepared/
```

---

## Deployed: `pll_bg` as the IN-LOOP fitness — and the consequence nobody asked for (2026-11-22)

The metric that was graded (13/13 on the leak-free CPSet head-to-tail pairs,
permutation p = 0.0059; within-complex median ρ +0.34 over 2501 SKEMPI mutations)
is now the loop's third evaluator: `--evaluator pll`, `agent/lnr/pll_inloop.py`.
It is wired, gated and verified end to end on real deposited complexes.

### What it changes in the loop

| | `proxy` | `boltz` | **`pll`** |
|---|---|---|---|
| cost / candidate | ~minutes (Rosetta) | ~2 min (GPU fold) | **29.9 µs** (33.4k seq/s) |
| noise | deterministic | sd ~3.7 REU/draw | **0.000e+00, bitwise** |
| resource | CPU | GPU | **CPU** |
| budget used | 24–40 | 24 | **400** (not binding) |
| ranks like truth? | **no** (measured) | it *is* the truth | yes (graded) |
| optimum | unknown | unknown | **known in closed form** |

Because the score is a table lookup, every noise defence the Boltz loop needs
switches itself off (`stochastic = False` → `n_reassess=0`, `best_sem=0.0`,
`best_recovered` monotone again). The finalist still carries
`recovered_binding_energy: null` — **nats are not REU and must not enter that
column** — and is still re-folded by `boltz_validate --mode mut-held` for the
believed number. That is now a genuine cross-fit: selection and reporting share
neither a measurement nor a model.

### THE FINDING: the objective is separable, so the inner search is solved

`pll_bg` is a **mean of independent per-position terms**. So the best ≤ k
substitution set is not something to search for — it is the k largest
per-position gains, read off the table (`TargetPSSM.best_subs`, asserted against
exhaustive enumeration in `test_pssm_best_subs_is_the_brute_force_optimum`).

On the first 3-target run (job 119392, k_mut=2) the random-EA arm hit the exact
optimum on 3/3, which looked like total saturation. It was not. **Full run, job
119443, 79 targets:**

```
arm            n   meanFit    meanWT  gainVsWT   meanOpt   regret  solved  evals2opt  evals
iter-ea:k2    79   -1.4768   -0.9528   -0.5240   -1.5436   0.0668     46%      118.6    246
```

**The random-EA arm solves 46% (36/79) and leaves real regret on the other 54%.**
So the comparison this module exists for IS runnable. The report flags saturation
automatically when *both* arms reach 0, rather than writing that tie up as a win.

### Longer peptides are harder — and the size confound runs the OTHER way

The n=12 partial suggested a length trend; I flagged it as unestablished (the
next target in, `2cnz` at 19 mutable positions, solved exactly). **At n=79 it
holds**, and the control makes it stronger rather than weaker:

```
                                       rho    perm p      (permutation null;
  length vs regret (raw, nats)       +0.343    0.0015       targets are independent
  length vs regret / available gain  +0.458    0.0002       rows, so a plain
  length vs available gain itself    -0.688    0.0002       shuffle is the null)
  length vs solved (0/1)             -0.474    0.0002
  n_mutable vs regret (raw)          +0.341    0.0017

  bin      n    solved  normRegret  meanGain      normRegret = fraction of the
  0-8     27     19/27       0.077    0.9164      AVAILABLE gain the search
  9-11    21     10/21       0.131    0.5212      failed to capture
  12-14   11      5/11       0.144    0.4086
  15-19   16      2/16       0.253    0.3300
  20+      4       0/4       0.274    0.3025
```

**Why the control matters here.** `pll_bg` is a *mean* over positions, so a
2-substitution edit is diluted by 1/L: available gain falls hard with length
(ρ −0.688). That confound *suppresses* raw regret on long peptides, so the naive
+0.343 understates the effect.

(Ranks are TIE-CORRECTED. Regret has 36 exact 0.0 values out of 79; an ordinal
ranking of that tie block gave ρ +0.316 / +0.427 in a first pass, and the
average-rank values above are the correct ones. `pll_regret.py` is the
reproducible version — `test_spearman_uses_average_ranks_so_ties_do_not_bias_it`
pins it.) Normalised, the EA misses ~8% of the available
gain at L≤8 and ~27% at L≥15 — monotone across all five bins. (This is the same
class of trap as `[[interface-dg-measures-size-not-affinity]]`, caught by running
the size null rather than trusting the raw correlation.)

**Design consequence, not just an observation:** at L≥20 the entire dynamic range
a k_mut=2 search can reach is ~0.30 nats, against 0.92 at L≤8. For long peptides
`k_mut` should scale with length, or `--pll-scope window`/`iface` should narrow
the denominator — otherwise the search is being graded on a range the metric
barely exposes.

### SETTLED: ~51% is the random-EA OPERATOR's ceiling, not a budget or generation limit

The 25-generation run never reached its evaluation cap (0/79 at the 400 cap, max
292, mean 246), so "46% solved" could have been a mere under-run. Job **119637**
reran all 79 at `n_gen=80` / `max_evals=900`, same `n_seed` so the seed pool is
identical and the comparison is paired per target:

```
PAIRED over 79 shared targets                solved   meanRegret  normRegret  meanEvals
  25 gens /  400 cap  (job 119443)          36 = 46%      0.0668       0.146        246
  80 gens /  900 cap  (job 119637)          40 = 51%      0.0548       0.128        360
  moved: 5 better, 0 worse, 74 BIT-IDENTICAL        sign test p = 1/32 = 0.031
  reached the 900 cap: 0/79
```

**3.2× the generations moved 5 of 79 targets and left 74 bit-identical**, with the
evaluation cap never reached in either run. So the limit is the operator: the EA
runs out of *new genomes to propose*, not out of budget — at `n_gen=80` it made
~980 proposals and only 360 were novel (63% duplicates, discarded). This confirms
the synthetic probe that showed 40/120/300 generations ending at identical regret.

Reporting the mean difference here would be the wrong summary ("improved by 0.012
nats"); the honest one is "moved 5 targets", which is why `pll_regret.compare`
does a sign test over the targets that moved and ignores the ties.

**This is the result that makes the LLM arm worth running.** On roughly half the
targets the random operator provably cannot reach a *known* optimum, and more
compute does not fix it — which is exactly the gap an LLM operator is supposed to
close. The comparison now has a real zero, a real baseline, and a reason to exist.

At GEN80 the length effect is slightly stronger and the subset slightly smaller:

```
  length vs regret (raw, nats)       +0.394   p 0.0004      bin      n  solved  normRegret
  length vs regret / available gain  +0.496   p <0.0001      9-11    21   12/21       0.101
  length vs available gain (CONFOUND)-0.688   p <0.0001     12-14    11    5/11       0.112
  length vs solved (0/1)             -0.477   p 0.0001      15-19    16    3/16       0.252
                                                              20+     4    0/4        0.274
  discriminating subset (regret > 0.05): 29/79 = 37%
```

### Other things the full run settled

* **The random-EA operator is weak even on a tiny separable problem.** `2b1n` is
  length 5 with every position mutable — 3,706 genomes — and the EA spent 256
  evaluations without finding the optimum (regret 0.347). The worst case is
  `3ayu` (L=10, regret 0.483 of 0.947 available: it captured only half).
  Length is not the whole story — 4 of the 12 worst targets are L≤8.
* **The action space binds on only 3/79** (`1t5z`, `2hwl`, `6seo`), so the
  `mutable` fix matters but is not load-bearing for the headline.
* **`best_subs` remains the better way to GET a candidate**: the exact optimum
  for one ~1 s table build, no search. The loop's value here is the
  matched-budget *comparison*, not the candidate.

### The discriminating subset, for when the LLM arm is run

32/79 targets (41%) have regret > 0.05 — these are where an LLM operator has room
to show a difference, and running the comparison only there costs ~40% of the
API budget for ~all of the signal:

Re-derived from the GEN80 rows (the ones to use — 29 targets, 37%):

```
3r42,2b1n,3n5u,5y59,4x3h,6fq4,6g86,5a29,1t3l,2hwl,6qg8,4k0u,1t5z,3brh,4z2o,
3vxw,3r7g,4w50,2qos,2wv5,4rs9,6seo,2f31,1ee5,4piq,5frs,4mn8,5onp,5wir
```

Running the LLM arm only here costs ~37% of the API budget for ~all of the
signal; the other 50 targets are already at regret 0 and can only produce a tie.

Regret is logged per generation in the trajectory and per arm in the finalist row
(`iter.regret`, `iter.evals_to_optimum`, `iter.optimum`), so this is re-readable
without recomputation. **It is the first time this project can say how good a
search is**, rather than only which arm scored higher — every prior in-loop
fitness had an unknown optimum.

### A fairness bug the first numbers had: whose optimum?

`1t5z`'s unrestricted optimum moves position 5, which carries no receptor
contact — and `_point_mutate` draws new positions **only** from the target's
interface set, so that genome is unreachable by construction. Regret there was
measuring a prohibition, not a search failure. Two position sets were being
conflated:

* which positions the score **averages over** (`scope` / `positions`), and
* which positions a search is **allowed to change** (`mutable`).

`position_gains` / `best_subs` / `optimum` now take them separately, `evolve`
passes its own `iface` set as `mutable`, and the unrestricted optimum is still
reported as `free_subs` / `free_fitness` / `free_minus_constrained` — so a
regret of 0 is never confused with "the metric's own best edit". Job 119413's
rows were archived as `*.UNCONSTRAINEDOPT.*` rather than deleted; 119443 is the
rerun under the corrected grading.

### Two bugs this found in the metric as it stood

1. **`TargetPSSM.score()` computed `pll_cplx`, not `pll_bg`.** The background
   subtraction is a per-column offset, so it is *not* a monotone relabelling: it
   reorders the per-position argmax and reorders two sequences of equal length
   whenever they differ in composition. On a random 12×21 table the two metrics
   pick **different optimal sequences**. Deploying `.score()` as written would
   have scored the graded metric's name with an ungraded metric's numbers.
   `score` / `best_sequence` / `top_k_at` / `best_subs` now all carry `metric`
   explicitly and default to `pll_bg`.
2. **Nothing checked that the loop and the table index the peptide the same
   way.** `env.LNRTarget` silently drops residues missing a backbone atom;
   LigandMPNN's `parse_PDB` does not. At equal length but one residue apart,
   every lookup reads the wrong row and **nothing raises**. `TargetPSSM.matches`
   now requires the two template sequences to be the *same string*, and the
   evaluator refuses the target otherwise. It fired on real data immediately:
   one LNR peptide is 18 residues to its own template's 19.

### The gate, and why it has two kinds of failure

`pll_inloop.py --selftest` runs before any search and separates:

* **CODE INVARIANTS** (fatal, per target): the fitness sign (the metric's optimum
  must be the population's minimum), bitwise determinism across independently
  built tables, the disk round-trip being exact, and a variant being exactly a
  table lookup.
* **DATA OBSERVATIONS** (reported, skipped, gated only in aggregate): whether a
  deposited peptide can be indexed consistently, and whether a native beats its
  own scrambles on its own pose. Over 93 real complexes some targets fail both;
  one odd structure must not abort the run. The aggregate gates are
  `--min-comp-frac 0.6` (below that the metric is reading composition
  fleet-wide) and `--max-unscorable-frac 0.25` (above that the pose set is
  wrong, not one structure).

Conflating them would have let a single structure abort a 93-target run — and,
worse, would have made the real invariants tempting to switch off.

### Open: the scope question is recorded, not decided

The graded form averages over **all** peptide positions. The LNR objective keeps
only the held window's interface (residues outside are released so the ring can
close), so `--pll-scope window` is arguably more faithful — but it was never
graded. Every row records all three scopes (`pll_all`, `pll_window`,
`pll_iface`) at a cost of microseconds, so the comparison is answerable from the
rows with no rerun. Default stays `all`.

### Jobs

```
sbatch/submit_lnr_pll_regret.sh                         the regret analysis (CPU, rows only)
  PLLREGRET_ROWS=...GEN80.jsonl PLLREGRET_MAX_EVALS=900 ...
$PY agent/lnr/pll_regret.py --rows <A> --compare <B> \
    --label A --compare-label B                         paired A-vs-B (B is the one judged)
sbatch/submit_lnr_seqmut_iter_pll.sh                    the loop (CPU, resumable)
  SEQITER_PLL_TARGETS=1bjr,1elr ...                     subset
  SEQITER_PLL_ARMS=iter-ea,iter-llm ...                 adds the LLM arm (spends API $)
  SEQITER_PLL_SCOPE=window ...                          the ungraded, more faithful scope
$PY agent/lnr/pll_inloop.py --selftest                  the gate alone
artefacts: seqmut_iter_pll_rows.jsonl, seqmut_iter_pll_traj.jsonl,
           seqmut_iter_pll_scores.jsonl (every sequence ever scored), pssm/*.json
jobs: 119392 (3-target validation, PASSED), 119413 (superseded, archived as
      *.UNCONSTRAINEDOPT.*), 119443 (79 targets, 25 gens), 119637 (79 targets,
      80 gens -> the operator-ceiling result), 119665 (regret analysis)
```

### RESULT: the LLM arm ran, and does NOT separate from random EA

Job 119773 (29 discriminating targets, both arms, `n_gen=80`, $3.00 cap) plus
job 120444 (the eval-matched EA control). **27 targets carry a funded LLM arm**;
`6qg8` and `6seo` do not and are excluded everywhere below — see "the credits
ran out" following.

| | mean evals | mean regret | exact optimum |
|---|---|---|---|
| EA, generation-matched (`n_gen=80`) | 342.7 | 0.1435 | 0/27 |
| EA, eval-matched (`n_gen=400`, cap 520) | 385.6 | 0.1415 | 1/27 |
| **LLM** (`n_gen=80`) | **516.0** | **0.1361** | **5/27** |

Paired, LLM vs the eval-matched EA: mean difference **−0.0054**, 15 better /
10 worse / 2 tied, sign test **p = 0.21**. On the more sensitive
"reached the exact optimum" endpoint the discordant pairs are 5 (LLM only) vs 1
(EA only, `5onp`), exact McNemar **p = 0.11**.

**Neither endpoint separates the operators at n=27.** Every comparison points
the same way — the LLM has lower regret, finds 5 exact optima to the EA's 1, and
explores more — but none of it reaches significance. The direction is consistent
and the evidence is weak; both statements are needed.

#### The arms CANNOT be budget-matched, and that is itself the finding

The LLM arm used more evaluations on **27/27** targets (516 vs 343, **+51%**),
so the generation-matched comparison flatters it. The control was meant to fix
that by giving the EA an eval budget instead of a generation budget — and it
could not: **5× the generations (80 → 400) moved the EA from 343 to 386 evals,
+13%**, never approaching the LLM's 516 (whose per-target *minimum*, 406, is
above the EA's mean).

The EA cannot spend the budget because `_point_mutate` keeps regenerating
genomes already in the population; it saturates at ~386 distinct proposals. So
"matched budget" is unreachable with this operator, and the operator ceiling
measured earlier is really a **proposal-diversity** ceiling, not a budget or
generation one. The LLM's one unambiguous advantage is that it proposes more
distinct genomes — which is the thing the EA cannot be tuned into doing.

#### Why n=27 is the ceiling on power here, and what would settle it

Adding the other 50 targets will **not** help: the discriminating subset was
chosen because the rest already sit at regret 0, where the LLM cannot do better
than tie. The movers are capped at ~27, which is why p stalls near 0.1–0.2.

What would separate them, in order of cost:

1. **Replicate the LLM arm with several seeds per target.** The pll evaluator is
   deterministic and the EA is seeded, so the LLM is the only stochastic
   component; k seeds per target turn one noisy draw per target into a mean and
   buy power without needing new targets. ~$1.66 per replicate.
2. **Give the EA a diversity-aware operator** (reject known genomes before
   scoring, scale `k_mut` with length). Free, and it is the honest control: the
   present EA loses partly by failing to use its budget.
3. **A harder target set** with more nonzero-regret targets.

#### The credits ran out mid-run — and the failure mode is the dangerous one

The last three targets processed (alphabetically) degraded: `6g86` took 12
errors of 80, then `6qg8` and `6seo` errored on **all 80 calls at $0.00**. A
direct probe returns `http_429: You have no credits remaining.` This is the
OpenAI account's billing limit, **not** this project's `--max-spend-usd` ledger
(which stopped at $1.66 of $3.00).

It matters because `llm_offspring` falls back to a **random-EA child** on error.
An unfunded LLM arm therefore writes rows, gets graded, and converges on its own
EA control — which is indistinguishable from the real "separable, saturated"
finding. `pll_llm_validity.py` exists for exactly this and flagged it
(`VALID`/`PARTIAL`/`INVALID_TAIL`); it is wired into `s4_lnr_pll_regret.sbatch`
as a gate, prints the API's own error text, and emits a ready-to-paste
`--exclude` list. **Add credits before any further LLM arm.**

Measured cost, for planning: **$0.0615/target** at `n_gen=80` (152 calls/target,
~$0.0004/call) — so 29 targets is ~$1.78 and all 79 would be ~$4.86.

#### Also settled: the scope question is NOT low-stakes

See the scope section above for the matrix. `all` and `iface` are effectively
one objective (3–6% divergence); **`window` is a different one (35–42%)**, and
the three scopes agree on the optimum for only 22/85 targets. Everything here
is graded under `all`, while `window` is what the LNR objective asks for, so
**these results do not transfer to `window`.** Deciding that is a judgement
about the objective; the measurement only proves the judgement matters.

### HANDOFF — state at 2026-11-23, picking this up in a new session

**Status: the `pll` evaluator is finished and verified, and the LLM arm HAS NOW
RUN** (jobs 119773 + the eval-matched control 120444). Committed and pushed on
branch `lnr/pll-inloop-evaluator`. 246 tests pass
(`PYTHONPATH=agent:pilot/stage0 $PY -m pytest agent/lnr/ -q` — note `$PY` is
`/zfsauton2/home/yixiz/Proteina-Complexa/.venv/bin/python`; the default `python`
on `lov4` has no pytest).

**Read "RESULT: the LLM arm ran" immediately above this section first.** In
short: the LLM arm does not separate from random EA (p = 0.21 on regret, 0.11 on
exact optima, n = 27), it cannot be budget-matched because the EA's operator
saturates, and **the OpenAI account has no credits left**, which cost the last
two targets and blocks any rerun.

#### What is done and believed

| | |
|---|---|
| `pll_bg` wired as in-loop fitness | `--evaluator pll`, verified end to end on real complexes |
| selftest gate | PASSES on 85/87 targets (determinism `0.000e+00`, sign, cache round-trip, lookup); composition holds fleet-wide at 81/85 = 95%, median +1.79 nats/res |
| throughput | 29.9 µs/seq, ~33k seq/s, CPU only |
| random-EA baseline | **51% of 79 targets reach the known optimum; ~13% of available gain left uncaptured. This is the OPERATOR's ceiling** (3.2× generations moved 5/79; neither run reached its eval cap) |
| length effect | ρ(length, normalised regret) **+0.496**, perm p < 0.0001, with the size confound ρ(length, available gain) = −0.688 reported beside it |

#### What is NOT done

1. ~~The LLM arm was never submitted.~~ **Submitted 2026-11-23 as job 119773**
   (29-target subset, both arms, `n_gen=80`, cap **$3.00**, 10 h). See
   "The $1.00 cap was structurally too small" below — the first attempt (119761)
   was cancelled for that reason, not because it failed.
2. ~~Nothing is committed.~~ **Committed on branch `lnr/pll-inloop-evaluator`**,
   pushed to `origin`. Four commits: the binding-metric layer, the pll in-loop
   layer, target exclusion, and the LLM funding gate.
3. **The scope question is recorded but not decided** — `all` (graded) vs
   `window`/`iface` (more faithful to the LNR objective, ungraded). All three are
   in every row, so deciding it needs no rerun. **Still open, and it is a
   judgement call about what the objective should be, not a computation.**
4. **`k_mut` does not scale with length**, which the 1/L dilution says it should.

#### To run the LLM arm (the obvious next step)

The data says exactly where to spend: the **29-target discriminating subset**
(regret > 0.05 after the generation increase). The other 50 are at regret 0 and
can only produce a tie, so restricting costs ~37% of the budget for ~all signal.

```bash
cd /zfsauton2/home/yixiz/lp2cp-agent      # from lov4; sbatch works directly
D=/zfsauton/scratch/yixiz/lp2cp_stage2/lnr
SEQITER_PLL_ARMS=iter-ea,iter-llm \
SEQITER_PLL_TARGETS=3r42,2b1n,3n5u,5y59,4x3h,6fq4,6g86,5a29,1t3l,2hwl,6qg8,4k0u,1t5z,3brh,4z2o,3vxw,3r7g,4w50,2qos,2wv5,4rs9,6seo,2f31,1ee5,4piq,5frs,4mn8,5onp,5wir \
SEQITER_PLL_N_GEN=80 SEQITER_PLL_M_OFF=12 SEQITER_PLL_MAX_EVALS=900 \
SEQITER_PLL_SEED_WINDOWS=$D/seqmut_iter_pll_rows.jsonl \
SEQITER_PLL_ROWS_FILE=seqmut_iter_pll_rows.LLM.jsonl \
SEQITER_PLL_TRAJ_FILE=seqmut_iter_pll_traj.LLM.jsonl \
SEQITER_PLL_MAX_SPEND=3.0 \
SEQITER_PLL_TIME=10:00:00 \
  sbatch/submit_lnr_seqmut_iter_pll.sh
```

##### The $1.00 cap in the original version of this command was too small

Not a judgement call — arithmetic. **The cap is per-RUN, not per-target:** the
client, and therefore the `SpendLedger`, is built once in `seqmut_iter.main()`
before the target loop and reused for every target. `llm_offspring` makes **one
call per generation** (plus up to one retry when proposals repeat), so a
29-target run at `n_gen=80` needs ~2,300–2,800 calls. At the client's own
2600-in/700-out estimate, `gpt-6-luna` is **$0.00061/call**, so the run projects
to **$1.42–$2.12** and $1.00 funds only **14–21 of the 29 targets**.

Why that is worse than just stopping early: on `BUDGET_CAP`, `llm_offspring`
**falls back to a random-EA child**. So the tail of the run is the EA arm
wearing an LLM label — graded, averaged in, and *converging on the control*.
The report would read "no difference", which is exactly the shape of the real
conclusion ("separable, saturated, not won"). And because the cap is spent in
target order, the contamination is the whole tail, not a random subset, so
pairing cannot repair it.

`SEQITER_PLL_TIME` exists because the LLM arm's wall clock is set by the API,
not the CPU: **~5 min/target measured** (119761), which does not fit the
script's 4 h `#SBATCH --time`. It is passed on the sbatch command line, which a
`#SBATCH` directive cannot read. This is *not* the `--export` hazard — that one
is specific to export lists, not resource flags.

A cap is a **ceiling, not a spend**: raising $1 → $3 does not change the
projected ~$1.70 actual cost, it only stops the run being truncated.

Then grade it (CPU, rows only — the LLM-vs-EA control prints automatically):

```bash
PLLREGRET_ROWS=$D/seqmut_iter_pll_rows.LLM.jsonl PLLREGRET_MAX_EVALS=900 \
PLLREGRET_OUT=$D/pll_regret_llm.json sbatch/submit_lnr_pll_regret.sh
```

**Read these before believing the output:**

* The job **gates on `OPENAI_API_KEY` and exits 0 if absent** — a missing key
  looks like a clean run that did nothing. Check the log says
  `iter-llm arm: backend=... cap=$...`, not `GATE: no OPENAI_API_KEY`.
* `SEQITER_PLL_SEED_WINDOWS` above reuses the 79 committed windows, which is what
  makes the run minutes instead of ~an hour. Without it every window is
  recomputed (~35–50 s/target) and the EA arm's seed pool stays identical anyway.
* **Both arms must be at `n_gen=80`** or the comparison is not matched — the EA
  baseline to compare against is `seqmut_iter_pll_rows.GEN80.jsonl`, not the
  25-generation file.
* `--max-spend-usd` is enforced by the ledger before each call, but it is a
  budget for the **whole run**, not per target or per arm. See above.
* If the LLM arm lands at regret 0 everywhere, the report says so itself
  ("separable … saturated, not won") — that is not a win, it is the objective
  being additive. **Check the funding verdict before accepting that reading**:
  an unfunded arm produces the identical output.

**The funding check is now automatic.** `s4_lnr_pll_regret.sbatch` runs
`pll_llm_validity.py` first and refuses to print an LLM-vs-EA comparison from a
run whose cap was reached or whose targets never called the model, exiting 0 so
no dependent strands. Pass the cap so it can say whether the budget was the
binding constraint:

```bash
PLLREGRET_ROWS=$D/seqmut_iter_pll_rows.LLM.jsonl PLLREGRET_MAX_EVALS=900 \
PLLREGRET_LLM_CAP=3.0 PLLREGRET_N_EXPECTED=29 \
PLLREGRET_OUT=$D/pll_regret_llm.json sbatch/submit_lnr_pll_regret.sh
```

Verdicts: `INVALID_TAIL` (budget cap or zero calls — absorbing and ordered,
refuses to grade), `PARTIAL` (declines/malformed over half the generations —
spread across targets, grade both with and without them), `VALID`. `ok_none` is
**not** a fallback: "I choose not to mutate" is a real answer from the model.
`PLLREGRET_FORCE=1` grades a rejected run for inspection only.

**A free matched-arm check, worth running once.** This run re-runs `iter-ea` on
the same 29 targets, and the EA arm is deterministic with the seed windows
reused — so it must reproduce `GEN80` bit-identically. Verified on `1ee5`:
regret 0.064, 282 evals, fitness −0.9521649118517178, subs `[[16,R],[18,H]]`,
all identical. If this ever disagrees, the two arms did not see the same
problem and the comparison is void:

```bash
$PY agent/lnr/pll_regret.py --rows $D/seqmut_iter_pll_rows.LLM.jsonl \
    --compare $D/seqmut_iter_pll_rows.GEN80.jsonl --arm iter-ea \
    --label LLMrun-EAarm --compare-label GEN80-EAarm
```

**`5a29` is not a binder.** Its peptide is `HHHHHHHSSGLVPRGSHA` — a His7 tag
plus the thrombin cleavage site `LVPRGS`, i.e. a crystallisation construct.
There is no binding signal for an operator to recover, so its regret is not
evidence about the operator. It is 1 of the 29 graded targets (~3.4% of the
weight) and sits at L=18, in the longest bin where the length effect is
measured. Audited across all 85 selftest-indexable targets against
His/thrombin/TEV/FLAG/myc/Strep/GS-linker patterns — it is the **only** one.
Drop it with `PLLREGRET_EXCLUDE_ARTIFACTS=1` (the reason travels with the id, in
`pll_regret.ARTIFACT_TARGETS`) and report headline numbers both ways.

#### Files (committed on branch `lnr/pll-inloop-evaluator`, pushed)

235 tests pass. Commits, oldest first: `acad786` binding-metric layer,
`701e89b` pll in-loop layer, `4b2fc21` target exclusion + the 5a29 audit,
`69b7080` the LLM funding gate.

```
new   agent/lnr/pll_inloop.py        the PLLScorer + --selftest gate
      agent/lnr/pll_regret.py        regret analysis + paired comparison,
                                     --exclude / --exclude-artifacts
      agent/lnr/pll_llm_validity.py  the iter-llm funding gate (see above)
      agent/lnr/test_pll_inloop.py   27 tests
      agent/lnr/test_pll_regret.py   22 tests
      agent/lnr/test_pll_llm_validity.py  10 tests
      sbatch/s4_lnr_seqmut_iter_pll.sbatch  + submit_lnr_seqmut_iter_pll.sh
      sbatch/s4_lnr_pll_regret.sbatch       + submit_lnr_pll_regret.sh
edit  agent/lnr/bindscore.py         TargetPSSM: metric-aware (DEFAULT pll_bg),
                                     matches(), position_gains(), best_subs(),
                                     mutable-vs-averaged position sets
      agent/lnr/seqmut_iter.py       OBJECTIVES registry, scorer_objective/_name,
                                     _regret, --evaluator pll + flags,
                                     prepare() skip, _summarize_pll
      agent/lnr/seqmut.py            MUT_SYSTEM_PLL
      sbatch/stage2_preamble.sh      SEQITER_PLL_* defaults
```

#### Three traps already paid for — do not re-introduce

1. **`TargetPSSM.score()` must carry `metric` explicitly.** `pll_bg` and
   `pll_cplx` pick *different* optimal sequences; the default is `pll_bg` because
   that is the graded one.
2. **Never write the pll fitness into `recovered_binding_energy`.** It is
   nats/residue; that column is REU. It stays in `inloop_pll` / `inloop_fitness`,
   and the REU number comes from re-folding the finalist.
3. **Regret must be graded inside the operator's action space** (`mutable`), or
   targets like `1t5z` are scored against an optimum `_point_mutate` cannot
   reach. Ranks in `pll_regret` are **tie-corrected** — 36/79 regrets are exactly
   0.0 and ordinal ranks biased ρ by ~0.03.
