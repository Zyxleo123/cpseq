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
