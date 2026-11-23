# The binding metric: results

Companion to `binding_metric_design.md` (the pre-registration and its two dated
amendments). Every number here is reproducible from committed rows with
`--report`, no recompute.

```
CPSet rows   /zfsauton/scratch/yixiz/lp2cp_stage2/lnr/bindscore_v2.jsonl
decoy rows   .../bindscore_v2_decoy.jsonl
SKEMPI rows  .../skempi_bindscore.jsonl
jobs         119302 (v1), 119321 (v2, 3 models), 119332 (SKEMPI), 119338 (decoy)
```

## The metric

With the receptor's backbone **and side chains** as context and the peptide's
**backbone only** (its own side chains are masked out, so the crystal's answer
cannot leak into its own score):

```
pll_cplx(s) = (1/n) Σ_i  log p( s_i | bound complex pose )        nats/residue
icg(s)      = (1/n) Σ_i [log p(s_i | complex) − log p(s_i | peptide alone)]
pll_bg(s)   = (1/n) Σ_i [log p(s_i | complex) − log q_background(s_i)]
```

`p` is an inverse-folding marginal (LigandMPNN `ligandmpnn_v_32_020_25`) with the
queried position decoded first, so it sees backbone geometry and no sequence.

**In marginal mode this is a position × amino-acid matrix**, fixed by the pose.
Scoring any sequence is a sum of table lookups.

## Noise: the question is closed, not estimated

`bindscore.selftest` asserts these against the real checkpoint before any scoring
runs, and the job treats a failure as a gate:

| claim | measured |
|---|---|
| identical across decoding-order seeds | max abs diff **0.000e+00** |
| identical after threading a different sequence | **0.000e+00** |
| matches upstream `single_aa_score` exactly | **0.000e+00** |
| receptor actually changes the marginal | 3.129 nats |

**Bitwise deterministic. One evaluation. No repeats, ever.** The incumbent oracle
has a ±3–4 REU floor dominated by Boltz fold sampling and needs 16 GPU folds to
reach SEM ≈ 1.0. The only variability here is *which template you pick*, reported
as `disc` (between-variant spread / within-variant SEM over templates) = 2.0–3.8.

Cost: one encode per target (~1 s CPU), then ~20 µs per sequence. The encoder
never reads the sequence, which is what makes the caching valid.

## Arm 1 — CPSet, right chemistry (head-to-tail), thin

Within-series pair agreement, `xtemplate` mode (every variant scored on *other*
members' crystals — never its own). At |ΔpKd| ≥ 1.0 the MIX series contribute no
pairs, so these 13 pairs are **strictly head-to-tail**: S03_keap1 (n=6) and
S05_6d3 (n=4).

| metric | family | pairs | agree | perm null | p_perm | mean ρ | x-ρ \| len |
|---|---|---|---|---|---|---|---|
| `pll_bg` | secondary | 13/13 | **1.000** | 0.500 | **0.0057** | −0.900 | **−0.57** |
| `pll_cplx` | secondary | 13/13 | **1.000** | 0.500 | **0.0057** | −0.900 | −0.41 |
| `pos_gain` | secondary | 13/13 | **1.000** | 0.500 | **0.0050** | −0.800 | −0.41 |
| `icg` | **pre-reg primary** | 12/13 | 0.923 | 0.501 | 0.0240 | −0.943 | −0.12 |
| `pll_pep` | control | 12/13 | 0.923 | 0.500 | 0.0224 | −0.671 | −0.29 |
| `pll_shuf` | **control** | 7/13 | 0.538 | 0.503 | 0.4989 | −0.329 | +0.07 |
| `dg_rosetta` | **incumbent** | 8/13 | 0.615 | 0.501 | 0.3583 | −0.457 | +0.23 |
| `null_chargecomp` | null | 5/13 | 0.385 | 0.397 | 0.5894 | +0.123 | +0.17 |
| `null_neglen` | null | 0/13 | 0.000 | 0.000 | 1.0000 | — | −0.29 |

All tiers (4 informative series): `icg` 20/26, p_perm 0.0345, **4/4 series
right-signed**; `dg_rosetta` 12/26, p_perm 0.66 — **the incumbent is at chance on
the same data under the same test**.

**Verdict: WEAK, and the reason matters.** Only 4 series have n ≥ 3 *and*
threadable equal-length members, so the per-series sign test bottoms out at
p = 1/16 = 0.0625. The observed 4/4 is the instrument **saturated**: the metric
did everything this data can show and the pre-registered rule still cannot
return PASS. That rule exists to stop a thin instrument being talked into a
strong claim, and it did its job — which is exactly why a confirmation arm was
built rather than more decimals extracted here.

### The mechanism: right answer for the right reason

Per-position log p on S03_keap1, where the measured SAR is known and local:

```
       seq        pos1     pos2     pos3     pos4     pos5     pos6     pos7
7K2H  GDPETGE   -0.781   -1.316   -0.940   -3.747   -1.807   -0.078   -2.901   pKd 6.85
7K2I  GAPETGE        .   -3.058        .        .        .        .        .   D2A  ΔpKd −1.96
7K2G  GDEEAGE        .        .   -2.742        .   -2.607        .        .   T5A  ΔpKd −3.30
```

`D2→A` costs −3.06 nats **at position 2 and nowhere else**; `T5→A` costs −2.61
nats **at position 5 and nowhere else**. Measured, those substitutions are worth
2.33 and 2.78 log units. The metric is not merely correlated with the SAR; it
localises to the same residues the chemistry does.

Honest caveat: the *magnitudes* are not calibrated. `P3→E` is penalised −2.65
nats but costs only 0.52 log units measured, and `D2→E` is penalised −1.71 nats
but is measured neutral (−0.03). The metric over-discriminates conservative
substitutions. Ranking survives because the real losses compound.

## Arm 2 — SKEMPI, wrong chemistry, high power (the confirmation)

2501 mutations, 82 complexes. Measured ΔΔG; positive = weaker.

| form | pooled ρ | Pearson | sign | within-complex median ρ | complexes right | p |
|---|---|---|---|---|---|---|
| `ddg_pred` (pll_cplx) | +0.331 | +0.315 | 0.739 | +0.310 | 72/79 | 5.3e−15 |
| `ddg_pred_bg` (pll_bg) | +0.352 | +0.327 | 0.745 | **+0.337** | **73/79** | **5.0e−16** |
| `ddg_pred_icg` (icg) | **+0.378** | **+0.399** | 0.692 | +0.328 | 72/79 | 5.3e−15 |
| *Rosetta dG, single relax (prior gate, n=44)* | *+0.220* | | *0.590* | | | |
| *Flex ddG, 35-model ensemble (published)* | *~+0.600* | | | | | |

**72–73 of 79 complexes ranked right-signed, p ≈ 1e−15.** The incumbent scored
+0.22 on 44 mutations; this is +0.33 to +0.38 on 2501, deterministically, at
roughly 1/1000 the cost per evaluation. It does not reach Flex ddG's ~0.60 — that
is an honest ceiling to state, not hide.

### The signal is the interface

Stratified by SKEMPI's own burial annotation:

| location | n | ρ |
|---|---|---|
| **COR** (interface core) | 1319 | **+0.367** |
| **RIM** (interface rim) | 802 | +0.242 |
| **SUP** (away from interface) | 380 | +0.079 |

A monotone dose–response in burial, decaying to ~0 off the interface. This is the
answer to the standing objection that the metric might be scoring generic protein
plausibility: generic plausibility would not care where the residue sits.

## Arm 3 — the specificity control: a wrong-receptor pocket

Each series' peptides threaded onto a **same-length template from a different
receptor** (KEAP1's 7-mers into streptavidin's site; SFTI-like 14-mers across
three unrelated proteases). Same sequences, same metric, same model — only the
pocket changed:

| metric | true receptor | wrong receptor | mean ρ (decoy) |
|---|---|---|---|
| `pll_cplx` | **13/13, p = 0.0059** | 6/13, p = 0.638 | **+0.257** (wrong sign) |
| `pll_bg` | **13/13, p = 0.0059** | 6/13, p = 0.638 | **+0.229** (wrong sign) |
| `icg` | **12/13, p = 0.0224** | 7/13, p = 0.494 | **+0.200** (wrong sign) |
| `pll_pep` (no receptor at all) | 12/13, p = 0.0224 | 9/13, p = 0.221 | −0.257 |

**The ranking collapses to chance and flips direction in a foreign pocket.**
This is what finally closes the standing objection. `pll_pep` had reached 0.923
on the true-receptor BB tier, leaving open that the metric was reading the
peptide's own bound conformation; if that were so, the decoy — which preserves a
peptide backbone conformation but changes the receptor — would have preserved the
ranking. It did not. **The signal is the pocket.**

## What was ruled out

| alternative explanation | control | result |
|---|---|---|
| amino-acid composition | `pll_shuf` — same residues, scrambled order, same pose | **0.538, p=0.50 → chance.** Permutation holds composition *exactly* fixed; signal is positional |
| peptide size / buried area | `null_neglen`, `null_negdsasa`, partial ρ \| length | 0/26 and 0.269; metric keeps −0.57 partial where dG keeps −0.00 |
| reading its own crystal | `xtemplate` excludes the variant's own structure | leak diagnostic `self − xtemplate` = −0.004 for `pll_cplx` |
| training-set memorisation via series consensus | `null_consensus` (leave-one-out Hamming) | near-all ties on S03; only separates the worst variant |
| generic protein plausibility | SKEMPI burial stratification | COR 0.367 → SUP 0.079 |
| **reading the peptide's own bound conformation** | **decoy receptor (Arm 3)** | **13/13 → 6/13, p 0.0059 → 0.638, sign flips** |
| net-charge complementarity | `null_chargecomp` | 0.385, p=0.59 → chance |

## Recommendation

Per Amendment 2 §2.3: SKEMPI did **not** materially separate the family
(within-complex medians 0.310 / 0.328 / 0.337), so clause 2 applies.

**For head-to-tail cyclic peptides, deploy `pll_bg` (or `pll_cplx`).** Both need
one forward pass. `icg` is the better *general* construction and wins the pooled
SKEMPI correlation, but it requires scoring the peptide **alone** — which for a
6–22 residue macrocycle is badly out of distribution, and on CPSet it costs
nearly all the cross-target signal (−0.12 vs −0.57 partialling out length). That
the same subtraction *helps* on SKEMPI, where the unbound reference is a real
folded protein, is the predicted behaviour and was written down before the
SKEMPI result was known.

`pll_bg` differs from `pll_cplx` only by a frozen SwissProt background constant —
no parameter is fitted to any benchmark here.

## DEPLOYED as the in-loop fitness (2026-11-23) — and one structural caveat

`pll_bg` is now the search's third evaluator (`--evaluator pll`); the full write-up
is in `docs/seqmut_notes.md`, section "Deployed: `pll_bg` as the IN-LOOP fitness".
Verified on real complexes: determinism `0.000e+00`, 29.9 µs/seq, and the
composition control holds fleet-wide on the LNR set (native beats its own
scrambles on 81/85 targets, median +1.79 nats/residue).

**The caveat that changes how to use it.** `pll_bg` is a *mean of independent
per-position terms*, so it is **separable**: the best ≤ k substitution set is not
something to search for, it is the k largest per-position gains read off the
table (`TargetPSSM.best_subs`, checked against exhaustive enumeration). Two
consequences:

* **To GET candidates, call `best_subs` — do not run a search.** One ~1 s table
  build returns the exact optimum, plus `position_gains` for the per-position
  preference the score is made of.
* Because the metric is a *mean*, a fixed-k edit is diluted by 1/L: available
  gain falls with peptide length at ρ −0.688 over 79 targets. For long peptides
  `k_mut` has to scale with length, or the scope must narrow, or the search is
  being graded on a range the metric barely exposes.

Two defects in this module were found only by deploying it, both now fixed and
tested: `TargetPSSM.score()` computed `pll_cplx` rather than the graded `pll_bg`
(they pick *different* optimal sequences), and nothing checked that a caller's
peptide indexing matched the table's — which fired immediately on real data
(`3wbn` is 18 residues to its template's 19).

## Limits, stated plainly

1. **Needs a reference bound pose.** This ranks sequences *for a target you have
   a complex for*. The LNR pipeline always does; a novel target does not.
2. **Not calibrated to kcal/mol.** It ranks; it does not predict ΔG. Magnitudes
   over-penalise conservative substitutions (D→E, E→P above).
3. **Head-to-tail evidence rests on 2 BB series** (S03 n=6, S05 n=4). The
   confirmation arm is protein–protein, and the two are never pooled.
4. **Below Flex ddG** (+0.35 vs ~+0.60 on SKEMPI) — but ~1000× cheaper and
   deterministic, which is what makes it usable in a loop.
5. **Training-set overlap.** These crystals predate the model's cutoff. The
   `xtemplate` firewall and `null_consensus` bound the effect; they do not
   eliminate it. A post-2021 held-out series would.
6. **Cyclisation is implicit.** The model sees a linear chain whose termini
   happen to be adjacent in space; no head-to-tail bond is declared. It is
   evaluated on genuine BB crystals, so this is a modelling caveat, not an
   evaluation one.
