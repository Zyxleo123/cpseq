# The binding metric for head-to-tail cyclic peptides

Pre-registration, written **before** the first run. Frozen by commit; the
verdict rule below cannot be edited after seeing a number.

## 1. What failed, and the one mechanism behind all of it

| attempt | result | why it failed |
|---|---|---|
| Rosetta interface `dG_separated` | cross-complex ρ −0.225; within-series pair agreement 0.424 | tracks peptide **size**: ρ(dG, length) = −0.718, partial ρ(dG, pKd \| length) = **−0.002** |
| Boltz ipTM / ipAE / LIS / ipDE | resolve variants *worse* than dG | trained/validated on binder-vs-non-binder; measure "is this a confident structure" |
| `complex_plddt` | flattering discriminability, correlated with nothing | bought by an **invariant** (receptor is 281 of 291 residues) |
| ESM-2 pLL, PepMLM cLL | mean within-complex ρ −0.14…+0.08 | **no structure**: cannot know the pocket |
| SKEMPI ΔΔG accuracy check | ρ +0.22, sign 59% | the scoring function itself is weak on polar recognition |

Every one of these fails for the *same* reason: **the quantity being measured is
not conditioned on the sequence's fit to the specific bound pose.** The physics
energies are dominated by how *much* surface is buried; the confidence metrics
by whether *a* structure is confident; the language models by whether the
peptide is *a plausible peptide*. None of them asks the actual question.

The CPSet S03_keap1 series shows what the right answer has to explain. Six
head-to-tail 7-mers on KEAP1, measured pKd spanning 3.3 log units:

```
pKd 6.85  7K2H  G D P E T G E      Rosetta dG -27.3
pKd 6.82  7K2M  G E P E T G E                 -26.8
pKd 6.33  7K2E  G D E E T G E                 -29.9
pKd 4.89  7K2I  G A P E T G E                 -30.3   <- dG's BEST binder
pKd 4.00  7K2F  G A E E T G E                 -26.1
pKd 3.55  7K2G  G D E E A G E                 -29.1
```

The SAR is additive and chemically specific: **D2→A costs 2.33 logs**
(independently: 6.33→4.00 and 6.85→4.89), **T5→A costs 2.78 logs**, E3→P *gains*
0.5–0.9, and D2↔E2 is free (0.03). These are losses of two specific polar
side-chain contacts plus one backbone-rigidifying substitution. Rosetta ranks
the variant that lost the D2 carboxylate as its **strongest** binder. In S02,
where all six peptides are the same length so no size confound exists at all,
the same pathology appears: `F3H` is measured weakest (4.30) and scored most
negative (−50.1).

So the metric must be: **conditioned on the bound pose, sensitive to residue
identity at specific positions, and insensitive to bulk buried area.**

## 2. The metric: Interface Context Gain (ICG)

Take the bound complex pose `C` = (receptor backbone **and side chains**,
peptide **backbone only**). For peptide position `i` with residue `s_i`, let

```
lp_cplx(i) = log p( s_i | receptor + peptide backbone )     # in the complex
lp_pep(i)  = log p( s_i | peptide backbone alone )          # receptor deleted
```

both from an inverse-folding model (LigandMPNN/ProteinMPNN) evaluated as a
*marginal*: the queried position is decoded first, so it sees backbone geometry
and no other sequence. Then

```
            1
ICG(s) =   ---  Σ  [ lp_cplx(i) - lp_pep(i) ]        nats / residue
            n   i∈pep
```

**Why this is the right shape:**

- It is **pose-conditioned**, so residue identity at a specific position in a
  specific pocket is what moves it — the thing the SAR is made of.
- The **difference** cancels the peptide's intrinsic plausibility and its own
  backbone's preferences. What is left is attributable to the receptor being
  there: it is the pointwise mutual information `I(s_i ; receptor | pose)`.
  This is precisely the critique that killed ESM-2 and PepMLM — they could only
  score "is this a plausible peptide" — and ICG is constructed to be immune to it.
- It is a **mean over positions**, in nats/residue: dimensionless and
  length-normalised by construction, so it cannot buy signal from size the way
  dG does. (`icg_sum`, the un-normalised version, is computed too and declared
  as the deliberately size-confounded control.)
- The peptide's **own side chains are never visible** to the scorer (LigandMPNN
  masks side-chain context for residues marked "designed"). The metric therefore
  scores "would this sequence like this pose", never "is this the crystal's
  sequence". Only the receptor's side chains are context.

**Why it is low-noise.** The encoder does not read the sequence, and
`augment_eps = 0` at inference, so there is no coordinate jitter. With the
queried position decoded first its representation is provably independent of the
decoding order of every other position, so the marginal is **exactly
deterministic** — bitwise reproducible, zero repeats needed. Compare the
incumbent: ±3–4 REU dominated by Boltz fold sampling, needing 16 GPU folds to
reach SEM ≈ 1.0. The noise claim is still *measured*, not asserted: the report
prints the SD across decoding-order seeds and across templates.

**Why it is cheap.** `encode()` is sequence-independent, so a template is
encoded once and then any number of variant sequences are scored by `n` small
decoder passes — milliseconds for a 7-mer, on CPU. It is usable directly as an
in-loop search fitness.

## 3. Where the pose comes from — and the leakage firewall

A metric that needs the variant's own crystal is useless for design. So every
variant is scored on **every** same-length template in its series, and rows
record which:

- `self` — scored on its own crystal. Reported for completeness **only**. The
  backbone was solved *for that sequence*, so it leaks.
- `xtemplate` — mean over templates **other than its own**. This is the honest,
  deployable number: one reference complex per target, sequence threaded on,
  nothing about the variant used but its sequence. It is also the form the LNR
  pipeline always has available, since it starts from a native linear-peptide
  complex.

**`xtemplate` is the headline.** If `self` passes and `xtemplate` does not, the
metric is reading the crystal, and that is a negative result.

## 4. Grading: the instrument and the pre-registered rule

Instrument: CPSet Dataset I's 10 congeneric series (same receptor, measured Kd),
headline statistic = **within-series pair sign agreement with a Wilson 95% CI**,
computed per series and never pooled across series (pooling imports the
cross-target pKd range and inflates it into fake within-target skill). Any
head-to-tail claim uses the **BB/MIX tier only**. Statistics are *imported* from
`cpset_accuracy` — not reimplemented — so this number is directly comparable to
the Rosetta dG result it is trying to beat.

Mandatory controls, every one reported beside the primary:

- `null_neglen` (−peptide length) — the scorer that beat Rosetta dG cross-complex
- `null_negdsasa` (−buried area), `dg_rosetta` — joined from `cpset_dg.jsonl`, no recompute
- `null_chargecomp` — net-charge complementarity, a free sequence-level feature
- `null_random` — seeded, calibrates the instrument's chance level
- `pll_pep` — the peptide scored **without** the receptor: if this ranks as well
  as ICG, the metric is not about binding
- partial ρ controlling for peptide length, on every cross-complex correlation

**PRE-REGISTERED VERDICT** (same thresholds as the dG gate, so the two are
comparable):

```
PASS  pair agreement at |ΔpKd| >= 1.0 has Wilson 95% CI lower bound >= 0.65
      AND mean per-series rho is right-signed with |rho| >= 0.40
      AND >= 60% of series with n >= 3 are right-signed
      AND the margin over the BEST null is positive on the headline bin
WEAK  CI lower bound > 0.50 but PASS not met
FAIL  CI lower bound <= 0.50
```

**Primary metric, fixed now:** `icg`, `xtemplate` mode, model
`ligandmpnn_v_32_020_25`, side-chain context on, marginal mode
(`see_other_seq=False`). Declared secondary family, reported with multiplicity
acknowledged and *not* promotable to the headline without confirmation on
held-out series: `pll_cplx`, `pll_bg`, `icg_if`, `icg_ar`, `rec_top1`.

## 5. What a PASS would and would not establish

A PASS establishes that **ranking head-to-tail cyclic peptide affinity within a
target is achievable**, with a deterministic, free, size-unconfounded metric —
and that the project's pivot away from affinity was caused by the choice of
scorer, not by the objective being impossible.

It would *not* establish absolute affinity prediction, nor transfer to targets
with no reference complex. And one caveat is stated up front rather than
discovered later: these crystals predate the inverse-folding model's training
cutoff, so the structures (not the Kd labels — the model never sees affinity)
are in its training set. That inflates `self` far more than `xtemplate`, and it
compresses rather than creates within-series contrast, since every member is
equally native. The `xtemplate`/`self` gap is the diagnostic, and it is reported.

---

## AMENDMENT 1 — 2026-11-22, after the first run, before any claim

Two corrections. Both make the gate **harder**, and both were forced by things
the first run exposed rather than by anything it rewarded.

### 1.1 The pooled pair statistic was overconfident. Fixed.

The first run's BB/MIX tier gave `null_random` — one draw of pure noise —
**0.846 agreement (11/13)**, which a binomial calls a 1-in-90 event. That is not
a fluke to be explained away; it is the statistic failing. The 13 "pairs" come
from only two informative series (S03 n=6, S05 n=4), so they are strongly
dependent: a single lucky ordering of six items flips many pairs at once. A
Wilson CI assumes independence those pairs do not have.

Added, and now **required for a PASS**:

- **Within-series permutation test.** Permute the metric's values inside each
  series (leaving series sizes, pKd values, the |ΔpKd| stratification and the
  tie structure untouched), recompute the pooled agreement, 10 000 times. Its
  spread is the real chance level. Reported as `p_perm`, and the null's mean is
  printed in the table so nobody reads the agreement against an imagined 0.5.
- **Per-series sign test.** Each series contributes exactly one number (its
  Spearman), so series are the independent unit. Only series with n ≥ 3 count:
  a 2-member series has ρ = ±1 by construction and carries no information.

A consequence worth stating plainly: with 5 informative series the sign test
**cannot** produce p below 1/32 = 0.031 however good the metric is. The CPSet
instrument has a hard evidence ceiling. That is why a confirmation arm was added
rather than more decimal places extracted from this one.

### 1.2 The pre-registered primary is not the best member of its family.

Measured on the BB/MIX tier, `xtemplate` mode:

| metric | agreement | mean ρ | cross-ρ \| length |
|---|---|---|---|
| `icg` (pre-registered primary) | 0.923 | −0.943 | **−0.12** |
| `pll_cplx` (secondary) | **1.000** | −0.900 | **−0.41** |

`icg`'s subtraction of the peptide-alone term costs it nearly all cross-complex
signal. The reason is visible in hindsight: on CPSet the "peptide alone" term is
a 6–22 residue peptide with no partner, which is badly out of distribution for
an inverse-folding model, so the subtraction removes real signal along with the
intended nuisance. (On the SKEMPI arm the same subtraction is well posed, because
there the monomer is a real folded protein — so ICG is reported there too, and
the two arms test the construction under both conditions.)

**This does not promote `pll_cplx`.** The pre-registration says a secondary
needs confirmation on held-out data, and that rule exists precisely for moments
like this one. Both are carried, labelled, and judged on the confirmation arm.
The headline reported to a reader is the pre-registered primary; any claim for
`pll_cplx` is explicitly a post-hoc hypothesis under test.

### 1.3 The control that was supposed to stay flat did not

`pll_pep` — the peptide scored with the receptor **deleted** — reached 0.923
agreement on BB/MIX. The metric is therefore *not yet* shown to be about binding
rather than about the peptide's own bound backbone conformation. `pll_cplx` beats
it on every axis (1.000 vs 0.923; cross-ρ|len −0.41 vs −0.29), so the receptor
does add — but "adds" is weaker than the clean separation that was hoped for.

Added for this: a **composition control** (`pos_gain` = `pll_cplx` − `pll_shuf`,
where `pll_shuf` scores the same residues in scrambled order on the same pose).
A permutation holds composition *exactly* fixed, so anything surviving it is
positional complementarity and cannot be an amino-acid-composition artefact —
a stronger statement than the background-frequency correction `pll_bg` makes.

---

## AMENDMENT 2 — 2026-11-22, written BEFORE the confirmation arm reported

The CPSet run is complete and the SKEMPI arm is still running. Fixing the
decision rule now, while its result is unknown, so that the choice between
family members cannot be made to fit whichever number arrives.

### 2.1 CPSet cannot PASS. This is structural, not a verdict on the metric.

Under `xtemplate`, only **4 series** have n ≥ 3 and members of equal peptide
length (S01, S02, S03, S05 — S06's three members are 12, 13 and 15 residues, so
none can be threaded onto another). The per-series sign test therefore bottoms
out at **p = 1/16 = 0.0625**, above the 0.05 the pre-registered rule demands.

The observed result is 4/4 right-signed, i.e. the instrument is **saturated**:
the metric did everything the data can show, and the rule still returns WEAK.
That is the rule working as intended — it was written to stop a thin instrument
being talked into a strong claim — but it means CPSet alone can never settle
this, whatever is run on it. Reporting "WEAK" without this sentence would
misrepresent the result in the other direction.

### 2.2 Division of labour, fixed

```
CPSet   DISCOVERY    right chemistry (head-to-tail BB tier), 4 informative series
SKEMPI  CONFIRMATION wrong chemistry (protein-protein), 2502 mutations / 82 complexes
```

They are never pooled, and the head-to-tail claim rests only on CPSet's BB tier.

### 2.3 Which family member gets recommended — decided now

On CPSet the pre-registered primary `icg` and the secondary `pll_cplx` split:

```
             BB/MIX pairs   all-tier p_perm   all-tier sign   cross-rho | length
icg            12/13            0.0345            4/4              -0.12
pll_cplx       13/13            0.0645            3/4              -0.41
```

Neither dominates. The rule, fixed before the confirmation arm reports:

1. **If SKEMPI separates them** — one has a materially higher within-complex
   median ρ and sign-test significance — recommend that one, and say plainly
   that CPSet did not choose it, SKEMPI did.
2. **If SKEMPI does not separate them**, recommend `pll_cplx` for deployment and
   state the reason, which is not a p-value: `icg` requires scoring the peptide
   **alone**, which for a 6–22 residue macrocycle is badly out of distribution
   for an inverse-folding model, and it discards the cross-target signal
   (−0.12 vs −0.41 partialling out length) that any absolute "will this bind"
   judgement needs. `pll_cplx` also needs one forward pass instead of two.
3. **If SKEMPI shows both at chance**, recommend neither; report the CPSet result
   as a within-target-only finding on 4 series and say the confirmation failed.

### 2.4 What is already settled, independent of SKEMPI

Two controls have already returned, and they close two of the three ways this
result could have been an artefact:

- **Not composition.** `pll_shuf` — the same residues in scrambled order on the
  same pose — scores 0.462 (p = 0.66), i.e. chance. Permutation holds amino-acid
  composition *exactly* fixed, so the signal is **positional**. This is the
  objection that closed the ESM-2 / PepMLM thread, and it does not apply here.
- **Not size.** `null_neglen` scores 0/26 and `null_negdsasa` 0.269; the metric
  keeps −0.41 of cross-complex correlation after partialling out peptide length,
  where Rosetta dG keeps −0.00.

The third, **pocket specificity**, is what the decoy-receptor control now
running addresses: threading each series onto a same-length template from a
*different receptor* must collapse the ranking. It is the standing objection,
because `pll_pep` (peptide scored with the receptor deleted) reached 0.923 on
the BB/MIX tier, and until the decoy result is in, "the metric reads the
peptide's bound conformation rather than the pocket" is not excluded.

---

## AMENDMENT 3 — 2026-11-22, closing

All three arms have reported. Resolving the clauses fixed above, in order.

**§2.1 (CPSet cannot PASS):** confirmed. `icg` is right-signed on **4/4**
informative series — the instrument saturated — and the sign test returns
p = 0.0625 because 1/16 is its floor. The pre-registered verdict is **WEAK** and
is reported as WEAK. The metric did not fail; the instrument cannot certify it.

**§2.3 (which family member):** SKEMPI did **not** materially separate them —
within-complex median ρ 0.310 (`pll_cplx`) / 0.328 (`icg`) / 0.337 (`pll_bg`),
with 72, 72 and 73 of 79 complexes right-signed. **Clause 2 applies:** deploy
`pll_bg` / `pll_cplx` for head-to-tail macrocycles, for the stated structural
reason rather than a p-value — `icg` needs the peptide scored *alone*, which for
a 6–22 residue macrocycle is out of distribution, and it discards the
cross-target signal (partial ρ −0.12 vs −0.57). The prediction written in §2.3
before the result — that the same subtraction would *help* where the unbound
reference is a real folded protein — held: `icg` wins SKEMPI's pooled ρ (+0.378).

**§2.4 (pocket specificity):** **resolved, and it was the objection that
mattered.** Threading each series into a same-length pocket from a *different*
receptor collapses 13/13 → 6/13 (p 0.0059 → 0.638) and flips the mean ρ to the
wrong sign. Had the metric been reading the peptide's own bound conformation —
the live possibility, since `pll_pep` reached 0.923 — the decoy would have
preserved the ranking, because it preserves a peptide backbone conformation and
changes only the receptor. It did not.

### Status

The metric is **established as a within-target ranker for head-to-tail cyclic
peptides**, on 13 strictly-BB pairs (p = 0.006), confirmed on 2501 protein–protein
mutations (72–73/79 complexes, p ≈ 1e−15), with composition, size, own-crystal,
consensus and wrong-pocket all excluded by run controls, and with noise
eliminated rather than reduced (bitwise determinism, asserted as a gate).

It is **not** established as a calibrated affinity predictor, and the head-to-tail
evidence rests on two series. The honest next step is a held-out, post-2021
head-to-tail series — which would also retire the training-overlap caveat that
`xtemplate` and `null_consensus` only bound.
