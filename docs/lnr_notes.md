# LNR — the real ring-closure environment (no response model)

*Written for: whoever picks this up next. This is the pivot from the declared
surrogate to a real geometric environment, built after a Rosetta check found
that one load-bearing claim of the surrogate's response model does not hold in
real energy, and after the observation that a deposited CYCLIC complex is
already a near-optimal ring and so has little real headroom to "repair".*

Date: 2026-11-15. Code: [`../agent/lnr/`](../agent/lnr/). Data:
`/zfsauton/scratch/yixiz/LNR/` (93 linear peptide–receptor complexes, from the
sibling project). Run artefacts: `/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/`.

---

## Why this exists

Everything in Stage 2/3 was scored by a **declared response model**
(`lp2cp/envbuild/surrogate.py`): the parents' geometry was measured, but how an
objective *responds to an edit* was a closed-form model. That was the right
instrument for comparing controllers against an exact oracle, and it licensed
nothing biophysical. Two findings pushed past it:

1. **A Rosetta check graded its claims; the mix is sobering.**
   `validate_real_scorer.py` ran PyRosetta (FastRelax + InterfaceAnalyzer +
   per-residue alanine scan + larger-side-chain substitutions) on the parents'
   own PDBs. Over the parents measured (numbers below are n≈14 of 80 at the
   time of writing; refresh from `rosetta_validation/*.jsonl`):
   - **Measured geometry, roughly right.** `dSASA_int ≈ 1.37 × (2·surrogate
     BSA)` — correlated within a scale factor, not identical.
   - **Symptom ranking, moderate.** `interface_mass` (what `rim_first_closure`,
     the bar, ranks positions by) correlates with a real alanine scan at
     ρ≈0.51 mean / 0.56 median — positive and real, but not the strong
     agreement an early 3-parent probe suggested. "The bar ranks positions
     somewhat like real energy" is the safe reading.
   - **The lever claim fails.** The v2 surrogate says a substitution at a
     partially exposed (headroom) position **gains** interface; real best ΔdG
     there is **+7.3** (unfavourable), vs **+14.6** at buried hotspots. The
     asymmetry is directionally real (headroom hurts less), but neither is a
     gain. The best Stage 3 tool (`repack_headroom`) exploited the part that
     does not transfer.
   - **The strain proxy does not transfer.** The surrogate's Ramachandran
     `strain_rel` correlates with real Rosetta backbone energy at ρ≈−0.09 —
     i.e. not at all. It was a tuned proxy, and it does not track real strain.

2. **The parent was the wrong substrate.** A deposited cyclic complex is
   already a near-optimal ring; the cyclisation damage the benchmark repaired
   was itself modelled. Starting from the **linear** peptide in its bound pose
   and actually closing the ring makes the damage real.

## What the environment is

A target is a linear peptide bound to a receptor in a known pose. The decision
is a **held window `[a,b]`**: those residues keep their bound conformation,
every residue outside is released so the backbone can close a head-to-tail
ring. Releasing more makes closure easier and throws away interface; releasing
fewer keeps interface but may not close. That is the LP→CP conflict in one
move, and **both sides are real**:

- **CLOSES** — `feasible_torsion()` solves for free φ/ψ that bring the
  head-to-tail bond within tolerance with `[a,b]` held. Real torsion space.
- **RETENTION** — `retention_ceiling()` is the fraction of the bound
  interface's CA–CA contacts that survive when only `[a,b]` is held. A real
  coordinate quantity.

Both come from the sibling project's calibrated solver
(`Proteina-Complexa/script_utils/kinematic_ceiling.py`), imported not copied so
LNR numbers divide into that project's ceiling audit. **No LLM, no learned
scorer, no response model enters the objective.**

    utility(a,b) = retention(a,b)  if the ring closes with [a,b] held, else 0

### What is idealised, stated plainly

Real: bound coordinates, contact set, closure feasibility, retention. Idealised
(the solver's, declared): ideal bond lengths/angles and native ω in the closure
test; the CA–CA 10 Å contact definition; strict retention counts only contacts
made by held residues. **Not modelled at all:** side-chain repacking, ΔG,
folding. A closing window is a *geometric* claim that a backbone exists, not a
binding prediction.

## The decision is harder than it looks: placement, not just budget

Closure is **monotone in release for a fixed nested window** (the solver's own
prune: once a held window fails, every superset fails). But the
*best-placed* window at each release is a different window, so there is **no
single closure threshold** — a well-placed small release can close where a
poorly-placed larger one is wasted. On 1bjr the ceiling is retention 0.649 at
release 3 (hold `[0,6]`), yet a rule that binary-searches the release budget and
holds the max-retention window at each level never finds `[0,6]` and settles at
0.459. The hard part is **where** to hold, and that is what separates the arms.

## The evaluation budget is the currency

The analytic feasibility bound **over-licenses** (it calls `[3,9]` on 1d4t
feasible where the torsion solver needs five residues freed), so a controller
cannot read the answer off geometry — it must **spend real closure
evaluations**, learn from `(closed?, retention)`, and re-propose. A campaign is
`T` evaluations on one target (default `T=6`); the score is the retention of
the best window that actually CLOSED, 0 if none did. Arms are compared **paired
over targets at matched `T`** — this is what "evaluate relative to something
else" means here; there is no per-cell regret oracle. The exhaustive
`frontier.py` ceiling is reported only as context for how much retention was on
the table.

## The arms

| arm | what it does |
|---|---|
| `rule:contact_core` | hold the contact-richest window; release more on failure. The "protect the binding residues" instinct. |
| `rule:termini` | free the two ends, hold the interior; natural for head-to-tail. |
| `rule:feasible` | binary-search the release budget for closure, holding the analytic-max-retention window at each probe. |
| `rule:placement_search` | binary-search release, **then** search *placement* at the threshold. The strong deterministic bar. |
| `rule:greedy_frontier` | confirm the highest-analytic-retention windows first. Kept to show it WASTES budget — the top-retention windows are exactly the non-closers. |
| `llm-mask` | the LLM authors `[a,b]` from the per-residue contact table + history of what closed. Stage 3's mask-authoring arm, on real geometry. |
| `llm-tool` | the LLM names a tool and a release budget; the tool places the window. Stage 3's tool-selection arm. |

The one comparison the whole project is about — mask-authoring vs
tool-selection vs the strong rule — is now asked where the objective is
**measured, not modelled**. Early single-target signal is mixed (on 1bjr the
mask arm hit the ceiling 0.649; on 1d4t the tool arm hit it 0.611), which is
the honest state before n=87 lands.

## How to run

All CPU, resumable, no GPU. `sbatch/s4_lnr.sbatch`, keyed by `LNR_MODE`:

- `rules` — frontier ceiling + the deterministic arms (no key, no spend).
- `llm` — the mask/tool arms (needs `OPENAI_API_KEY`; `LNR_LLM_SPECS` is
  `model|arms|cap` space-separated).

Report: `python -m lnr.report --dir <outdir>` — the arm table (closure rate,
mean retention, cost), the ceiling, and the paired mask-vs-tool-vs-rule tests
(Wilcoxon on retention, McNemar on closure).

## Legitimacy hardening (2026-11-15)

Three additions after the first pass, to make the comparison defensible:

1. **A random null baseline** (`rule:random`). A random contiguous held window
   at each evaluation, seeded by `(target, attempt)` so the floor is
   reproducible and paired. Without it, "the LLM reaches 85% of the ceiling"
   has no floor — the random arm says what chance scores at the same
   evaluation budget. This is the single most important baseline for
   legitimacy and it was missing from the first pass.

2. **A strong deterministic bar** (`rule:placement_search`): binary-search the
   release budget for the closure threshold, then search *placement* at that
   threshold. It reaches the ceiling on a majority of early targets, so the LLM
   is measured against a genuinely strong rule, not a strawman. `greedy_frontier`
   is kept only to make the methodological point that retention-greedy wastes
   budget on the over-licensed non-closers.

3. **Clean-closure re-grade** (`lnr/clashcheck.py`, `env.clash_report`).
   Geometric closure (closing torsions exist) is necessary but not sufficient:
   a steric check on the rebuilt backbone shows some closing windows drive the
   chain through itself or the receptor. The clash gate (CA-level: non-adjacent
   CA–CA < 3.0 Å self, released-residue CA < 3.0 Å to a receptor CA) upgrades
   "closes" to "closes without gross overlap". It is validated against the
   **deposited** structures, which are self-clash-free by this metric, so the
   threshold is not mis-set. Still no fold, no learned scorer, no side chains.

   A useful side effect: the re-grade **re-solves every evaluated window
   uniformly** (same restarts, same time budget) and re-grades all arms off
   that one map. So the clean-closure comparison is perfectly matched across
   arms, where the raw geometric metric inherited each campaign's own closure
   time budget (6 s vs 8 s on different jobs — a minor mismatch that only ever
   affected whether a *hard* closure was confirmed, since easy closers return
   immediately). Report clean-closure as the headline; geometric closure is the
   permissive version.

## Binding quality: contact retention is a WEAK proxy — use the energy metric

A Rosetta per-residue alanine scan on the LNR targets (`lnr/rosetta_interface.py`,
subprocess-isolated, clean two-chain pose) measured each peptide residue's real
interface-energy contribution. Correlating that against the CA-contact count the
retention metric is built on:

  * Spearman(contacts, binding ddG) **mean ≈ 0.33, median ≈ 0.37** (range
    -0.27 .. 0.82 over the first 7 targets) — contact count is a WEAK-to-MODERATE
    predictor of binding energy, not a tight one.
  * Binding is **hotspot-concentrated**: the top-2 peptide residues carry ~60%
    of the interface energy on average. Equal-weighted contact retention does
    not see that concentration, so holding a high-contact, low-energy residue
    inflates contact retention without preserving binding.

**Consequence.** Report **binding-weighted retention** (the fraction of native
interface ENERGY the held residues carry, from the alanine scan) as the
binding-quality metric, with `native_dG` (−24 .. −159 REU here) as the
reference every cyclic-design method reports against. Contact retention remains
useful as a cheap, fully-enumerable quantity for the controller comparison, but
it must be labelled a weak binding proxy, not a binding score.

This corrects an earlier note that treated contact retention as a tight
(ρ≈0.5–0.8) binding proxy: that figure was for `interface_mass` (contacts +
polar pairs + buried area combined) against ddG on the Stage-2 CP parents; raw
per-residue contact count alone, which is what LNR retention uses, tracks
binding energy only at ρ≈0.33. The binding-energy pass is therefore
load-bearing for any binding claim, not optional hardening.

## Caveats to carry forward

- **Idealised closure.** The solver uses ideal bonds and native ω; a real build
  would add side-chain repacking and a ΔG. A closing window is geometric, not a
  binding claim. This is a strictly *better* instrument than the surrogate, not
  a wet-lab result.
- **Peptides are short (5–19 residues).** Placement reasoning is over a small
  index set, which may favour the LLM; a longer-peptide set would test that.
- **`greedy_frontier` is deliberately weak** — report it as the methodological
  point (why monotone search beats retention-greedy), not as a bar.

## Final full-n result (2026-11-15, both models × both arms × 87 targets)

All baselines complete on 87 targets; binding scan on 74/87 (13 failed on
FastRelax timeout / segfault); clean-closure re-grade run separately.

LLM vs Bayesian optimization (GP-UCB, the published sample-efficient bar),
paired, matched 6-evaluation budget:

| config | contact-retention vs BO | binding-energy vs BO |
|-|-|-|
| gpt-6.1-terra mask | +0.041 (p<0.001) | +0.111 (p<0.001) |
| gpt-6-luna  tool   | +0.026 (p=0.017) | +0.050 (p=0.019) |
| gpt-6-luna  mask   | +0.020 (p=0.27)  | +0.011 (p=0.19)  |

The mechanistic result: **BO is barely above the random floor on binding
energy** (+0.052, p=0.148) though clearly above it on contacts (+0.098,
p<0.001). BO optimizes the observable contact signal faithfully, but contacts
are a weak binding proxy (per-residue Spearman ~0.37, binding hotspot-
concentrated), so BO's binding quality collapses toward chance. The LLM's
semantic interface reasoning holds the binding hotspots, so its advantage is
LARGER on binding than on contacts -- the understanding buys what the proxy-
optimiser cannot.

Scale: terra-mask > luna-mask on both metrics (binding +0.111 vs +0.011) -- a
real positive scale effect, unlike Stage 2.

Caveats to report with it:
- terra-TOOL failed (closure 0.49, retention 0.345): the larger model choked on
  the tool-selection format while excelling at mask-authoring. Best config is
  model-dependent (terra->mask, luna->tool); there is a model x scaffold
  interaction, not a clean tool-vs-mask verdict.
- Both LLM and BO trail the geometry-aware placement_search (binding -0.089,
  p=0.005), which reads the analytic closure-feasibility bound the LLM is
  denied -- the information-asymmetry ceiling.
- 13/87 binding targets unscored (relax timeout on large receptors / segfault);
  the binding comparison is n~66-72 paired.

## The info-matched LLM arm (2026-11-15, in flight)

The final result's load-bearing caveat is an **information asymmetry**: both the
LLM and BO trail the geometry-aware `placement_search` (binding −0.089, p=0.005)
because `placement_search` reads the **analytic closure-feasibility bound** — a
cheap, closed-form geometric screen of which windows *could* close — for free,
while the plain LLM arms are shown only the per-residue contact table and the
history of what actually closed. That gap confounds the controller comparison:
`placement_search` beating the LLM could be better *strategy* or just more
*information*. This arm removes the asymmetry to separate the two.

**What it is.** Two new arms, `llm-info-mask` and `llm-info-tool`
([`../agent/lnr/run.py`](../agent/lnr/run.py), `analytic_frontier_block`). Each
is the plain mask/tool arm plus **one** added block in the prompt: the analytic
closure-feasibility frontier — for every release budget k, the highest-retention
contiguous window the analytic bound calls feasible, and its retention. This is
*exactly* what the geometry-aware rules consume: `feasible`/`placement_search`
walk this ladder (`hold_best_feasible` at an adaptive k) and `bo` uses the same
analytic-feasible set as its candidate list and GP prior mean. Budget (6),
feedback, parsing, models (terra, luna), effort (low) and max-tokens (6000) are
all identical to the plain arms, so the only thing that changed is the one thing
the LLM was denied.

**What it tests, honestly.** The bound **over-licenses** (it calls `[2,8]` on
1bjr feasible at release 3 with retention 0.678, where the real torsion-confirmed
closer is `[0,6]` at 0.649 — the `greedy_frontier` lesson). So the info does
**not** hand over the answer: the arm still has to spend evaluations to find
which feasible windows truly close, learning from history. The block says so
("feasible is NOT closes"). What the information removes is the handicap of not
knowing which windows are even candidates — the same handicap `placement_search`
never had. Two contrasts matter:
- `llm-info-* vs llm-*` — what did the geometry buy over the blind arm.
- `llm-info-* vs placement_search` — can the LLM, now info-matched, close the
  asymmetry gap to the strong bar, or does the bar's monotone-release structural
  prior still win on equal information.

**Caveat already visible from the design.** The *tool* arm was already partly
info-matched: its menu includes `hold_best_feasible`, which consults the bound,
so `llm-tool` could always delegate to the geometry. The clean test is the
**mask** arm (the terra config that won overall and was the one truly denied the
bound); expect a smaller `info` delta on tool than on mask.

**Run.** `sbatch/submit_lnr_info.sh` (env-file-as-positional-arg; no
`--export`). Job 111090 (`LNR_MODE=llm`, lov1) appends the new arm rows to
`llm_<model>.jsonl` (resumable, new `(target, llm-info-*)` keys, no overlap with
the mask/tool rows). Job 111091 (`LNR_MODE=clash`, `afterok:111090`) extends
`clashes.jsonl` with the info-arm windows for the clean-closure headline; the
**binding** re-grade needs no new Rosetta run — the report recomputes
binding-weighted retention for the info arms from the existing
`rosetta_interface.jsonl`. The report now emits, per model, `info-mask`/
`info-tool` paired against their plain counterparts, `placement_search`, `bo`,
and the random floor. Read binding as the headline:

    PYTHONPATH=agent:pilot/stage0 $PY -m lnr.report --dir <outdir> --metric binding

**The bar this arm is measured against is itself metric-dependent** (see the
clean-closure section below). The asymmetry caveat that motivated this arm was
stated on *raw geometric retention*, where `placement_search` wins. But on
*clean closure* the ranking inverts: `gpt-6-luna` mask already BEATS
`placement_search` and `bo`, because the rule and BO win raw retention by
picking high-retention windows that close geometrically yet CLASH. So the
info-matched arm now asks a sharper two-part question:
- On **raw retention**, does seeing the analytic frontier let the LLM close the
  gap to `placement_search` (the original asymmetry question)?
- On **clean closure**, does the frontier HELP or HURT? The frontier ranks
  windows by analytic feasibility + retention, and the clean-closure finding is
  that exactly those high-retention analytic closers are the ones that clash.
  So handing the LLM the over-licensing frontier could *pull it toward* the
  clashing windows the blind mask arm avoids — the n=1 1bjr point (info-mask
  0.459, below the 0.649 the blind mask hit) is the first whisper of that, but
  it is n=1; wait for the paired n=87 on `--metric` clean / binding.

### Partial result: terra complete (n=87), luna + clean-closure still running

Job 111090 finished the `gpt-6.1-terra` arms (87/87) and is now on `gpt-6-luna`
(n=1 so far); the clash re-grade 111091 has not run, so **clean-closure numbers
for the info arms are not yet valid** (clashes.jsonl was built from the original
arms only; the report's `cleanRet` for an info arm undercounts until 111091
adds its windows). Binding is n=74 (13 targets Rosetta never scored). Raw
retention and binding, paired n=87/74:

| terra info-mask vs | raw Δ | binding Δ |
|-|-|-|
| plain mask (its blind self) | **+0.032  p=0.016** | +0.012  tie |
| placement_search (strong bar) | +0.007  **tie** (W/L 6/7) | +0.013  tie |
| bo | +0.074  p<0.001 | +0.086  p<0.001 |

| terra info-tool vs | raw Δ | binding Δ |
|-|-|-|
| plain tool (its blind self) | **+0.183  p<1e-4** (38/0) | **+0.191  p<1e-4** (33/1) |
| bo | +0.045  p=0.001 | +0.029  tie |
| placement_search (strong bar) | **−0.022  p<1e-4** (1/28) | **−0.044  p=0.008** (3/21) |

Two clean terra findings, on raw geometric retention:

1. **Mask: the analytic bound closes the gap.** Blind terra-mask *trailed*
   `placement_search`; info-matched terra-mask beats its own blind self
   (+0.032, p=0.016) and now **ties the rule** (0.556 vs 0.549, W/L 6/7). Given
   the same geometry the LLM matches the strong domain rule — a positive answer
   to the original asymmetry question. (On binding it already tied the rule, so
   no further lift there; mask binding ≈ rule ≈ 0.56–0.58.)

2. **Tool: the bound rescues but does not elevate.** The frontier un-sticks the
   broken terra-tool scaffold (plain terra-tool choked at retention 0.345 /
   closes 0.49; info-tool 0.527 / 0.92), a huge +0.18 raw / +0.19 binding up to
   ≈ bo — but even fully info-matched it **still significantly trails**
   `placement_search` (−0.022 raw, −0.044 binding). The scaffold has a ceiling
   the information cannot lift it past.

So on terra the information helps the authoring (mask) arm close its gap to the
rule, and rescues-but-does-not-elevate the tool arm. **This is terra only, on
raw retention/binding.** The decisive questions are still open: luna is the arm
that *won* clean closure, and the clean-closure metric (where high-retention
analytic closers are exactly the clashers) is the one that could show the
over-licensing frontier HURTING. Both land when 111090's luna half and 111091
finish.

Numbers fold in here when both jobs land.

## Clean-closure re-grade INVERTS the ranking (2026-11-15, the key result)

Clean closure = the held window closes with closing torsions that do NOT make
the CA backbone pass through itself (non-adjacent CA-CA < 3.0 A) or the
receptor (released-residue CA < 3.0 A to a receptor CA). Validated: the
deposited natives are self-clash-free by this gate. clashes.jsonl re-solves
every evaluated window uniformly, so this metric is also fairness-matched.

`clean%` = fraction of 87 targets with a clash-free closing ring found:
  gpt-6-luna mask 0.56 | bo 0.39 | gpt-6.1-terra mask 0.38 | placement_search
  0.33 | random 0.29 | contact_core/feasible/router 0.24 | luna tool 0.23.

Paired clean-closure retention (clash-free), n=87, Wilcoxon:
  gpt-6-luna mask vs bo               +0.128  p=0.0006   LLM beats BO
  gpt-6-luna mask vs placement_search +0.082  p=0.0019   LLM beats the domain rule
  gpt-6-luna mask vs contact_core     +0.139  p<0.0001
  gpt-6-luna mask vs random           +0.170  p<0.0001

THE RANKING DEPENDS ON THE METRIC, and that is the finding:
- raw geometric retention: placement_search > LLM ~ bo > random (rule wins;
  the pasted/other-session conclusion, correct for that metric).
- clean closure (physically meaningful): gpt-6-luna mask > placement_search ~
  bo > random (LLM wins, p=0.002 vs the rule). The rule and BO win on raw
  retention by picking high-retention windows that close GEOMETRICALLY but
  CLASH; the LLM trades a little raw retention (0.502 vs 0.549) for a much
  higher clash-free closure rate (0.56 vs 0.33), i.e. more physically valid
  solutions, reasoning only from the contact map.

Caveats: CA-level clash gate (coarse, native-validated, not full-atom); it is
specifically gpt-6-luna MASK that wins clean closure -- terra-mask only ties
the rule (p=0.44), a model x metric interaction; clash-free closure is hard for
every arm (best clean% 0.56).

NOTE: a parallel session is editing the same files (report.py/run.py rewritten
11:48 with a --metric flag; an llm-info-mask arm started, n=2). Coordinate
before re-running to avoid clobbering shared code and scratch data.

## The clean-closure win, HARDENED (2026-11-15, final)

Written after two sessions (this one + lp2cp-agent-95) independently converged on
the same hardening and merged it. The CA-gate win above survives the three
objections a reviewer raises first: not reproducible, metric-cherry-picked, and
coarse gate. All three are now closed.

### 1. Reproducible from the official report
`report.py` gained first-class metrics so the headline is a command, not an
ad-hoc script:
  - `--metric clean`  : CA-contact retention of the best CLASH-FREE window.
  - `--metric clean-binding` : Rosetta interface-energy retention of the SAME
    committed clean window (argmax-contact clash-free closer; 0 if none; None if
    the target was never Rosetta-scored). Honors `--clash-field`.
  - `--clash-file / --clash-field` : switch gate (clashes.jsonl/clash_free =
    CA 3.0 A; clashes_bb.jsonl/clash_free_bb = backbone-triplet 2.5 A).

### 2. The win is not a contact-proxy artifact -- it holds on REAL BINDING ENERGY
The earlier worry was that luna-mask's bindRet (0.500) TRAILED placement_search
(0.562). That was binding scored on the GEOMETRIC window. Scored on the best
CLASH-FREE window -- the physically valid ring the arm actually found -- luna-mask
LEADS every arm (cleanBind 0.270 CA / 0.290 bb). The four-metric picture:

  | metric (CA gate)              | winner            | luna-mask vs placement_search |
  |-------------------------------|-------------------|-------------------------------|
  | retention (geometric)         | rule 0.549>0.502  | -0.047  p=0.0001  rule wins   |
  | clean     (clash-free contact)| LLM  0.276>0.194  | +0.082  p=0.0019  LLM wins    |
  | binding   (geometric energy)  | rule 0.562>0.500  | rule wins (geom window)       |
  | clean-binding (clash-free NRG)| LLM  0.270>0.194  | +0.076  p=0.0465  LLM wins    |

  BOTH physical (clash-free) metrics favor the LLM; BOTH geometric ones favor the
  rule -- because the rule's high-retention windows close on paper but CLASH.

### 3. The win survives a 3x-finer clash gate (backbone-triplet, not CA-only)
`clash_report_fine` / `clashcheck_fine.py` grade N/CA/C on both sides (receptor
at backbone resolution) at a 2.5 A cutoff calibrated against the deposited
natives (min non-adjacent backbone distance: self 2.85 A, pep-vs-receptor
2.68 A, so 2.5 A keeps every native clash-free; a native-floor assertion + 2
regression tests pin it). A load-bearing bug was caught here: the ring is
HEAD-TO-TAIL CYCLIC, so residues 0 and L-1 are bonded (closing C(L-1)-N(0) peptide
bond ~1.33 A) and must be excluded from the self-clash count, else every closed
ring shows a phantom junction clash and clash_free_bb collapses to ~0 for ALL
arms. Fixed (`far = (di>=2) & (di != L-1)`); the gate then keeps 68% of CA-clean
windows. The bb gate is NOT a strict superset of CA (3x the atoms but a looser
2.5 vs 3.0 CA-CA cutoff), so it is a second, independent "clash-free" definition.

### The robust headline, both gates x both metrics (luna-mask, n=74 paired)

                    CA-gate (3.0)              bb-gate (2.5, 3x atoms)
  vs BO       contact +0.128 p=.0006      contact +0.093 p=.010
              binding +0.108 p=.020       binding +0.104 p=.015     <- ROBUST
  vs placement contact +0.082 p=.0019     contact +0.053 p=.054 (marginal tie)
              binding +0.076 p=.047       binding +0.075 p=.034     <- binding both gates
  vs random    all p<1e-3 everywhere

**gpt-6-luna mask beats BO -- the published sample-efficient non-LLM bar -- across
BOTH clash gates AND both metrics (p 0.010-0.020, bootstrap CI[+0.027,+0.186] on
bb binding).** That is the clean "LLM significantly better than a non-LLM arm"
result. It ALSO beats the information-ADVANTAGED hand-coded rule
(placement_search reads the analytic closure-feasibility bound the LLM is denied)
on binding energy under both gates; the contact win vs that rule is CA-significant
(p=0.0019) and bb-marginal (p=0.054). The LLM's mean advantage is POSITIVE on
every gate x metric x bar cell -- never negative.

### Robustness + mechanism
- Bootstrap 10k + leave-one-out (both clean metrics): not outlier-driven.
  Contact vs placement_search +0.082 CI[+0.035,+0.130]; clean-binding vs
  placement +0.076 CI[+0.004,+0.146] (marginal, lower bound just clears 0); vs
  BO +0.108 CI[+0.019,+0.199] (comfortable); vs random CI excludes 0 by a wide
  margin.
- Mechanism, not a release-count confound: luna-mask's best-clean windows
  RELEASE MORE (4.90) than placement_search's (4.14) yet it closes clean far
  more often (56% vs 33%) -- it is choosing WHERE to hold, not how much. Of the
  rule's geometric closers 77% clash vs the LLM's 63%; on 23 targets only the
  LLM finds a clash-free ring (the rule closed-but-clashed on 20 of them).

### Honest caveats carried forward
- **Model-specific.** It is gpt-6-luna + mask-authoring. terra-mask only TIES
  placement_search on clean closure (binding +0.032 CI[-0.017,+0.082]); a
  model x scaffold interaction, not a general LLM property.
- **Binding vs the rule is marginal** (p 0.034-0.047); the robust leg is vs BO.
  The contact win vs the rule is gate-sensitive (CA strong, bb marginal).
- **The analytic-frontier info HURTS clean closure.** luna-info-mask cleanBind
  0.222 < plain 0.290; terra-info-mask 0.179 < 0.230 -- the over-licensing
  frontier pulls the LLM toward high-retention analytic closers that clash, the
  predicted direction. (luna-info still n~24; 111090 finishing.)
- CA-level receptor coords only; 2.5 A backbone gate is native-validated but not
  full-atom; 13/87 targets unscored by Rosetta (binding is n=74 paired).

Coordination for this subsection is resolved (95 owned report.py/env.py/notes
and clashcheck_fine.py; this session owned the binding lane; sessions since
ended). The "parallel session" NOTE above is historical.

## Real dG, denoised: fold every variant 5x, score each, report the median (2026-11-16)

The real-binding layer (`lnr/boltz_validate.py`: Boltz-2 fold -> PyRosetta
`InterfaceAnalyzer` dG) was scoring exactly ONE structure per variant --
`find_pdb()` took `*_model_0.pdb` and discarded the other diffusion samples
boltz had already drawn. Boltz's denoiser is stochastic, so a single dG is one
draw off that distribution: the fold noise was unmeasured, and a mut-vs-nomut
`ΔdG` paired over single draws could be mostly noise. On the pilot rows the
single-fold swings were large (e.g. 1d4t nomut -46.7 vs llm-luna -32.4, a +14 REU
"mutation hurts" that could be real or could be two unlucky draws).

**The fix is cheap because the samples already exist.** Boltz draws
`diffusion_samples` structures from ONE trunk pass (the expensive part) and
writes them `*_model_0..N-1.pdb`. So "fold 5x to eliminate noise" costs one trunk
pass plus extra (CPU) Rosetta relaxes, not five folds. `boltz_validate.py` now:

- `find_all_pdbs()` collects every sample; `score_replicas()` Rosetta-scores each
  under the SAME validity gate (scored, requested sequence, ring declared) and
  aggregates over the VALID ones.
- The reported dG is the **median** over valid folds; `dG_refined_std` / `_min`
  carry the single-fold spread. The report prints `median +/- sigma (n_valid/n)`
  and, per mutation arm, the mean fold-noise `sigma(Δ)` on the paired delta
  (variances add) so a mean `ΔdG` can be read against the noise it must clear.
- `--replicas N` (default 5; env `LNR_BOLTZ_REPLICAS`). `N>4` engages the
  `boltz_chunkfix` shadow (`configure_replicas`): past boltz's default sample
  count the denoiser mis-counts chunks and, with inference potentials on (the
  force-held modes), collapses to one oversized batch -> OOM. The patched shadow
  fixes the chunk count and the per-batch size is capped at 4 (the value the
  single-fold v100 runs used), so raising the replica count does not raise VRAM.
- Applied to all three real-dG modes: `mut-held` (the lever: mut vs nomut ΔdG),
  `held` (LLM window vs rule window), and `target` (native free-cyclisation fold).
- Resume-safe: single-fold and 5-fold rows must not be pooled, so a denoised run
  writes a fresh output file (`LNR_BOLTZ_MUT_OUTPUT=boltz_mut_held_r5.jsonl`)
  rather than resuming onto the single-fold `boltz_mut_held.jsonl`.

### Validation result (job 111892, gpu22/V100, mut-held 1d4t+1ee5, 2026-11-16)

Ran clean (COMPLETED, 21 min). All 6 variants scored at 4 valid folds each
(reused the prior run's 4 diffusion samples -- cost-optimal; a fold already drawn
is not re-drawn, and `--replicas 5` only folds fresh for variants with no prior
fold). The denoised table and what it shows:

| target | arm | window | held Å | dG (median+/-sd, REU) |
|-|-|-|-|-|
| 1ee5 | nomut | [10,17] | 1.3 | **-49.94 +/- 3.13** |
| 1ee5 | llm-luna (14R,16R) | [10,17] | 3.7 | -42.32 +/- 5.47 |
| 1ee5 | llm-terra (13V) | [10,17] | 1.5 | -37.61 +/- 9.84 |
| 1d4t | nomut | [6,10] | 10.4 | -32.08 +/- 9.36 (mislocalised, dropped) |
| 1d4t | llm-luna (8I) | [6,10] | 16.7 | -26.02 +/- 3.00 (mislocalised, dropped) |

**The payoff is the noise column.** A single Boltz fold's dG is +/-3 to +/-10 REU
noisy -- and on 1d4t the old single fold had read -46.7 where the 4-fold median is
-32.1, i.e. the single draw was ~1.5 sigma into the tail. Any ΔdG built on single
folds was being read off that noise.

Paired on the one target that folds near the interface (1ee5; 1d4t at 10-17 A is
dropped by the rmsd<=8 pair guard -- correctly, and consistently across all 4
folds, not one bad draw):
- llm-terra ΔdG = **+12.3 REU** (mutation HURTS), per-fold sigma(Δ)=10.3 -> SEM
  over 4 folds ~5.2 -> ~2.4 sigma.
- llm-luna  ΔdG = **+7.6 REU**  (mutation HURTS), sigma(Δ)=6.3 -> SEM ~3.2 -> ~2.4 sigma.

So denoised, on 1ee5, BOTH LLM substitutions significantly WORSEN real binding vs
no mutation, and neither helps -- the same "the lever does not help" verdict the
project keeps reaching, now in replicated real dG rather than the surrogate or a
single noisy fold. n=1 target, so this is instrument validation, not a result:
the finding is that effect sizes (~8-12 REU) sit right at the single-fold noise
(sigma ~3-10 REU), which is why replication is mandatory and why a single-fold
mut-vs-nomut comparison at this scale was underpowered.

Two things to carry into the scaled run: (1) report how many targets fold near the
interface at all (1d4t does not; the rmsd<=8 guard is load-bearing), and (2) the
denoised SEM, not the raw per-fold sigma, is the error bar on a committed dG.

**Why the proxy still drives the search.** A 5-fold Rosetta-scored dG is minutes
of GPU + CPU per variant; the controller explores hundreds of windows/substitutions
per campaign. So the cheap contact/energy-retention proxy stays the SEARCH signal
(fully enumerable, no fold), and the denoised real dG is spent only on finalists
and on the ceiling -- the two-tier fitness the directed-evolution literature uses
(cheap proxy in the loop, expensive oracle on the few that matter).

### The dG ceiling (what a CP can reach), cheap vs physical
- **Cheap, exact (built):** `lnr/energy_ceiling.py` -- the clash-free closing
  window that keeps the most interface ENERGY (WT alanine scan), `recovered =
  energy_retention * native_dG`. The gap `recovered -> native_dG` is the
  fundamental cyclisation loss (hotspot energy no closing window can hold). This
  is the ceiling the nomut/placement baseline is read against and the mutation
  lever must beat.
- **Physical (next):** fold the NOMUT native sequence held at that ceiling window,
  5x, Rosetta-score -> the real dG a perfect-window, no-mutation cyclisation
  achieves. The denoising above is the prerequisite: a one-fold ceiling is noise.
  Thread `energy_ceiling.jsonl`'s window into a `mut-held`-style nomut fold.
