"""Iterative-search mutation arms for LNR: an LLM evolutionary loop vs a matched
random-EA control, both scored by the SAME cheap proxy oracle the single-shot
seqmut arms use, with the best finalist emitted as a standard seqmut row so the
existing Boltz 4-fold job validates it unchanged.

Why this exists (and what it is NOT)
------------------------------------
`seqmut.py` runs the WEAKEST version of LLM-guided design: one shot, no feedback.
The literature on LLM-as-optimizer is unanimous that the lever is ITERATION with
a fitness signal fed back. This module adds that loop, and -- critically -- the
CONTROL that makes an LLM win real: a random-mutation/crossover EA under an
IDENTICAL budget. A loop beats a single shot even with a random operator (more
tries, elitist memory), so the only honest question is whether the LLM operator
beats the random operator at MATCHED proxy evaluations.

Grounding (informed, not copied)
--------------------------------
- Wang et al. 2501.09274 ("LLM is Secretly a Protein Sequence Optimizer"): the
  template -- population, LLM proposes by mutation/crossover conditioned on shown
  fitness, elitist selection, a Hamming budget (here k_mut), and a random-operator
  EA control under the SAME hyperparameters. This module's `iter-ea` arm is that
  control.
- Zhang et al. 2604.19440 ("What Makes an LLM a Good Optimizer?"): the EA loop
  shape (top-q elite, fitness-proportional parent sampling, show parents WITH
  their fitness and instruct the model to beat them) AND, more importantly, what
  to MEASURE. Their finding: a good LLM optimizer is a LOCAL REFINER -- what
  separates it from drift is the BREAKTHROUGH rate (generations that improve the
  best-so-far) and the rate at which offspring beat the frontier they were shown,
  NOT novelty/diversity. So every generation logs those, for both arms, to a
  trajectory file -- the diagnosis of WHETHER the LLM is optimizing, redrawable
  without recompute.
- Lu et al. 2605.02933 (R2SAEA) and StructEvo: the two-tier economics. The cheap
  proxy (`seqmut.recovered_binding_energy`, a pose-preserving Rosetta scan) runs
  IN the loop; the expensive true oracle (Boltz diffusion_samples=4 -> Rosetta)
  is spent only on the finalist per arm. We emit the finalist as a seqmut row so
  the existing `boltz_validate.py --mode mut-held` picks it up with no change.

Budget semantics (the matched quantity)
---------------------------------------
The expensive resource in the loop is a proxy evaluation (a Rosetta mutant scan,
~minutes). Both arms are capped at `--max-evals` DISTINCT genomes scored for that
arm. A genome already in the shared scan cache (e.g. scored by the other arm, or
a single-shot run) still counts against this arm's budget -- so both arms receive
exactly `max_evals` evaluations' worth of information; the cross-arm cache only
saves wall-clock, never budget. This is the FEs-matched principle of Wang/R2SAEA.

Determinism / resumability
--------------------------
The random-EA arm is seeded by (target, arm, k_mut), so a resubmission replays the
same trajectory and every scan is a cache hit until new territory. The LLM arm is
not bit-reproducible (sampling temperature), but identical re-proposed sets are
free (cached), and a finished (target, arm) is skipped whole. A time limit costs a
resubmission, not a restart (the LNR convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import seqmut as S  # noqa: E402  the shared scorer / cache / window / prompts

AA20 = S.AA20
EPS = 1e-6

ITER_LLM_ARMS = ("iter-llm", "iter-llm-ala", "iter-llm-hi")
ITER_EA_ARM = "iter-ea"
ITER_ARMS = ITER_LLM_ARMS + (ITER_EA_ARM,)

# `iter-llm-hi` is the SAME operator and prompt at a higher reasoning effort and
# correspondingly fewer generations, so the two LLM arms cost about the same in
# dollars. It exists because the only unambiguous advantage measured for the LLM
# arm so far is that it proposes more DISTINCT genomes -- a diversity effect that
# a better-reasoning model need not improve at all. Separating "thinks harder"
# from "proposes more" needs the dollars held fixed and the generations varied,
# which is what this arm does.
HI_EFFORT_ARM = "iter-llm-hi"


# ---------------------------------------------------------------------------
# Genome helpers.  A genome is a sorted tuple of (pos, aa); () is nomut.
# ---------------------------------------------------------------------------

def _norm(subs) -> tuple:
    """Canonicalise a substitution set: ints, upper AA, one entry per position,
    sorted. Mirrors subs_key's identity so cache lookups and dedup agree."""
    seen = {}
    for i, a in subs:
        i = int(i)
        a = str(a).upper()
        if a in AA20:
            seen[i] = a  # last write per position wins (crossover may collide)
    return tuple(sorted(seen.items()))


def _iface_positions(t) -> list:
    return [i for i in range(t.length)
            if i < len(t.per_res_contacts) and t.per_res_contacts[i] > 0] \
        or list(range(t.length))


# ---------------------------------------------------------------------------
# The proxy-scored, budget-capped population.
# ---------------------------------------------------------------------------

class Population:
    """Genome -> cached fitness, with a hard distinct-evaluation budget.

    fitness = recovered_binding_energy (REU); MORE NEGATIVE IS BETTER, so the
    frontier is the minimum. A genome that fails to score (scan error/timeout) is
    remembered with recovered=None so it is neither re-scored nor ever elite.

    NOISY FITNESS (the measured failure this class now defends against)
    ------------------------------------------------------------------
    Under `--evaluator boltz` a fitness is a MEASUREMENT, not a value: the median
    of 4 stochastic folds, whose own spread is sd ~3.7 REU (SEM ~1.9), and whose
    run-to-run reproducibility on the SAME variant was measured at ~4 REU
    (1bjr wild-type across independent draws: -20.9 / -24.3 / -26.8 / -26.9).
    Taking the argmin of N such draws is the optimizer's curse: the winner is
    optimistic by ~1-2 SEM purely by selection. Measured on the first real-score
    loop, 61% of the apparent gain vs wild-type evaporated when the finalist was
    re-folded with an independent seed (1bjr iter-llm: -8.01 -> -0.57 REU).

    So a member keeps ALL its observations, not one:
      obs       every independent draw of this genome's fitness
      recovered the running MEAN (the unbiased point estimate; what gets reported)
      sem       standard error of that mean (from repeats when n>=2, else the
                evaluator's own within-draw SEM via `scorer.sem(row)`)
    and selection uses an interval, not the point:
      elite()  ranks OPTIMISTICALLY (mean - z_explore*sem) -- a noisy candidate
               stays in contention long enough to be re-measured (UCB-style).
      best()   commits PESSIMISTICALLY (mean + z_commit*sem) -- the finalist we
               spend the real oracle on must be good because it IS good, not
               because it was measured luckily (LCB-style).
    With a deterministic evaluator (the proxy) sem == 0 and both rules collapse to
    the old argmin, so the proxy path is behaviourally unchanged.
    """

    def __init__(self, t, window, cache, k_mut, relax_repeats, timeout, max_evals,
                 scorer=None, z_explore: float = 1.0, z_commit: float = 1.0):
        self.t = t
        self.window = window
        self.cache = cache
        self.k_mut = k_mut
        self.relax_repeats = relax_repeats
        self.timeout = timeout
        self.max_evals = max_evals
        # The evaluator is pluggable: default is the pose-preserving Rosetta
        # retention proxy; `--evaluator boltz` passes a BoltzScorer (real Boltz x4
        # -> Rosetta dG). Both expose .score(t, subs, window) -> (fitness, row),
        # fitness more-negative=better, None when unscored. Default kept so the
        # proxy path (and its no-GPU unit tests) is byte-identical.
        if scorer is None:
            from lnr.boltz_inloop import ProxyScorer
            scorer = ProxyScorer(cache, relax_repeats, timeout)
        self.scorer = scorer
        self.z_explore = float(z_explore)
        self.z_commit = float(z_commit)
        # subs_key -> {"subs", "obs": [float], "recovered": mean|None, "sem", "gen"}
        self.members: dict = {}
        self.evals = 0            # evaluations THIS arm has spent (budget)
        self.reassessments = 0    # of those, re-measurements of a known genome

    def budget_left(self) -> bool:
        return self.evals < self.max_evals

    def _scorer_sem(self, row) -> float:
        """The evaluator's own within-draw SEM, used while a genome has a single
        observation. Deterministic evaluators report 0."""
        fn = getattr(self.scorer, "sem", None)
        if fn is None:
            return 0.0
        try:
            v = fn(row)
        except Exception:  # noqa: BLE001  a sem estimate must never fail a run
            return 0.0
        return float(v) if isinstance(v, (int, float)) and v == v else 0.0

    def _record(self, key, g, rec, row, gen: int):
        """Fold one observation into a member (creating it on first sight)."""
        m = self.members.get(key)
        if m is None:
            m = self.members[key] = {"subs": g, "obs": [], "recovered": None,
                                     "sem": 0.0, "gen": gen}
        if rec is None:
            return m
        m["obs"].append(float(rec))
        n = len(m["obs"])
        m["recovered"] = sum(m["obs"]) / n
        if n >= 2:
            mean = m["recovered"]
            var = sum((x - mean) ** 2 for x in m["obs"]) / (n - 1)
            m["sem"] = (var ** 0.5) / (n ** 0.5)
        else:
            m["sem"] = self._scorer_sem(row)
        return m

    def evaluate(self, subs, gen: int):
        """Score `subs` with the evaluator (cached). Returns fitness (or None).
        Counts one unit of budget the first time this arm sees the genome; returns
        None without scoring once the budget is spent on a genome not yet seen."""
        g = _norm(subs)
        key = S.subs_key(g)
        if key in self.members:
            return self.members[key]["recovered"]
        if not self.budget_left():
            return None
        rec, row = self.scorer.score(self.t, list(g), self.window)
        self.evals += 1
        self._record(key, g, rec, row, gen)
        return self.members[key]["recovered"]

    def reassess(self, member, gen: int):
        """Re-measure an ALREADY-scored genome with a fresh independent draw and
        fold it into that member's running mean. This is the direct fix for the
        optimizer's curse: without it the incumbent frontier is whatever single
        draw happened to come out lowest, it is never corrected, and every later
        generation is conditioned on it.

        Costs one unit of the SAME budget an offspring would cost, so the arms
        stay fitness-evaluation matched. A no-op (returns None, charges nothing)
        on a deterministic evaluator, where a redraw cannot differ."""
        if not getattr(self.scorer, "stochastic", False):
            return None
        if member is None or not self.budget_left():
            return None
        g = member["subs"]
        key = S.subs_key(g)
        before = member["recovered"]
        rec, row = self.scorer.score(self.t, list(g), self.window,
                                     draw=len(member["obs"]))
        self.evals += 1
        self.reassessments += 1
        if rec is None:
            return None
        self._record(key, g, rec, row, gen)
        return round(self.members[key]["recovered"] - before, 3)

    def evaluate_many(self, subs_list, gen: int, n_workers: int = 1) -> dict:
        """Score a batch of genomes, evaluating the NEW ones concurrently when the
        evaluator supports it (the CPU proxy: each worker blocks on its own
        PyRosetta subprocess -> real parallelism; the GPU Boltz scorer saturates the
        one GPU, so it is called with n_workers=1). Cached/duplicate genomes are
        free. Budget is charged per distinct NEW genome, exactly as the serial path
        -- parallelism changes wall-clock only, never which genomes get evaluated.
        Returns {subs_key: fitness} for every requested genome now in the pool."""
        results, to_score, pending = {}, [], set()
        for subs in subs_list:
            g = _norm(subs)
            key = S.subs_key(g)
            if key in self.members:
                results[key] = self.members[key]["recovered"]
                continue
            if key in pending:
                continue                                   # dup within the batch
            if self.evals + len(to_score) >= self.max_evals:
                continue                                   # budget exhausted
            to_score.append((key, g))
            pending.add(key)
        if not to_score:
            return results

        def _run(g):
            return self.scorer.score(self.t, list(g), self.window)

        if n_workers and n_workers > 1 and len(to_score) > 1:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(to_score))) as ex:
                out = list(ex.map(lambda kg: _run(kg[1]), to_score))
        else:
            out = [_run(g) for _, g in to_score]
        for (key, g), (rec, row) in zip(to_score, out):
            self.evals += 1
            self._record(key, g, rec, row, gen)
            results[key] = self.members[key]["recovered"]
        return results

    def scored(self) -> list:
        return [m for m in self.members.values() if m["recovered"] is not None]

    def lcb(self, m) -> float:
        """Pessimistic score (commit): a noisy member is charged for its noise."""
        return m["recovered"] + self.z_commit * m.get("sem", 0.0)

    def ucb(self, m) -> float:
        """Optimistic score (explore): a noisy member keeps its benefit of doubt,
        so it stays elite long enough to be re-measured rather than being
        dismissed -- or crowned -- on one draw."""
        return m["recovered"] - self.z_explore * m.get("sem", 0.0)

    def best(self):
        """The genome we would COMMIT to (finalist / believed-number fold)."""
        v = self.scored()
        return min(v, key=self.lcb) if v else None

    def leader(self):
        """The genome at the head of the optimistic order -- the incumbent the
        elite block shows and the one worth re-measuring."""
        v = self.scored()
        return min(v, key=self.ucb) if v else None

    def best_recovered(self):
        b = self.best()
        return b["recovered"] if b else None

    def elite(self, frac: float, min_n: int = 2, max_n: int = 12) -> list:
        """Top-q elite in OPTIMISTIC order (mean - z*sem, most negative first)."""
        v = sorted(self.scored(), key=self.ucb)
        if not v:
            return []
        n = max(min_n, int(round(frac * len(v))))
        return v[: min(max_n, min(n, len(v)))]

    def tried(self, exclude_keys=(), limit: int = 14) -> list:
        """Scored members NOT in `exclude_keys` -- the failure memory the LLM brief
        was missing. Showing only the elite left the model blind to what it had
        already tried and lost: 34-45% of its proposals re-proposed a genome
        already in the pool, and it fixated on the incumbent's positions (on
        1elr, 6 of 7 post-seed proposals carried the elite's `1:Q`).

        ORDER: NEAR-MISSES FIRST, then the worst for contrast. This block used to
        be sorted worst-first, which spent the whole budget of rows on the least
        informative genomes in the pool -- edits that are obviously bad say
        nothing about where the next gain is, while the sets sitting just behind
        the frontier are exactly the ones whose near-success localises it. Giving
        both ends also keeps the dynamic range visible, so "this edit is bad"
        and "this edit is nearly the best" are distinguishable rows.

        Duplicate AVOIDANCE is no longer this block's job -- it cannot be, at 14
        rows out of a pool of hundreds -- see `measured_keys`.
        """
        ex = set(exclude_keys)
        v = [m for m in self.scored() if S.subs_key(m["subs"]) not in ex]
        v.sort(key=lambda m: m["recovered"])      # best (most negative) first
        if len(v) <= limit:
            return v
        n_near = limit - limit // 3               # 2/3 near-misses, 1/3 worst
        return v[:n_near] + v[-(limit - n_near):]

    def measured_keys(self) -> list:
        """EVERY genome already in the population, as raw subs keys, best-first
        (anything attempted but unscored last).

        The brief instructs the model that each proposal "must be one NOT listed
        anywhere above" -- an instruction that was unsatisfiable by construction
        while the only things listed were 12 elite + 14 tried rows out of a pool
        that reaches ~500. The measured 34-45% duplicate rate is that contract
        failing, not the model ignoring it, and every duplicate costs a proposal
        slot (they are dropped, not scored). The full key list is a few hundred
        short strings -- cheap next to a wasted generation.

        Keys are RAW (the wild-type is ""), so this list is exactly the set the
        duplicate filter tests against; rendering "" as "(none)" is the display
        layer's job. Returning a prettified key here would silently exempt the
        wild-type from the filter.
        """
        def rank(m):
            r = m.get("recovered")
            return (1, 0.0) if r is None else (0, r)
        return [S.subs_key(m["subs"])
                for m in sorted(self.members.values(), key=rank)]


# ---------------------------------------------------------------------------
# Random-EA operator (the matched control): crossover + point mutation.
# ---------------------------------------------------------------------------

def _regret(best, opt) -> float | None:
    """How far the arm's best genome is from the fitness's KNOWN optimum.

    Both are in the loop's more-negative-is-better convention, so regret is
    `best - optimum` and is >= 0, with 0.0 meaning the arm found the exact
    optimum. None whenever the evaluator cannot supply an optimum -- which is
    every evaluator except the PLL table, and the reason this project has never
    been able to say how good its searches actually are.
    """
    if best is None or not opt:
        return None
    return round(float(best) - float(opt["fitness"]), 4)


def _rank_weights(n: int) -> list:
    """Linear rank weights, best (index 0) heaviest -- the fitness-proportional
    parent sampling of 2604.19440 without the sign trouble of a negative REU."""
    return [n - i for i in range(n)]


def _sample_parents(elite, rng, k=2):
    if len(elite) <= 1:
        return [elite[0]] * k if elite else []
    w = _rank_weights(len(elite))
    return rng.choices(elite, weights=w, k=k)


def _crossover(p1: tuple, p2: tuple, k_mut: int, rng) -> dict:
    d: dict = {}
    for i, a in p1:
        d[i] = a
    for i, a in p2:
        if i in d:
            if rng.random() < 0.5:
                d[i] = a            # position clash: inherit from one parent
        else:
            d[i] = a
    if len(d) > k_mut:              # trim to the Hamming budget
        keep = rng.sample(list(d), k_mut)
        d = {i: d[i] for i in keep}
    return d


def _point_mutate(d: dict, t, k_mut: int, iface, rng, p=0.7) -> dict:
    """Add / replace / drop one substitution, keeping within budget and valid."""
    if rng.random() > p:
        return d
    moves = []
    if len(d) < k_mut:
        moves.append("add")
    if d:
        moves += ["replace", "drop"]
    if not moves:
        return d
    move = rng.choice(moves)
    if move == "drop":
        d.pop(rng.choice(list(d)))
    elif move == "add":
        free = [i for i in iface if i not in d]
        if free:
            i = rng.choice(free)
            wt = t.seq[i] if i < len(t.seq) else "A"
            d[i] = rng.choice([a for a in AA20 if a != wt])
    else:  # replace the AA at an existing position
        i = rng.choice(list(d))
        wt = t.seq[i] if i < len(t.seq) else "A"
        d[i] = rng.choice([a for a in AA20 if a not in (wt, d[i])] or
                          [a for a in AA20 if a != d[i]])
    return d


def ea_offspring(elite, t, k_mut: int, iface, rng, m: int, known=()) -> list:
    """m children, each (genome, parent_recovered) where parent is the better of
    its two crossover parents -- the frontier that child is asked to beat.

    `known` is every genome already evaluated. A child that repeats one is
    RESAMPLED, exactly as a repeated LLM proposal is re-asked: crossover between
    similar elites collides often, and without this the EA quietly produces fewer
    distinct evaluations than the LLM arm (measured 18 vs 24 of a 24 budget on a
    dry run) -- which would hand the LLM arm extra oracle calls and void the
    matched-budget comparison. Capped resampling, so a saturated neighbourhood
    still returns a child rather than spinning."""
    out, known = [], set(known)
    for _ in range(m):
        p = _sample_parents(elite, rng, k=2)
        if not p:
            break
        parent_rec = min(pp["recovered"] for pp in p)   # the stronger parent
        child = None
        for _try in range(8):
            d = _crossover(p[0]["subs"], p[1]["subs"], k_mut, rng)
            cand = _norm(_point_mutate(d, t, k_mut, iface, rng).items())
            if S.subs_key(cand) not in known:
                child = cand
                break
            p = _sample_parents(elite, rng, k=2) or p   # try a different pairing
        if child is None:
            child = cand                                 # neighbourhood saturated
        known.add(S.subs_key(child))
        out.append((child, parent_rec))
    return out


# ---------------------------------------------------------------------------
# LLM operator: show the elite WITH their proxy fitness; ask for better sets.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WHAT THE IN-LOOP FITNESS ACTUALLY IS.
#
# Three evaluators now exist and they do not measure the same thing or even share
# a unit. The brief has to say which one the model is being graded by, because
# telling it the wrong one handicaps the LLM arm against the random-EA control,
# which reads no brief at all and so cannot be misled. This table is the single
# place that mapping lives; `scorer_objective` resolves a scorer to one entry.
# ---------------------------------------------------------------------------
OBJECTIVES = {
    "proxy": dict(
        units="REU",
        system_attr="MUT_SYSTEM",
        scored_by=(
            "Candidates are scored by RECOVERED BINDING ENERGY (energy-weighted "
            "interface retention x native dG, REU). MORE NEGATIVE IS BETTER -- "
            "more of the interface energy kept on a ring that closes."),
        propose_mode=(
            "Keep the NATIVE BINDING MODE (the scorer repacks locally; it cannot "
            "reward a changed fold)."),
    ),
    "folded": dict(
        units="REU",
        system_attr="MUT_SYSTEM_FOLDED",
        scored_by=(
            "Candidates are scored by the REAL FOLDED binding energy: the "
            "cyclized peptide is folded from its sequence (backbone held at the "
            "retained window), and the predicted ring's receptor interface is "
            "scored (structure prediction x4 -> Rosetta dG, REU, median over "
            "folds). MORE NEGATIVE IS BETTER."),
        propose_mode=(
            "Your edits are judged on the complex they fold into, so a change "
            "that improves how the peptide folds or binds is rewarded -- not "
            "only local interface tweaks."),
    ),
    "pll": dict(
        units="nats/res",
        system_attr="MUT_SYSTEM_PLL",
        scored_by=(
            "Candidates are scored by POSE-CONDITIONED INTERFACE LIKELIHOOD: an "
            "inverse-folding model reads the deposited complex (receptor backbone "
            "and side chains, peptide BACKBONE ONLY) and reports, per position, "
            "the log-odds of the residue you put there against its background "
            "frequency, averaged over the peptide. The number below is that "
            "average NEGATED, so that -- as in every other column in this "
            "project -- MORE NEGATIVE IS BETTER. Units are nats/residue, not "
            "REU, and a difference of 0.05 is real: the score is exact, with no "
            "measurement noise at all."),
        propose_mode=(
            "The score reads only how well each residue suits the pocket it "
            "faces in the FIXED deposited pose. It cannot see strain, packing "
            "clashes, or a changed fold, so a bulky residue earns nothing for "
            "being bulky -- only for matching the shape and chemistry of the "
            "receptor atoms across from it."),
    ),
}


def scorer_objective(scorer) -> dict:
    """Resolve an evaluator to its OBJECTIVES entry.

    Keyed on the scorer's own declared `name`, falling back to the historical
    `folds`/`is_gpu` sniffing so the pre-existing evaluators and the GPU-less
    test doubles that only set `is_gpu` keep resolving exactly as before.
    """
    name = getattr(scorer, "name", None)
    if name == "pll":
        return OBJECTIVES["pll"]
    folded = bool(getattr(scorer, "folds", getattr(scorer, "is_gpu", False)))
    return OBJECTIVES["folded" if folded else "proxy"]


def scorer_name(scorer) -> str:
    """The short evaluator tag written into finalist rows ('proxy'/'boltz'/'pll')."""
    name = getattr(scorer, "name", None)
    if name:
        return str(name)
    return "boltz" if getattr(scorer, "is_gpu", False) else "proxy"


def _elite_block(elite, noisy: bool = False, units: str = "REU") -> str:
    """The frontier. With a stochastic evaluator each score carries its standard
    error and how many independent measurements it rests on, so the model can see
    that a 1 REU lead backed by one noisy draw is not a lead."""
    if noisy:
        rows = [f"  rank  score({units})   +/-SE   n  substitutions (pos:AA)",
                "  ----  ----------  ------  --  ----------------------"]
        for r, m in enumerate(elite, 1):
            sk = S.subs_key(m["subs"]) or "(none / wild-type)"
            rows.append(f"  {r:4d}  {m['recovered']:+10.2f}  {m.get('sem', 0.0):6.2f}  "
                        f"{len(m.get('obs') or [1]):2d}  {sk}")
    else:
        rows = [f"  rank  score({units})  substitutions (pos:AA)",
                "  ----  --------------  ----------------------"]
        for r, m in enumerate(elite, 1):
            sk = S.subs_key(m["subs"]) or "(none / wild-type)"
            rows.append(f"  {r:4d}  {m['recovered']:+14.2f}  {sk}")
    return "\n".join(rows)


def _tried_block(tried, units: str = "REU") -> str:
    """The failure memory: sets already measured that did NOT make the frontier."""
    if not tried:
        return ""
    cells = [f"{S.subs_key(m['subs']) or '(none)'} ({m['recovered']:+.2f})"
             for m in tried]
    return (f"Measured, not on the frontier -- nearest misses first, worst last "
            f"({units}):\n  " + "; ".join(cells) + "\n\n")


def _measured_block(keys, cap: int = 600) -> str:
    """The COMPLETE list of genomes already evaluated.

    Scores are deliberately absent: this block answers "has this been tried?",
    which is the question the duplicate-avoidance instruction actually needs,
    and nothing more. Keeping fitness out of it means the block cannot be read
    as a per-position credit assignment -- that is a separate, deliberately
    unpulled lever (see docs/seqmut_notes.md, "the credit table").
    """
    keys = [k if k else "(none)" for k in keys]
    if not keys:
        return ""
    shown = keys[:cap]
    more = (f"\n  ... and {len(keys) - len(shown)} more (the list is truncated; "
            f"prefer positions marked unexplored below)" if len(keys) > cap else "")
    return (f"ALL {len(keys)} sets measured so far -- every one of these is spent, "
            f"and a repeat is DISCARDED without being scored, costing you the "
            f"slot:\n  " + "; ".join(shown) + more + "\n\n")


def _coverage_block(keys, t, mutable=()) -> str:
    """Per position, which amino acids have been measured there -- and which
    positions are still untouched.

    The complement of `_measured_block` and the actionable half: a list of spent
    genomes tells the model what to avoid, whereas this tells it where room is
    left. Also scores-free, so it reports coverage and not credit.
    """
    import re as _re
    seen: dict[int, set] = {}
    for k in keys:
        if not k or k == "(none)":
            continue
        for tok in _re.split(r"[,\s;]+", k):
            mm = S._SUB_TOK.match(tok.strip()) if tok.strip() else None
            if mm:
                seen.setdefault(int(mm.group(1)), set()).add(mm.group(2).upper())
    pos = sorted(mutable) if mutable else list(range(t.length))
    if not pos:
        return ""
    rows, untouched, exhausted = [], [], []
    for i in pos:
        aas = seen.get(i)
        if not aas:
            untouched.append(i)
            continue
        wt = t.seq[i] if i < len(t.seq) else ""
        left = [a for a in AA20 if a not in aas and a != wt]
        if left:
            rows.append(f"{i}:{''.join(sorted(left))}")
        else:
            exhausted.append(i)
    # Phrased as what is LEFT, not what is spent: the spent half is already the
    # measured block above, and a list of remaining options is the only form of
    # this that a proposal can be read straight off.
    out = ("Single substitutions NOT yet measured, by position (the room that is "
           "left):\n  " + ("; ".join(rows) if rows else "(none)") + "\n")
    if untouched:
        out += ("  NEVER EDITED at all -- every amino acid is untried here: " +
                ", ".join(str(i) for i in untouched) + "\n")
    if exhausted:
        out += ("  EXHAUSTED as single edits (only combinations remain): " +
                ", ".join(str(i) for i in exhausted) + "\n")
    return out + "\n"


def iter_brief(t, window, k_mut, elite, m, gen, wt_ala=None, folded=False,
               tried=(), noise_sd: float = 0.0, wt_score=None,
               objective=None, measured=(), mutable=(), rec_ctx=None) -> str:
    a, b = window["a"], window["b"]
    note = (" The bindE column is each residue's current interface energy "
            "(more positive = it carries more binding)." if wt_ala else "")
    # The receptor side of the interface. The objective text asks the model to
    # match the chemistry across from each position; until this column existed
    # the brief never said what that chemistry was.
    faces_note = (" The 'faces' column lists the receptor residues each peptide "
                  "position contacts (one-letter + residue number), nearest "
                  "first -- that is the pocket a substitution at that position "
                  "has to suit." if rec_ctx else "")
    # `objective` is the explicit form; `folded` is kept as the legacy selector so
    # existing callers and tests are unaffected.
    obj = objective or OBJECTIVES["folded" if folded else "proxy"]
    units = obj["units"]
    # The in-loop fitness differs by evaluator, and the brief must describe the
    # REAL objective: the proxy threads onto the fixed native pose (edits cannot
    # change the fold), whereas the folded scorer re-folds the mutant (edits that
    # change folding/binding are exactly what is rewarded). Telling the model the
    # wrong one handicaps it versus the random-EA control, which reads no brief.
    scored_by, propose_mode = obj["scored_by"], obj["propose_mode"]
    # Tell the model the score is a MEASUREMENT. Without this it reads a 1 REU
    # lead as a result and spends the next generation polishing it -- which is
    # exactly what the first real-score run did (1bjr gen 2: six proposals, all
    # single substitutions at the incumbent's position 2).
    noise = ""
    if noise_sd > 0:
        noise = (
            f"IMPORTANT -- these scores are NOISY MEASUREMENTS, not exact values: "
            f"repeat measurement of the same sequence moves it by about "
            f"{noise_sd:.1f} {units} (see the +/-SE column). Two sets within ~"
            f"{2 * noise_sd:.1f} {units} of each other are TIED, not ranked. Do not "
            f"chase a lead smaller than that; look for an edit that should be "
            f"clearly better for a structural reason.\n\n")
    # The exploration floor. Pure local refinement around the incumbent is how the
    # loop stalled: >90% of offspring failed to beat the frontier they were shown
    # and the best genome was already found by generation 1 in 9 of 16 runs.
    n_new_pos = max(1, m // 2)
    elite_pos = sorted({int(i) for mm in elite for i, _ in mm["subs"]})
    pos_note = (f"Positions already on the frontier: "
                f"{', '.join(str(i) for i in elite_pos)}. " if elite_pos else "")
    return (
        f"Generation {gen}. Peptide length {t.length} (indices 0..{t.length-1}), "
        f"{t.n_contacts} interface contacts. On cyclization the residues in window "
        f"[{a},{b}] are HELD at their bound pose and keep the interface; residues "
        f"outside are released so the ring can close.\n\n"
        f"Per-residue table:{note}{faces_note}\n"
        f"{S._residue_table(t, wt_ala, rec_ctx=rec_ctx)}\n\n"
        # The wild-type is THE bar (every headline number is paired against
        # nomut), but it is often not in the elite -- naming it as "the row marked
        # (none)" then points at a row that is not on screen.
        f"{scored_by}\n\n"
        + (f"The wild-type peptide, unmodified, scores {wt_score:+.2f} {units}. "
           f"That is the bar every candidate is measured against.\n\n"
           if isinstance(wt_score, (int, float)) else
           "'(none)' below is the wild-type sequence, the bar to beat.\n\n")
        + f"Best substitution sets found so far:\n\n"
        f"{_elite_block(elite, noisy=noise_sd > 0, units=units)}\n\n"
        f"{_tried_block(tried, units=units)}"
        f"{_measured_block(measured)}"
        f"{_coverage_block(measured, t, mutable)}"
        f"{noise}"
        f"Study what the strong sets share and where the weak ones waste edits, "
        f"then propose {m} NEW candidate substitution set(s) that you expect to "
        f"score MORE NEGATIVE than the best above. Each set uses at most {k_mut} "
        f"substitution(s). {propose_mode}\n\n"
        f"Requirements for the {m} sets:\n"
        f"- Every set must be absent from the measured list above. That list is "
        f"COMPLETE, so this is checkable: check it.\n"
        f"- At least {n_new_pos} of them must edit a position that is NOT on the "
        f"frontier. {pos_note}The frontier may be sitting on a measurement "
        f"artifact, so part of the budget has to test elsewhere -- prefer "
        f"positions listed as never edited, or amino acid classes not yet tried "
        f"at a position.\n"
        f"- Spread the rest: refine the strong sets, but no two of your sets "
        f"should differ only in the amino acid at one shared position.\n\n"
        f"Output EXACTLY {m} line(s), each one candidate and nothing else:\n"
        f"<mut>pos:AA,pos:AA</mut>   (0-indexed position, one-letter amino acid), "
        f"or <mut>none</mut> for the wild-type."
    )


def parse_many(text: str, L: int, k_mut: int, m: int) -> list:
    """Parse every <mut>...</mut> tag into a list of genomes (each a list of
    (pos,AA)). Dedupes identical sets, keeps order, clamps to m. '<mut>none</mut>'
    contributes the empty genome []. Invalid tokens are dropped (rejection sampling,
    as Wang et al.)."""
    out, keys = [], set()
    for body in S._MUT.findall(text or ""):
        body = body.strip()
        if body.lower() in ("none", "", "-", "no", "nomut"):
            g = []
        else:
            g, seen = [], set()
            for tok in __import__("re").split(r"[,\s;]+", body):
                tok = tok.strip()
                mm = S._SUB_TOK.match(tok) if tok else None
                if not mm:
                    continue
                pos, aa = int(mm.group(1)), mm.group(2).upper()
                if 0 <= pos < L and aa in AA20 and pos not in seen:
                    seen.add(pos)
                    g.append((pos, aa))
                if len(g) >= k_mut:
                    break
        key = S.subs_key(g)
        if key in keys:
            continue
        keys.add(key)
        out.append(g)
        if len(out) >= m:
            break
    return out


def llm_offspring(llm, t, window, k_mut, elite, m, gen, wt_ala, meta, rng,
                  iface, max_retries=1, save_raw=False, folded=False,
                  known=(), tried=(), noise_sd: float = 0.0, wt_score=None,
                  objective=None, rec_ctx=None) -> list:
    """One LLM call returning up to m new genomes, each paired with the best elite
    recovered shown (the frontier it was asked to beat). Falls back to a random-EA
    child if the model declines/malforms, so a dead generation does not stall the
    loop (recorded in meta['llm_status'] counts).

    `known` is every genome key already in the population. Proposals that repeat
    one are DROPPED, not scored: a repeat costs the generation a proposal slot and
    teaches the search nothing (measured at 34-45% of all LLM proposals). When
    repeats leave the generation short, the model is re-asked ONCE with the
    repeats named -- a cheap, in-context correction -- before the EA fallback.

    `known` is now also SHOWN to the model (in full) rather than used only to
    filter afterwards. It was always available here; withholding it is what made
    the brief's "propose something not listed above" instruction unsatisfiable,
    because only ~26 of up to ~500 measured genomes were ever listed.
    """
    from lp2cp.enums import StepStatus

    best_rec = elite[0]["recovered"] if elite else None
    obj = objective or OBJECTIVES["folded" if folded else "proxy"]
    brief = iter_brief(t, window, k_mut, elite, m, gen, wt_ala=wt_ala, folded=folded,
                       tried=tried, noise_sd=noise_sd, wt_score=wt_score,
                       objective=obj, measured=known, mutable=iface,
                       rec_ctx=rec_ctx)
    # The system prompt has to describe the same fitness as the brief, or the model
    # reasons about a scorer it is not being graded by.
    system = getattr(S, obj["system_attr"], S.MUT_SYSTEM)
    msgs = [{"role": "user", "content": brief}]
    known = set(known)
    fresh, seen_keys = [], set()
    status = "malformed"
    for r in range(max_retries + 1):
        reply = llm.complete(system, msgs)
        meta["llm_calls"] += 1
        meta["llm_cost_usd"] = round(meta["llm_cost_usd"] + reply.cost_usd, 6)
        if reply.status == StepStatus.DECLINED:
            status = "declined"
            break
        if reply.status in (StepStatus.ERROR, StepStatus.BUDGET_CAP):
            status = reply.status.value.lower()
            # Keep the FIRST failure's text. Without it a row records only
            # `error: 80` with $0.00 cost, and the cause -- quota exhausted, a
            # bad key, a rate limit, a timeout -- is indistinguishable from the
            # outside; diagnosing job 119773 needed a fresh API probe hours
            # later to learn it was "no credits remaining". One short string
            # makes the row self-explanatory. First, not last, because later
            # calls in an outage report the same thing and the first is the
            # one with the original context.
            if not meta.get("llm_error_detail"):
                meta["llm_error_detail"] = (reply.detail or "")[:300]
            break
        if save_raw:
            meta.setdefault("llm_raw", []).append((reply.text or "")[:600])
        parsed = parse_many(reply.text, t.length, k_mut, m)
        repeats = []
        for g in parsed:
            key = S.subs_key(g)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key in known:
                repeats.append(key or "(none)")
                meta["n_dup_proposed"] = meta.get("n_dup_proposed", 0) + 1
                continue
            fresh.append(g)
        if parsed:
            status = "ok" if fresh else "all_duplicates"
        if len(fresh) >= m or r >= max_retries:
            break
        # Short this generation: say exactly why and ask again, once.
        if repeats:
            why = ("Those were already measured: " + ", ".join(repeats[:8]) +
                   ". Repeats are discarded. ")
        else:
            why = "Could not parse. "
        msgs += [{"role": "assistant", "content": reply.text},
                 {"role": "user", "content":
                  f"{why}Give {m - len(fresh)} DIFFERENT substitution set(s) that "
                  f"appear nowhere above, each as <mut>pos:AA,pos:AA</mut> (or "
                  f"<mut>none</mut>), one per line and nothing else."}]
    meta["status_counts"][status] = meta["status_counts"].get(status, 0) + 1
    genomes = fresh[:m]
    if not genomes:   # keep the loop moving with a random child this generation
        genomes = [g for g, _ in ea_offspring(elite, t, k_mut, iface, rng, 1)]
        meta["fallback_gens"] = meta.get("fallback_gens", 0) + 1
    return [(g, best_rec) for g in genomes]


# ---------------------------------------------------------------------------
# The evolutionary loop (shared by both arms).
# ---------------------------------------------------------------------------

def _seed_population(pop: Population, t, k_mut, iface, rng, n_seed, n_workers=1):
    """Initial pool: wild-type + random single/double mutants, scored under the
    budget (the new ones in parallel).

    `rng` MUST be the arm-independent seeding RNG (see `evolve`): every arm has to
    start from the SAME pool or the matched-budget comparison is confounded before
    an operator has proposed anything."""
    seeds = [[]]
    tried = {S.subs_key([])}
    guard = 0
    while len(seeds) < n_seed and guard < n_seed * 20:
        guard += 1
        kk = rng.randint(1, max(1, k_mut))
        pos = rng.sample(iface, min(kk, len(iface)))
        g = []
        for i in pos:
            wt = t.seq[i] if i < len(t.seq) else "A"
            g.append((i, rng.choice([a for a in AA20 if a != wt])))
        key = S.subs_key(_norm(g))
        if key in tried:
            continue
        tried.add(key)
        seeds.append(g)
    pop.evaluate_many(seeds, gen=0, n_workers=n_workers)


def evolve(arm, t, window, cache, *, k_mut, n_seed, n_gen, m_off, max_evals,
           elite_frac, relax_repeats, timeout, llm=None, wt_ala=None,
           max_retries=1, save_raw=False, n_workers=1, scorer=None,
           z_explore=1.0, z_commit=1.0, reassess=True):
    """Run one arm on one target. Returns (population, trajectory, meta). `scorer`
    is the pluggable evaluator (proxy by default; a BoltzScorer for real-score-in-
    loop). A GPU scorer saturates the one GPU, so its folds run serially."""
    iface = _iface_positions(t)
    # TWO RNGs, and the split is load-bearing.
    #   seed_rng is keyed WITHOUT the arm, so iter-ea and iter-llm* start from an
    #   IDENTICAL random pool and the matched-budget comparison is paired. Keying
    #   it on the arm (what this did before) handed each arm a different starting
    #   point: measured mean seed-pool gap between arms 7.66 REU, up to 19.7 on
    #   1jrr -- several times the arm difference the experiment is trying to read.
    #   op_rng is keyed WITH the arm, so the EA's operator draws stay independent
    #   of the LLM arm's fallback draws and each arm still replays deterministically.
    seed_rng = random.Random(int.from_bytes(hashlib.sha256(
        f"{t.target_id}|seed|{k_mut}|{n_seed}".encode()).digest()[:8], "big"))
    rng = random.Random(int.from_bytes(hashlib.sha256(
        f"{t.target_id}|{arm}|{k_mut}".encode()).digest()[:8], "big"))
    # Does the in-loop scorer re-fold the mutant? If so the LLM brief must describe
    # the folded objective, not the fixed-pose proxy. `folds` is the explicit flag;
    # fall back to is_gpu (BoltzScorer / the GPU fake both re-fold) if unset.
    folded = bool(getattr(scorer, "folds", getattr(scorer, "is_gpu", False)))
    noisy = bool(getattr(scorer, "stochastic", False))
    objective = scorer_objective(scorer)
    # The receptor side of each position, read ONCE per target (it parses the
    # PDB) and reused for every generation's brief. Only the LLM arms read it;
    # the EA arm reads no brief at all, so this costs the control nothing.
    rec_ctx = S.receptor_context(t) if arm in ITER_LLM_ARMS else None
    if arm in ITER_LLM_ARMS and rec_ctx is None:
        print(f"[seqmut-iter] {t.target_id:8s} receptor contact context "
              f"unavailable -- brief falls back to contact counts only",
              flush=True)
    if getattr(scorer, "is_gpu", False):
        n_workers = 1
    # THE KNOWN OPTIMUM, when the evaluator can supply one. An exactly solvable
    # fitness (the additive per-position PLL table) turns "which arm scored
    # better" into "which arm reached the optimum, and in how many evaluations" --
    # a regret with a real zero. Computed ONCE per (target, arm), outside the
    # budget, because reading it off the table is not an evaluation. No previous
    # in-loop fitness in this project could support this measurement.
    opt = None
    opt_fn = getattr(scorer, "optimum", None)
    if callable(opt_fn):
        try:
            # `iface` is exactly the set `_point_mutate` draws new positions
            # from, so the optimum is graded inside the operator's action space.
            opt = opt_fn(t, k_mut, window=window, mutable=iface)
        except Exception as exc:  # noqa: BLE001  a diagnostic must never fail a run
            print(f"[seqmut-iter] {t.target_id} optimum unavailable: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    pop = Population(t, window, cache, k_mut, relax_repeats, timeout, max_evals,
                    scorer=scorer, z_explore=z_explore, z_commit=z_commit)
    meta = {"llm_calls": 0, "llm_cost_usd": 0.0, "status_counts": {},
            "model_id": getattr(llm, "model_id", None) if llm else None,
            "effort": getattr(llm, "effort", None) if llm else None}

    if opt:
        meta["optimum"] = opt
        print(f"[seqmut-iter] {t.target_id:8s} {arm:12s} KNOWN OPTIMUM "
              f"fitness={opt['fitness']:+.4f} (wt {opt['wt_fitness']:+.4f}) "
              f"subs={opt['subs']} over {opt.get('n_mutable')} mutable "
              f"position(s) -- the search is graded by regret against this"
              + (f"; unrestricted the metric would pick {opt['free_subs']} at "
                 f"{opt['free_fitness']:+.4f}, which the operator cannot reach"
                 if opt.get("free_minus_constrained") else ""),
              flush=True)

    _seed_population(pop, t, k_mut, iface, seed_rng, n_seed, n_workers=n_workers)
    traj = [{"gen": 0, "evals": pop.evals, "best_recovered": pop.best_recovered(),
             "pop": len(pop.scored()), "phase": "seed",
             "regret": _regret(pop.best_recovered(), opt)}]

    for gen in range(1, n_gen + 1):
        if not pop.budget_left():
            break
        # Re-measure the incumbent BEFORE proposing against it. One unit of the
        # same budget both arms spend, and the only thing that stops a lucky draw
        # from owning the frontier for the rest of the run. `shift` is the live
        # readout of the optimizer's curse: positive = the leader got worse once
        # measured again.
        shift = None
        if reassess and noisy:
            shift = pop.reassess(pop.leader(), gen)
            if not pop.budget_left():
                break
        elite = pop.elite(elite_frac)
        if not elite:
            break
        frontier = elite[0]["recovered"]
        # MAX elite SEM, not mean/median, on purpose: the quantity feeds a "these
        # two are TIED" statement, and the measured failure is fixation on a
        # sub-noise lead, so over-declaring ties pushes the model toward the
        # exploration it was not doing. Erring the other way reproduces the bug.
        noise_sd = (max((m.get("sem", 0.0) for m in elite), default=0.0)
                    if noisy else 0.0)
        if arm in ITER_LLM_ARMS:
            elite_keys = {S.subs_key(m["subs"]) for m in elite}
            wt_m = pop.members.get(S.subs_key([]))
            children = llm_offspring(llm, t, window, k_mut, elite, m_off, gen,
                                     wt_ala, meta, rng, iface,
                                     max_retries=max_retries, save_raw=save_raw,
                                     folded=folded,
                                     known=pop.measured_keys(),
                                     tried=pop.tried(exclude_keys=elite_keys),
                                     noise_sd=noise_sd,
                                     wt_score=(wt_m or {}).get("recovered"),
                                     objective=objective, rec_ctx=rec_ctx)
        else:
            children = ea_offspring(elite, t, k_mut, iface, rng, m_off,
                                    known=set(pop.members))

        # scan the generation's NEW offspring in parallel (independent genomes);
        # budget + diagnostics are identical to the serial path.
        pre = set(pop.members)
        results = pop.evaluate_many([c for c, _ in children], gen, n_workers=n_workers)
        scored_this, beat_frontier, improved_parent, newly = 0, 0, 0, 0
        for child, parent_rec in children:
            key = S.subs_key(_norm(child))
            rec = results.get(key)
            if rec is None:
                continue
            if key not in pre:
                newly += 1
            scored_this += 1
            if rec < frontier - EPS:
                beat_frontier += 1
            if parent_rec is not None and rec < parent_rec - EPS:
                improved_parent += 1

        best_after = pop.best_recovered()
        bm = pop.best()
        traj.append({
            "gen": gen, "evals": pop.evals, "pop": len(pop.scored()),
            "frontier_shown": round(frontier, 3),
            "best_recovered": None if best_after is None else round(best_after, 3),
            # 2604.19440 diagnostics, symmetric across arms:
            "breakthrough": bool(best_after is not None and best_after < frontier - EPS),
            "n_proposed": len(children), "n_scored": scored_this, "n_new": newly,
            # fraction of scored offspring that advanced the shown frontier:
            "beat_frontier_rate": round(beat_frontier / scored_this, 3) if scored_this else 0.0,
            # fraction that beat their own parent (local refinement):
            "refine_rate": round(improved_parent / scored_this, 3) if scored_this else 0.0,
            # noise bookkeeping: how far the re-measured leader moved (the live
            # optimizer's-curse readout), and how well-measured the committed best is
            "incumbent_shift": shift,
            "best_sem": None if bm is None else round(bm.get("sem", 0.0), 3),
            "best_n_obs": None if bm is None else len(bm.get("obs") or []),
            "n_dup_proposed": meta.get("n_dup_proposed", 0),
            # distance from the KNOWN optimum (None when the fitness has no
            # closed-form optimum); 0.0 means this arm has solved the target
            "regret": _regret(best_after, opt),
        })
    if llm is not None:
        # True only if the client still had reasoning_effort enabled after every
        # call -- see the `effort_sent` note in `_finalist_row`.
        meta["effort_sent"] = bool(getattr(llm, "_send_effort", True))
        if meta.get("effort") and not meta["effort_sent"]:
            print(f"[seqmut-iter] {t.target_id:8s} {arm:12s} WARNING: the API "
                  f"REJECTED reasoning_effort={meta['effort']}; this arm ran at "
                  f"the model's default effort and is NOT a high-effort arm",
                  flush=True)
    return pop, traj, meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_llm_clients(llm_arms, *, backend, model, effort, hi_effort,
                     max_spend_usd, max_tokens):
    """Build one client per EFFORT LEVEL and divide the spend cap between them.

    Returns `(efforts, llm_by_effort)` where `efforts` maps an effort level to
    the arms that run at it. Arms sharing an effort share a client, because the
    client owns the ledger and the ledger is what the cap is about.

    THE CAP IS PER CLIENT. Handing each client the full `--max-spend-usd` would
    make the run's real ceiling a multiple of the one asked for, against a
    convention (docs/seqmut_notes.md) where every spend figure is per-run. Worse,
    a cap that IS reached does not stop the run: `llm_offspring` substitutes a
    random-EA child on BUDGET_CAP, so an overspent arm silently becomes its own
    control for the rest of the run, in target order. The cap is therefore split
    and every share printed.

    Kept separate from `run` so the division is testable without a manifest, an
    output path, or a search.
    """
    efforts: dict[str, list] = {}
    for a in llm_arms:
        efforts.setdefault(hi_effort if a == HI_EFFORT_ARM else effort,
                           []).append(a)
    llm_by_effort = {}
    if not llm_arms:
        return efforts, llm_by_effort
    if not backend:
        raise SystemExit(f"[seqmut-iter] LLM arms {llm_arms} need --backend/--model")
    # Imported here, not at module scope, to match `run`: `lnr.run` pulls in the
    # LLM backends, and the EA-only path must stay importable without them.
    from lnr import run as R
    share = max_spend_usd / len(efforts) if max_spend_usd else 0.0
    for eff, arms_here in efforts.items():
        llm_by_effort[eff] = R.make_llm(backend, model, eff, share, max_tokens)
        print(f"[seqmut-iter] llm backend={backend} model={model} "
              f"effort={eff} arms={arms_here} cap=${share:.2f} "
              f"({len(efforts)} ledger(s), ${max_spend_usd:.2f} total)",
              flush=True)
    return efforts, llm_by_effort


def _probe_call_cost(llm_by_effort, efforts, targets, args) -> int:
    """Price one call per effort level, then print the generations each arm can
    afford. Exits without searching.

    This exists because `--hi-n-gen` cannot be guessed. A high-effort call costs
    more than a low-effort one by an amount that depends on the model and on how
    much it chooses to think, so equal generations is NOT equal spend -- and the
    consequence of overspending is not a clean stop: on BUDGET_CAP
    `llm_offspring` substitutes a random-EA child, so the arm degrades into its
    own control partway through, in target order. Measuring one call costs about
    a tenth of a cent and makes the dollar-match arithmetic checkable before a
    run is submitted, which is the convention the rest of this module follows.
    """
    from lp2cp.enums import StepStatus
    if not targets:
        print("[probe] no targets loaded -- nothing to price", flush=True)
        return 1
    t = targets[0]
    win = {"a": 0, "b": max(0, t.length - 1), "retention": 1.0, "clean": True}
    # A brief of realistic SIZE, since input tokens are most of the cost: a
    # full-looking elite block, failure memory and measured list, not an empty
    # generation-1 brief that would price the run too low.
    elite = [{"subs": ((i % t.length, "W"),), "recovered": -1.5 - 0.01 * i,
              "sem": 0.0, "obs": [-1.5], "gen": 1} for i in range(12)]
    measured = [S.subs_key([(i % t.length, a)])
                for i, a in enumerate(AA20 * 8)]
    rec_ctx = S.receptor_context(t)
    print(f"[probe] pricing on {t.target_id} (L={t.length}), m_off={args.m_off}",
          flush=True)
    rows = []
    for effort, arms_here in efforts.items():
        llm = llm_by_effort[effort]
        brief = iter_brief(t, win, args.k_mut, elite, args.m_off, 7,
                           objective=OBJECTIVES["pll"], tried=elite[6:],
                           measured=measured, mutable=list(range(t.length)),
                           rec_ctx=rec_ctx, wt_score=-1.2)
        reply = llm.complete(getattr(S, "MUT_SYSTEM_PLL", S.MUT_SYSTEM),
                             [{"role": "user", "content": brief}])
        ok = reply.status == StepStatus.OK
        sent = bool(getattr(llm, "_send_effort", True))
        n_parsed = len(parse_many(reply.text, t.length, args.k_mut, args.m_off))
        rows.append((effort, arms_here, reply.cost_usd, ok, sent, n_parsed,
                     len(brief)))
        print(f"[probe] effort={effort:6s} status={reply.status.value:10s} "
              f"${reply.cost_usd:.6f}/call  brief={len(brief)}c  "
              f"parsed={n_parsed}/{args.m_off} proposals  "
              f"reasoning_effort_accepted={sent}"
              + ("" if ok else f"  detail={(reply.detail or '')[:160]}"),
              flush=True)
    n_t = len(targets)
    cap = args.max_spend_usd
    share = cap / len(efforts) if cap else 0.0
    print(f"\n[probe] {n_t} target(s), cap ${cap:.2f} split ${share:.2f} per "
          f"effort ledger. Calls per generation ~1 (plus up to one re-ask).",
          flush=True)
    for effort, arms_here, cost, ok, sent, _, _ in rows:
        if not ok or cost <= 0:
            print(f"[probe] effort={effort}: cannot price (status not OK or "
                  f"$0 cost) -- do NOT submit this arm", flush=True)
            continue
        afford = int(share / (cost * n_t)) if share else 0
        at_n_gen = cost * n_t * args.n_gen
        print(f"[probe] effort={effort:6s} arms={arms_here}: "
              f"${cost * n_t:.4f}/generation across all targets -> "
              f"--n-gen {args.n_gen} would cost ${at_n_gen:.2f}; "
              f"${share:.2f} affords n_gen={afford}"
              + (f"  <-- use --hi-n-gen {afford}" if HI_EFFORT_ARM in arms_here
                 else ""), flush=True)
    print("[probe] no search was run (--probe-call-cost)", flush=True)
    return 0


def _load_windows(path: Path) -> dict:
    """Reuse the committed window per target from an existing seqmut rows file, so
    the iterative job scores on the SAME window as the single-shot run (and the
    nomut proxy baseline is identical). `committed_clean_window` uses randomized
    clash restarts, so recomputing could drift the window; reading it does not."""
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cw = r.get("committed_window")
        if cw and r.get("target_id") not in out:
            out[r["target_id"]] = {"a": int(cw[0]), "b": int(cw[1]),
                                   "retention": r.get("contact_retention", 0.0),
                                   "clean": r.get("window_clean")}
    return out


def _finalist_row(t, arm, window, cache, pop, traj, meta, k_mut, relax_repeats,
                  timeout, closure_arm):
    """Best genome -> a standard seqmut row (so boltz_validate mut-held folds it),
    augmented with iterative metadata and the trajectory summary."""
    best = pop.best()
    subs = list(best["subs"]) if best else []
    a, b = window["a"], window["b"]
    name = scorer_name(pop.scorer)
    if name != "proxy":
        # The in-loop number is already the fitness we selected on; do NOT
        # recompute the Rosetta proxy. Build the row from the best genome so the
        # believed-number report re-folds it with an INDEPENDENT draw.
        obj = scorer_objective(pop.scorer)
        best_rec = pop.best_recovered()
        # Whether the in-loop fitness may be written into
        # `recovered_binding_energy` AT ALL. The Boltz fitness is a Rosetta
        # energy, so it belongs there; the PLL fitness is nats/residue, and
        # putting it there would silently pool two different units in the same
        # report column -- the exact class of mistake that produced this
        # project's "interface dG measures size" result. It stays out, under its
        # own key, and the report re-folds the finalist for the REU number.
        is_reu = bool(getattr(pop.scorer, "is_reu", True))
        row = {"target_id": t.target_id, "mut_arm": arm, "length": t.length,
               "k_mut": k_mut, "subs": [[int(i), aa.upper()] for i, aa in subs],
               "subs_key": S.subs_key(subs), "committed_window": [a, b],
               "window_clean": window.get("clean"),
               "contact_retention": round(window.get("retention", 0.0), 4),
               "native_dG": None, "energy_retention": None,
               "recovered_binding_energy": best_rec if is_reu else None,
               "inloop_evaluator": name,
               "inloop_fitness": best_rec,
               "inloop_units": obj["units"]}
        if name == "boltz":
            row["inloop_dG"] = best_rec          # the pre-existing key, unchanged
        if name == "pll" and best_rec is not None:
            # the metric in ITS OWN orientation (higher = better), beside the
            # negated fitness, so no consumer has to guess which way round it is
            row["inloop_pll"] = round(-float(best_rec), 6)
            row["inloop_metric"] = getattr(pop.scorer, "metric", None)
            row["inloop_scope"] = getattr(pop.scorer, "scope", None)
    else:
        row = S.score_target(t, arm, window, cache, k_mut, relax_repeats, timeout,
                             subs_override=subs)
    row["closure_arm"] = closure_arm
    if meta.get("model_id"):
        row["model_id"] = meta["model_id"]
    # arm accounting + the "good optimizer" trajectory summary
    gens = [d for d in traj if d.get("phase") != "seed"]
    shifts = [d["incumbent_shift"] for d in gens if d.get("incumbent_shift") is not None]
    row["iter"] = {
        "n_evals": pop.evals, "n_gen_run": len(gens), "n_seed": traj[0]["pop"],
        "best_gen": best["gen"] if best else None,
        "best_recovered": pop.best_recovered(),
        # how well-measured the committed finalist is, and what re-measuring the
        # incumbent kept correcting -- the optimizer's-curse accounting
        "best_sem": round(best.get("sem", 0.0), 3) if best else None,
        "best_n_obs": len(best.get("obs") or []) if best else None,
        "n_reassess": pop.reassessments,
        # grading against the fitness's KNOWN optimum, when it has one. `regret`
        # 0.0 = this arm found the exact best genome available under its Hamming
        # budget; `evals_to_optimum` is how much of the budget that took.
        "optimum": meta.get("optimum"),
        "regret": _regret(pop.best_recovered(), meta.get("optimum")),
        "evals_to_optimum": next(
            (d["evals"] for d in traj
             if d.get("regret") is not None and d["regret"] <= 0.0), None),
        "mean_incumbent_shift": round(sum(shifts) / len(shifts), 3) if shifts else None,
        "n_dup_proposed": meta.get("n_dup_proposed", 0),
        "seed_best": traj[0]["best_recovered"],
        "breakthrough_rate": round(sum(d["breakthrough"] for d in gens) / len(gens), 3) if gens else 0.0,
        "mean_beat_frontier": round(sum(d["beat_frontier_rate"] for d in gens) / len(gens), 3) if gens else 0.0,
        "mean_refine_rate": round(sum(d["refine_rate"] for d in gens) / len(gens), 3) if gens else 0.0,
        "llm_calls": meta["llm_calls"], "llm_cost_usd": meta["llm_cost_usd"],
        "status_counts": meta["status_counts"], "fallback_gens": meta.get("fallback_gens", 0),
        "llm_error_detail": meta.get("llm_error_detail", ""),
        "effort": meta.get("effort"),
        # Whether `reasoning_effort` was actually ACCEPTED by the API. The
        # OpenAI client drops the parameter and silently retries without it on a
        # 400 (openai_api.py), so an arm labelled high-effort can run at the
        # model's default while every other column looks normal -- the hi-effort
        # arm would then be a duplicate of the low-effort arm wearing a
        # different name, and "effort does not matter" is exactly the plausible
        # conclusion it would produce. Recorded per row so the grader can gate
        # on it instead of trusting the label.
        "effort_sent": meta.get("effort_sent"),
    }
    if meta.get("model_id"):
        row["llm_cost_usd"] = meta["llm_cost_usd"]
        row["llm_status"] = ("ok" if meta["status_counts"].get("ok") else
                             next(iter(meta["status_counts"]), "malformed"))
        # --save-raw collected every reply; persist them or the flag does nothing
        # (seqmut.py's single-shot path already writes llm_raw, this one dropped
        # it). Needed to check the brief is actually obeyed -- that proposals stop
        # repeating known sets and some of them leave the frontier's positions.
        if meta.get("llm_raw"):
            row["llm_raw"] = meta["llm_raw"]
    return row


def run(args):
    from lnr import env as E
    from lnr import run as R

    targets = E.load_manifest(args.manifest, args.pdb_dir)
    targets.sort(key=lambda t: t.target_id)
    want = [s.strip() for s in getattr(args, "targets", "").split(",") if s.strip()]
    if want:
        targets = [t for t in targets if t.target_id in set(want)]
    if args.limit:
        targets = targets[: args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ITER_ARMS:
            raise SystemExit(f"[seqmut-iter] unknown arm {a!r}; allowed: {ITER_ARMS}")
    llm_arms = [a for a in arms if a in ITER_LLM_ARMS]
    # `run` is called directly by tests with a SimpleNamespace, so every new
    # option is read defensively -- the same reason `evaluator` is below.
    hi_effort = getattr(args, "hi_effort", "high")
    hi_n_gen = getattr(args, "hi_n_gen", 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traj_path = Path(args.traj) if args.traj else out.parent / "seqmut_iter_traj.jsonl"
    cache = S.ScanCache(Path(args.scan_cache) if args.scan_cache
                        else out.parent / "seqmut_iter_scans.jsonl")
    if args.seed_wt:
        print(f"[seqmut-iter] seeded {cache.seed_wt(Path(args.seed_wt))} WT scans "
              f"from {args.seed_wt}", flush=True)
    for sf in [s.strip() for s in args.seed_scans.split(",") if s.strip()]:
        print(f"[seqmut-iter] seeded {cache.seed_scans(Path(sf))} prior scans "
              f"from {sf}", flush=True)

    windows = _load_windows(Path(args.seed_windows)) if args.seed_windows else {}
    if windows:
        print(f"[seqmut-iter] reusing {len(windows)} committed windows from "
              f"{args.seed_windows}", flush=True)

    efforts, llm_by_effort = make_llm_clients(
        llm_arms, backend=args.backend, model=args.model, effort=args.effort,
        hi_effort=hi_effort, max_spend_usd=args.max_spend_usd,
        max_tokens=args.max_tokens)
    llm = llm_by_effort.get(args.effort)

    if getattr(args, "probe_call_cost", False):
        return _probe_call_cost(llm_by_effort, efforts, targets, args)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["target_id"], r["mut_arm"],
                              r.get("model_id"), r.get("k_mut")))
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[seqmut-iter] {len(targets)} targets, arms={arms}, k_mut={args.k_mut}, "
          f"n_seed={args.n_seed}, n_gen={args.n_gen}, m_off={args.m_off}, "
          f"max_evals={args.max_evals}, {len(done)} rows done", flush=True)

    # The pluggable in-loop evaluator. proxy (default) = pose-preserving Rosetta
    # retention; boltz = real Boltz x4 -> Rosetta dG, shared across arms/targets
    # (cached by variant_id) so the matched budget and cross-arm reuse hold.
    scorer = None
    if getattr(args, "evaluator", "proxy") == "pll":
        from lnr.pll_inloop import PLLScorer
        scorer = PLLScorer(
            metric=args.pll_metric, scope=args.pll_scope,
            model_name=args.pll_model, device=args.pll_device,
            pssm_dir=(args.pll_pssm_dir or out.parent / "pssm"),
            cache_path=(args.pll_cache or out.parent / "seqmut_iter_pll_scores.jsonl"))
        print(f"[seqmut-iter] evaluator=pll metric={args.pll_metric} "
              f"scope={args.pll_scope} model={args.pll_model} "
              f"(deterministic, CPU; the optimum is known in closed form)",
              flush=True)
    if getattr(args, "evaluator", "proxy") == "boltz":
        from lnr.boltz_inloop import BoltzScorer
        scorer = BoltzScorer(
            cache_path=(args.boltz_cache or out.parent / "seqmut_iter_boltz_cache.jsonl"),
            work=(args.boltz_work or out.parent / "boltz_inloop"),
            replicas=args.boltz_replicas, seed=args.boltz_seed)
        print(f"[seqmut-iter] evaluator=boltz replicas={args.boltz_replicas} "
              f"seed={args.boltz_seed} (real-score-in-loop; GPU)", flush=True)

    model_id = args.model if llm_arms else None
    fh = out.open("a")
    tfh = traj_path.open("a")
    for t in targets:
        # Commit the window ONCE per target (backbone-only closure is mutation-
        # invariant). Reuse the single-shot window if we have it; else compute it.
        if t.target_id in windows:
            win = windows[t.target_id]
        else:
            camp = R.run_campaign(args.closure_arm, t, args.budget, llm=None,
                                  max_seconds=args.max_seconds)
            win = S.committed_clean_window(t, camp["history"],
                                           max_seconds=args.max_seconds)
        if win is None:
            print(f"[seqmut-iter] {t.target_id:8s} no closing window -- skipped",
                  flush=True)
            continue
        # Let the evaluator pay its one-time per-target setup and declare the
        # target usable BEFORE any budget is spent. Without this a scorer that
        # cannot build a reference pose returns None for every genome, the arm
        # burns its whole seed budget, and the log is indistinguishable from a
        # search that simply found nothing.
        prep = getattr(scorer, "prepare", None)
        if callable(prep):
            why = prep(t, win)
            if why:
                print(f"[seqmut-iter] {t.target_id:8s} evaluator "
                      f"{scorer_name(scorer)} cannot score this target -- "
                      f"skipped: {why}", flush=True)
                continue
        wt = cache.get(t.target_id, [])
        wt_ala = (wt or {}).get("per_residue_ddG_ala")
        # The -ala arm needs the WT per-residue interface energies. In proxy mode
        # the nomut scan already populated them; under the Boltz evaluator nothing
        # Rosetta-scans the WT, so without this the -ala arm would silently see no
        # bindE column and collapse to the plain iter-llm arm. Compute it once
        # (pose-preserving WT alanine scan, cheap) only when that arm is requested.
        if wt_ala is None and "iter-llm-ala" in arms:
            scan = S.mutant_scan(t, [], cache, args.relax_repeats,
                                 args.per_target_timeout)
            wt_ala = (scan or {}).get("per_residue_ddG_ala")
            print(f"[seqmut-iter] {t.target_id:8s} WT alanine scan for -ala arm: "
                  f"{'ok' if wt_ala else 'unavailable'}", flush=True)

        # A matched nomut row per target (the pairing base the report folds as
        # N_cyc). Under the Boltz evaluator we DON'T compute the Rosetta proxy for
        # nomut -- the report re-folds it -- so we emit just the foldable stub.
        if (t.target_id, "nomut", None, args.k_mut) not in done:
            if scorer is not None and scorer_name(scorer) != "proxy":
                nrow = {"target_id": t.target_id, "mut_arm": "nomut",
                        "length": t.length, "k_mut": args.k_mut, "subs": [],
                        "subs_key": "", "committed_window": [win["a"], win["b"]],
                        "window_clean": win.get("clean"),
                        "contact_retention": round(win.get("retention", 0.0), 4),
                        "native_dG": None, "recovered_binding_energy": None,
                        "inloop_evaluator": scorer_name(scorer),
                        "closure_arm": args.closure_arm}
                # A deterministic, free evaluator CAN score the wild-type, and the
                # wild-type is the bar every headline number is paired against --
                # so record it rather than leaving the pairing base empty.
                if not getattr(scorer, "is_gpu", False):
                    f0, r0 = scorer.score(t, [], win)
                    nrow["inloop_fitness"] = f0
                    nrow["inloop_units"] = scorer_objective(scorer)["units"]
                    if f0 is not None and scorer_name(scorer) == "pll":
                        nrow["inloop_pll"] = round(-float(f0), 6)
            else:
                nrow = S.score_target(t, "nomut", win, cache, args.k_mut,
                                      args.relax_repeats, args.per_target_timeout)
                nrow["closure_arm"] = args.closure_arm
            fh.write(json.dumps(nrow) + "\n"); fh.flush()
            print(f"[seqmut-iter] {t.target_id:8s} nomut        "
                  f"win=[{win['a']},{win['b']}] recovered={nrow.get('recovered_binding_energy')}",
                  flush=True)

        for arm in arms:
            if (t.target_id, arm, model_id if arm in ITER_LLM_ARMS else None,
                    args.k_mut) in done:
                continue
            t0 = time.time()
            # The hi-effort arm runs its own client (its own effort, its own
            # ledger) and its own generation count, which is how the two LLM
            # arms are matched on DOLLARS instead of on generations.
            arm_effort = hi_effort if arm == HI_EFFORT_ARM else args.effort
            arm_llm = llm_by_effort.get(arm_effort)
            arm_n_gen = (hi_n_gen if arm == HI_EFFORT_ARM and hi_n_gen
                         else args.n_gen)
            pop, traj, meta = evolve(
                arm, t, win, cache, k_mut=args.k_mut, n_seed=args.n_seed,
                n_gen=arm_n_gen, m_off=args.m_off, max_evals=args.max_evals,
                elite_frac=args.elite_frac, relax_repeats=args.relax_repeats,
                timeout=args.per_target_timeout,
                llm=(arm_llm if arm in ITER_LLM_ARMS else None),
                wt_ala=(wt_ala if arm == "iter-llm-ala" else None),
                max_retries=args.max_retries, save_raw=args.save_raw,
                n_workers=args.n_workers, scorer=scorer,
                z_explore=getattr(args, "z_explore", 1.0),
                z_commit=getattr(args, "z_commit", 1.0),
                reassess=not getattr(args, "no_reassess", False))
            for d in traj:
                d2 = dict(d, target_id=t.target_id, arm=arm, model_id=meta.get("model_id"))
                tfh.write(json.dumps(d2) + "\n")
            tfh.flush()
            row = _finalist_row(t, arm, win, cache, pop, traj, meta, args.k_mut,
                                args.relax_repeats, args.per_target_timeout,
                                args.closure_arm)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            it = row["iter"]
            print(f"[seqmut-iter] {t.target_id:8s} {arm:12s} "
                  f"evals={it['n_evals']} gens={it['n_gen_run']} "
                  f"best={it['best_recovered']} +/-{it['best_sem']} "
                  f"(n={it['best_n_obs']}, seed {it['seed_best']}) "
                  f"bt_rate={it['breakthrough_rate']} beat={it['mean_beat_frontier']} "
                  f"reassess={it['n_reassess']}/shift={it['mean_incumbent_shift']} "
                  f"dup={it['n_dup_proposed']} "
                  f"subs={row.get('subs')} ${it['llm_cost_usd']:.4f} "
                  f"{round(time.time()-t0)}s", flush=True)
    fh.close()
    tfh.close()
    cache.close()
    if scorer is not None:
        scorer.close()
    _summarize(out, traj_path)
    print("[seqmut-iter] done", flush=True)


# ---------------------------------------------------------------------------
# Report: finalists vs nomut, and the matched-budget LLM-vs-EA control.
# ---------------------------------------------------------------------------

def _summarize(out: Path, traj_path: Path = None):
    rows = [json.loads(l) for l in Path(out).read_text().splitlines() if l.strip()]
    _summarize_rows(rows)


def report_dir(d: Path):
    """Aggregate every seqmut_iter*_rows.jsonl in a directory (the EA control job
    and the LLM-arm job each write their own), dedup on (arm_label, target), and
    print the combined table -- so the KEY CONTROL (LLM vs random-EA) reads out
    across the sibling jobs."""
    rows = []
    for f in sorted(Path(d).glob("seqmut_iter*_rows.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen = {}
    for r in rows:
        seen[(S._arm_label(r), r.get("target_id"))] = r
    print(f"[seqmut-iter] aggregated {len(rows)} rows -> {len(seen)} (arm,target) "
          f"cells from {d}")
    _summarize_rows(list(seen.values()))


def _summarize_pll(rows):
    """The report for a fitness measured in nats, not REU -- and graded against a
    KNOWN optimum.

    This table answers a question the REU table cannot even ask. The columns that
    matter are `regret` (distance from the exact best <= k_mut substitution set,
    0 = solved) and `solved` (the fraction of targets where the arm reached it).
    An arm at regret 0 everywhere has not won the experiment -- it has shown the
    INSTRUMENT IS SATURATED: the fitness is a sum of independent per-position
    terms, so the optimum is the top-k per-position gains and any hill-climber
    finds it. When both arms read 100% solved, the LLM-vs-EA comparison at this
    k_mut is measuring nothing, and the next move is a non-separable objective,
    not more budget.
    """
    pll = [r for r in rows if r.get("inloop_evaluator") == "pll"
           and r.get("inloop_fitness") is not None]
    if not pll:
        return
    by = {}
    for r in pll:
        by.setdefault(S._arm_label(r), {})[r.get("target_id")] = r
    print("\n" + "=" * 110)
    print("SEQMUT-ITER -- POSE-CONDITIONED LIKELIHOOD fitness (nats/residue, "
          "NEGATED so more negative = better).")
    print("regret = distance from the EXACT optimum under the Hamming budget; "
          "0.0000 = the arm found it. solved = fraction of targets at regret 0.")
    print("=" * 110)
    print(f"{'arm':30s} {'n':>3s} {'meanFit':>9s} {'meanWT':>9s} {'gainVsWT':>9s} "
          f"{'meanOpt':>9s} {'regret':>8s} {'solved':>7s} {'evals2opt':>10s} "
          f"{'evals':>6s} {'cost$':>8s}")
    wt_of = {r["target_id"]: r.get("inloop_fitness") for r in rows
             if r.get("mut_arm") == "nomut" and r.get("inloop_fitness") is not None}
    for label in sorted(by):
        cells = list(by[label].values())
        its = [c.get("iter") or {} for c in cells]
        fit = [c["inloop_fitness"] for c in cells]
        opts = [(i.get("optimum") or {}).get("fitness") for i in its]
        opts = [o for o in opts if o is not None]
        regs = [i.get("regret") for i in its if i.get("regret") is not None]
        wts = [wt_of.get(c["target_id"]) for c in cells]
        wts = [w for w in wts if w is not None]
        e2o = [i.get("evals_to_optimum") for i in its
               if i.get("evals_to_optimum") is not None]
        mf = sum(fit) / len(fit)
        mw = sum(wts) / len(wts) if wts else float("nan")
        ev = sum(i.get("n_evals", 0) for i in its) / len(its) if its else 0
        co = sum(i.get("llm_cost_usd", 0.0) or 0.0 for i in its)
        # The nomut baseline carries no loop and therefore no optimum or regret.
        # Printing nan there reads as a broken computation rather than as a
        # column that does not apply, so it is dotted out.
        mo = f"{sum(opts)/len(opts):9.4f}" if opts else f"{'.':>9s}"
        mr = f"{sum(regs)/len(regs):8.4f}" if regs else f"{'.':>8s}"
        sv = (f"{sum(1 for r in regs if r <= 0.0)/len(regs):7.0%}" if regs
              else f"{'.':>7s}")
        me = f"{sum(e2o)/len(e2o):10.1f}" if e2o else f"{'.':>10s}"
        print(f"{label:30s} {len(cells):3d} {mf:9.4f} {mw:9.4f} "
              f"{mf - mw:+9.4f} {mo} {mr} {sv} {me} {ev:6.0f} {co:8.3f}")
    # Where the operator's ACTION SPACE, not the search, is the binding
    # constraint: the graded optimum is restricted to the positions the operator
    # may touch, so a target whose unrestricted optimum is better is one where no
    # amount of searching could have reached the metric's own best edit. Reported
    # because it changes what a regret of 0 means on that target.
    gaps = [(tid, (c.get("iter") or {}).get("optimum") or {})
            for lab, cs in by.items() if lab.split(":")[0] != "nomut"
            for tid, c in cs.items()]
    gaps = [(tid, o) for tid, o in gaps if o.get("free_minus_constrained")]
    if gaps:
        print(f"\n  ACTION SPACE binds on {len(gaps)} target(s): the metric's "
              f"unrestricted best edit lies outside the positions the operator "
              f"may change, so regret there is a bound, not a search failure.")
        for tid, o in sorted(gaps)[:8]:
            print(f"    {tid:8s} graded opt {o['subs']} {o['fitness']:+.4f} vs "
                  f"free {o['free_subs']} {o['free_fitness']:+.4f} "
                  f"(delta {o['free_minus_constrained']:+.4f})")

    # The matched-budget control, on the fitness the arms were actually selected
    # on. Reported over SHARED targets only, so it is paired.
    # (not a comprehension with a shadowed name: `cells` and `row` stay distinct
    # so the flattening is readable)
    ea = {}
    for lab, cells in by.items():
        if lab.split(":")[0] == ITER_EA_ARM:
            ea.update(cells)
    for label in sorted(by):
        if label.split(":")[0] not in ITER_LLM_ARMS:
            continue
        shared = sorted(set(ea) & set(by[label]))
        if not shared:
            continue
        d = [by[label][k]["inloop_fitness"] - ea[k]["inloop_fitness"] for k in shared]
        rl = [(by[label][k].get("iter") or {}).get("regret") for k in shared]
        re_ = [(ea[k].get("iter") or {}).get("regret") for k in shared]
        rl = [x for x in rl if x is not None]
        re_ = [x for x in re_ if x is not None]
        print(f"\n  CONTROL {label} vs {ITER_EA_ARM} on {len(shared)} shared "
              f"target(s): mean delta fitness {sum(d)/len(d):+.4f} nats "
              f"(negative = the LLM operator wins)")
        if rl and re_:
            print(f"    mean regret: LLM {sum(rl)/len(rl):.4f} vs "
                  f"EA {sum(re_)/len(re_):.4f}"
                  + ("   -- BOTH AT 0: the objective is separable and this "
                     "comparison is saturated, not won"
                     if max(rl) <= 0.0 and max(re_) <= 0.0 else ""))


def _summarize_rows(rows):
    _summarize_pll(rows)
    by = {}
    for r in rows:
        by.setdefault(S._arm_label(r), {})[r.get("target_id")] = r
    # Per-target CEILING: the WT native interface dG (retention=1 -- all interface
    # energy kept). The proxy recovered = retention x native_dG approaches it; a
    # mutation that STRENGTHENS native_dG can exceed it, so it is a reference bar,
    # not an absolute cap. Always reported next to recovered so a number is
    # interpretable [[seqmut-iter-report-ceiling]].
    ceil = {r["target_id"]: r.get("native_dG") for r in rows
            if r.get("mut_arm") == "nomut" and r.get("native_dG") is not None}
    print("\n" + "=" * 110)
    print("SEQMUT-ITER -- recovered binding energy (retention x native_dG), REU; "
          "more negative = stronger. CEIL = WT native_dG at retention=1.")
    print("=" * 110)
    print(f"{'arm':30s} {'n':>3s} {'meanRecov':>10s} {'meanSEM':>8s} {'meanCeil':>9s} "
          f"{'%ceil':>6s} {'bestGen':>8s} {'btRate':>7s} {'beatFr':>7s} "
          f"{'refine':>7s} {'shift':>7s} {'dup':>5s} {'evals':>6s} {'cost$':>8s}")
    for label in sorted(by):
        cells = list(by[label].values())
        v = [c for c in cells if c.get("recovered_binding_energy") is not None]
        if not v:
            # A non-REU in-loop fitness deliberately leaves this column empty
            # rather than pooling two units in it; the number is in the table
            # above, and the REU number comes from re-folding the finalist.
            ev = {c.get("inloop_evaluator") for c in cells} - {None}
            why = ("(fitness is not REU -- see the table above; the REU number "
                   "comes from boltz_validate --mode mut-held)"
                   if ev and all(e == "pll" for e in ev) else "(no scored rows)")
            print(f"{label:30s} {len(cells):3d}  {why}")
            continue
        mr = sum(c["recovered_binding_energy"] for c in v) / len(v)
        cv = [ceil[c["target_id"]] for c in v if ceil.get(c["target_id"]) is not None]
        mc = sum(cv) / len(cv) if cv else float("nan")
        pc = (100.0 * mr / mc) if cv and mc else float("nan")
        its = [c.get("iter") for c in v if c.get("iter")]
        if its:
            bt = sum(i["breakthrough_rate"] for i in its) / len(its)
            bf = sum(i["mean_beat_frontier"] for i in its) / len(its)
            rf = sum(i["mean_refine_rate"] for i in its) / len(its)
            ev = sum(i["n_evals"] for i in its) / len(its)
            co = sum(i["llm_cost_usd"] for i in its)
            bg = sum((i["best_gen"] or 0) for i in its) / len(its)
            sems = [i.get("best_sem") for i in its if i.get("best_sem") is not None]
            se = sum(sems) / len(sems) if sems else 0.0
            shs = [i.get("mean_incumbent_shift") for i in its
                   if i.get("mean_incumbent_shift") is not None]
            sh = f"{sum(shs)/len(shs):+7.2f}" if shs else f"{'.':>7s}"
            dup = sum(i.get("n_dup_proposed", 0) or 0 for i in its)
            print(f"{label:30s} {len(v):3d} {mr:10.2f} {se:8.2f} {mc:9.2f} {pc:6.0f} "
                  f"{bg:8.1f} {bt:7.2f} {bf:7.2f} {rf:7.2f} {sh} {dup:5d} "
                  f"{ev:6.0f} {co:8.3f}")
        else:
            print(f"{label:30s} {len(v):3d} {mr:10.2f} {'':8s} {mc:9.2f} {pc:6.0f} "
                  f"  (baseline, no loop)")

    # Paired deltas vs the matching-closure nomut, and the KEY control: the LLM
    # arm vs the random-EA arm at matched budget, over shared targets.
    def closure_of(cells):
        return (next(iter(cells.values())).get("closure_arm") or "placement_search")
    nomut = {closure_of(c): c for lab, c in by.items() if lab.split(":")[0] == "nomut"}

    def paired(a_cells, b_cells):
        sh = sorted(set(a_cells) & set(b_cells))
        d = [(a_cells[p]["recovered_binding_energy"] - b_cells[p]["recovered_binding_energy"])
             for p in sh
             if a_cells[p].get("recovered_binding_energy") is not None
             and b_cells[p].get("recovered_binding_energy") is not None]
        return d

    if nomut:
        print("\n  paired vs nomut (matching closure; more negative Δ = more binding kept):")
        for label in sorted(by):
            if label.split(":")[0] == "nomut":
                continue
            base = nomut.get(closure_of(by[label]))
            if not base:
                continue
            d = paired(by[label], base)
            if d:
                print(f"    {label:28s} n={len(d):2d} mean_d={sum(d)/len(d):+.2f} REU "
                      f"(better/worse={sum(x<-EPS for x in d)}/{sum(x>EPS for x in d)})")

    ea = {lab: c for lab, c in by.items() if lab.split(":")[0] == ITER_EA_ARM}
    if ea:
        ea_cells = next(iter(ea.values()))
        print("\n  KEY CONTROL -- LLM arm vs random-EA at matched budget "
              "(more negative Δ = LLM operator wins):")
        for label in sorted(by):
            if not label.split(":")[0] in ITER_LLM_ARMS:
                continue
            d = paired(by[label], ea_cells)
            if d:
                print(f"    {label:28s} vs {next(iter(ea)):18s} n={len(d):2d} "
                      f"mean_d={sum(d)/len(d):+.2f} REU "
                      f"(LLM better/worse={sum(x<-EPS for x in d)}/{sum(x>EPS for x in d)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--out", default="", help="finalist rows jsonl (resumable)")
    ap.add_argument("--traj", default="", help="per-generation trajectory jsonl "
                    "(default <out dir>/seqmut_iter_traj.jsonl)")
    ap.add_argument("--scan-cache", default="")
    ap.add_argument("--seed-wt", default="", help="rosetta_interface.jsonl -> WT scans")
    ap.add_argument("--seed-scans", default="", help="comma-sep prior scan caches "
                    "(reuse single-shot seqmut Rosetta work read-only)")
    ap.add_argument("--seed-windows", default="", help="seqmut_rows.jsonl to reuse "
                    "the committed window per target (keeps the proxy comparable)")
    ap.add_argument("--arms", default="iter-ea",
                    help=f"comma-sep from {ITER_ARMS}")
    ap.add_argument("--closure-arm", default="placement_search")
    ap.add_argument("--budget", type=int, default=6, help="closure campaign budget")
    ap.add_argument("--k-mut", type=int, default=2)
    # evolutionary knobs (defaults small: validate before scaling)
    ap.add_argument("--n-seed", type=int, default=8, help="initial pool size")
    ap.add_argument("--n-gen", type=int, default=6, help="generations")
    ap.add_argument("--m-off", type=int, default=6, help="offspring per generation")
    ap.add_argument("--max-evals", type=int, default=40,
                    help="MATCHED budget: distinct proxy evaluations per (target,arm)")
    ap.add_argument("--elite-frac", type=float, default=0.3)
    # Noisy-fitness controls. Only engaged by a stochastic evaluator (boltz); with
    # the deterministic proxy sem==0 and all three are no-ops.
    ap.add_argument("--z-explore", type=float, default=1.0,
                    help="elite ranked OPTIMISTICALLY by mean - z*SEM, so a noisy "
                         "candidate survives to be re-measured instead of being "
                         "crowned or dismissed on one draw")
    ap.add_argument("--z-commit", type=float, default=1.0,
                    help="finalist chosen PESSIMISTICALLY by mean + z*SEM, so the "
                         "genome we spend the real oracle on is not just the "
                         "luckiest draw (the measured optimizer's curse: 61%% of "
                         "the first run's apparent gain did not survive a redraw)")
    ap.add_argument("--no-reassess", action="store_true",
                    help="do NOT re-measure the incumbent each generation "
                         "(restores the old behaviour: one draw owns the frontier "
                         "for the whole run)")
    ap.add_argument("--n-workers", type=int, default=4,
                    help="parallel proxy scans per generation (<= cpus-per-task); "
                    "wall-clock only, never changes which genomes are evaluated")
    # The pose-conditioned-likelihood evaluator (CPU, deterministic, exactly
    # solvable). See agent/lnr/pll_inloop.py for why it exists.
    ap.add_argument("--pll-metric", default="pll_bg",
                    choices=["pll_bg", "pll_cplx"],
                    help="which form of the metric is the fitness; pll_bg is the "
                         "pre-registered and graded one")
    ap.add_argument("--pll-scope", default="all", choices=["all", "window", "iface"],
                    help="positions the score averages over. 'all' is the GRADED "
                         "form and the default; 'window'/'iface' are recorded in "
                         "every row either way, so the comparison needs no rerun")
    ap.add_argument("--pll-model", default="ligandmpnn_v_32_020_25")
    ap.add_argument("--pll-device", default="cpu")
    ap.add_argument("--pll-pssm-dir", default="",
                    help="where per-target PSSM tables are cached (default "
                         "<out dir>/pssm)")
    ap.add_argument("--pll-cache", default="",
                    help="jsonl of every scored sequence (default "
                         "<out dir>/seqmut_iter_pll_scores.jsonl)")
    ap.add_argument("--evaluator", default="proxy", choices=["proxy", "boltz", "pll"],
                    help="proxy = pose-preserving Rosetta retention (CPU, cheap); "
                    "boltz = REAL Boltz x4 -> Rosetta dG in the loop (GPU, ~2min/cand); "
                    "pll = pose-conditioned interface likelihood (CPU, ~30us/cand, "
                    "bitwise deterministic, optimum known in closed form)")
    ap.add_argument("--boltz-replicas", type=int, default=4,
                    help="Boltz diffusion samples per candidate (the denoised unit)")
    ap.add_argument("--boltz-seed", type=int, default=1234,
                    help="the LOOP's Boltz fold seed; the report re-folds with a "
                    "DIFFERENT seed for an independent draw (the noise firewall)")
    ap.add_argument("--boltz-work", default="", help="Boltz scratch dir (default <out>/boltz_inloop)")
    ap.add_argument("--boltz-cache", default="", help="Boltz fold cache jsonl "
                    "(default <out dir>/seqmut_iter_boltz_cache.jsonl)")
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--per-target-timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--targets", default="", help="comma-sep target ids to restrict "
                    "to (applied before --limit); empty = all")
    ap.add_argument("--backend", default="", choices=["", "openai", "api", "scripted"])
    ap.add_argument("--model", default="")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-spend-usd", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--hi-effort", default="high",
                    help=f"reasoning effort for the {HI_EFFORT_ARM} arm")
    ap.add_argument("--hi-n-gen", type=int, default=0,
                    help=f"generations for the {HI_EFFORT_ARM} arm (0 = --n-gen). "
                         f"Set this BELOW --n-gen to dollar-match the arms: a "
                         f"high-effort call costs more, so equal generations is "
                         f"not equal spend. Price it with --probe-call-cost.")
    ap.add_argument("--probe-call-cost", action="store_true",
                    help="make one brief-shaped call per configured LLM arm, print "
                         "the measured $/call and the generation count each arm "
                         "can afford under --max-spend-usd, then exit without "
                         "running a search")
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--report-only", default="", help="summarize an existing out "
                    "jsonl and exit (no run)")
    ap.add_argument("--report-dir", default="", help="aggregate every "
                    "seqmut_iter*_rows.jsonl in this dir (EA + LLM sibling jobs) "
                    "and print the combined table, then exit (no run)")
    args = ap.parse_args()

    if args.report_dir:
        report_dir(Path(args.report_dir))
        return 0
    if args.report_only:
        _summarize(Path(args.report_only))
        return 0
    if not (args.manifest and args.pdb_dir and args.out):
        raise SystemExit("[seqmut-iter] --manifest, --pdb-dir, --out required "
                         "(or --report-only)")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
