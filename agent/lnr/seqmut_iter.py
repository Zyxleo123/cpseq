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

ITER_LLM_ARMS = ("iter-llm", "iter-llm-ala")
ITER_EA_ARM = "iter-ea"
ITER_ARMS = ITER_LLM_ARMS + (ITER_EA_ARM,)


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
    """Genome -> cached proxy fitness, with a hard distinct-evaluation budget.

    fitness = recovered_binding_energy (REU); MORE NEGATIVE IS BETTER, so the
    frontier is the minimum. A genome that fails to score (scan error/timeout) is
    remembered with recovered=None so it is neither re-scored nor ever elite.
    """

    def __init__(self, t, window, cache, k_mut, relax_repeats, timeout, max_evals,
                 scorer=None):
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
        self.members: dict = {}   # subs_key -> {"subs": tuple, "recovered": float|None, "gen": int}
        self.evals = 0            # distinct genomes THIS arm has scored (budget)

    def budget_left(self) -> bool:
        return self.evals < self.max_evals

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
        rec, _row = self.scorer.score(self.t, list(g), self.window)
        self.evals += 1
        self.members[key] = {"subs": g, "recovered": rec, "gen": gen}
        return rec

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
            rec, _row = self.scorer.score(self.t, list(g), self.window)
            return rec

        if n_workers and n_workers > 1 and len(to_score) > 1:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(to_score))) as ex:
                recs = list(ex.map(lambda kg: _run(kg[1]), to_score))
        else:
            recs = [_run(g) for _, g in to_score]
        for (key, g), rec in zip(to_score, recs):
            self.evals += 1
            self.members[key] = {"subs": g, "recovered": rec, "gen": gen}
            results[key] = rec
        return results

    def scored(self) -> list:
        return [m for m in self.members.values() if m["recovered"] is not None]

    def best(self):
        v = self.scored()
        return min(v, key=lambda m: m["recovered"]) if v else None

    def best_recovered(self):
        b = self.best()
        return b["recovered"] if b else None

    def elite(self, frac: float, min_n: int = 2, max_n: int = 12) -> list:
        """Top-q elite by recovered (most negative first)."""
        v = sorted(self.scored(), key=lambda m: m["recovered"])
        if not v:
            return []
        n = max(min_n, int(round(frac * len(v))))
        return v[: min(max_n, min(n, len(v)))]


# ---------------------------------------------------------------------------
# Random-EA operator (the matched control): crossover + point mutation.
# ---------------------------------------------------------------------------

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


def ea_offspring(elite, t, k_mut: int, iface, rng, m: int) -> list:
    """m children, each (genome, parent_recovered) where parent is the better of
    its two crossover parents -- the frontier that child is asked to beat."""
    out = []
    for _ in range(m):
        p = _sample_parents(elite, rng, k=2)
        if not p:
            break
        child = _crossover(p[0]["subs"], p[1]["subs"], k_mut, rng)
        child = _point_mutate(child, t, k_mut, iface, rng)
        parent_rec = min(pp["recovered"] for pp in p)   # the stronger parent
        out.append((_norm(child.items()), parent_rec))
    return out


# ---------------------------------------------------------------------------
# LLM operator: show the elite WITH their proxy fitness; ask for better sets.
# ---------------------------------------------------------------------------

def _elite_block(elite) -> str:
    rows = ["  rank  recovered(REU)  substitutions (pos:AA)",
            "  ----  --------------  ----------------------"]
    for r, m in enumerate(elite, 1):
        sk = S.subs_key(m["subs"]) or "(none / wild-type)"
        rows.append(f"  {r:4d}  {m['recovered']:+14.2f}  {sk}")
    return "\n".join(rows)


def iter_brief(t, window, k_mut, elite, m, gen, wt_ala=None, folded=False) -> str:
    a, b = window["a"], window["b"]
    note = (" The bindE column is each residue's current interface energy "
            "(more positive = it carries more binding)." if wt_ala else "")
    # The in-loop fitness differs by evaluator, and the brief must describe the
    # REAL objective: the proxy threads onto the fixed native pose (edits cannot
    # change the fold), whereas the folded scorer re-folds the mutant (edits that
    # change folding/binding are exactly what is rewarded). Telling the model the
    # wrong one handicaps it versus the random-EA control, which reads no brief.
    if folded:
        scored_by = (
            "These are the best substitution sets found so far, scored by the REAL "
            "FOLDED binding energy: the cyclized peptide is folded from its sequence "
            "(backbone held at the retained window), and the predicted ring's "
            "receptor interface is scored (structure prediction x4 -> Rosetta dG, "
            "REU, median over folds). MORE NEGATIVE IS BETTER.")
        propose_mode = (
            "Your edits are judged on the complex they fold into, so a change that "
            "improves how the peptide folds or binds is rewarded -- not only local "
            "interface tweaks.")
    else:
        scored_by = (
            "These are the best substitution sets found so far, scored by RECOVERED "
            "BINDING ENERGY (energy-weighted interface retention x native dG, REU). "
            "MORE NEGATIVE IS BETTER -- more of the interface energy kept on a ring "
            "that closes.")
        propose_mode = (
            "Keep the NATIVE BINDING MODE (the scorer repacks locally; it cannot "
            "reward a changed fold).")
    return (
        f"Generation {gen}. Peptide length {t.length} (indices 0..{t.length-1}), "
        f"{t.n_contacts} interface contacts. On cyclization the residues in window "
        f"[{a},{b}] are HELD at their bound pose and keep the interface; residues "
        f"outside are released so the ring can close.\n\n"
        f"Per-residue table:{note}\n{S._residue_table(t, wt_ala)}\n\n"
        f"{scored_by} '(none)' is the wild-type sequence, the bar to beat:\n\n"
        f"{_elite_block(elite)}\n\n"
        f"Study what the strong sets share and where the weak ones waste edits, "
        f"then propose {m} NEW candidate substitution set(s) that you expect to "
        f"score MORE NEGATIVE than the best above. Each set uses at most {k_mut} "
        f"substitution(s). {propose_mode} Vary the sets; small, targeted changes "
        f"to the strong sets usually beat starting over.\n\n"
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
                  iface, max_retries=1, save_raw=False, folded=False) -> list:
    """One LLM call returning up to m new genomes, each paired with the best elite
    recovered shown (the frontier it was asked to beat). Falls back to a random-EA
    child if the model declines/malforms, so a dead generation does not stall the
    loop (recorded in meta['llm_status'] counts)."""
    from lp2cp.enums import StepStatus

    best_rec = elite[0]["recovered"] if elite else None
    brief = iter_brief(t, window, k_mut, elite, m, gen, wt_ala=wt_ala, folded=folded)
    system = S.MUT_SYSTEM_FOLDED if folded else S.MUT_SYSTEM
    msgs = [{"role": "user", "content": brief}]
    genomes = []
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
            break
        if save_raw:
            meta.setdefault("llm_raw", []).append((reply.text or "")[:600])
        genomes = parse_many(reply.text, t.length, k_mut, m)
        if genomes:
            status = "ok"
            break
        if r < max_retries:
            msgs += [{"role": "assistant", "content": reply.text},
                     {"role": "user", "content":
                      f"Could not parse. Output exactly {m} line(s), each "
                      "<mut>pos:AA,pos:AA</mut> or <mut>none</mut>, nothing else."}]
    meta["status_counts"][status] = meta["status_counts"].get(status, 0) + 1
    if not genomes:   # keep the loop moving with a random child this generation
        genomes = [g for g, _ in ea_offspring(elite, t, k_mut, iface, rng, 1)]
        meta["fallback_gens"] = meta.get("fallback_gens", 0) + 1
    return [(g, best_rec) for g in genomes]


# ---------------------------------------------------------------------------
# The evolutionary loop (shared by both arms).
# ---------------------------------------------------------------------------

def _seed_population(pop: Population, t, k_mut, iface, rng, n_seed, n_workers=1):
    """Initial pool: wild-type + random single/double mutants, scored under the
    budget (the new ones in parallel). Shared seeding keeps the two arms' starting
    pools comparable."""
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
           max_retries=1, save_raw=False, n_workers=1, scorer=None):
    """Run one arm on one target. Returns (population, trajectory, meta). `scorer`
    is the pluggable evaluator (proxy by default; a BoltzScorer for real-score-in-
    loop). A GPU scorer saturates the one GPU, so its folds run serially."""
    iface = _iface_positions(t)
    seed = int.from_bytes(hashlib.sha256(
        f"{t.target_id}|{arm}|{k_mut}".encode()).digest()[:8], "big")
    rng = random.Random(seed)
    # Does the in-loop scorer re-fold the mutant? If so the LLM brief must describe
    # the folded objective, not the fixed-pose proxy. `folds` is the explicit flag;
    # fall back to is_gpu (BoltzScorer / the GPU fake both re-fold) if unset.
    folded = bool(getattr(scorer, "folds", getattr(scorer, "is_gpu", False)))
    if getattr(scorer, "is_gpu", False):
        n_workers = 1
    pop = Population(t, window, cache, k_mut, relax_repeats, timeout, max_evals,
                    scorer=scorer)
    meta = {"llm_calls": 0, "llm_cost_usd": 0.0, "status_counts": {},
            "model_id": getattr(llm, "model_id", None) if llm else None}

    _seed_population(pop, t, k_mut, iface, rng, n_seed, n_workers=n_workers)
    traj = [{"gen": 0, "evals": pop.evals, "best_recovered": pop.best_recovered(),
             "pop": len(pop.scored()), "phase": "seed"}]

    for gen in range(1, n_gen + 1):
        if not pop.budget_left():
            break
        elite = pop.elite(elite_frac)
        if not elite:
            break
        frontier = elite[0]["recovered"]
        if arm in ITER_LLM_ARMS:
            children = llm_offspring(llm, t, window, k_mut, elite, m_off, gen,
                                     wt_ala, meta, rng, iface,
                                     max_retries=max_retries, save_raw=save_raw,
                                     folded=folded)
        else:
            children = ea_offspring(elite, t, k_mut, iface, rng, m_off)

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
        })
    return pop, traj, meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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
    if getattr(pop.scorer, "is_gpu", False):
        # Real-score-in-loop: the finalist's in-loop number is already a Boltz dG;
        # do NOT recompute the Rosetta proxy. Build the row from the best genome so
        # the believed-number report re-folds it with an INDEPENDENT draw.
        row = {"target_id": t.target_id, "mut_arm": arm, "length": t.length,
               "k_mut": k_mut, "subs": [[int(i), aa.upper()] for i, aa in subs],
               "subs_key": S.subs_key(subs), "committed_window": [a, b],
               "window_clean": window.get("clean"),
               "contact_retention": round(window.get("retention", 0.0), 4),
               "native_dG": None, "energy_retention": None,
               "recovered_binding_energy": pop.best_recovered(),
               "inloop_evaluator": "boltz", "inloop_dG": pop.best_recovered()}
    else:
        row = S.score_target(t, arm, window, cache, k_mut, relax_repeats, timeout,
                             subs_override=subs)
    row["closure_arm"] = closure_arm
    if meta.get("model_id"):
        row["model_id"] = meta["model_id"]
    # arm accounting + the "good optimizer" trajectory summary
    gens = [d for d in traj if d.get("phase") != "seed"]
    row["iter"] = {
        "n_evals": pop.evals, "n_gen_run": len(gens), "n_seed": traj[0]["pop"],
        "best_gen": best["gen"] if best else None,
        "best_recovered": pop.best_recovered(),
        "seed_best": traj[0]["best_recovered"],
        "breakthrough_rate": round(sum(d["breakthrough"] for d in gens) / len(gens), 3) if gens else 0.0,
        "mean_beat_frontier": round(sum(d["beat_frontier_rate"] for d in gens) / len(gens), 3) if gens else 0.0,
        "mean_refine_rate": round(sum(d["refine_rate"] for d in gens) / len(gens), 3) if gens else 0.0,
        "llm_calls": meta["llm_calls"], "llm_cost_usd": meta["llm_cost_usd"],
        "status_counts": meta["status_counts"], "fallback_gens": meta.get("fallback_gens", 0),
    }
    if meta.get("model_id"):
        row["llm_cost_usd"] = meta["llm_cost_usd"]
        row["llm_status"] = ("ok" if meta["status_counts"].get("ok") else
                             next(iter(meta["status_counts"]), "malformed"))
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

    llm = None
    if llm_arms:
        if not args.backend:
            raise SystemExit(f"[seqmut-iter] LLM arms {llm_arms} need --backend/--model")
        llm = R.make_llm(args.backend, args.model, args.effort,
                         args.max_spend_usd, args.max_tokens)
        print(f"[seqmut-iter] llm backend={args.backend} model={args.model} "
              f"cap=${args.max_spend_usd}", flush=True)

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
            if scorer is not None and getattr(scorer, "is_gpu", False):
                nrow = {"target_id": t.target_id, "mut_arm": "nomut",
                        "length": t.length, "k_mut": args.k_mut, "subs": [],
                        "subs_key": "", "committed_window": [win["a"], win["b"]],
                        "window_clean": win.get("clean"),
                        "contact_retention": round(win.get("retention", 0.0), 4),
                        "native_dG": None, "recovered_binding_energy": None,
                        "closure_arm": args.closure_arm}
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
            pop, traj, meta = evolve(
                arm, t, win, cache, k_mut=args.k_mut, n_seed=args.n_seed,
                n_gen=args.n_gen, m_off=args.m_off, max_evals=args.max_evals,
                elite_frac=args.elite_frac, relax_repeats=args.relax_repeats,
                timeout=args.per_target_timeout,
                llm=(llm if arm in ITER_LLM_ARMS else None),
                wt_ala=(wt_ala if arm == "iter-llm-ala" else None),
                max_retries=args.max_retries, save_raw=args.save_raw,
                n_workers=args.n_workers, scorer=scorer)
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
                  f"best={it['best_recovered']} (seed {it['seed_best']}) "
                  f"bt_rate={it['breakthrough_rate']} beat={it['mean_beat_frontier']} "
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


def _summarize_rows(rows):
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
    print(f"{'arm':30s} {'n':>3s} {'meanRecov':>10s} {'meanCeil':>9s} "
          f"{'%ceil':>6s} {'bestGen':>8s} {'btRate':>7s} {'beatFr':>7s} "
          f"{'refine':>7s} {'evals':>6s} {'cost$':>8s}")
    for label in sorted(by):
        cells = list(by[label].values())
        v = [c for c in cells if c.get("recovered_binding_energy") is not None]
        if not v:
            print(f"{label:30s} {len(cells):3d}  (no scored rows)")
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
            print(f"{label:30s} {len(v):3d} {mr:10.2f} {mc:9.2f} {pc:6.0f} "
                  f"{bg:8.1f} {bt:7.2f} {bf:7.2f} {rf:7.2f} {ev:6.0f} {co:8.3f}")
        else:
            print(f"{label:30s} {len(v):3d} {mr:10.2f} {mc:9.2f} {pc:6.0f} "
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
    ap.add_argument("--n-workers", type=int, default=4,
                    help="parallel proxy scans per generation (<= cpus-per-task); "
                    "wall-clock only, never changes which genomes are evaluated")
    ap.add_argument("--evaluator", default="proxy", choices=["proxy", "boltz"],
                    help="proxy = pose-preserving Rosetta retention (CPU, cheap); "
                    "boltz = REAL Boltz x4 -> Rosetta dG in the loop (GPU, ~2min/cand)")
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
