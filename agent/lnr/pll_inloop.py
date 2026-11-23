"""IN-LOOP evaluator #3: the pose-conditioned interface likelihood (`pll_bg`) as
the search fitness, in place of the Boltz->Rosetta oracle.

Why this evaluator exists
-------------------------
The loop has had two fitnesses and both were wrong in a measured way:

  * `proxy` (pose-preserving Rosetta retention) DISAGREES with the real folded
    score -- on 1bjr it ranked the edits as big wins while the folded score
    ranked nomut best and every edit worse. A search on it optimises an artefact.
  * `boltz` (real Boltz x4 -> Rosetta dG) is the truth we report, but as an
    IN-LOOP fitness it is a draw, not a value: per-variant fold sd ~3.7 REU, the
    same variant re-measured across runs moved ~4 REU, and 61% of the apparent
    gain vs wild-type evaporated on an independent redraw. Selection on it is
    mostly selection on noise, and each evaluation costs a GPU fold.

`pll_bg` is the metric that was pre-registered, graded and (per
`docs/binding_metric_results.md`) actually ranks: 13/13 on the leak-free CPSet
head-to-tail pairs (permutation p = 0.0059, incumbent dG 8/13), within-complex
median rho +0.34 on 2501 SKEMPI mutations. Here it becomes the in-loop fitness.
Three things change as a result:

  1. NOISE GOES TO EXACTLY ZERO. In marginal mode the encoder never reads the
     peptide's sequence, so one reference complex fixes an `n x 21` table and
     scoring is a table lookup -- bitwise identical across seeds and runs. Every
     noise defence the Boltz loop needs (re-measurement, optimistic/pessimistic
     selection intervals) becomes a no-op, and `stochastic = False` turns them
     off. `--selftest` asserts the determinism rather than assuming it.
  2. THE BUDGET STOPS BINDING. A fold costs minutes of GPU; a lookup costs tens
     of microseconds, so `--max-evals` can be raised by orders of magnitude and
     the job runs on the CPU partition instead of queueing behind real work.
  3. THE OPTIMUM BECOMES KNOWN. The metric is a sum of independent per-position
     terms, so the best `<= k_mut` substitution set is read off the table by
     `TargetPSSM.best_subs` instead of searched for. That converts the
     LLM-vs-random-EA question from "which arm scored better" into "which arm
     reached the KNOWN optimum, and how fast" -- a regret, with a zero. No
     previous fitness in this project could be graded that way.

What this evaluator is NOT
--------------------------
It is not the believed number and it does not replace the report. It needs a
reference BOUND POSE (the target's own deposited complex) and it ranks without
being calibrated in kcal/mol. The finalist is still emitted as a standard seqmut
row and still re-folded by `boltz_validate --mode mut-held` for the believed
number -- which is now a genuine cross-fit: selection and reporting no longer
share a measurement, or even a model.

A note on the window, and on what is recorded
---------------------------------------------
The metric scores a SEQUENCE against a POSE; the committed window [a,b] does not
enter it. But the LNR objective keeps only the window's interface (residues
outside are released so the ring can close), so "should the average run over the
window only?" is a real question -- and `--pll-scope window` is NOT the form that
was graded. Rather than decide it by assertion, every row records all three
scopes (all / window / interface) at a cost of microseconds, so the comparison is
redrawable from the rows with no recomputation. `scope` selects which one is the
fitness; the default is `all`, the graded form.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import bindscore as bs   # noqa: E402  the metric + TargetPSSM
from lnr import seqmut as S       # noqa: E402  apply_subs / subs_key

SCOPES = ("all", "window", "iface")


def _window_positions(t, window):
    """The held window as a position list, clamped to the peptide."""
    if not window:
        return None
    a, b = int(window.get("a", 0)), int(window.get("b", t.length - 1))
    a, b = max(0, min(a, b)), min(t.length - 1, max(a, b))
    return list(range(a, b + 1))


class PoseUnavailable(RuntimeError):
    """This target has no usable reference pose for the metric (no complex, or a
    peptide chain whose indexing does not match the loop's target object)."""


class PLLScorer:
    """In-loop fitness = -pll_bg(sequence | bound pose). Drop-in for
    `seqmut_iter.Population` via `.score`.

    THE SIGN. The metric is higher-is-better (more nats of pose-conditioned
    preference); `Population` is written for Rosetta's more-NEGATIVE-is-better
    convention. The fitness is therefore the NEGATED metric, in one place, here,
    and every row carries both `pll` (as the metric defines it) and `fitness`
    (negated) so a reader never has to infer which way round a number is. Sign
    slips are this project's recurring bug class; `--selftest` asserts that the
    native sequence's fitness is below a scrambled sequence's.
    """

    is_gpu = False       # CPU table lookups -> offspring can be scored in parallel
    folds = False        # scores a sequence against a FIXED reference pose
    stochastic = False   # bitwise deterministic -> no re-measurement, sem == 0
    name = "pll"
    units = "nats/res"
    is_reu = False       # the fitness is NOT a Rosetta energy; see _finalist_row

    def __init__(self, metric="pll_bg", scope="all", model_name=bs.PRIMARY_MODEL,
                 device="cpu", pssm_dir=None, cache_path=None, allowed=None):
        if metric not in bs.TargetPSSM.PSSM_METRICS:
            raise ValueError(f"metric {metric!r} not in {bs.TargetPSSM.PSSM_METRICS}")
        if scope not in SCOPES:
            raise ValueError(f"scope {scope!r} not in {SCOPES}")
        self.metric = metric
        self.scope = scope
        self.model_name = model_name
        self.device = device
        self.pssm_dir = Path(pssm_dir) if pssm_dir else None
        self.path = Path(cache_path) if cache_path else None
        self.allowed = allowed
        self._scorer = None                 # the torch model, built on first need
        self._pssm: dict = {}               # target_id -> TargetPSSM
        self._failed: dict = {}             # target_id -> reason
        self._lock = threading.Lock()       # offspring are scored on a thread pool
        self._fh = None
        self.n_scored = 0
        self.n_built = 0

    # -- the reference pose -------------------------------------------------
    def _mpnn(self):
        if self._scorer is None:
            self._scorer = bs.MPNNScorer(model_name=self.model_name,
                                         device=self.device)
        return self._scorer

    def _pssm_path(self, t):
        if not self.pssm_dir:
            return None
        return self.pssm_dir / f"pssm_{t.target_id}_{t.pep_chain}_{self.model_name}.json"

    def pssm(self, t):
        """The target's table, built once (or read from disk) and then reused.

        Raises `PoseUnavailable` with a reason the job log can print. The
        ALIGNMENT CHECK is the load-bearing part: `env.LNRTarget` builds its
        sequence with a parser that silently drops any residue missing a backbone
        atom, while the table is built by LigandMPNN's parser. If those two
        disagree the lookups read the wrong rows and every score is confidently
        wrong with nothing raising, so the table is rejected unless its template
        peptide is the SAME STRING as the loop's target sequence.
        """
        with self._lock:
            if t.target_id in self._pssm:
                return self._pssm[t.target_id]
            if t.target_id in self._failed:
                raise PoseUnavailable(self._failed[t.target_id])
            try:
                p = self._build(t)
            except PoseUnavailable as exc:
                self._failed[t.target_id] = str(exc)
                raise
            except Exception as exc:  # noqa: BLE001  any parse/model failure
                self._failed[t.target_id] = f"{type(exc).__name__}: {exc}"
                raise PoseUnavailable(self._failed[t.target_id]) from exc
            self._pssm[t.target_id] = p
            return p

    def _build(self, t):
        cached = self._pssm_path(t)
        p = None
        if cached and cached.is_file():
            try:
                p = bs.TargetPSSM.load(cached)
                if p.n != t.length:
                    p = None            # stale table: rebuild rather than trust
            except Exception:           # noqa: BLE001  a corrupt cache is not fatal
                p = None
        if p is None:
            if not t.pdb_path or not Path(t.pdb_path).is_file():
                raise PoseUnavailable(f"no reference complex at {t.pdb_path!r}")
            p = bs.TargetPSSM.from_complex(
                t.pdb_path, pep_chain=t.pep_chain, rec_chains=[t.rec_chain],
                model_name=self.model_name, scorer=self._mpnn())
            self.n_built += 1
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                p.save(cached)
        ok, detail = p.matches(t.seq)
        if not ok:
            raise PoseUnavailable(f"reference pose does not match the target "
                                  f"peptide: {detail}")
        p._align_detail = detail
        return p

    def prepare(self, t, window=None):
        """Pay the one-time build OUTSIDE the search budget and report usability.

        Returns None when the target is scorable, else a human-readable reason.
        `seqmut_iter.run` calls this (when present) and SKIPS an unusable target
        instead of letting the loop spend its budget on a scorer that can only
        return None -- which would look like a search that found nothing.
        """
        try:
            p = self.pssm(t)
        except PoseUnavailable as exc:
            return str(exc)
        return None if p is not None else "unknown failure building the PSSM"

    # -- the fitness --------------------------------------------------------
    def sem(self, row) -> float:
        """Exactly zero: the metric is a table lookup, not a measurement."""
        return 0.0

    def _scopes(self, p, t, seq, window):
        """All three scopes for one sequence. Microseconds each, so they are all
        recorded and the scope question stays answerable from the rows."""
        out = {"pll_all": p.score(seq, metric=self.metric)}
        wpos = _window_positions(t, window)
        try:
            out["pll_window"] = (p.score(seq, metric=self.metric, positions=wpos)
                                 if wpos else None)
        except ValueError:
            out["pll_window"] = None
        try:
            out["pll_iface"] = p.score(seq, metric=self.metric, interface_only=True)
        except ValueError:
            out["pll_iface"] = None
        return out

    def score(self, t, subs, window, draw: int = 0):
        """(fitness_or_None, row). `draw` is accepted and IGNORED: a redraw of a
        deterministic metric is the same number, which is why `stochastic` is
        False and the loop never asks for one."""
        seq = S.apply_subs(t.seq, subs)
        row = {"target_id": t.target_id, "evaluator": self.name,
               "metric": self.metric, "scope": self.scope,
               "subs": [[int(i), str(a).upper()] for i, a in subs],
               "subs_key": S.subs_key(subs), "peptide_sequence": seq,
               "window": [window.get("a"), window.get("b")] if window else None,
               "model": self.model_name}
        try:
            p = self.pssm(t)
        except PoseUnavailable as exc:
            row["error"] = str(exc)
            return None, row
        try:
            vals = self._scopes(p, t, seq, window)
        except (ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            return None, row
        row.update(vals)
        pll = vals.get(f"pll_{self.scope}" if self.scope != "all" else "pll_all")
        if pll is None:
            row["error"] = f"scope {self.scope!r} has no scorable position"
            return None, row
        row["pll"] = pll
        row["n_scored"] = len(p._positions(self.scope == "iface",
                                           _window_positions(t, window)
                                           if self.scope == "window" else None))
        # THE ONE PLACE THE SIGN FLIPS.
        row["fitness"] = -float(pll)
        row["template"] = p.template
        self._append(row)
        return row["fitness"], row

    # -- what only a solvable metric can do ---------------------------------
    def optimum(self, t, k_mut, window=None, positions=None, mutable=None):
        """The EXACT best `<= k_mut` substitution set, and its fitness.

        Not a search: the metric is additive over positions, so this is read off
        the table. Returned so the loop can log its REGRET against it -- the
        measurement the Boltz fitness cannot support, since there the optimum is
        unknown. Returns None if the target is unscorable.

        `mutable` is the position set the SEARCH is allowed to change (the loop
        passes its interface set). The optimum is restricted to it, because an
        optimum the operator cannot reach is not a fair bar -- measured on 1t5z,
        where the unrestricted optimum moves a non-interface residue the EA can
        never propose. The unrestricted optimum is still reported, as
        `free_*`, since it is what the metric itself would pick.
        """
        try:
            p = self.pssm(t)
        except PoseUnavailable:
            return None
        pos = positions
        if pos is None and self.scope == "window":
            pos = _window_positions(t, window)
        iface_only = (self.scope == "iface")
        kw = dict(allowed=self.allowed, positions=pos, metric=self.metric,
                  interface_only=iface_only)
        subs, pll = p.best_subs(t.seq, int(k_mut), mutable=mutable, **kw)
        wt = p.score(t.seq, metric=self.metric, positions=pos,
                     interface_only=iface_only)
        out = {"subs": [[int(i), a] for i, a in subs],
               # the canonical key, so "did the arm find the optimum?" is a
               # string comparison and not a re-derivation
               "subs_key": S.subs_key(subs), "pll": pll,
               "fitness": -float(pll), "wt_pll": wt, "wt_fitness": -float(wt),
               "k_mut": int(k_mut), "scope": self.scope, "metric": self.metric,
               "n_mutable": (None if mutable is None else len(set(mutable))),
               "n_gain_positions": len(p.position_gains(
                   t.seq, mutable=mutable, **kw))}
        if mutable is not None:
            # what the metric would pick with no restriction, and whether the
            # restriction cost anything -- so a regret of 0 is not confused with
            # "the metric's own best edit"
            fsubs, fpll = p.best_subs(t.seq, int(k_mut), **kw)
            out["free_subs"] = [[int(i), a] for i, a in fsubs]
            out["free_fitness"] = -float(fpll)
            out["free_minus_constrained"] = round(-float(fpll) - (-float(pll)), 6)
        return out

    def hint(self, t, k_mut=1, top=8, window=None):
        """Per-position preference: the single best edit at each position, best
        first -- the metric explaining itself.

        FOR DIAGNOSTICS AND THE SELFTEST ONLY. This deliberately does NOT go into
        the LLM brief. The brief's whole purpose is to measure whether an LLM
        operator optimises a fitness better than a random operator at a matched
        budget; handing the LLM arm the metric's own per-position gradient -- which
        contains the exact answer, since the metric is additive -- would let it
        read the optimum off the page while the EA arm still has to search. The
        arms would no longer differ by operator, and the comparison would measure
        nothing. Both arms see fitness values and nothing else.
        """
        try:
            p = self.pssm(t)
        except PoseUnavailable:
            return []
        pos = _window_positions(t, window) if self.scope == "window" else None
        g = p.position_gains(t.seq, allowed=self.allowed, positions=pos,
                             metric=self.metric,
                             interface_only=(self.scope == "iface"))
        return [(int(i), a, round(float(v), 3)) for i, a, v in g[:top]]

    def _append(self, row):
        """Append one row and count it. Under the lock: the loop scores a
        generation's offspring on a thread pool, and neither the counter's
        read-modify-write nor an unsynchronised write+flush pair is safe there --
        the visible symptom would be interleaved half-lines in the provenance
        file, which only shows up as a json decode error much later."""
        with self._lock:
            self.n_scored += 1
            if self.path is None:
                return
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a")
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ---------------------------------------------------------------------------
# SELFTEST: the job's gate. Everything below runs against the REAL checkpoint
# and the REAL deposited complexes, because the claims that matter cannot be
# checked with a fake matrix: that the table is sequence-independent (so caching
# it is legitimate), that it is bitwise reproducible, that the loop's target
# object and the table agree on what residue sits at each index, and that the
# metric is not reading amino-acid composition.
# ---------------------------------------------------------------------------

def _scramble(seq, seed=0):
    import random as _r
    r = _r.Random(seed)
    out = list(seq)
    r.shuffle(out)
    return "".join(out)


def selftest(manifest, pdb_dir, targets=(), metric="pll_bg", scope="all",
             model_name=bs.PRIMARY_MODEL, device="cpu", pssm_dir=None,
             k_mut=2, n_shuf=24, min_comp_frac=0.6, max_unscorable_frac=0.25):
    """Returns 0 if the gate passes. Prints one block per target.

    TWO KINDS OF FAILURE, DELIBERATELY NOT POOLED. Running this over 93 real
    deposited complexes made the distinction unavoidable:

      CODE INVARIANTS -- the sign of the fitness, bitwise determinism, the disk
        round-trip, and a variant being exactly a table lookup. These are
        properties of the implementation. They hold on every target or something
        is wrong with the code, so ANY failure fails the gate.

      DATA OBSERVATIONS -- whether a given deposited peptide can be indexed
        consistently by both parsers, and whether a given native sequence beats
        its own scrambles on its own pose. These are properties of each
        structure. On real data some targets fail them: one peptide in the LNR
        test set is 18 residues to its own template's 19 (a residue missing a
        backbone atom), and some natives genuinely score below their scrambles.
        Those targets are REPORTED AND SKIPPED, not fatal -- the loop skips them
        through `prepare` anyway. What IS fatal is the AGGREGATE going wrong:
        if the native beats its scrambles on fewer than `min_comp_frac` of
        targets the metric is reading composition fleet-wide, and if more than
        `max_unscorable_frac` are unindexable the pose set itself is wrong.

    Conflating the two would let one odd structure abort a 93-target run, and --
    worse -- would make the real code invariants easy to switch off out of
    annoyance.
    """
    from lnr import env as E

    ts = E.load_manifest(str(manifest), str(pdb_dir))
    ts.sort(key=lambda x: x.target_id)
    want = {s for s in targets if s}
    if want:
        ts = [x for x in ts if x.target_id in want]
    if not ts:
        print(f"GATE: no targets from {manifest} / {pdb_dir}")
        return 1
    print(f"[pll-selftest] {len(ts)} target(s), metric={metric} scope={scope} "
          f"model={model_name} device={device}")

    sc = PLLScorer(metric=metric, scope=scope, model_name=model_name,
                   device=device, pssm_dir=pssm_dir)
    hard, unscorable, comp = [], [], []     # code invariants / data / data
    ok_targets = 0
    t_build = time.time()
    for t in ts:
        print(f"\n--- {t.target_id}  pep_chain={t.pep_chain} "
              f"rec_chain={t.rec_chain}  L={t.length} ---")
        why = sc.prepare(t)
        if why:
            print(f"  UNSCORABLE (skipped, not fatal): {why}")
            unscorable.append(f"{t.target_id}: {why}")
            continue
        p = sc.pssm(t)
        win = {"a": 0, "b": t.length - 1}
        print(f"  alignment  : {getattr(p, '_align_detail', '?')}")
        print(f"  native seq : {t.seq}")

        fit, row = sc.score(t, [], win)
        if fit is None:
            unscorable.append(f"{t.target_id}: wild-type unscorable: "
                              f"{row.get('error')}")
            print(f"  UNSCORABLE (skipped): {row.get('error')}")
            continue
        def _f(v):
            return "n/a" if v is None else f"{v:+.4f}"
        # all three scopes, every time: the scope question stays answerable from
        # the log and the rows without rerunning anything
        print(f"  wt         : pll={row['pll']:+.4f}  fitness={fit:+.4f}   "
              f"[all {_f(row.get('pll_all'))}  window {_f(row.get('pll_window'))}"
              f"  iface {_f(row.get('pll_iface'))}]")

        # [1] SIGN. The metric's optimum must be the loop's minimum.
        opt = sc.optimum(t, k_mut, window=win)
        if not (opt["fitness"] <= fit + 1e-12):
            hard.append(f"{t.target_id}: SIGN -- optimum fitness {opt['fitness']} "
                        f"is not <= wild-type {fit}")
            print("  FAIL [1] sign: the optimum is not the minimum")
        else:
            print(f"  [1] sign   : OK  optimum fitness {opt['fitness']:+.4f} "
                  f"<= wt {fit:+.4f}; subs {opt['subs']} "
                  f"({opt['n_gain_positions']} position(s) improvable)")

        # [2] DETERMINISM, bitwise, across an independently built table. This is
        # what licenses switching off every noise defence in the loop.
        sc2 = PLLScorer(metric=metric, scope=scope, model_name=model_name,
                        device=device)
        f2, _ = sc2.score(t, [], win)
        f3, _ = sc.score(t, [], win)
        if not (f2 == fit == f3):
            hard.append(f"{t.target_id}: DETERMINISM -- {fit!r} vs {f2!r} vs {f3!r}")
            print(f"  FAIL [2] determinism: {fit!r} / {f2!r} / {f3!r}")
        else:
            print(f"  [2] determ.: OK  delta 0.000e+00 across independently "
                  f"built tables")

        # [3] DISK ROUND-TRIP. The job caches tables; a cached table that scores
        # differently would make a resubmission silently incomparable.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            q = Path(td, "t.json")
            p.save(q)
            back = bs.TargetPSSM.load(q)
            d = abs(back.score(t.seq, metric=metric) - p.score(t.seq, metric=metric))
            if d > 0:
                hard.append(f"{t.target_id}: PSSM disk round-trip differs by {d}")
                print(f"  FAIL [3] round-trip: delta {d:.3e}")
            else:
                print("  [3] cache  : OK  disk round-trip is exact")

        # [4] COMPOSITION. The same residues in a scrambled order on the same
        # pose must score WORSE on average, or the metric is reading composition
        # -- the objection that closed the ESM-2 / PepMLM thread.
        sh = [p.score(_scramble(t.seq, s), metric=metric) for s in range(n_shuf)]
        mean_sh = sum(sh) / len(sh)
        gain = row["pll"] - mean_sh
        n_beat = sum(1 for v in sh if v >= row["pll"])
        comp.append((t.target_id, gain, n_beat))
        if gain <= 0:
            print(f"  [4] compos.: BELOW CHANCE on this target (reported, not "
                  f"fatal): native {row['pll']:+.4f} <= scrambled mean "
                  f"{mean_sh:+.4f}; {n_beat}/{n_shuf} scrambles beat it")
        else:
            print(f"  [4] compos.: OK  native {row['pll']:+.4f} vs scrambled "
                  f"mean {mean_sh:+.4f} (+{gain:.4f}); {n_beat}/{n_shuf} "
                  f"scrambles beat the native")

        # [5] SEQUENCE-INDEPENDENCE OF THE TABLE. The whole cost argument rests
        # on the encoder never reading the peptide's sequence, so a variant must
        # be a pure lookup: score(mutant) must equal the table's own arithmetic.
        subs = [(int(i), a) for i, a in opt["subs"]] or [(0, "W")]
        fm, _ = sc.score(t, subs, win)
        expect = p.score(S.apply_subs(t.seq, subs), metric=metric)
        if abs((-fm) - expect) > 0:
            hard.append(f"{t.target_id}: LOOKUP -- {-fm!r} != {expect!r}")
            print(f"  FAIL [5] lookup: {-fm!r} != {expect!r}")
        else:
            print("  [5] lookup : OK  a variant is exactly a table lookup")

        ok_targets += 1

    # [6] THROUGHPUT, measured not asserted: this is the number that decides how
    # much budget the loop can afford.
    t0 = time.time()
    n = 0
    for t in ts:
        if t.target_id in sc._failed:
            continue
        p = sc.pssm(t)
        win = {"a": 0, "b": t.length - 1}
        import random as _r
        r = _r.Random(0)
        for _ in range(2000):
            q = "".join(r.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(t.length))
            p.score(q, metric=metric)
            n += 1
    dt = time.time() - t0
    print(f"\n[pll-selftest] throughput: {n} sequences in {dt:.3f} s "
          f"= {n / max(dt, 1e-9):,.0f} seq/s ({dt / max(n, 1) * 1e6:.1f} us/seq)")
    print(f"[pll-selftest] tables built: {sc.n_built} "
          f"(total setup {time.time() - t_build:.1f} s incl. all checks)")
    sc.close()

    # -- the verdict ------------------------------------------------------
    print(f"\n[pll-selftest] {ok_targets}/{len(ts)} target(s) scorable")
    if unscorable:
        frac = len(unscorable) / len(ts)
        print(f"\n  SKIPPED (data, not code) -- {len(unscorable)}/{len(ts)} "
              f"= {frac:.0%} of targets. The loop skips these the same way, so "
              f"they cost nothing; they are listed so the set is known.")
        for f in unscorable:
            print(f"    skip {f}")
    pos = [g for _, g, _ in comp if g > 0]
    comp_frac = len(pos) / len(comp) if comp else 0.0
    if comp:
        gains = sorted(g for _, g, _ in comp)
        med = gains[len(gains) // 2]
        print(f"\n  COMPOSITION, AGGREGATE: native beats its own scrambles on "
              f"{len(pos)}/{len(comp)} = {comp_frac:.0%} of scorable targets; "
              f"median gain {med:+.4f}, mean "
              f"{sum(gains)/len(gains):+.4f} nats/residue.")
        below = [tid for tid, g, _ in comp if g <= 0]
        if below:
            print(f"  below chance on: {', '.join(below)} -- reported, not fatal: "
                  f"the claim is fleet-level, and a single deposited peptide that "
                  f"scores below its own permutations is a statement about that "
                  f"structure, not about the metric.")

    verdict = []
    if hard:
        verdict.append(f"{len(hard)} CODE-INVARIANT failure(s)")
    if ok_targets == 0:
        verdict.append("no scorable target at all")
    if comp and comp_frac < min_comp_frac:
        verdict.append(f"composition: only {comp_frac:.0%} of targets beat their "
                       f"own scrambles (floor {min_comp_frac:.0%}) -- at this "
                       f"rate the metric is reading amino-acid composition")
    if unscorable and len(unscorable) / len(ts) > max_unscorable_frac:
        verdict.append(f"{len(unscorable)/len(ts):.0%} of targets unindexable "
                       f"(ceiling {max_unscorable_frac:.0%}) -- the pose set or "
                       f"the manifest is wrong, not one structure")
    for f in hard:
        print(f"  HARD FAIL {f}")
    if verdict:
        print("\n[pll-selftest] gate failed on: " + "; ".join(verdict))
    print(f"PLL SELFTEST {'PASSED' if not verdict else 'FAILED'}")
    return 0 if not verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default="/zfsauton/scratch/yixiz/LNR/test.txt")
    ap.add_argument("--pdb-dir", default="/zfsauton/scratch/yixiz/LNR/pdbs")
    ap.add_argument("--targets", default="", help="comma-separated subset")
    ap.add_argument("--metric", default="pll_bg", choices=list(bs.TargetPSSM.PSSM_METRICS))
    ap.add_argument("--scope", default="all", choices=list(SCOPES))
    ap.add_argument("--model", default=bs.PRIMARY_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pssm-dir", default="")
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--n-shuf", type=int, default=24)
    ap.add_argument("--min-comp-frac", type=float, default=0.6,
                    help="minimum fraction of targets whose native must beat its "
                         "own scrambles; below this the metric is reading "
                         "composition fleet-wide and the gate fails")
    ap.add_argument("--max-unscorable-frac", type=float, default=0.25,
                    help="above this fraction of unindexable targets the pose set "
                         "itself is wrong and the gate fails")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("nothing to do; pass --selftest")
    return selftest(a.manifest, a.pdb_dir,
                    targets=[s.strip() for s in a.targets.split(",") if s.strip()],
                    metric=a.metric, scope=a.scope, model_name=a.model,
                    device=a.device, pssm_dir=(a.pssm_dir or None),
                    k_mut=a.k_mut, n_shuf=a.n_shuf,
                    min_comp_frac=a.min_comp_frac,
                    max_unscorable_frac=a.max_unscorable_frac)


if __name__ == "__main__":
    raise SystemExit(main())
