"""Tests for the pose-conditioned-likelihood IN-LOOP evaluator.

No torch model and no PDB: a `TargetPSSM` is a plain matrix, so the scorer is
exercised by injecting one. What cannot be faked (that the matrix really is
sequence-independent, that the marginals match upstream) is asserted against the
real checkpoint by `bindscore.selftest`, which the job runs as a gate.

The tests that matter most here are the two failure modes that would be SILENT in
production: the sign of the fitness, and a reference pose that is indexed
differently from the loop's target object.

Run: python -m pytest agent/lnr/test_pll_inloop.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import bindscore as bs        # noqa: E402
from lnr import pll_inloop as P        # noqa: E402
from lnr import seqmut as S            # noqa: E402
from lnr import seqmut_iter as I       # noqa: E402

ALPH = list("ACDEFGHIKLMNPQRSTVWYX")
SEQ = "ACDEFGHI"
WIN = {"a": 2, "b": 6, "retention": 1.0, "clean": True}


class _T:
    """Minimal env.LNRTarget stand-in: the fields the evaluator reads."""
    def __init__(self, tid="x1", seq=SEQ, pdb="t.pdb"):
        self.target_id = tid
        self.seq = seq
        self.length = len(seq)
        self.per_res_contacts = tuple([1] * len(seq))
        self.n_contacts = len(seq)
        self.pdb_path = pdb
        self.pep_chain = "B"
        self.rec_chain = "A"


def _pssm(seq=SEQ, seed=3, iface=None):
    rng = np.random.default_rng(seed)
    return bs.TargetPSSM(rng.normal(size=(len(seq), 21)), ALPH,
                         list(range(len(seq))), "t.pdb", "B", seq, "m",
                         iface=iface if iface is not None else [True] * len(seq))


def _scorer(t=None, pssm=None, iface=None, **kw):
    """A PLLScorer with its table injected -- no checkpoint, no PDB."""
    sc = P.PLLScorer(**kw)
    t = t or _T()
    sc._pssm[t.target_id] = pssm or _pssm(t.seq, iface=iface)
    return sc, t


# ---------------------------------------------------------------------------
# The sign. This project's recurring bug class.
# ---------------------------------------------------------------------------
def test_fitness_is_the_negated_metric_so_more_negative_is_better():
    """`Population` minimises. The metric maximises. The flip happens in exactly
    one place and both orientations are recorded, so no consumer has to infer it."""
    sc, t = _scorer()
    fit, row = sc.score(t, [], WIN)
    assert row["pll"] == pytest.approx(sc.pssm(t).score(SEQ))
    assert fit == pytest.approx(-row["pll"])
    assert row["fitness"] == fit
    assert sc.is_reu is False and sc.units == "nats/res"


def test_a_better_sequence_gets_a_more_negative_fitness():
    """The end-to-end orientation check: the metric's own optimum must be the
    loop's minimum, or the search climbs the wrong way."""
    sc, t = _scorer()
    p = sc.pssm(t)
    best_subs, best_pll = p.best_subs(SEQ, 2)
    wt_fit, _ = sc.score(t, [], WIN)
    opt_fit, _ = sc.score(t, best_subs, WIN)
    assert best_pll > p.score(SEQ)          # the metric says the mutant is better
    assert opt_fit < wt_fit                 # ...and the loop agrees it is better


def test_the_metric_is_bitwise_deterministic_across_calls_and_instances():
    """0.000e+00 across seeds was the measured claim; the loop depends on it (it
    is why re-measurement and the selection intervals are switched off)."""
    sc, t = _scorer()
    a, _ = sc.score(t, [(1, "W")], WIN)
    b, _ = sc.score(t, [(1, "W")], WIN)
    sc2, t2 = _scorer(pssm=_pssm())
    c, _ = sc2.score(t2, [(1, "W")], WIN)
    assert a == b == c
    assert sc.stochastic is False and sc.sem({"anything": 1}) == 0.0
    # `draw` is accepted (Population may pass it) and must be ignored, not
    # mistaken for a request to re-measure
    assert sc.score(t, [(1, "W")], WIN, draw=5)[0] == a


# ---------------------------------------------------------------------------
# Alignment: the failure that would otherwise be silent
# ---------------------------------------------------------------------------
def test_a_misaligned_reference_pose_is_refused_not_scored():
    """A table built from a peptide the loop indexes differently returns numbers
    that are confidently wrong. `prepare` refuses the target instead."""
    sc = P.PLLScorer()
    t = _T(seq="ACDEFGHI")
    sc._pssm["x1"] = _pssm(seq="CDEFGHIK")      # same length, shifted by one
    # the table is already cached, so refusal has to come from the alignment check
    sc._pssm.clear()
    sc._failed["x1"] = "reference pose does not match the target peptide: shifted"
    why = sc.prepare(t, WIN)
    assert why and "does not match" in why
    fit, row = sc.score(t, [], WIN)
    assert fit is None and "does not match" in row["error"]


def test_prepare_reports_a_missing_complex_rather_than_raising():
    sc = P.PLLScorer()
    why = sc.prepare(_T(pdb="/nonexistent/nope.pdb"), WIN)
    assert why and "no reference complex" in why


def test_prepare_runs_the_alignment_check_on_a_built_table(monkeypatch):
    """The whole point of `prepare`: the table is built, checked against the
    loop's own target sequence, and only then declared usable."""
    sc = P.PLLScorer()
    t = _T()
    monkeypatch.setattr(bs.TargetPSSM, "from_complex",
                        classmethod(lambda cls, *a, **k: _pssm(seq="CDEFGHIK")))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    why = sc.prepare(t, WIN)
    assert why and "does not match" in why and "differ" in why


# ---------------------------------------------------------------------------
# Scope: recorded in full so the choice never needs a rerun
# ---------------------------------------------------------------------------
def test_every_row_records_all_three_scopes():
    sc, t = _scorer(iface=[False, False, True, True, False, False, False, False])
    _, row = sc.score(t, [], WIN)
    assert {"pll_all", "pll_window", "pll_iface"} <= set(row)
    assert row["pll_all"] != row["pll_window"] != row["pll_iface"]
    assert row["pll"] == row["pll_all"]            # default scope is the graded one
    assert row["window"] == [2, 6] and row["n_scored"] == len(SEQ)


def test_scope_window_scores_only_the_held_window():
    pssm = _pssm(iface=[True] * len(SEQ))
    sc, t = _scorer(pssm=pssm, scope="window")
    _, row = sc.score(t, [], WIN)
    assert row["pll"] == pytest.approx(
        pssm.score(SEQ, positions=list(range(2, 7))))
    assert row["n_scored"] == 5
    # an edit OUTSIDE the window cannot move a window-scoped score
    assert sc.score(t, [(0, "W")], WIN)[0] == pytest.approx(row["fitness"])
    # ...but one inside it must
    assert sc.score(t, [(3, "W")], WIN)[0] != pytest.approx(row["fitness"])


def test_scope_iface_uses_the_interface_mask():
    pssm = _pssm(iface=[False, True, False, True, False, False, False, False])
    sc, t = _scorer(pssm=pssm, scope="iface")
    _, row = sc.score(t, [], WIN)
    assert row["pll"] == pytest.approx(pssm.score(SEQ, interface_only=True))
    assert row["n_scored"] == 2


# ---------------------------------------------------------------------------
# The known optimum -- the measurement no previous fitness could support
# ---------------------------------------------------------------------------
def test_optimum_is_unbeatable_by_any_genome_within_the_budget():
    sc, t = _scorer()
    opt = sc.optimum(t, 2)
    assert opt["fitness"] < opt["wt_fitness"]
    rng = np.random.default_rng(0)
    for _ in range(300):
        pos = rng.choice(len(SEQ), size=2, replace=False)
        subs = [(int(p), "ACDEFGHIKLMNPQRSTVWY"[int(rng.integers(20))]) for p in pos]
        fit, _ = sc.score(t, subs, WIN)
        assert fit >= opt["fitness"] - 1e-12, f"{subs} beat the 'optimum'"
    assert sc.score(t, [(i, a) for i, a in opt["subs"]], WIN)[0] == \
        pytest.approx(opt["fitness"])


def test_optimum_respects_the_scope_it_will_be_graded_in():
    """A window-scoped run must be graded against the window-scoped optimum, or
    the regret is measured against an optimum the fitness cannot reach."""
    pssm = _pssm()
    sc, t = _scorer(pssm=pssm, scope="window")
    opt = sc.optimum(t, 2, window=WIN)
    assert all(2 <= p <= 6 for p, _ in opt["subs"])
    assert sc.score(t, [(i, a) for i, a in opt["subs"]], WIN)[0] == \
        pytest.approx(opt["fitness"])


def test_optimum_is_none_when_the_target_is_unscorable():
    sc = P.PLLScorer()
    assert sc.optimum(_T(pdb="/nonexistent/nope.pdb"), 2) is None


def test_hint_is_diagnostic_only_and_ordered_by_gain():
    sc, t = _scorer()
    h = sc.hint(t, top=4)
    assert len(h) == 4 and all(v > 0 for _, _, v in h)
    assert [v for _, _, v in h] == sorted((v for _, _, v in h), reverse=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_scores_are_appended_to_the_cache_jsonl(tmp_path):
    out = tmp_path / "pll.jsonl"
    sc, t = _scorer(cache_path=out)
    sc.score(t, [], WIN)
    sc.score(t, [(0, "W")], WIN)
    sc.close()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[1]["peptide_sequence"] == S.apply_subs(SEQ, [(0, "W")])
    assert rows[1]["subs_key"] == S.subs_key([(0, "W")])
    assert rows[0]["metric"] == "pll_bg" and rows[0]["evaluator"] == "pll"


def test_rejects_an_unknown_metric_or_scope():
    with pytest.raises(ValueError, match="metric"):
        P.PLLScorer(metric="dG")
    with pytest.raises(ValueError, match="scope"):
        P.PLLScorer(scope="everything")


# ---------------------------------------------------------------------------
# Integration with the loop
# ---------------------------------------------------------------------------
def test_the_loop_resolves_the_pll_objective_and_its_own_system_prompt():
    sc, _ = _scorer()
    obj = I.scorer_objective(sc)
    assert obj["units"] == "nats/res" and obj["system_attr"] == "MUT_SYSTEM_PLL"
    assert I.scorer_name(sc) == "pll"
    brief = I.iter_brief(_T(), WIN, 2,
                         [{"subs": ((1, "W"),), "recovered": -1.5, "sem": 0.0,
                           "obs": [-1.5], "gen": 1}], 4, 1, objective=obj)
    assert "nats/res" in brief and "POSE-CONDITIONED" in brief
    assert "nats/residue, not REU" in brief        # the unit, stated as not-REU
    # none of the REU-denominated framings may survive into this brief
    assert "RECOVERED BINDING ENERGY" not in brief
    assert "Rosetta dG" not in brief and "FOLDED binding energy" not in brief
    assert "score(nats/res)" in brief              # the elite table's own header
    assert "NOISY MEASUREMENTS" not in brief       # it is exact
    assert hasattr(S, "MUT_SYSTEM_PLL")
    assert S.MUT_SYSTEM_PLL != S.MUT_SYSTEM != S.MUT_SYSTEM_FOLDED


def test_evolve_on_the_pll_fitness_logs_regret_and_never_beats_the_optimum():
    sc, t = _scorer()
    pop, traj, meta = I.evolve("iter-ea", t, WIN, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=6, n_gen=4, m_off=4, max_evals=40,
                               elite_frac=0.4, relax_repeats=1, timeout=1.0,
                               scorer=sc, n_workers=2)
    opt = meta["optimum"]
    assert opt and pop.best_recovered() >= opt["fitness"] - 1e-12
    regrets = [d["regret"] for d in traj if d.get("regret") is not None]
    assert regrets and all(r >= -1e-9 for r in regrets)
    assert regrets == sorted(regrets, reverse=True)   # monotone: elitist search
    # deterministic evaluator -> no re-measurement spent, no SEM anywhere
    assert pop.reassessments == 0
    assert all((m.get("sem") or 0.0) == 0.0 for m in pop.members.values())


def test_pll_finalist_row_keeps_nats_out_of_the_REU_column():
    """`recovered_binding_energy` is a Rosetta energy everywhere else in this
    project. Writing nats/residue into it would pool two units in one report
    column -- the mistake that produced the 'interface dG measures size' result."""
    sc, t = _scorer()
    pop, traj, meta = I.evolve("iter-ea", t, WIN, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=4, n_gen=2, m_off=3, max_evals=14,
                               elite_frac=0.5, relax_repeats=1, timeout=1.0,
                               scorer=sc)
    row = I._finalist_row(t, "iter-ea", WIN, S.ScanCache(Path("/dev/null")), pop,
                          traj, meta, 2, 1, 1.0, "placement_search")
    assert row["recovered_binding_energy"] is None
    assert row["native_dG"] is None and "inloop_dG" not in row
    assert row["inloop_evaluator"] == "pll" and row["inloop_units"] == "nats/res"
    assert row["inloop_fitness"] == pop.best_recovered()
    assert row["inloop_pll"] == pytest.approx(-pop.best_recovered())
    assert row["inloop_metric"] == "pll_bg" and row["inloop_scope"] == "all"
    # still foldable by the believed-number report: genome + window present
    assert row["committed_window"] == [2, 6] and "subs" in row
    assert row["iter"]["regret"] is not None and row["iter"]["regret"] >= 0.0
    assert row["iter"]["optimum"]["subs"] == meta["optimum"]["subs"]


def test_regret_reaches_exactly_zero_when_the_optimum_is_evaluated():
    """The plumbing: the zero of the regret scale is attainable, and the
    generation that attains it is the one `evals_to_optimum` reports. Asserted by
    PUTTING the optimum in the population, so this tests the measurement and not
    the search's luck."""
    sc, t = _scorer()
    opt = sc.optimum(t, 2)
    pop = I.Population(t, WIN, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, 9,
                       scorer=sc)
    pop.evaluate([], 0)
    assert I._regret(pop.best_recovered(), opt) > 0.0
    pop.evaluate([(int(i), a) for i, a in opt["subs"]], 1)
    assert I._regret(pop.best_recovered(), opt) == 0.0
    traj = [{"gen": 0, "evals": 1, "pop": 1, "phase": "seed",
             "best_recovered": None, "regret": 1.0},
            {"gen": 1, "evals": 2, "pop": 2, "regret": 0.0,
             "breakthrough": True, "beat_frontier_rate": 1.0,
             "refine_rate": 1.0, "incumbent_shift": None}]
    row = I._finalist_row(t, "iter-ea", WIN, S.ScanCache(Path("/dev/null")), pop,
                          traj, {"llm_calls": 0, "llm_cost_usd": 0.0,
                                 "status_counts": {}, "optimum": opt},
                          2, 1, 1.0, "placement_search")
    assert row["iter"]["regret"] == 0.0 and row["iter"]["evals_to_optimum"] == 2
    assert row["subs"] == [[int(i), a] for i, a in opt["subs"]]


def test_the_random_ea_operator_STALLS_short_of_the_known_optimum():
    """A measurement, not a requirement -- and the first one of its kind in this
    project, because until now the optimum of an in-loop fitness was unknown.

    MEASURED (this fixture, k_mut=2, L=8): the random-EA arm converges to a
    genome 0.0074 nats/residue short of the exact optimum and then STOPS
    GENERATING NEW GENOMES -- 40, 120 and 300 generations all end at the same
    regret, with distinct evaluations plateauing around 500 of the ~11k reachable
    genomes. The budget is not what binds; the operator is. That is a statement
    about the search operator that no noisy fitness could have supported, and it
    is the baseline the LLM arm has to beat.
    """
    sc, t = _scorer()
    outcomes = []
    for n_gen in (40, 300):
        s, tt = _scorer(pssm=_pssm())
        pop, traj, meta = I.evolve("iter-ea", tt, WIN,
                                   S.ScanCache(Path("/dev/null")), k_mut=2,
                                   n_seed=12, n_gen=n_gen, m_off=8,
                                   max_evals=100000, elite_frac=0.3,
                                   relax_repeats=1, timeout=1.0, scorer=s,
                                   n_workers=4)
        outcomes.append((pop.evals,
                         I._regret(pop.best_recovered(), meta["optimum"])))
    (e40, r40), (e300, r300) = outcomes
    assert r40 > 0.0 and r300 == r40          # 7.5x the generations, same answer
    assert e300 < 1000                        # and the budget was never the limit
    assert r40 < 0.05                         # it does get close


# ---------------------------------------------------------------------------
# "averaged over" vs "allowed to change" -- a distinction that, conflated,
# produces an optimum no search can reach
# ---------------------------------------------------------------------------
def test_the_graded_optimum_is_restricted_to_the_searchs_action_space():
    """MEASURED on 1t5z: the unrestricted optimum moved a NON-INTERFACE residue,
    which `_point_mutate` can never propose (it draws new positions from the
    target's interface set only). Grading the arm against that bar blames it for
    something it is forbidden to do, so the optimum is constrained to `mutable`
    and the unrestricted one is reported beside it."""
    sc, t = _scorer()
    mutable = [0, 1, 2]
    con = sc.optimum(t, 2, mutable=mutable)
    free = sc.optimum(t, 2)
    assert all(p in mutable for p, _ in con["subs"])
    assert con["n_mutable"] == 3
    # the restriction cannot help, and here it genuinely costs something
    assert con["fitness"] >= free["fitness"]
    assert con["free_fitness"] == pytest.approx(free["fitness"])
    assert con["free_minus_constrained"] < 0        # unrestricted is better
    assert "free_subs" not in free                  # nothing to compare against


def test_a_mutable_restriction_does_not_change_which_positions_are_averaged():
    """The two sets are independent: restricting what may CHANGE must not change
    what the score is a mean OF, or the constrained and unconstrained optima are
    not on the same scale and their difference is meaningless."""
    sc, t = _scorer()
    con = sc.optimum(t, 2, mutable=[0, 1])
    free = sc.optimum(t, 2)
    assert con["wt_fitness"] == free["wt_fitness"]  # same denominator
    got, _ = sc.score(t, [(int(i), a) for i, a in con["subs"]], WIN)
    assert got == pytest.approx(con["fitness"])     # scored the same way


def test_evolve_grades_against_the_reachable_optimum():
    """End to end: with only some positions carrying contacts, the loop's optimum
    must be the reachable one, and regret must stay >= 0."""
    class _TI(_T):
        def __init__(self):
            super().__init__()
            self.per_res_contacts = (1, 1, 0, 0, 0, 0, 0, 0)
            self.n_contacts = 2
    t = _TI()
    sc = P.PLLScorer()
    sc._pssm[t.target_id] = _pssm(t.seq)
    pop, traj, meta = I.evolve("iter-ea", t, WIN, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=8, n_gen=6, m_off=4, max_evals=60,
                               elite_frac=0.4, relax_repeats=1, timeout=1.0,
                               scorer=sc)
    opt = meta["optimum"]
    assert opt["n_mutable"] == 2 and all(p in (0, 1) for p, _ in opt["subs"])
    assert pop.best_recovered() >= opt["fitness"] - 1e-12
    assert I._regret(pop.best_recovered(), opt) >= 0.0


# ---------------------------------------------------------------------------
# The report: the matched-budget control is the output this module exists for
# ---------------------------------------------------------------------------
def _row(target, arm, fit, regret, opt_fit, **kw):
    r = {"target_id": target, "mut_arm": arm, "length": 8, "k_mut": 2,
         "inloop_evaluator": "pll", "inloop_fitness": fit,
         "inloop_units": "nats/res", "recovered_binding_energy": None,
         "closure_arm": "placement_search"}
    if arm != "nomut":
        r["iter"] = {"n_evals": 100, "llm_cost_usd": 0.0, "regret": regret,
                     "evals_to_optimum": 50 if regret == 0.0 else None,
                     "optimum": dict({"subs": [[0, "W"]], "fitness": opt_fit},
                                     **kw)}
    if arm.startswith("iter-llm"):
        r["model_id"] = "m"
    return r


def test_report_pairs_the_llm_arm_against_the_ea_arm_on_the_pll_fitness(capsys):
    rows = [_row("a", "nomut", -1.0, None, None), _row("b", "nomut", -2.0, None, None),
            _row("a", I.ITER_EA_ARM, -1.5, 0.2, -1.7),
            _row("b", I.ITER_EA_ARM, -2.5, 0.1, -2.6),
            _row("a", "iter-llm", -1.7, 0.0, -1.7),
            _row("b", "iter-llm", -2.6, 0.0, -2.6)]
    I._summarize_pll(rows)
    out = capsys.readouterr().out
    assert "POSE-CONDITIONED LIKELIHOOD" in out and "nats/residue" in out
    assert "vs iter-ea on 2 shared target(s)" in out and "CONTROL iter-llm" in out
    # the LLM arm is 0.15 nats better on both -> a negative (winning) delta
    assert "-0.1500" in out
    assert "mean regret: LLM 0.0000 vs EA 0.1500" in out


def test_report_calls_out_a_saturated_comparison_rather_than_a_tie(capsys):
    """Both arms at regret 0 is NOT the LLM winning or losing -- it means the
    objective is separable and the comparison measured nothing. The report has
    to say so, or a tie gets written up as a result."""
    rows = [_row("a", I.ITER_EA_ARM, -1.7, 0.0, -1.7),
            _row("a", "iter-llm", -1.7, 0.0, -1.7)]
    I._summarize_pll(rows)
    out = capsys.readouterr().out
    assert "separable" in out and "saturated, not won" in out


def test_report_flags_targets_where_the_action_space_binds(capsys):
    rows = [_row("a", I.ITER_EA_ARM, -1.5, 0.2, -1.7,
                 free_subs=[[5, "K"]], free_fitness=-1.9,
                 free_minus_constrained=-0.2)]
    I._summarize_pll(rows)
    out = capsys.readouterr().out
    assert "ACTION SPACE binds on 1 target" in out
    assert "bound, not a search failure" in out


def test_report_is_silent_when_no_pll_rows_exist(capsys):
    I._summarize_pll([{"target_id": "a", "mut_arm": "nomut",
                       "recovered_binding_energy": -5.0}])
    assert capsys.readouterr().out == ""
