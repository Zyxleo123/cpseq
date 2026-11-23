"""Tests for the regret analysis. Pure arithmetic on synthetic rows.

The two that matter: average ranks (regret is full of exact 0.0 ties, and
ordinal ranks would silently bias rho), and the NORMALISATION -- raw regret is
not comparable across targets because `pll_bg` is a mean over positions, so a
fixed-k edit is diluted by 1/L.

Run: python -m pytest agent/lnr/test_pll_regret.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import pll_regret as PR  # noqa: E402


def _row(tid, length, regret, wt=-1.0, opt=-2.0, arm="iter-ea", n_mutable=None,
         n_evals=200, e2o=None, free=None):
    o = {"subs": [[0, "W"]], "fitness": opt, "wt_fitness": wt,
         "n_mutable": n_mutable if n_mutable is not None else length}
    if free is not None:
        o["free_minus_constrained"] = free
    return {"target_id": tid, "mut_arm": arm, "length": length,
            "inloop_evaluator": "pll", "inloop_fitness": opt + regret,
            "iter": {"regret": regret, "optimum": o, "n_evals": n_evals,
                     "n_gen_run": 25, "evals_to_optimum": e2o}}


def _write(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# --- statistics -------------------------------------------------------------
def test_spearman_uses_average_ranks_so_ties_do_not_bias_it():
    """Regret has many exact 0.0 values. With ordinal ranks their arbitrary
    order leaks into rho; with average ranks a block of ties is uninformative,
    which is the truth."""
    x = [1, 2, 3, 4]
    tied = [0.0, 0.0, 0.0, 0.0]
    assert PR.spearman(x, tied) == 0.0          # a block of ties says nothing
    assert PR.spearman(x, [4, 3, 2, 1]) == pytest.approx(-1.0)
    # A tied pair CANNOT reach |rho| = 1, and that is the point: ordinal ranks
    # would return 1.0 here and claim a perfect ordering the data does not have.
    half = [0.0, 0.0, 1.0, 1.0]
    assert PR.spearman(x, half) == pytest.approx(0.894427, abs=1e-5)
    assert PR.spearman(x, half) < 1.0


def test_perm_p_is_near_one_for_noise_and_small_for_a_real_effect():
    x = list(range(30))
    assert PR.perm_p(x, list(x), n=2000) < 0.01
    assert PR.perm_p(x, [0.0] * 30, n=500) > 0.5


def test_perm_p_is_nan_on_too_few_points():
    import math
    assert math.isnan(PR.perm_p([1, 2], [1, 2]))


# --- loading ----------------------------------------------------------------
def test_load_skips_nomut_and_ungraded_rows(tmp_path):
    rows = [_row("a", 8, 0.0), {"target_id": "b", "mut_arm": "nomut"},
            {"target_id": "c", "mut_arm": "iter-ea", "iter": {"regret": None}},
            {"target_id": "d", "mut_arm": "iter-ea", "length": 8,
             "iter": {"regret": 0.1}}]        # regret but NO optimum -> ungraded
    got = PR.load(_write(tmp_path, rows))
    assert [r["target_id"] for r in got] == ["a"]


def test_normalised_regret_is_the_fraction_of_available_gain_missed(tmp_path):
    """The headline quantity. wt -1.0, opt -2.0 -> 1.0 nats available; a regret
    of 0.25 means a quarter of the reachable improvement was left behind."""
    got = PR.load(_write(tmp_path, [_row("a", 8, 0.25, wt=-1.0, opt=-2.0)]))
    assert got[0]["gain"] == pytest.approx(1.0)
    assert got[0]["norm_regret"] == pytest.approx(0.25)
    assert got[0]["solved"] == 0.0


def test_a_zero_gain_target_does_not_divide_by_zero(tmp_path):
    got = PR.load(_write(tmp_path, [_row("a", 8, 0.0, wt=-1.0, opt=-1.0)]))
    assert got[0]["norm_regret"] == 0.0


def test_solved_is_regret_exactly_zero(tmp_path):
    got = PR.load(_write(tmp_path, [_row("a", 8, 0.0), _row("b", 8, 1e-9)]))
    assert [r["solved"] for r in got] == [1.0, 0.0]


# --- the tables -------------------------------------------------------------
def test_by_length_bins_and_counts(tmp_path):
    rows = [_row("a", 5, 0.0), _row("b", 7, 0.2), _row("c", 10, 0.1),
            _row("d", 25, 0.3)]
    g = PR.by_length(PR.load(_write(tmp_path, rows)))
    got = {b["bin"]: (b["n"], b["n_solved"]) for b in g}
    assert got["0-8"] == (2, 1) and got["9-11"] == (1, 0) and got["20+"] == (1, 0)
    assert "12-14" not in got            # empty bins are omitted, not zero-filled


def test_correlations_always_report_the_size_confound(tmp_path):
    """The confound row is not optional: a raw regret-vs-length rho read without
    it is the 'interface dG measures size' mistake again."""
    rows = [_row(f"t{i}", 5 + i, 0.01 * i) for i in range(12)]
    cors = PR.correlations(PR.load(_write(tmp_path, rows)), n_perm=200)
    names = [c["name"] for c in cors]
    assert any("THE CONFOUND" in n for n in names)
    assert any("available gain" in n for n in names)
    assert all(c["n"] == 12 for c in cors)


def test_discriminating_subset_is_ordered_by_regret(tmp_path):
    rows = [_row("lo", 8, 0.01), _row("mid", 8, 0.20), _row("hi", 8, 0.40)]
    d = PR.discriminating(PR.load(_write(tmp_path, rows)), thresh=0.05)
    assert d == ["hi", "mid"]            # "lo" is below threshold


def test_budget_check_distinguishes_a_cap_from_an_under_run(tmp_path):
    under = PR.load(_write(tmp_path, [_row("a", 8, 0.1, n_evals=246)]))
    assert PR.budget_check(under, max_evals=400)["n_at_cap"] == 0
    at = PR.load(_write(tmp_path, [_row("a", 8, 0.1, n_evals=400)]))
    assert PR.budget_check(at, max_evals=400)["n_at_cap"] == 1


# --- the report -------------------------------------------------------------
def test_report_warns_when_the_run_was_not_budget_limited(tmp_path, capsys):
    """MEASURED on job 119443: 0/79 targets reached the 400-eval cap, so '46%
    solved' was a generation limit. The report must say so, or an under-run gets
    written up as an operator ceiling."""
    rows = [_row(f"t{i}", 8 + i, 0.1, n_evals=246) for i in range(6)]
    assert PR.report(_write(tmp_path, rows), n_perm=200, max_evals=400) == 0
    out = capsys.readouterr().out
    assert "NOT budget-limited" in out
    assert "must not be reported as an operator ceiling" in out


def test_report_flags_the_suppressed_confound(tmp_path, capsys):
    """Available gain shrinking with length SUPPRESSES raw regret-vs-length, so
    the normalised rho is larger. The report has to name which one is the real
    effect size."""
    # regret FLAT in nats, available gain shrinking with length -- the real
    # mechanism. Raw rho is then ~0 (all regrets tied) while the normalised one
    # is strongly positive, so the raw number would hide the whole effect.
    rows = [_row(f"t{i}", 5 + 2 * i, 0.05, wt=-1.0,
                 opt=-1.0 - (1.0 - 0.08 * i)) for i in range(11)]
    PR.report(_write(tmp_path, rows), n_perm=400, max_evals=400)
    out = capsys.readouterr().out
    assert "SUPPRESSES" in out and "normalised one is the real" in out
    rl = PR.load(_write(tmp_path, rows))
    cors = PR.correlations(rl, n_perm=200)
    assert abs(cors[0]["rho"]) == 0.0          # raw: nothing to see
    assert cors[1]["rho"] > 0.9                # normalised: the whole effect
    assert cors[2]["rho"] < -0.9               # the confound itself


def test_report_writes_a_json_summary_and_gates_on_empty(tmp_path, capsys):
    rows = [_row(f"t{i}", 8, 0.1, e2o=50) for i in range(4)]
    j = tmp_path / "out" / "s.json"
    assert PR.report(_write(tmp_path, rows), out_json=j, n_perm=200) == 0
    d = json.loads(j.read_text())
    assert d["n"] == 4 and d["iter-ea"]["n_solved"] == 0
    assert d["iter-ea"]["by_length"] and d["iter-ea"]["correlations"]
    assert PR.report(_write(tmp_path, [{"mut_arm": "nomut"}]), n_perm=10) == 1
    assert "GATE" in capsys.readouterr().out


# --- paired comparison -----------------------------------------------------
def test_compare_is_paired_and_counts_targets_that_MOVED(tmp_path, capsys):
    """MEASURED on jobs 119443 vs 119637: 3.2x the generations moved 5 of 79
    targets and left 74 bit-identical. A mean difference would have called that
    'improved by 0.012 nats'; the honest statement is 'moved 5 targets'."""
    a = _write(tmp_path / "a", [_row("t1", 8, 0.2), _row("t2", 8, 0.1),
                                _row("t3", 8, 0.0), _row("only_a", 8, 0.5)])
    b = _write(tmp_path / "b", [_row("t1", 8, 0.0), _row("t2", 8, 0.1),
                                _row("t3", 8, 0.0), _row("only_b", 8, 0.5)])
    out = PR.compare(a, b, label_a="25 gens", label_b="80 gens")
    assert out["n_shared"] == 3                   # only_a / only_b excluded
    assert out["n_better"] == 1 and out["n_worse"] == 0
    assert out["n_unchanged"] == 2
    assert out["sign_p"] == pytest.approx(0.5)    # one moved target, one-sided
    assert out["25 gens"]["solved"] == 1 and out["80 gens"]["solved"] == 2
    txt = capsys.readouterr().out
    assert "bit-identical" in txt and "better on: t1" in txt


def test_compare_calls_out_an_operator_ceiling(capsys, tmp_path):
    rows_a = [_row(f"t{i}", 8, 0.1) for i in range(20)]
    rows_b = [_row(f"t{i}", 8, 0.0 if i == 0 else 0.1) for i in range(20)]
    PR.compare(_write(tmp_path / "a", rows_a), _write(tmp_path / "b", rows_b))
    out = capsys.readouterr().out
    assert "OPERATOR's ceiling" in out


def test_compare_gates_on_no_shared_targets(tmp_path, capsys):
    a = _write(tmp_path / "a", [_row("x", 8, 0.1)])
    b = _write(tmp_path / "b", [_row("y", 8, 0.1)])
    assert PR.compare(a, b) is None
    assert "GATE" in capsys.readouterr().out


def test_report_names_targets_where_the_action_space_binds(tmp_path, capsys):
    rows = [_row("a", 8, 0.1, free=-0.05), _row("b", 8, 0.1)]
    PR.report(_write(tmp_path, rows), n_perm=100)
    out = capsys.readouterr().out
    assert "action space binds on 1/2: a" in out
    assert "bound, not a failure" in out


# --- exclusion --------------------------------------------------------------
def test_load_drops_excluded_targets_everywhere(tmp_path):
    """Exclusion happens in `load`, not per table, so every statistic
    downstream sees the same n. A target dropped from the correlations but kept
    in the solve rate would make two numbers in one report disagree."""
    rows = [_row("keep1", 8, 0.0), _row("drop", 18, 0.5), _row("keep2", 9, 0.1)]
    p = _write(tmp_path / "x", rows)
    assert len(PR.load(p)) == 3
    kept = PR.load(p, exclude={"drop"})
    assert [r["target_id"] for r in kept] == ["keep1", "keep2"]
    # and an id that is not present is simply inert, not an error
    assert len(PR.load(p, exclude={"nosuchtarget"})) == 3


def test_artifact_targets_names_why_not_just_which():
    """The curated set must carry its reason: a bare id list invites someone to
    drop a target because it is inconvenient. 5a29's peptide is an expression
    tag, which is a statement about the DATA, not about the result."""
    assert "5a29" in PR.ARTIFACT_TARGETS
    why = PR.ARTIFACT_TARGETS["5a29"]
    assert "tag" in why.lower() and "LVPRGS" in why


def test_exclusion_is_reported_and_changes_n(tmp_path, capsys):
    """An excluded target is printed, never silent -- a reader comparing two
    reports has to be able to see which set each one is over."""
    rows = [_row("5a29", 18, 0.5), _row("keep1", 8, 0.0), _row("keep2", 9, 0.1)]
    p = _write(tmp_path / "y", rows)
    assert PR.report(p, exclude={"5a29"}) == 0
    out = capsys.readouterr().out
    assert "EXCLUDED 5a29" in out
    assert "tag" in out.lower()
    assert "rows: 2" in out


def test_compare_excludes_from_the_shared_set(tmp_path):
    """The paired comparison must not quietly re-admit an excluded target via
    the shared-target intersection."""
    a = _write(tmp_path / "a", [_row("5a29", 18, 0.5), _row("t1", 8, 0.2)])
    b = _write(tmp_path / "b", [_row("5a29", 18, 0.1), _row("t1", 8, 0.0)])
    out = PR.compare(a, b, exclude={"5a29"})
    ids = {t for t in (out.get("better") or [])} | {t for t in (out.get("worse") or [])}
    assert "5a29" not in ids
    assert out["n_shared"] == 1
