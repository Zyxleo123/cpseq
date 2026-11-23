"""Tests for the cross-scope comparison. Fakes the PSSM, so no model and no pose.

The ones that matter are the two ways a 0 can lie: a scope with no reachable
gain (nothing to win, so every optimum ties) and a target whose three position
sets are identical (the scopes are the same objective on it). Both must be
excluded from the fleet mean, or "the scopes agree" gets reported for targets
that cannot disagree.

Run: python -m pytest agent/lnr/test_pll_scope.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import pll_scope as PS  # noqa: E402


class _T:
    def __init__(self, seq, tid="t1"):
        self.seq, self.target_id, self.length = seq, tid, len(seq)


class _PI:
    """Stands in for pll_inloop: only _window_positions is used."""
    @staticmethod
    def _window_positions(t, window):
        if not window:
            return None
        return list(range(int(window["a"]), int(window["b"]) + 1))


class _PSSM:
    """Scores a sequence as the mean over the scope's positions of a per-index
    table. `iface_pos` is what interface_only selects."""
    def __init__(self, n, table, iface_pos):
        self.n, self.table, self.iface_pos = n, table, iface_pos

    def _positions(self, interface_only=False, positions=None):
        if positions is not None:
            return list(positions)
        if interface_only:
            return list(self.iface_pos)
        return list(range(self.n))

    def score(self, seq, metric=None, positions=None, interface_only=False):
        pos = self._positions(interface_only, positions)
        return sum(self.table[i][seq[i]] for i in pos) / len(pos)


class _Scorer:
    def __init__(self, scope, pssm, opt_subs, metric="pll_bg"):
        self.scope, self._pssm, self._opt, self.metric = scope, pssm, opt_subs, metric

    def pssm(self, t):
        return self._pssm

    def optimum(self, t, k_mut, window=None, mutable=None):
        subs = self._opt
        return {"subs": subs,
                "subs_key": ",".join(f"{i}:{a}" for i, a in subs)}


def _build(table, iface_pos, opt_by_scope, seq="AA"):
    p = _PSSM(len(seq), table, iface_pos)
    sc = {s: _Scorer(s, p, opt_by_scope[s]) for s in PS.SCOPES}
    return _T(seq), p, sc


def test_identical_position_sets_are_flagged_as_nondiscriminating():
    """If the window spans the peptide and every residue is interface, the three
    scopes ARE one objective on this target. A 0 regret here is a tautology."""
    tbl = [{"A": 0.0, "W": 1.0}, {"A": 0.0, "W": 1.0}]
    t, p, sc = _build(tbl, iface_pos=[0, 1],
                      opt_by_scope={s: [[0, "W"]] for s in PS.SCOPES})
    r = PS.compare_target(sc, t, {"a": 0, "b": 1}, [0, 1], 1, _PI)
    assert r["position_sets_identical"] is True
    assert r["all_scopes_agree"] is True
    assert PS.summarize([r])["n_discriminating"] == 0


def test_differing_position_sets_discriminate():
    tbl = [{"A": 0.0, "W": 1.0}, {"A": 0.0, "W": 1.0}]
    t, p, sc = _build(tbl, iface_pos=[0],
                      opt_by_scope={s: [[0, "W"]] for s in PS.SCOPES})
    r = PS.compare_target(sc, t, {"a": 1, "b": 1}, [0, 1], 1, _PI)
    assert r["position_sets_identical"] is False
    assert r["n_positions"] == {"all": 2, "window": 1, "iface": 1}


def test_regret_is_the_forgone_fraction_of_the_true_scopes_gain():
    """Deploying the window optimum when `all` is true should forgo exactly the
    share of `all`'s gain that the window optimum does not reach."""
    # position 0 is worth +1.0, position 1 is worth +0.5. window = {1}.
    tbl = [{"A": 0.0, "W": 1.0, "F": 0.0}, {"A": 0.0, "W": 0.5, "F": 0.0}]
    t, p, sc = _build(tbl, iface_pos=[0, 1],
                      opt_by_scope={"all": [[0, "W"]],        # best for `all`
                                    "window": [[1, "W"]],    # best for window
                                    "iface": [[0, "W"]]})
    r = PS.compare_target(sc, t, {"a": 1, "b": 1}, [0, 1], 1, _PI)
    # true=all: wt 0.0, best (0:W) = 1.0/2 = 0.5 -> gain 0.5.
    # deploying window's (1:W) gives 0.5/2 = 0.25 -> forgone 0.25/0.5 = 0.5
    assert r["regret"]["window"]["all"] == pytest.approx(0.5)
    assert r["regret"]["all"]["all"] == 0.0
    # true=window: only position 1 counts. wt 0, best 0.5 -> gain 0.5.
    # deploying all's (0:W) changes nothing at position 1 -> forgone 100%
    assert r["regret"]["all"]["window"] == pytest.approx(1.0)
    assert r["all_scopes_agree"] is False


def test_a_scope_with_no_reachable_gain_is_marked_degenerate():
    """Nothing to win is not the same as the scopes agreeing, so it must not
    enter the fleet mean as a 0."""
    # Under `window` (= position 1) no letter beats native 'A'.
    tbl = [{"A": 0.0, "W": 1.0}, {"A": 0.0, "W": -1.0}]
    t, p, sc = _build(tbl, iface_pos=[0],
                      opt_by_scope={"all": [[0, "W"]], "window": [],
                                    "iface": [[0, "W"]]})
    r = PS.compare_target(sc, t, {"a": 1, "b": 1}, [0, 1], 1, _PI)
    assert "window" in r["degenerate_scopes"]
    s = PS.summarize([r])
    # every x->window cell has no usable target, so it is None, not 0.0
    assert all(s["matrix"][x]["window"] is None for x in PS.SCOPES)
    # ...while the `all` column is still measured
    assert s["matrix"]["all"]["all"] == 0.0


def test_summarize_separates_the_discriminating_subset():
    tbl = [{"A": 0.0, "W": 1.0, "F": 0.0}, {"A": 0.0, "W": 0.5, "F": 0.0}]
    t1, _, sc1 = _build(tbl, [0, 1], {s: [[0, "W"]] for s in PS.SCOPES})
    nondisc = PS.compare_target(sc1, t1, {"a": 0, "b": 1}, [0, 1], 1, _PI)
    t2, _, sc2 = _build(tbl, [0, 1], {"all": [[0, "W"]], "window": [[1, "W"]],
                                      "iface": [[0, "W"]]})
    disc = PS.compare_target(sc2, t2, {"a": 1, "b": 1}, [0, 1], 1, _PI)
    s = PS.summarize([nondisc, disc])
    assert s["n"] == 2 and s["n_discriminating"] == 1
    # pooling the two dilutes the signal; the discriminating matrix must not
    assert s["matrix_discriminating"]["all"]["window"] == pytest.approx(1.0)
    assert s["matrix"]["all"]["window"] == pytest.approx(0.5)   # diluted by the tie


def test_diagonal_is_always_zero():
    tbl = [{"A": 0.0, "W": 1.0, "F": 0.2}, {"A": 0.0, "W": 0.5, "F": 0.9}]
    t, _, sc = _build(tbl, [0], {"all": [[0, "W"]], "window": [[1, "F"]],
                                 "iface": [[0, "W"]]})
    r = PS.compare_target(sc, t, {"a": 1, "b": 1}, [0, 1], 1, _PI)
    for s in PS.SCOPES:
        assert r["regret"][s][s] == 0.0
