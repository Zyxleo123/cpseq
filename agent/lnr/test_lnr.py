"""Fast unit tests for the LNR experiment logic. No torsion solves here.

The real closure solver is exercised by the campaigns; these pin the pure
decision logic that would otherwise fail silently and be read as a model
result -- the parsing, the binary-search release policy, the tool placements,
and the within-campaign score cache.

Run: python -m pytest agent/lnr/test_lnr.py -q   (needs numpy on the path)
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import run, tools  # noqa: E402


def test_parse_hold():
    assert run.parse_hold("reasoning <hold>2,7</hold>", 10)[0] == (2, 7)
    # out of order is sorted
    assert run.parse_hold("<hold>7,2</hold>", 10)[0] == (2, 7)
    # clamped to the peptide
    assert run.parse_hold("<hold>0,99</hold>", 10)[0] == (0, 9)
    # last tag wins
    assert run.parse_hold("<hold>1,2</hold> no <hold>3,4</hold>", 10)[0] == (3, 4)
    assert run.parse_hold("no tag", 10) == (None, "no_tag")


def test_parse_tool():
    assert run.parse_tool("<tool>hold_best_feasible 5</tool>")[0] == ("hold_best_feasible", 5)
    assert run.parse_tool("<tool>release_termini, 3</tool>")[0] == ("release_termini", 3)
    assert run.parse_tool("<tool>not_a_tool 2</tool>") == (None, "unknown_tool")
    assert run.parse_tool("nothing") == (None, "no_tag")


def test_next_release_binary_search_converges_and_is_monotone():
    """The release policy must bisect toward the smallest closing budget and
    never re-ask below a known failure or above a known success."""
    L = 16
    # no history -> midpoint
    assert run._next_release([], L) == (0 + L - 1) // 2
    # a failure at k=7 pushes the probe above 7
    assert run._next_release([{"release": 7, "closes": False}], L) > 7
    # a success at k=9 pushes the probe at or below 9
    assert run._next_release([{"release": 9, "closes": True}], L) <= 9
    # bracketed: fail at 5, close at 9 -> probe strictly inside (6..8)
    hist = [{"release": 5, "closes": False}, {"release": 9, "closes": True}]
    k = run._next_release(hist, L)
    assert 6 <= k <= 8
    # converged: fail at 7, close at 8 -> returns the minimum closing budget 8
    hist = [{"release": 7, "closes": False}, {"release": 8, "closes": True}]
    assert run._next_release(hist, L) == 8


class _Res:
    def __init__(self, length, per_res):
        self.length = length
        self.per_res_contacts = tuple(per_res)
        self.seq = "A" * length


def test_hold_contact_core_picks_the_richest_window():
    # contacts concentrated at indices 2-4; releasing 2 should keep [1,4] or [2,5]
    t = _Res(7, [0, 1, 9, 9, 9, 1, 0])
    a, b = tools.hold_contact_core(t, 2)        # hold length 5
    assert b - a + 1 == 5
    assert a <= 2 and b >= 4                     # the rich stretch is inside


def test_bo_cold_start_and_paired_determinism():
    """The GP-UCB baseline must (1) with no history pick the top-retention
    analytic-feasible window, (2) never re-propose a tried window while untried
    feasible ones remain, and (3) be deterministic -- no RNG -- so it is paired
    across runs. A fake ranked-window cache stands in for the solver so this
    test pays no torsion solve."""
    from lnr import env as E

    t = make_tiny_target()
    # Inject a deterministic analytic-feasible candidate list (retention, a, b),
    # best retention first, as ranked_analytic_windows would return it.
    ranked = [(0.80, 1, 4), (0.55, 0, 3), (0.40, 2, 5), (0.20, 3, 5)]
    object.__setattr__(t, "_ranked_cache", ranked)

    # (1) cold start -> the top-retention feasible window
    assert run._bo_window(t, []) == (1, 4)

    # (3) determinism: identical inputs -> identical output, twice
    assert run._bo_window(t, []) == (1, 4)

    # (2) with the top window tried and NOT closing, BO must move elsewhere and
    # never re-propose a tried window while untried feasible ones remain.
    hist = [{"a": 1, "b": 4, "release": 2, "closes": False, "retention": 0.80}]
    nxt = run._bo_window(t, hist)
    assert nxt != (1, 4)
    assert nxt in {(0, 3), (2, 5), (3, 5)}

    # exhaust the feasible set with all-closers; BO re-proposes the best closer
    hist = [{"a": a, "b": b, "release": t.length - (b - a + 1),
             "closes": True, "retention": r} for (r, a, b) in ranked]
    assert run._bo_window(t, hist) == (1, 4)      # highest-retention closer


def test_info_arm_shows_the_analytic_frontier_and_plain_arm_does_not():
    """The info-matched arms must be shown EXACTLY the analytic feasibility
    frontier the geometry-aware rules read, and the plain arms must not. A
    fake ranked-window cache stands in for the solver so no torsion solve runs.
    """
    t = make_tiny_target()                       # length 6
    # (retention, a, b) as ranked_analytic_windows returns it, best first.
    # releases here: [1,4]->2, [0,3]->2, [2,5]->2, [0,4]->1, [1,5]->1.
    ranked = [(0.90, 0, 4), (0.80, 1, 4), (0.70, 1, 5),
              (0.55, 0, 3), (0.40, 2, 5)]
    object.__setattr__(t, "_ranked_cache", ranked)

    block = run.analytic_frontier_block(t)
    # per-release maxima: release 1 -> [0,4] (0.90 beats [1,5] 0.70);
    # release 2 -> [1,4] (0.80 beats [0,3] 0.55 and [2,5] 0.40).
    assert "[0,4]" in block and "0.900" in block
    assert "[1,4]" in block and "0.800" in block
    # release 0 has no feasible window in the cache -> marked as such
    assert "(no feasible window)" in block
    # it is labelled an UPPER BOUND, not a closure result
    assert "UPPER BOUND" in block and "feasible is NOT closes" in block

    # routing + brief: info in the arm name turns the block on, and only then
    info_mask = run.mask_brief(t, [], 6, info=True)
    plain_mask = run.mask_brief(t, [], 6, info=False)
    assert "closure-feasibility frontier" in info_mask
    assert "closure-feasibility frontier" not in plain_mask
    info_tool = run.tool_brief(t, [], 6, info=True)
    assert "closure-feasibility frontier" in info_tool
    # the arm-name convention the runner keys off
    assert ("info" in "llm-info-mask") and ("info" not in "llm-mask")
    assert "llm-info-tool".endswith("tool") and "llm-info-mask".endswith("mask")


def test_bb_clash_gate_excludes_the_cyclic_closure_junction():
    """The backbone-triplet clash gate must treat the ring as HEAD-TO-TAIL
    CYCLIC: residues 0 and L-1 are bonded by the closing peptide bond, so their
    legitimately-short backbone distance must NOT be counted as a self-clash.
    Without this the gate flags every closed ring at its own junction and the
    clean-closure signal collapses to ~0 for all arms. A genuine non-adjacent
    overlap must still be caught. Pure geometry -- no torsion solve."""
    import numpy as np

    from lnr import env as E

    def coords(positions):
        p = np.asarray(positions, float)
        return p.copy(), p.copy(), p.copy()      # N=CA=C per residue, same point

    # L=5 ring. res4 sits 0.6 A from res0 (the closure bond). Everything else far.
    Nn, CAn, Cn = coords([(0, 0, 0), (10, 0, 0), (20, 0, 0),
                          (30, 0, 0), (0.6, 0, 0)])
    rec = np.zeros((0, 3))
    self_cl, rec_cl, self_min, rec_min = E._bb_clashes(
        Nn, CAn, Cn, rec, a=1, b=3, self_cut_A=2.5, rec_cut_A=2.5)
    # the ONLY sub-cutoff pair is the 0<->L-1 wrap -> excluded -> clash-free
    assert self_cl == 0, f"closure junction wrongly counted ({self_cl})"

    # now also put res3 0.6 A from res1 (|i-j|=2, a real non-adjacent overlap)
    Nn, CAn, Cn = coords([(0, 0, 0), (10, 0, 0), (20, 0, 0),
                          (10.6, 0, 0), (0.6, 0, 0)])
    self_cl, _, _, _ = E._bb_clashes(Nn, CAn, Cn, rec, a=0, b=4,
                                     self_cut_A=2.5, rec_cut_A=2.5)
    assert self_cl > 0, "genuine non-adjacent backbone overlap not caught"


def test_bb_clash_gate_receptor_counts_only_released_residues():
    """A held residue sits at its deposited (native, clash-free) pose, so its
    closeness to the receptor is binding geometry, not a clash -- only RELEASED
    residues are checked against the receptor."""
    import numpy as np

    from lnr import env as E
    pos = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)]
    Nn = CAn = Cn = np.asarray(pos, float)
    rec = np.array([[0.0, 2.0, 0.0]])            # 2.0 A from res0

    # res0 RELEASED (window [1,3]) -> its closeness to the receptor is a clash
    _, rec_rel, _, _ = E._bb_clashes(Nn, CAn, Cn, rec, a=1, b=3,
                                     self_cut_A=2.5, rec_cut_A=2.5)
    assert rec_rel > 0
    # res0 HELD (window [0,3]) -> native binding geometry, not counted
    _, rec_held, _, _ = E._bb_clashes(Nn, CAn, Cn, rec, a=0, b=3,
                                      self_cut_A=2.5, rec_cut_A=2.5)
    assert rec_held == 0


def test_release_termini_frees_the_ends():
    t = _Res(10, [1] * 10)
    a, b = tools.release_termini(t, 4)           # free 4, split 2/2
    assert (a, b) == (2, 7)
    a, b = tools.release_termini(t, 3)           # free 3, extra to C-term
    assert (a, b) == (1, 7)


def test_window_cache_scores_a_repeat_window_once(monkeypatch=None):
    """Re-proposing a window must not pay the torsion solver twice."""
    import lnr.env as E

    calls = {"n": 0}
    real = E.score_window

    class _Fake(E.LNRTarget):
        pass

    # a minimal target: monkeypatch score_window to count real calls
    from lnr import run as R

    t = make_tiny_target()

    def counting(tt, a, b, **kw):
        calls["n"] += 1
        return E.WindowScore(a=a, b=b, window_len=b - a + 1,
                             release_count=tt.length - (b - a + 1),
                             analytic_feasible=True, closes=(a == 0),
                             residual_A=0.0, retention=0.5,
                             utility=0.5 if a == 0 else 0.0, seconds=0.0)

    orig = E.score_window
    E.score_window = counting
    try:
        # contact_core with a fixed tiny target; budget 4 but the binary search
        # will revisit windows, which must hit the cache.
        row = R.run_campaign("termini", t, budget=4)
        # every distinct (a,b) scored at most once
        windows = {(h["a"], h["b"]) for h in row["history"]}
        assert calls["n"] == len(windows)
    finally:
        E.score_window = orig


def make_tiny_target():
    import numpy as np

    from lnr import env as E
    L = 6
    ca = np.array([[float(i), 0.0, 0.0] for i in range(L)])
    rec = np.array([[0.0, 3.0, 0.0]])
    return E.LNRTarget(
        target_id="tiny", pdb_path="", pep_chain="B", rec_chain="A",
        seq="A" * L, N=ca.copy(), CA=ca.copy(), C=ca.copy(), rec_ca=rec,
        contacts=frozenset({(0, 0), (1, 0)}),
        per_res_contacts=(1, 1, 0, 0, 0, 0),
    )
