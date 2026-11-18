"""Tests for the believed-number report. Pure: synthetic folded jsonl, no Boltz.

Run: python -m pytest agent/lnr/test_seqmut_iter_report.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import seqmut_iter_report as RPT  # noqa: E402


def _fold(tid, arm, dG, std=1.0, n=4, rmsd=4.0, ok=True):
    return {"target_id": tid, "arm_label": arm, "dG_refined": dG,
            "dG_refined_std": std, "n_valid_folds": n, "n_folds": 4,
            "held_rmsd": rmsd, "window": [0, 5],
            "score_status": "ok" if ok else "fail",
            "sequence_matches": True, "ring_declared": True}


def _write(tmp, rows, name="boltz_iter_mut_held.jsonl"):
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_valid_gate_and_sem():
    assert RPT._valid(_fold("t", "a", -20.0))
    assert not RPT._valid(_fold("t", "a", -20.0, rmsd=12.0))   # mislocalised fold
    assert not RPT._valid(_fold("t", "a", None))               # never scored
    assert not RPT._valid(None)
    # SEM = std/sqrt(n_valid)
    assert RPT._sem(_fold("t", "a", -20.0, std=2.0, n=4)) == 1.0
    assert RPT._sem(_fold("t", "a", -20.0, std=None, n=4)) is None   # legacy single-fold


def test_load_folded_and_linear(tmp_path):
    p = _write(tmp_path, [_fold("1abc", "iter-ea:k2", -30.0),
                          _fold("1abc", "nomut:k2", -25.0)])
    folded = RPT.load_folded(p)
    assert folded[("1abc", "iter-ea:k2")]["dG_refined"] == -30.0
    lr = tmp_path / "lin.jsonl"
    lr.write_text(json.dumps({"target_id": "1abc", "linear_dG": -40.0}) + "\n"
                  + json.dumps({"target_id": "1xyz", "linear_dG": None}) + "\n")
    nlin = RPT.load_linear(lr)
    assert nlin == {"1abc": -40.0}                             # None linear_dG skipped


def test_paired_uses_only_both_valid_complexes():
    folded = {
        ("t1", "iter-llm:luna:k2"): _fold("t1", "iter-llm:luna:k2", -32.0),
        ("t1", "iter-ea:k2"): _fold("t1", "iter-ea:k2", -28.0),
        ("t2", "iter-llm:luna:k2"): _fold("t2", "iter-llm:luna:k2", -20.0),
        ("t2", "iter-ea:k2"): _fold("t2", "iter-ea:k2", -22.0),
        # t3: EA fold mislocalised -> the pair is dropped, never pooled
        ("t3", "iter-llm:luna:k2"): _fold("t3", "iter-llm:luna:k2", -50.0),
        ("t3", "iter-ea:k2"): _fold("t3", "iter-ea:k2", -10.0, rmsd=20.0),
    }
    rows = RPT._paired(folded, "iter-llm:luna:k2", "iter-ea:k2")
    tids = {t for t, *_ in rows}
    assert tids == {"t1", "t2"}                                # t3 dropped
    d = {t: dd for t, dd, *_ in rows}
    assert abs(d["t1"] - (-4.0)) < 1e-9                        # llm stronger on t1
    assert abs(d["t2"] - (+2.0)) < 1e-9                        # ea stronger on t2


def test_report_runs_end_to_end_with_decomposition_and_null(tmp_path, capsys):
    rows = []
    # 3 complexes: N_lin strongest, nomc(=ceiling) weaker (cyc cost), llm recovers,
    # ea recovers less -> llm should win the null.
    for tid, (nl, ncyc, llm, ea) in {
        "1aaa": (-60, -40, -52, -44),
        "1bbb": (-50, -35, -47, -38),
        "1ccc": (-70, -50, -61, -53),
    }.items():
        rows += [_fold(tid, "nomut:k2", ncyc),
                 _fold(tid, "iter-llm:luna:k2", llm),
                 _fold(tid, "iter-ea:k2", ea)]
    mh = _write(tmp_path, rows)
    lr = tmp_path / "lin.jsonl"
    lr.write_text("\n".join(json.dumps({"target_id": t, "linear_dG": v})
                            for t, v in [("1aaa", -60), ("1bbb", -50), ("1ccc", -70)]) + "\n")
    RPT.report(mh, lr)
    out = capsys.readouterr().out
    assert "believed number" in out
    assert "KEY NULL" in out and "iter-ea:k2" in out
    assert "decomposition" in out
    # llm beats ea on all 3 complexes -> "better/worse=3/0"
    assert "3/0" in out


def test_report_handles_no_folds(tmp_path, capsys):
    RPT.report(tmp_path / "missing.jsonl", "")
    assert "no folded rows" in capsys.readouterr().out
