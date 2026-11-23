"""Unit tests for the SKEMPI accuracy gate. Selection, aggregation and correlation
are tested with fakes; the PyRosetta relax is not run.

Run: python -m pytest agent/lnr/test_skempi_accuracy.py -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from lnr import skempi_accuracy as S  # noqa: E402


def test_split_pos_handles_icode_and_negative():
    assert S._split_pos("38") == (38, "")
    assert S._split_pos("38A") == (38, "A")
    assert S._split_pos("-2") == (-2, "")
    assert S._split_pos("x") == (None, None)


def _mut(wt, chain, pos, mut):
    return types.SimpleNamespace(wt=wt, mut=mut, chain=chain, position=pos,
                                 pdb_site=(chain, pos), __str__=lambda s: f"{wt}{chain}{pos}{mut}")


def _rec(pdb_id, chains1, chains2, muts, ddg, interface=True):
    return types.SimpleNamespace(
        complex_id=f"{pdb_id}_{chains1}_{chains2}", pdb_id=pdb_id,
        chains1=chains1, chains2=chains2, mutations=muts, ddg=ddg,
        k=len(muts), at_interface=interface, locations=("COR",) if interface else ("SUR",))


def test_select_filters_and_reports(tmp_path):
    (tmp_path / "1abc.pdb").write_text("ATOM\n")
    records = [
        _rec("1ABC", "A", "B", [_mut("L", "A", "45", "G")], 1.2),          # keep
        _rec("1ABC", "A", "B", [_mut("L", "A", "45", "G"),
                                _mut("I", "B", "10", "A")], 0.5),          # not_single
        _rec("1ABC", "A", "B", [_mut("L", "A", "45", "G")], 0.3, interface=False),  # not_interface
        _rec("9ZZZ", "A", "B", [_mut("L", "A", "45", "G")], 0.3),          # no_pdb
        _rec("1ABC", "A", "B", [_mut("X", "A", "45", "G")], 0.3),          # nonstandard wt
    ]
    sel, reasons = S.select_records(records, tmp_path)
    assert reasons["kept"] == 1 and reasons["not_single"] == 1
    assert reasons["not_interface"] == 1 and reasons["no_pdb"] == 1
    assert reasons["nonstandard_aa"] == 1
    s = sel[0]
    assert s["interface"] == "A_B" and s["keep_chains"] == "AB"
    assert s["mut_chain"] == "A" and s["mut_resnum"] == 45 and s["mut"] == "G"
    assert s["measured_ddg"] == 1.2


def test_select_max_records_spreads_across_complexes(tmp_path):
    for p in ("1aaa", "1bbb"):
        (tmp_path / f"{p}.pdb").write_text("ATOM\n")
    recs = []
    for pid in ("1AAA", "1BBB"):
        for pos in range(10, 16):
            recs.append(_rec(pid, "A", "B", [_mut("L", "A", str(pos), "G")], 0.1 * pos))
    sel, _ = S.select_records(recs, tmp_path, max_records=4)
    assert len(sel) == 4
    # round-robin => both complexes represented, not 4 of one
    assert len({s["pdb_id"] for s in sel}) == 2


def test_aggregate_predictions_pairs_mut_minus_wt():
    rows = [
        {"complex_id": "1ABC_A_B", "label": "WT", "replica": 0, "dG": -10.0, "valid": True},
        {"complex_id": "1ABC_A_B", "label": "WT", "replica": 1, "dG": -12.0, "valid": True},
        {"complex_id": "1ABC_A_B", "label": "LA45G", "replica": 0, "dG": -5.0,
         "valid": True, "measured_ddg": 2.0},
        {"complex_id": "1ABC_A_B", "label": "LA45G", "replica": 1, "dG": -7.0,
         "valid": True, "measured_ddg": 2.0},
        {"complex_id": "1ABC_A_B", "label": "LA45G", "replica": 2, "dG": None,
         "valid": False},
    ]
    preds = S.aggregate_predictions(rows)
    assert len(preds) == 1
    p = preds[0]
    # ddg_pred = mean(mut -5,-7) - mean(WT -10,-12) = -6 - (-11) = +5
    assert p["ddg_pred"] == pytest.approx(5.0)
    assert p["measured_ddg"] == 2.0 and p["n_mut"] == 2 and p["n_wt"] == 2


def test_correlations_known_values():
    assert S._pearson([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert S._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert S._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert S._pearson([1, 2], [1, 2]) is None


def test_score_builds_wt_once_per_complex_and_is_resumable(tmp_path, monkeypatch):
    # fake load_records: 2 mutations on the same complex + 1 on another
    import skempi
    recs = [
        _rec("1AAA", "A", "B", [_mut("L", "A", "45", "G")], 1.0),
        _rec("1AAA", "A", "B", [_mut("I", "A", "50", "A")], -0.5),
        _rec("1BBB", "A", "B", [_mut("V", "A", "7", "D")], 0.2),
    ]
    for p in ("1aaa", "1bbb"):
        (tmp_path / f"{p}.pdb").write_text("ATOM\n")
    monkeypatch.setattr(skempi, "load_records", lambda c: (recs, {"csv_rows": 3}))

    calls = []

    def fake(task):
        calls.append((task["complex_id"], task["label"], task["replica"]))
        # dG: WT=-10; mutant depends on measured sign so correlation is positive
        dg = -10.0 if task["label"] == "WT" else -10.0 + 1.0
        return {"complex_id": task["complex_id"], "label": task["label"],
                "replica": task["replica"], "dG": dg, "valid": True,
                "measured_ddg": task.get("measured_ddg")}

    out = tmp_path / "s.jsonl"
    # 2 complexes: WT(1AAA) x2reps + 2 muts x2 + WT(1BBB) x2 + 1 mut x2 = 10 tasks
    n = S.score(str(tmp_path / "skempi.csv"), tmp_path, out, max_records=0,
                replicas=2, workers=1, _scorer=fake)
    assert n == 10
    # WT scored once per complex (not once per mutation)
    wt_1aaa = [c for c in calls if c[0] == "1AAA_A_B" and c[1] == "WT"]
    assert len(wt_1aaa) == 2    # 2 replicas, not 4 (would be 2 muts x 2)
    # resumable
    assert S.score(str(tmp_path / "skempi.csv"), tmp_path, out, max_records=0,
                   replicas=2, workers=1, _scorer=fake) == 0


def test_report_runs(tmp_path, capsys):
    rows = []
    # construct a positive correlation: measured and pred both increase together
    for i, pid in enumerate(["1AAA_A_B", "1BBB_A_B", "1CCC_A_B", "1DDD_A_B"]):
        rows.append({"complex_id": pid, "label": "WT", "replica": 0,
                     "dG": -10.0, "valid": True})
        rows.append({"complex_id": pid, "label": f"m{i}", "replica": 0,
                     "dG": -10.0 + i, "valid": True, "measured_ddg": float(i)})
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    S.report(p)
    out = capsys.readouterr().out
    assert "ACCURACY GATE" in out and "VERDICT" in out
    assert "Spearman" in out


def test_max_ca_filter_drops_oversized_complexes(tmp_path):
    def _pdb(path, n_ca, chain="A"):
        lines = [f"ATOM  {i:5d}  CA  ALA {chain}{i:4d}    0.0 0.0 0.0" for i in range(1, n_ca + 1)]
        Path(path).write_text("\n".join(lines) + "\n")
    _pdb(tmp_path / "1sml.pdb", 50)
    _pdb(tmp_path / "1big.pdb", 500)
    recs = [
        _rec("1SML", "A", "B", [_mut("L", "A", "10", "G")], 1.0),
        _rec("1BIG", "A", "B", [_mut("L", "A", "10", "G")], 1.0),
    ]
    # need a B chain too for keep_chains count; add it
    for f, nA in (("1sml.pdb", 50), ("1big.pdb", 500)):
        with (tmp_path / f).open("a") as fh:
            for i in range(1, 6):
                fh.write(f"ATOM  {9000+i:5d}  CA  ALA B{i:4d}    0.0 0.0 0.0\n")
    sel, reasons = S.select_records(recs, tmp_path, max_ca=200)
    assert reasons["too_big"] == 1 and reasons["kept"] == 1
    assert sel[0]["pdb_id"] == "1SML"


def test_fail_record_is_valid_false():
    t = {"complex_id": "1X_A_B", "label": "WT", "replica": 2}
    r = S._fail(t, "worker died (exitcode -9)")
    assert r["valid"] is False and r["complex_id"] == "1X_A_B"
    assert r["label"] == "WT" and r["replica"] == 2 and "died" in r["error"]
