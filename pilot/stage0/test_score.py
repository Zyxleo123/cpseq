"""Arithmetic tests for the Stage 0 scorer.

A bug here invalidates every number the pilot produces, and none of it is
checked by the pipeline running to completion -- a wrong regret still scores.
So the arithmetic is pinned against hand-computed values.

    PYTHONPATH=. python test_score.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import score

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _oracle_dir(candidates: list[dict]) -> Path:
    vals = [c["value"] for c in candidates if c["value"] is not None]
    blob = [{
        "decision_id": "t__X", "slice": "t", "split": "test", "pdb_id": "X", "k": 1,
        "oracle_value": min(vals), "worst_value": max(vals),
        "median_value": sorted(vals)[len(vals) // 2],
        "spread": max(vals) - min(vals), "candidates": candidates,
    }]
    d = Path(tempfile.mkdtemp())
    (d / "decisions_oracle.json").write_text(json.dumps(blob))
    return d


def _cand(cid: str, value):
    return {"cand_id": cid, "sites": [["A", "1"]], "value": value,
            "value_mean": value, "n_records": 1 if value is not None else 0,
            "substitutions": [], "additive_from_singles": None, "epistasis": None}


def _row(cid, valid=True, status="ok"):
    return [{"decision_id": "t__X", "arm": "a", "order": "main", "model_id": "m",
             "cand_id": cid, "choice_valid": valid, "status": status,
             "seconds": 1, "n_retries": 0, "presented_index": 0}]


def main() -> int:
    vals = {"c000": 0.5, "c001": -1.0, "c002": 3.0, "c003": 1.5}
    cands = [_cand(c, v) for c, v in vals.items()]
    oracle = score.load_oracle(_oracle_dir(cands))

    print("regret / normalised regret / oracle flag")
    for cid, want_reg, want_norm, want_oracle in [
        ("c001", 0.0, 0.000, True), ("c000", 1.5, 0.375, False),
        ("c003", 2.5, 0.625, False), ("c002", 4.0, 1.000, False),
    ]:
        rec = next(iter(score.per_decision(_row(cid), oracle).values()))
        check(f"{cid} regret={want_reg}",
              abs(rec["regret"] - want_reg) < 1e-9
              and abs(rec["norm_regret"] - want_norm) < 1e-9
              and rec["is_oracle"] == want_oracle,
              f"got {rec['regret']:.3f}/{rec['norm_regret']:.3f}")

    print("analytic random equals mean(value) - oracle, exactly")
    sr = next(iter(score.synth_random(oracle).values()))
    want = float(np.mean(list(vals.values()))) - min(vals.values())
    check("random_expected", abs(sr["regret"] - want) < 1e-9,
          f"got {sr['regret']:.4f} want {want:.4f}")

    print("an unparseable answer is not scored as zero regret")
    rec = next(iter(score.per_decision(_row(None, False, "invalid_no_tag"),
                                       oracle).values()))
    summ = score.summarise([rec])
    check("invalid stays unscored",
          rec["regret"] is None and summ["n_valid"] == 0
          and summ["invalid_rate"] == 1.0)

    print("paired test direction: A better => negative median difference")
    a = [{"decision_id": f"d{i}", "valid": True, "regret": 0.5} for i in range(12)]
    b = [{"decision_id": f"d{i}", "valid": True, "regret": 2.0} for i in range(12)]
    res = score.paired_test(a, b)
    check("direction", res["median_paired_diff"] < 0
          and res["direction"] == "A better" and res["wins"] == 12,
          f"median={res['median_paired_diff']}")

    print("a valid pick on an unmeasured position is not a model failure")
    oracle2 = score.load_oracle(_oracle_dir(cands + [_cand("c999", None)]))
    rec = next(iter(score.per_decision(_row("c999"), oracle2).values()))
    summ = score.summarise([rec])
    check("valid but unscorable",
          rec["valid"] and not rec["scorable"]
          and summ["invalid_rate"] == 0.0 and summ["n_valid"] == 0,
          f"coverage={summ['coverage_of_measured']}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all scorer arithmetic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
