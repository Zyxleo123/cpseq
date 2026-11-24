"""Tests for the LLM-arm validity gate. Synthetic rows, no API, no cluster.

The one that matters is `test_budget_cap_is_invalid_not_merely_noted`: the
whole point of the module is that a capped run must NOT be gradeable, because
the failure converges on the control arm and therefore looks like a result.

Run: python -m pytest agent/lnr/test_pll_llm_validity.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import pll_llm_validity as V  # noqa: E402


def _row(tid, *, arm="iter-llm", regret=0.1, n_gen=80, calls=80, cost=0.05,
         status_counts=None, fallback_gens=0, effort=None, effort_sent=None):
    return {"target_id": tid, "mut_arm": arm,
            "iter": {"regret": regret, "n_gen_run": n_gen, "llm_calls": calls,
                     "llm_cost_usd": cost, "fallback_gens": fallback_gens,
                     "effort": effort, "effort_sent": effort_sent,
                     # `is not None`, not `or`: an explicitly EMPTY dict means
                     # "no statuses recorded" (a target that never called) and
                     # must not be silently replaced by the all-ok default.
                     "status_counts": (status_counts
                                       if status_counts is not None
                                       else {"ok": n_gen})}}


def _write(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_clean_run_is_valid(tmp_path):
    p = _write(tmp_path / "a", [_row("t1"), _row("t2")])
    v = V.verdict(V.load(p), cap=10.0)
    assert v["verdict"] == "VALID"
    assert v["n_clean"] == 2 and v["n_fatal_budget_cap"] == 0


def test_budget_cap_is_invalid_not_merely_noted(tmp_path):
    """A capped target ran as random-EA under an LLM label. It must flip the
    verdict, because the resulting tie is indistinguishable from the real
    scientific conclusion it would otherwise be mistaken for."""
    rows = [_row("t1"),
            _row("t2", status_counts={"ok": 10, "budget_cap": 70},
                 fallback_gens=70, cost=0.0)]
    p = _write(tmp_path / "b", rows)
    v = V.verdict(V.load(p), cap=1.0)
    assert v["verdict"] == "INVALID_TAIL"
    assert v["fatal_targets"] == ["t2"]


def test_exhausted_cap_flips_verdict_even_if_no_row_says_budget_cap(tmp_path):
    """Spend at the cap is itself disqualifying: the next call would have
    failed, so the run ended for lack of money, not for lack of generations."""
    rows = [_row("t1", cost=0.5), _row("t2", cost=0.5)]
    p = _write(tmp_path / "c", rows)
    v = V.verdict(V.load(p), cap=1.0)
    assert v["verdict"] == "INVALID_TAIL"
    assert v["cap_exhausted"] is True
    # ...and the same rows under a cap that was never threatened are fine
    assert V.verdict(V.load(p), cap=100.0)["verdict"] == "VALID"


def test_no_usable_proposal_is_invalid_however_it_happened(tmp_path):
    """Two shapes, one meaning: the arm is the EA arm with a different name.

    `t2` made no calls at all. `t3` made all 80 and every one failed -- which is
    what quota exhaustion looks like (job 119773: 6qg8/6seo, 80 errors, $0.00
    billed). Keying on call COUNT would file t3 as merely suspect although it
    carries no LLM signal whatsoever, so the predicate is usable SUCCESSES.
    """
    rows = [_row("t1"),
            _row("t2", calls=0, cost=0.0, status_counts={}, fallback_gens=80),
            _row("t3", calls=80, cost=0.0, status_counts={"error": 80},
                 fallback_gens=80)]
    p = _write(tmp_path / "d", rows)
    v = V.verdict(V.load(p), cap=10.0)
    assert v["verdict"] == "INVALID_TAIL"
    assert v["never_called_targets"] == ["t2", "t3"]
    # neither is double-counted as "suspect", and both are named for exclusion
    assert v["suspect_targets"] == []
    assert v["exclude_arg"] == "t2,t3"


def test_api_error_text_is_surfaced(tmp_path, capsys):
    """The row keeps the first failure's text so the cause is readable without
    a fresh API probe -- diagnosing 119773 needed one to learn 'no credits'."""
    rows = [_row("t1", calls=80, cost=0.0, status_counts={"error": 80},
                 fallback_gens=80)]
    rows[0]["iter"]["llm_error_detail"] = "http_429: You have no credits remaining."
    p = _write(tmp_path / "dd", rows)
    V.report(p, cap=3.0)
    out = capsys.readouterr().out
    assert "no credits remaining" in out
    assert "--exclude t1" in out


def test_mostly_malformed_is_partial_not_invalid(tmp_path):
    """Declines and malformed replies are a property of the model, spread
    across targets rather than absorbing, so they degrade rather than
    disqualify -- a different verdict with a different remedy."""
    rows = [_row("t1"),
            _row("t2", status_counts={"ok": 20, "malformed": 60},
                 fallback_gens=60)]
    p = _write(tmp_path / "e", rows)
    v = V.verdict(V.load(p), cap=10.0)
    assert v["verdict"] == "PARTIAL"
    assert v["suspect_targets"] == ["t2"]


def test_ok_none_is_not_a_fallback(tmp_path):
    """'I choose not to mutate' is a real answer, not a failure to answer."""
    rows = [_row("t1", status_counts={"ok": 40, "ok_none": 40})]
    p = _write(tmp_path / "f", rows)
    r = V.load(p)[0]
    assert r["n_ok"] == 80 and r["n_fatal"] == 0
    assert V.verdict(V.load(p), cap=10.0)["verdict"] == "VALID"


def test_projection_recommends_a_cap_from_measured_cost(tmp_path):
    """The useful output of a capped run is the number that makes the next one
    valid, which is only knowable by measuring."""
    p = _write(tmp_path / "g", [_row("t1", cost=0.10), _row("t2", cost=0.20)])
    proj = V.projected_cap(V.load(p), 29)
    assert proj["mean_cost_per_target"] == 0.15
    assert proj["projected_total_usd"] == 4.35
    assert proj["recommended_cap_usd"] == 5.66   # 30% headroom


def test_only_the_named_arm_is_counted(tmp_path):
    """The EA control shares the file and has no LLM accounting; counting it
    would dilute every fraction here."""
    rows = [_row("t1"), _row("t1", arm="iter-ea", calls=0, cost=0.0,
                             status_counts={})]
    p = _write(tmp_path / "h", rows)
    assert [r["target_id"] for r in V.load(p)] == ["t1"]
    assert V.verdict(V.load(p), cap=10.0)["verdict"] == "VALID"


def test_report_returns_2_on_invalid_so_a_job_can_branch(tmp_path, capsys):
    rows = [_row("t1", status_counts={"ok": 5, "budget_cap": 75},
                 fallback_gens=75)]
    p = _write(tmp_path / "i", rows)
    rc = V.report(p, cap=1.0)
    out = capsys.readouterr().out
    assert rc == 2
    assert "INVALID_TAIL" in out
    assert "separable" in out      # names the specific misreading
    assert V.report(_write(tmp_path / "j", [_row("t1")]), cap=10.0) == 0


def test_no_rows_is_gated_and_names_the_key_trap(tmp_path, capsys):
    p = _write(tmp_path / "k", [_row("t1", arm="iter-ea")])
    rc = V.report(p, cap=1.0)
    assert rc == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_a_rejected_reasoning_effort_invalidates_the_hi_effort_arm(tmp_path, capsys):
    """An effort the API refused is a silent duplicate of the low-effort arm.

    The OpenAI client drops `reasoning_effort` and retries without it on a 400,
    so `iter-llm-hi` can look entirely healthy -- calls made, money spent, no
    fallbacks -- while running at the model's default effort. Graded naively it
    would say "effort does not help", which is the same shape as the real
    conclusion and therefore has to be gated, not noted.
    """
    rows = [_row("t1", arm="iter-llm-hi", effort="high", effort_sent=True),
            _row("t2", arm="iter-llm-hi", effort="high", effort_sent=False)]
    p = _write(tmp_path / "eff", rows)
    v = V.verdict(V.load(p, arm="iter-llm-hi"))
    assert v["verdict"] == "INVALID_EFFORT"
    assert v["effort_dropped_targets"] == ["t2"]
    assert v["efforts_claimed"] == ["high"]
    # and it must be a HARD gate, so the job can refuse to grade
    assert V.report(p, arm="iter-llm-hi") == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "does not help" in out


def test_rows_without_the_effort_field_are_not_penalised(tmp_path):
    """`effort_sent` is absent on every row written before the field existed.

    Absence is not evidence that the effort was dropped, so it must not flip a
    verdict -- otherwise adding the check retroactively invalidates the runs
    already recorded in docs/seqmut_notes.md.
    """
    p = _write(tmp_path / "old", [_row("t1"), _row("t2")])
    assert V.verdict(V.load(p))["verdict"] == "VALID"


def test_low_effort_arm_is_unaffected_by_the_effort_check(tmp_path):
    """effort=low that WAS sent is fine; the check keys on rejection only."""
    p = _write(tmp_path / "lo",
               [_row("t1", effort="low", effort_sent=True)])
    assert V.verdict(V.load(p))["verdict"] == "VALID"
