"""Fast unit tests for the sequence-mutation lever. No Rosetta, no solver.

These pin the pure scoring (which MUST match report.py byte-for-byte), the
mutation arms, the resumable scan cache, and the scorer wiring -- the parts that
would otherwise fail silently and be read as a result. The real mutant scan is a
subprocess PyRosetta call exercised under SLURM; here it is pre-seeded into the
cache so the scorer runs with no Rosetta on the path.

Run: python -m pytest agent/lnr/test_seqmut.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import seqmut as S  # noqa: E402


class _T:
    """Minimal stand-in for env.LNRTarget: the fields seqmut actually reads."""
    def __init__(self, tid, seq, per_res):
        self.target_id = tid
        self.seq = seq
        self.length = len(seq)
        self.per_res_contacts = tuple(per_res)
        self.n_contacts = sum(per_res)
        self.pdb_path = ""
        self.pep_chain = "B"
        self.rec_chain = "A"


# --- pure scoring -----------------------------------------------------------

def test_energy_weighted_retention_matches_report_formula():
    # positive ddG = favourable; negatives and zeros contribute nothing.
    pr = {0: 2.0, 1: -1.0, 2: 3.0, 3: 0.0, 4: 1.0}   # total positive = 6.0
    assert abs(S.energy_weighted_retention(pr, 0, 2) - (2.0 + 3.0) / 6.0) < 1e-9
    assert abs(S.energy_weighted_retention(pr, 2, 4) - (3.0 + 1.0) / 6.0) < 1e-9
    # string keys (as JSON round-trips them) are coerced, like report.py
    assert abs(S.energy_weighted_retention({"0": 2.0, "2": 3.0}, 0, 2)
               - (2.0 + 3.0) / 5.0) < 1e-9
    # all-nonpositive -> denominator falls back to 1.0, numerator 0 -> 0.0
    assert S.energy_weighted_retention({0: -1.0, 1: 0.0}, 0, 1) == 0.0


def test_recovered_binding_energy_sign_and_none():
    pr = {0: 2.0, 1: 3.0, 2: 1.0}                     # total positive = 6.0
    # native_dG is negative (favourable); recovered = retention * native_dG < 0
    assert abs(S.recovered_binding_energy(pr, -60.0, 0, 1) - (5.0 / 6.0 * -60.0)) < 1e-9
    # holding more hotspot energy => MORE NEGATIVE (better)
    more = S.recovered_binding_energy(pr, -60.0, 0, 2)   # retention 1.0
    less = S.recovered_binding_energy(pr, -60.0, 0, 0)   # retention 2/6
    assert more < less
    # unscored mutant -> None, never silently 0
    assert S.recovered_binding_energy(pr, None, 0, 1) is None


def test_subs_key_is_order_independent_and_apply_subs():
    assert S.subs_key([(3, "A"), (1, "k")]) == S.subs_key([(1, "K"), (3, "a")])
    assert S.subs_key([]) == ""
    assert S.apply_subs("ACDEF", [(0, "W"), (4, "G")]) == "WCDEG"
    # out-of-range / non-standard are ignored, not crashed
    assert S.apply_subs("ACDEF", [(9, "W"), (1, "Z")]) == "ACDEF"


# --- mutation arms ----------------------------------------------------------

def test_nomut_is_empty_and_random_mut_is_seeded_distinct_and_nonwt():
    t = _T("x1", "ACDEFGHIKL", [0, 5, 9, 9, 3, 0, 1, 8, 2, 0])
    win = {"a": 1, "b": 6, "retention": 0.5, "clean": True}
    assert S.propose_subs("nomut", t, win, 2, t.seq) == []
    r1 = S.propose_subs("random_mut", t, win, 2, t.seq)
    r2 = S.propose_subs("random_mut", t, win, 2, t.seq)
    assert r1 == r2                                   # deterministic (paired floor)
    assert len(r1) == 2
    pos = [i for i, _ in r1]
    assert len(set(pos)) == 2                         # distinct positions
    for i, aa in r1:
        assert aa in S.AA20 and aa != t.seq[i]        # a real, non-WT substitution
        assert t.per_res_contacts[i] > 0              # drawn from interface positions


def test_random_mut_respects_budget_and_k_zero():
    t = _T("x2", "ACDEFG", [1, 1, 1, 1, 1, 1])
    assert S.propose_subs("random_mut", t, {"a": 0, "b": 5}, 0, t.seq) == []
    assert len(S.propose_subs("random_mut", t, {"a": 0, "b": 5}, 3, t.seq)) == 3


def test_conservative_avoids_hotspots_and_anti_targets_them():
    t = _T("c1", "ACDEFGHIKL", [2, 9, 2, 9, 2, 2, 2, 9, 2, 2])   # all interface
    win = {"a": 0, "b": 9}
    wt_ala = {0: 0.1, 1: 5.0, 2: 0.2, 3: 6.0, 4: -0.5,
              5: 0.0, 6: 0.3, 7: 4.0, 8: 0.1, 9: 0.2}
    cons = S.propose_subs("conservative", t, win, 2, t.seq, wt_ala=wt_ala)
    assert sorted(i for i, _ in cons) == [4, 5]      # the two least-binding sites
    for i, aa in cons:
        assert aa in S.AA20 and aa != t.seq[i]
    anti = S.propose_subs("anti", t, win, 2, t.seq, wt_ala=wt_ala)
    assert sorted(i for i, _ in anti) == [1, 3]      # the two hotspots
    assert all(aa == "A" for _, aa in anti)          # knocked to alanine


def test_conservative_without_ala_degrades_to_random():
    t = _T("c2", "ACDEFG", [1, 1, 1, 1, 1, 1])
    got = S.propose_subs("conservative", t, {"a": 0, "b": 5}, 2, t.seq, wt_ala=None)
    assert len(got) == 2                             # falls back to random_mut


def test_seed_scans_reuses_prior_cache_and_persists(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({"target_id": "z", "subs_key": "3:W",
                               "native_dG": -50.0,
                               "per_residue_ddG_ala": {"3": 1.0}}) + "\n")
    c = S.ScanCache(tmp_path / "dst.jsonl")
    assert c.seed_scans(src) == 1
    assert c.get("z", [(3, "W")])["native_dG"] == -50.0
    c.close()
    c2 = S.ScanCache(tmp_path / "dst.jsonl")          # persisted -> resumable
    assert c2.get("z", [(3, "W")]) is not None


# --- resumable cache --------------------------------------------------------

def test_scan_cache_roundtrip_resume_and_seed_wt(tmp_path):
    p = tmp_path / "scans.jsonl"
    c = S.ScanCache(p)
    row = {"target_id": "x1", "subs_key": S.subs_key([(2, "W")]),
           "native_dG": -50.0, "per_residue_ddG_ala": {"2": 1.0}}
    c.put(row)
    assert c.get("x1", [(2, "W")])["native_dG"] == -50.0
    c.close()
    # resume: a fresh cache over the same file sees the persisted row
    c2 = S.ScanCache(p)
    assert c2.get("x1", [(2, "W")]) is not None
    assert c2.get("x1", []) is None                   # a different key is absent
    # seed WT from a rosetta_interface-style file -> the empty-subs key
    rif = tmp_path / "rosetta_interface.jsonl"
    rif.write_text('{"target_id": "x1", "native_dG": -42.0, '
                   '"per_residue_ddG_ala": {"0": 2.0}}\n'
                   '{"target_id": "x2", "error": "timeout"}\n')
    n = c2.seed_wt(rif)
    assert n == 1                                     # only the scored target
    wt = c2.get("x1", [])
    assert wt["native_dG"] == -42.0 and wt["seeded_from_wt"] is True


# --- scorer wiring (cache pre-seeded, so no PyRosetta) -----------------------

def test_score_target_uses_mutant_scan_and_same_window_across_arms(tmp_path):
    t = _T("x1", "ACDEFGHIKL", [0, 5, 9, 9, 3, 0, 1, 8, 2, 0])
    win = {"a": 1, "b": 4, "retention": 0.5, "clean": True}
    c = S.ScanCache(tmp_path / "scans.jsonl")
    # WT scan (nomut): hotspots at 2,3 inside the window
    c.put({"target_id": "x1", "subs_key": "",
           "native_dG": -60.0,
           "per_residue_ddG_ala": {"1": 1.0, "2": 2.0, "3": 2.0, "6": 1.0}})
    nomut = S.score_target(t, "nomut", win, c, 2, 1, 1.0)
    # WT energy retention = (1+2+2)/(1+2+2+1) over [1,4] = 5/6 (stored rounded 4dp)
    assert abs(nomut["energy_retention"] - 5.0 / 6.0) < 1e-3
    assert abs(nomut["recovered_binding_energy"] - (5.0 / 6.0 * -60.0)) < 1e-2
    assert nomut["committed_window"] == [1, 4]

    # a mutation that re-scans to a STRONGER, better-placed interface: the mutant
    # scan is what is scored, NOT the WT table (the correctness point).
    subs = [(4, "W")]
    c.put({"target_id": "x1", "subs_key": S.subs_key(subs),
           "native_dG": -75.0,
           "per_residue_ddG_ala": {"1": 1.0, "2": 2.0, "3": 2.0, "4": 3.0, "6": 1.0}})
    mut = S.score_target(t, "llm", win, c, 2, 1, 1.0, subs_override=subs)
    # mutant energy retention over [1,4] = (1+2+2+3)/(1+2+2+3+1) = 8/9, native -75
    assert abs(mut["energy_retention"] - 8.0 / 9.0) < 1e-3
    assert abs(mut["recovered_binding_energy"] - (8.0 / 9.0 * -75.0)) < 1e-2
    # same committed window for both arms -> closure is substitution-invariant
    assert mut["committed_window"] == nomut["committed_window"]
    # and the mutant kept MORE absolute binding energy (more negative)
    assert mut["recovered_binding_energy"] < nomut["recovered_binding_energy"]


def test_score_target_reports_unscored_mutant_as_none_not_zero(tmp_path):
    t = _T("x1", "ACDEF", [1, 1, 1, 1, 1])
    win = {"a": 0, "b": 2, "retention": 0.4, "clean": False}
    c = S.ScanCache(tmp_path / "scans.jsonl")
    c.put({"target_id": "x1", "subs_key": S.subs_key([(1, "W")]),
           "error": "timeout"})                        # mutant scan failed
    row = S.score_target(t, "llm", win, c, 1, 1, 1.0, subs_override=[(1, "W")])
    assert row["recovered_binding_energy"] is None     # never silently 0
    assert row["energy_retention"] is None
    assert row["scan_error"] == "timeout"


# --- LLM mutation arm: parsing ----------------------------------------------

def test_parse_subs_valid_none_and_clamp():
    # valid, both "3:W" and "3W" token forms, last tag wins
    assert S.parse_subs("x <mut>3:W, 5:L</mut>", 10, 3)[0] == [(3, "W"), (5, "L")]
    assert S.parse_subs("<mut>3W 5L</mut>", 10, 3)[0] == [(3, "W"), (5, "L")]
    assert S.parse_subs("<mut>1:A</mut> no <mut>2:G</mut>", 10, 3)[0] == [(2, "G")]
    # explicit decline-to-mutate is VALID and distinct from malformed
    assert S.parse_subs("reason <mut>none</mut>", 10, 3) == ([], "ok_none")
    # budget clamp: only the first k_mut valid substitutions are kept
    assert S.parse_subs("<mut>0:W,1:L,2:F,3:Y</mut>", 10, 2)[0] == [(0, "W"), (1, "L")]


def test_parse_subs_drops_bad_tokens_and_flags_unparseable():
    # out of range, non-standard residue (B,Z,X), duplicate position -> dropped
    got, st = S.parse_subs("<mut>99:W, 2:B, 3:L, 3:F</mut>", 5, 5)
    assert got == [(3, "L")] and st == "ok"            # 99 oob, B invalid, 3 dup
    # no tag / no valid token -> None so the caller retries then records malformed
    assert S.parse_subs("I think position 3 to W", 10, 3) == (None, "no_tag")
    assert S.parse_subs("<mut>garbage here</mut>", 10, 3) == (None, "no_valid_subs")


# --- LLM mutation arm: wiring (scripted backend, no spend) -------------------

def _scripted(replies):
    from lp2cp.llm.anthropic_api import ScriptedClient
    return ScriptedClient(replies, model_id="scripted")


def test_llm_propose_subs_parses_and_reports_status():
    t = _T("x1", "ACDEFGHIKL", [0, 5, 9, 9, 3, 0, 1, 8, 2, 0])
    win = {"a": 1, "b": 4, "retention": 0.5, "clean": True}
    subs, meta = S.llm_propose_subs(_scripted(["<mut>2:W,3:Y</mut>"]), t, win, 2)
    assert subs == [(2, "W"), (3, "Y")]
    assert meta["llm_status"] == "ok" and meta["n_proposed"] == 2
    assert meta["llm_cost_usd"] == 0.0 and meta["model_id"] == "scripted"
    # explicit no-mutation
    subs, meta = S.llm_propose_subs(_scripted(["<mut>none</mut>"]), t, win, 2)
    assert subs == [] and meta["llm_status"] == "ok_none"


def test_llm_propose_subs_retries_then_marks_malformed():
    t = _T("x1", "ACDEFGHIKL", [0, 5, 9, 9, 3, 0, 1, 8, 2, 0])
    win = {"a": 1, "b": 4, "retention": 0.5, "clean": True}
    # both the first reply and the retry are unparseable -> malformed, empty subs
    subs, meta = S.llm_propose_subs(
        _scripted(["no tag at all", "still nothing"]), t, win, 2, max_retries=1)
    assert subs == [] and meta["llm_status"] == "malformed"
    assert meta["llm_calls"] == 2                      # it used the retry


def test_mut_brief_states_the_pose_preserving_assumption_and_window():
    t = _T("x1", "ACDEFGHIKL", [0, 5, 9, 9, 3, 0, 1, 8, 2, 0])
    win = {"a": 2, "b": 6, "retention": 0.5, "clean": True}
    plain = S.mut_brief(t, win, 2, wt_ala=None)
    assert "[2,6]" in plain and "<mut>" in plain
    assert "NATIVE BINDING MODE" in plain              # the scorer's assumption, stated
    assert "bindE" not in plain                        # no evidence column on the plain arm
    # the -ala evidence arm adds the WT alanine-scan binding column
    ala = S.mut_brief(t, win, 2, wt_ala={0: 2.0, 2: 3.0})
    assert "bindE" in ala and "+3.00" in ala
