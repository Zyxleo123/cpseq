"""Tests for the CPSet within-target accuracy gate. No PyRosetta, no network.

The pure layer is what decides the headline, so it is what is tested: the sign
convention, the pair statistic, the Wilson interval, the pre-registered verdict,
and the structure-prep rules (chains from files not CSV, CYX->CYS, caps dropped
and counted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lnr import cpset_accuracy as ca


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _truth(entries):
    """entries: {pdb: (pkd, series)} -> a truth dict shaped like load_dataset1."""
    return {p: {"pdb": p, "pkd": pk, "series": s, "chain_type": "BB",
                "pep_len": 7, "polar_pct": 50.0, "kd_m": 1e-6,
                "chain_id_csv": "A:P"}
            for p, (pk, s) in entries.items()}


def _atom(serial, name, res, chain, resseq, altloc=" ", record="ATOM  "):
    # columns: 1-6 record, 7-11 serial, 13-16 name, 17 altloc, 18-20 resname,
    # 22 chain, 23-26 resseq
    return (f"{record}{serial:>5d} {name:<4s}{altloc}{res:>3s} {chain}"
            f"{resseq:>4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}\n")


# ---------------------------------------------------------------------------
# frozen table integrity
# ---------------------------------------------------------------------------

def test_frozen_series_table_is_self_consistent():
    all_pdbs = [p for s in ca.SERIES for p in s["pdbs"]]
    assert len(all_pdbs) == len(set(all_pdbs)), "a PDB appears in two series"
    assert len(all_pdbs) == 38, "the frozen table should cover 38 of the 50"
    assert len(ca.SERIES) == 10
    # every tier label is one of the three, and matches its own members
    for s in ca.SERIES:
        assert s["tier"] in ("BB", "MIX", "SS")
        assert len(s["pdbs"]) >= 2
    # the headline KEAP1 series is the head-to-tail one
    keap = next(s for s in ca.SERIES if s["id"] == "S03_keap1")
    assert keap["tier"] == "BB"
    assert set(keap["pdbs"]) == {"7K2E", "7K2F", "7K2G", "7K2H", "7K2I", "7K2M"}


def test_total_within_series_pairs_is_70():
    tot = sum(len(s["pdbs"]) * (len(s["pdbs"]) - 1) // 2 for s in ca.SERIES)
    assert tot == 70
    bb = sum(len(s["pdbs"]) * (len(s["pdbs"]) - 1) // 2
             for s in ca.SERIES if s["tier"] in ("BB", "MIX"))
    assert bb == 28


def test_series_tier_classification():
    assert ca.series_tier(["BB", "BB"]) == "BB"
    assert ca.series_tier(["BB+SS", "BB"]) == "BB"
    assert ca.series_tier(["BB+SS", "SS*1"]) == "MIX"
    assert ca.series_tier(["SS*1", "SS*2"]) == "SS"


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------

def test_pairs_are_within_series_only():
    truth = _truth({"A1": (9.0, "sX"), "A2": (6.0, "sX"),
                    "B1": (8.0, "sY"), "B2": (5.0, "sY")})
    pairs = ca.within_series_pairs(truth)
    assert len(pairs) == 2
    assert {p["series"] for p in pairs} == {"sX", "sY"}
    # never a cross-series pair, which is what would import the easy contrast
    for p in pairs:
        assert truth[p["strong"]]["series"] == truth[p["weak"]]["series"]


def test_pair_orientation_follows_measured_pkd():
    truth = _truth({"WEAK": (4.0, "s"), "STRONG": (8.0, "s")})
    (pair,) = ca.within_series_pairs(truth)
    assert pair["strong"] == "STRONG" and pair["weak"] == "WEAK"
    assert pair["dpkd"] == pytest.approx(4.0)


def test_pairs_exclude_singletons_and_unscored():
    truth = _truth({"A1": (9.0, "s"), "A2": (6.0, "s"), "LONE": (7.0, None)})
    assert len(ca.within_series_pairs(truth)) == 1
    # restricting the pool to scored complexes drops the pair entirely
    assert ca.within_series_pairs(truth, pdbs={"A1"}) == []


def test_dpkd_threshold_filters_coin_flip_pairs():
    truth = _truth({"A": (8.46, "s"), "B": (8.44, "s"), "C": (5.00, "s")})
    assert len(ca.within_series_pairs(truth, min_dpkd=0.0)) == 3
    # the 0.02 pair (the real 6O21/4K1E case) must drop out
    kept = ca.within_series_pairs(truth, min_dpkd=0.3)
    assert len(kept) == 2
    assert all(p["dpkd"] >= 0.3 for p in kept)


# ---------------------------------------------------------------------------
# the sign convention -- the project's recurring bug class
# ---------------------------------------------------------------------------

def test_a_perfect_scorer_scores_1_and_a_sign_flipped_one_scores_0():
    truth = _truth({"S": (9.0, "s"), "M": (7.0, "s"), "W": (4.0, "s")})
    pairs = ca.within_series_pairs(truth)
    # perfect: stronger binder (higher pKd) has the MORE NEGATIVE dG
    good = {"S": -40.0, "M": -25.0, "W": -10.0}
    assert ca.pair_agreement(pairs, good)["agreement"] == 1.0
    # a sign slip is caught, not rewarded
    bad = {k: -v for k, v in good.items()}
    assert ca.pair_agreement(pairs, bad)["agreement"] == 0.0


def test_ties_count_as_failures():
    truth = _truth({"A": (9.0, "s"), "B": (5.0, "s")})
    pairs = ca.within_series_pairs(truth)
    flat = {"A": -20.0, "B": -20.0}
    res = ca.pair_agreement(pairs, flat)
    assert res["n_pairs"] == 1 and res["n_correct"] == 0


def test_missing_scores_are_skipped_not_counted_wrong():
    truth = _truth({"A": (9.0, "s"), "B": (5.0, "s")})
    pairs = ca.within_series_pairs(truth)
    res = ca.pair_agreement(pairs, {"A": -20.0})
    assert res["n_pairs"] == 0 and res["skipped"] == 1
    assert res["agreement"] is None


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = ca.wilson95(35, 50)
    assert lo < 0.70 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_interval_narrows_with_n():
    narrow = ca.wilson95(70, 100)
    wide = ca.wilson95(7, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_at_a_coin_flip_straddles_one_half():
    lo, hi = ca.wilson95(35, 70)
    assert lo < 0.5 < hi, "70 coin flips must not look better than chance"


def test_wilson_handles_zero_n():
    assert ca.wilson95(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# the pre-registered verdict
# ---------------------------------------------------------------------------

def _stats(ci_lo, mean_rho, frac_neg_rhos):
    return {"headline": {"ci95": (ci_lo, 1.0), "n_pairs": 28},
            "series_rho": [mean_rho], "series_rho_n3": frac_neg_rhos}


def test_verdict_fails_at_a_coin_flip():
    label, _ = ca.verdict(_stats(0.48, -0.9, [-0.9, -0.9]))
    assert label == "FAIL", "a CI touching 0.5 cannot be a pass at any rho"


def test_verdict_passes_only_when_all_three_conditions_hold():
    assert ca.verdict(_stats(0.70, -0.55, [-0.5, -0.6, -0.7]))[0] == "PASS"
    # strong pairs but flat per-series rho => WEAK, not PASS
    assert ca.verdict(_stats(0.70, -0.10, [-0.5, -0.6, -0.7]))[0] == "WEAK"
    # strong rho but pairs only marginally above chance => WEAK
    assert ca.verdict(_stats(0.55, -0.80, [-0.5, -0.6, -0.7]))[0] == "WEAK"
    # majority of series with the WRONG sign => WEAK
    assert ca.verdict(_stats(0.70, -0.55, [+0.5, +0.6, -0.7]))[0] == "WEAK"


def test_verdict_reasons_always_report_all_three_criteria():
    _, reasons = ca.verdict(_stats(0.70, -0.55, [-0.5]))
    assert len(reasons) == 3
    assert any("CI95 lower bound" in r for r in reasons)
    assert any("per-series rho" in r for r in reasons)
    assert any("negative rho" in r for r in reasons)


def test_verdict_survives_empty_stats():
    label, reasons = ca.verdict({})
    assert label == "FAIL" and len(reasons) == 3


# ---------------------------------------------------------------------------
# structure preparation
# ---------------------------------------------------------------------------

def test_chains_are_read_from_files_not_from_the_csv(tmp_path):
    """The 1SLE case: the CSV says BD:M, the files hold D and P. Trusting the
    CSV gives InterfaceAnalyzer a spec matching nothing, with no error."""
    pro = tmp_path / "x_protein.pdb"
    cp = tmp_path / "x_CP.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "D", 1))
    cp.write_text(_atom(2, "CA", "GLY", "P", 1))
    assert ca.derive_chains(str(pro), str(cp)) == ("D", "P")


def test_derive_chains_rejects_collisions_and_multichain_peptides(tmp_path):
    pro = tmp_path / "p.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "A", 1))
    same = tmp_path / "c1.pdb"
    same.write_text(_atom(2, "CA", "GLY", "A", 1))
    assert ca.derive_chains(str(pro), str(same)) is None, "collision not caught"
    two = tmp_path / "c2.pdb"
    two.write_text(_atom(2, "CA", "GLY", "P", 1) + _atom(3, "CA", "GLY", "Q", 1))
    assert ca.derive_chains(str(pro), str(two)) is None


def test_prepare_renames_cyx_drops_caps_and_counts_both(tmp_path):
    pro = tmp_path / "x_protein.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "A", 1) + _atom(2, "CA", "LEU", "A", 2))
    cp = tmp_path / "x_CP.pdb"
    cp.write_text(
        _atom(10, "C", "ACE", "P", 0, record="HETATM")
        + _atom(11, "CA", "CYX", "P", 1)
        + _atom(12, "CA", "HIS", "P", 2)
        + _atom(13, "CA", "CYX", "P", 3)
        + _atom(14, "N", "NHE", "P", 4, record="HETATM")
    )
    out, prep = ca.prepare_pdb(str(pro), str(cp), tmp_path / "merged.pdb")
    assert out is not None, prep
    text = Path(out).read_text()
    assert "CYX" not in text, "CYX must be renamed so Rosetta finds the disulfide"
    assert text.count("CYS") == 2
    assert "ACE" not in text and "NHE" not in text
    assert prep["renamed"] == {"CYX": 2}
    assert prep["dropped_caps"] == {"ACE": 1, "NHE": 1}
    assert prep["n_res_rec"] == 2 and prep["n_res_pep"] == 3
    assert prep["interface"] == "A_P"
    # HETATM is rewritten to ATOM so a renamed cap-neighbour is never a ligand
    assert "HETATM" not in text


def test_prepare_builds_multichain_receptor_interface_spec(tmp_path):
    pro = tmp_path / "x_protein.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "A", 1) + _atom(2, "CA", "LEU", "B", 1))
    cp = tmp_path / "x_CP.pdb"
    cp.write_text(_atom(3, "CA", "GLY", "C", 1))
    out, prep = ca.prepare_pdb(str(pro), str(cp), tmp_path / "m.pdb")
    assert out is not None
    assert prep["interface"] == "AB_C", "the 6D3X dimer case"


def test_prepare_drops_secondary_altlocs(tmp_path):
    pro = tmp_path / "x_protein.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "A", 1, altloc="A")
                   + _atom(2, "CA", "ALA", "A", 1, altloc="B"))
    cp = tmp_path / "x_CP.pdb"
    cp.write_text(_atom(3, "CA", "GLY", "P", 1))
    out, prep = ca.prepare_pdb(str(pro), str(cp), tmp_path / "m.pdb")
    assert out is not None
    assert prep["altloc_dropped"] == 1
    assert prep["n_res_rec"] == 1


def test_prepare_fails_loudly_when_cleaning_empties_a_side(tmp_path):
    pro = tmp_path / "x_protein.pdb"
    pro.write_text(_atom(1, "CA", "ALA", "A", 1))
    cp = tmp_path / "x_CP.pdb"
    cp.write_text(_atom(2, "C", "ACE", "P", 1, record="HETATM"))
    out, prep = ca.prepare_pdb(str(pro), str(cp), tmp_path / "m.pdb")
    assert out is None and "error" in prep


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_regroup_merges_identical_receptors_and_splits_different_ones():
    sigs = {"A1": "MKVLAAGIVGLNT" * 6, "A2": "MKVLAAGIVGLNT" * 6,
            "B1": "QQWWEERRTTYYU" * 6}
    groups, borderline = ca.regroup(sigs, threshold=0.95)
    sets = sorted((sorted(v) for v in groups.values()), key=len, reverse=True)
    assert sets[0] == ["A1", "A2"]
    assert ["B1"] in sets
    assert borderline == []


def test_regroup_reports_borderline_identities_rather_than_merging_them():
    base = "MKVLAAGIVGLNTAAGIVGLNTMKVLAAGIVG"
    near = base[:24] + "WWWWWWWW"   # ~75% identical: not merged, but reported
    groups, borderline = ca.regroup({"A": base, "B": near}, threshold=0.95)
    assert len(groups) == 2
    assert borderline and borderline[0][2] < 0.95


# ---------------------------------------------------------------------------
# aggregation and stats
# ---------------------------------------------------------------------------

def test_aggregate_averages_replicas_and_reports_the_rulers_noise():
    rows = [
        {"pdb": "A", "replica": 0, "dG": -20.0, "valid": True, "pkd": 9.0,
         "series": "s", "tier": "BB", "scorer": "rosetta"},
        {"pdb": "A", "replica": 1, "dG": -24.0, "valid": True, "pkd": 9.0,
         "series": "s", "tier": "BB", "scorer": "rosetta"},
        {"pdb": "B", "replica": 0, "dG": -10.0, "valid": True, "pkd": 5.0,
         "series": "s", "tier": "BB", "scorer": "rosetta"},
        {"pdb": "C", "replica": 0, "valid": False, "error": "segfault"},
    ]
    agg = ca.aggregate(rows)
    assert set(agg) == {"A", "B"}, "invalid rows must not enter the aggregate"
    assert agg["A"]["dG"] == pytest.approx(-22.0)
    assert agg["A"]["n"] == 2 and agg["A"]["sd"] > 0
    assert agg["B"]["sem"] is None, "a single replica has no SEM to claim"


def test_compute_stats_separates_within_series_from_cross_complex():
    # Two real series; within each, the scorer is PERFECT. Across series the
    # dG scale is deliberately inverted, so a pooled statistic would look awful
    # while the within-series statistic is 1.0. This is the inflation guard.
    truth = {}
    for pdb, pkd, sid in (("7K2H", 6.85, "S03_keap1"), ("7K2G", 3.55, "S03_keap1"),
                          ("6D3X", 10.29, "S05_6d3"), ("6D3Y", 8.70, "S05_6d3")):
        truth[pdb] = {"pdb": pdb, "pkd": pkd, "series": sid, "chain_type": "BB"}
    agg = {
        "7K2H": {"dG": -30.0, "sem": 0.5, "sd": 1.0, "n": 3, "pkd": 6.85,
                 "series": "S03_keap1", "tier": "BB"},
        "7K2G": {"dG": -20.0, "sem": 0.5, "sd": 1.0, "n": 3, "pkd": 3.55,
                 "series": "S03_keap1", "tier": "BB"},
        "6D3X": {"dG": -5.0, "sem": 0.5, "sd": 1.0, "n": 3, "pkd": 10.29,
                 "series": "S05_6d3", "tier": "BB"},
        "6D3Y": {"dG": -1.0, "sem": 0.5, "sd": 1.0, "n": 3, "pkd": 8.70,
                 "series": "S05_6d3", "tier": "BB"},
    }
    stats = ca.compute_stats(agg, truth)
    assert stats["bins"][0.0]["all"]["agreement"] == 1.0
    assert stats["bins"][1.0]["all"]["n_pairs"] == 2
    # the cross-complex number is the inverted one and must NOT leak into
    # the within-series result
    assert stats["cross_complex"]["pearson"] > 0
    assert stats["noise"]["resolution"] is not None


def test_prep_flags_surface_caps_and_length_mismatches_by_series():
    agg = {
        "1VWB": {"series": "S01_strep", "prep": {"caps": {"ACE": 3, "NHE": 1},
                                                 "n_res_pep": 6, "pep_len_csv": 6}},
        "1SLD": {"series": "S01_strep", "prep": {"caps": {"ACE": 3},
                                                 "n_res_pep": 6, "pep_len_csv": 6}},
        "6O21": {"series": "S08_fcqrs", "prep": {"caps": {},
                                                 "n_res_pep": 14, "pep_len_csv": 11}},
        "7K2E": {"series": "S03_keap1", "prep": {"caps": {},
                                                 "n_res_pep": 7, "pep_len_csv": 7}},
    }
    flags = ca.prep_flags(agg)
    assert set(flags) == {"S01_strep", "S08_fcqrs"}
    assert sorted(flags["S01_strep"]["capped"]) == ["1SLD", "1VWB"]
    assert flags["S08_fcqrs"]["len_mismatch"] == ["6O21(14vs11)"]
    # the clean head-to-tail series must NOT be flagged
    assert "S03_keap1" not in flags


def test_prep_flags_ignore_complexes_outside_any_series():
    agg = {"3PP4": {"series": None, "prep": {"caps": {"NHE": 1}}}}
    assert ca.prep_flags(agg) == {}


def test_resolution_uses_within_series_spread_not_the_cross_target_one():
    """The guard against flattering the ruler with the wrong contrast.

    Two series whose members are nearly identical in dG (so within-series
    resolution is poor) but which sit far apart on the dG scale (so the
    cross-target spread is huge). Honest resolution must be small; the inflated
    variant must be much larger, and must be labelled as such.
    """
    truth, agg = {}, {}
    for pdb, pkd, sid, dg in (("7K2H", 6.85, "S03_keap1", -20.0),
                              ("7K2G", 3.55, "S03_keap1", -20.5),
                              ("6D3X", 10.29, "S05_6d3", -200.0),
                              ("6D3Y", 8.70, "S05_6d3", -200.5)):
        truth[pdb] = {"pdb": pdb, "pkd": pkd, "series": sid, "chain_type": "BB"}
        agg[pdb] = {"dG": dg, "sem": 1.0, "sd": 2.0, "n": 3, "pkd": pkd,
                    "series": sid, "tier": "BB", "prep": {}}
    nz = ca.compute_stats(agg, truth)["noise"]
    assert nz["resolution"] < 1.0, "within-series variants are NOT resolvable here"
    assert nz["resolution_cross_target_INFLATED"] > 50
    assert nz["within_series_sd"] < nz["cross_target_sd"]


def test_resolution_is_none_when_no_series_has_two_scored_members():
    truth = {"3PP4": {"pdb": "3PP4", "pkd": 5.15, "series": None,
                      "chain_type": "SS*1"}}
    agg = {"3PP4": {"dG": -20.0, "sem": None, "sd": 0.0, "n": 1, "pkd": 5.15,
                    "series": None, "tier": "-", "prep": {}}}
    nz = ca.compute_stats(agg, truth)["noise"]
    assert nz["within_series_sd"] is None and nz["resolution"] is None


def test_compute_stats_bb_tier_is_a_subset_of_all():
    truth = {p: {"pdb": p, "pkd": k, "series": s, "chain_type": "x"}
             for p, k, s in (("7K2H", 6.85, "S03_keap1"),
                             ("7K2G", 3.55, "S03_keap1"),
                             ("4GW1", 6.37, "S02_qfdlstrrlk"),
                             ("5EUK", 4.30, "S02_qfdlstrrlk"))}
    agg = {p: {"dG": -10.0 * truth[p]["pkd"], "sem": 0.1, "sd": 0.2, "n": 2,
               "pkd": truth[p]["pkd"], "series": truth[p]["series"],
               "tier": ca.TIER_OF[truth[p]["series"]]} for p in truth}
    stats = ca.compute_stats(agg, truth)
    allp = stats["bins"][0.0]["all"]["n_pairs"]
    bbp = stats["bins"][0.0]["bb"]["n_pairs"]
    assert bbp < allp, "the SS series must not be in the BB/MIX tier"
    assert bbp == 1


# ---------------------------------------------------------------------------
# ground-truth loader
# ---------------------------------------------------------------------------

def test_load_dataset1_parses_and_attaches_series(tmp_path):
    p = tmp_path / "dataset1.csv"
    p.write_text(
        "PDB code,Chain id,Peptide length,Polar interface (%),Kd (M),pKd,Chain type\n"
        "7K2E,A:P,7,48.6,4.70E-07,6.33,BB\n"
        "1SLE,BD:M,8,77.4,6.70E-07,6.17,SS*1\n"
        "3PP4,HL:P,24,54.65,7.00E-06,5.15,SS*1\n"
    )
    t = ca.load_dataset1(p)
    assert t["7K2E"]["pkd"] == pytest.approx(6.33)
    assert t["7K2E"]["series"] == "S03_keap1"
    assert t["1SLE"]["series"] == "S01_strep"
    assert t["3PP4"]["series"] is None, "3PP4 is a singleton, not in a series"
    # the CSV chain field is carried but is NOT the source of truth for chains
    assert t["1SLE"]["chain_id_csv"] == "BD:M"


def test_higher_pkd_means_stronger_binding_is_encoded_consistently(tmp_path):
    """Guards the convention itself: pKd = -log10(Kd), so a smaller Kd must
    give a larger pKd. A silent inversion here would invert the whole gate."""
    p = tmp_path / "dataset1.csv"
    p.write_text(
        "PDB code,Chain id,Peptide length,Polar interface (%),Kd (M),pKd,Chain type\n"
        "7K2H,A:P,7,50.96,1.40E-07,6.85,BB\n"
        "7K2G,B:P,7,47.51,2.80E-04,3.55,BB\n"
    )
    t = ca.load_dataset1(p)
    assert t["7K2H"]["kd_m"] < t["7K2G"]["kd_m"]
    assert t["7K2H"]["pkd"] > t["7K2G"]["pkd"]
    import math as _m
    for rec in t.values():
        assert abs(-_m.log10(rec["kd_m"]) - rec["pkd"]) < 0.02
