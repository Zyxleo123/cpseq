"""Unit tests for the CPU re-relax denoiser. The aggregation, resolution math,
discovery, resumability and reporting are tested with a FAKE scorer (a
deterministic noisy relax), so no PyRosetta, no fold PDBs.

Run: python -m pytest agent/lnr/test_rescore_denoise.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from lnr import rescore_denoise as D  # noqa: E402


def test_aggregate_median_std_sem():
    a = D.aggregate([10.0, 12.0, 14.0, 16.0])
    assert a["n"] == 4 and a["median"] == 13.0
    # sem = std / sqrt(n); shrinks as n grows
    a2 = D.aggregate([10.0, 12.0, 14.0, 16.0] * 4)   # same spread, 4x the n
    assert a2["sem"] < a["sem"]
    assert D.aggregate([])["median"] is None


def test_resolution_ratio():
    # variants well separated (medians -40,-30,-20) vs tiny within SEM -> discrim high
    hi = D.resolution([{"median": -40.0, "sem": 1.0}, {"median": -30.0, "sem": 1.0},
                       {"median": -20.0, "sem": 1.0}])
    assert hi["discrim"] > 5
    # same separation but huge within SEM -> discrim near the separation/noise
    lo = D.resolution([{"median": -40.0, "sem": 10.0}, {"median": -30.0, "sem": 10.0},
                       {"median": -20.0, "sem": 10.0}])
    assert lo["discrim"] < hi["discrim"]
    assert D.resolution([{"median": -1.0, "sem": 1.0}])["discrim"] is None  # n<2


def test_aggregate_clustered_separates_fold_and_relax_noise():
    # 2 folds; within each fold the relaxes are tight (sd ~0.5), but the two
    # fold MEANS are 6 apart -> fold noise dominates and the honest SEM reflects
    # the fold spread, NOT a sqrt(all-draws) shrinkage.
    by_model = {0: [-40.5, -40.0, -39.5],    # fold mean -40, relax sd ~0.41
                1: [-34.5, -34.0, -33.5]}     # fold mean -34, relax sd ~0.41
    a = D.aggregate_clustered(by_model)
    assert a["n_folds"] == 2 and a["n_draws"] == 6
    assert a["relax_sd"] == pytest.approx(0.408, abs=0.01)
    assert a["fold_sd"] == pytest.approx(3.0, abs=0.01)   # pstdev([-40,-34]) = 3
    # clustered SEM = fold_sd / sqrt(2) = 2.12 -- NOT std(all 6)/sqrt(6) (~0.95)
    assert a["sem"] == pytest.approx(3.0 / (2 ** 0.5), abs=0.01)
    assert a["sem"] > 2.0    # the honest SEM, dominated by fold spread


def test_denoised_vs_cached_uses_clustered_sem():
    # fold means differ by variant; re-relax does NOT beat the fold floor
    rows = []
    for vid, base in (("1x_a", -40.0), ("1x_b", -30.0), ("1x_c", -20.0)):
        for m in range(4):
            fold_offset = (m - 1.5) * 2.0          # folds spread ~2.2 REU
            for r in range(3):
                rows.append({"variant_id": vid, "model": m, "repeat": r,
                             "valid": True,
                             "dG_refined": base + fold_offset + (r - 1) * 0.3})
    cache = {vid: {"target_id": "1x", "dG_refined": base, "dG_refined_std": 8.0,
                   "n_valid_folds": 4, "peptide_sequence": "AAAA"}
             for vid, base in (("1x_a", -40.0), ("1x_b", -30.0), ("1x_c", -20.0))}
    out = D.denoised_vs_cached(rows, cache)
    assert "1x" in out and out["1x"]["n_variants"] == 3
    v0 = out["1x"]["variants"][0]["denoised"]
    # the honest SEM is the fold-clustered one, well above a sqrt(12) pooling
    assert v0["fold_sd"] > v0["relax_sd"]            # fold noise is the bigger part
    assert v0["sem"] == pytest.approx(v0["fold_sd"] / 2.0, abs=0.05)  # /sqrt(4)


# --- discovery + scoring with a fake relax ----------------------------------

def _make_fold(work: Path, vid: str, n_models=4):
    pred = (work / f"heldout_{vid}" / "boltz_results_x" / "predictions" / "held_x")
    pred.mkdir(parents=True)
    for k in range(n_models):
        (pred / f"held_x_model_{k}.pdb").write_text("ATOM\nEND\n")
    return pred


def _fake_scorer_factory(base_by_vid):
    """Deterministic 'relax': dG = base(variant) + small per-(model,repeat) jitter,
    so re-relaxing shrinks SEM exactly as the real thing should."""
    def fake(task):
        vid, model_k, rep, pdb, peptide, refined = task
        jit = ((model_k * 7 + rep * 3) % 5 - 2) * 0.4     # deterministic +/- jitter
        return {"variant_id": vid, "model": model_k, "repeat": rep, "pdb": pdb,
                "valid": True, "dG_refined": base_by_vid[vid] + jit}
    return fake


def test_score_discovers_variants_and_is_resumable(tmp_path):
    work = tmp_path / "work"
    for vid in ("1x_6_9_aaa", "1x_6_9_bbb"):
        _make_fold(work, vid)
    cache = tmp_path / "cache.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in [
        {"variant_id": "1x_6_9_aaa", "target_id": "1x", "draw": 0,
         "peptide_sequence": "AAAAAAAAAA", "dG_refined": -30.0,
         "dG_refined_std": 8.0, "n_valid_folds": 4},
        {"variant_id": "1x_6_9_bbb", "target_id": "1x", "draw": 0,
         "peptide_sequence": "AAAAAAAAAA", "dG_refined": -20.0,
         "dG_refined_std": 8.0, "n_valid_folds": 4},
    ]) + "\n")
    out = tmp_path / "den.jsonl"
    fake = _fake_scorer_factory({"1x_6_9_aaa": -30.0, "1x_6_9_bbb": -20.0})
    # 2 variants x 4 models x 3 repeats = 24 draws
    n = D.score(work, cache, out, tmp_path / "ref", repeats=3, workers=1,
                _scorer=fake)
    assert n == 24
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len({(r["variant_id"], r["model"], r["repeat"]) for r in rows}) == 24
    # resumable: a second pass runs nothing
    assert D.score(work, cache, out, tmp_path / "ref", repeats=3, workers=1,
                   _scorer=fake) == 0


def test_score_respects_target_and_max_variants(tmp_path):
    work = tmp_path / "work"
    for vid in ("1x_6_9_a", "1x_6_9_b", "1x_6_9_c", "2y_0_1_d"):
        _make_fold(work, vid, n_models=2)
    cache = tmp_path / "cache.jsonl"
    cache.write_text("\n".join(json.dumps(
        {"variant_id": vid, "target_id": vid.split("_")[0], "draw": 0,
         "peptide_sequence": "AAAA", "dG_refined": -25.0, "dG_refined_std": 6.0,
         "n_valid_folds": 4})
        for vid in ("1x_6_9_a", "1x_6_9_b", "1x_6_9_c", "2y_0_1_d")) + "\n")
    out = tmp_path / "den.jsonl"
    fake = _fake_scorer_factory({v: -25.0 for v in
                                 ("1x_6_9_a", "1x_6_9_b", "1x_6_9_c", "2y_0_1_d")})
    # only target 1x, at most 2 variants of it: 2 variants x 2 models x 2 repeats = 8
    n = D.score(work, cache, out, tmp_path / "ref", repeats=2, workers=1,
                targets=["1x"], max_variants=2, _scorer=fake)
    assert n == 8
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["variant_id"] for r in rows} == {"1x_6_9_a", "1x_6_9_b"}


def test_report_runs_and_prints_the_noise_decomposition(tmp_path, capsys):
    # fold-dominated noise -> the report must say CPU re-relax does not help and
    # name fold sampling as the dominant source
    rows = []
    for vid, base in (("1x_a", -40.0), ("1x_b", -30.0), ("1x_c", -20.0)):
        for m in range(4):
            fold_offset = (m - 1.5) * 3.0          # big fold spread
            for r in range(4):
                rows.append({"variant_id": vid, "model": m, "repeat": r,
                             "valid": True,
                             "dG_refined": base + fold_offset + (r - 1.5) * 0.2})
    rp = tmp_path / "den.jsonl"
    rp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cache = tmp_path / "cache.jsonl"
    # cached std must be self-consistent: it is the spread over the same 4 single-
    # relax folds, i.e. ~ the fold spread (pstdev of offsets -4.5,-1.5,1.5,4.5 = 3.35)
    cache.write_text("\n".join(json.dumps(
        {"variant_id": vid, "target_id": "1x", "draw": 0, "peptide_sequence": "A",
         "dG_refined": base, "dG_refined_std": 3.35, "n_valid_folds": 4})
        for vid, base in (("1x_a", -40.0), ("1x_b", -30.0), ("1x_c", -20.0))) + "\n")
    D.report(rp, cache)
    out = capsys.readouterr().out
    assert "RE-RELAX DENOISER" in out and "VERDICT" in out
    assert "WHERE THE NOISE LIVES" in out
    # fold noise (~3.35) >> relax noise (~0.12) and cached~denoised SEM, so the
    # verdict must say CPU re-relax does not help
    assert "does NOT help" in out
