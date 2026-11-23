"""Unit tests for the interface-confidence extraction. Synthetic Boltz output,
no GPU, no real folds.

Run: python -m pytest agent/lnr/test_confidence_metrics.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lnr import confidence_metrics as C  # noqa: E402


def _pdb(path, chains):
    """chains = [(chain_id, n_res)] -> a CA-only PDB in that chain order."""
    lines, n = [], 0
    for ch, nres in chains:
        for i in range(1, nres + 1):
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA {ch}{i:4d}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def _fold(tmp, stem, k, *, rec_n=6, pep_n=3, inter_pae=2.0, intra_pae=1.0,
          iptm_pr=0.4, iptm_rp=0.9):
    """One synthetic model: receptor chain A then peptide chain B."""
    pdb = tmp / f"{stem}_model_{k}.pdb"
    _pdb(pdb, [("A", rec_n), ("B", pep_n)])
    total = rec_n + pep_n
    pae = np.full((total, total), intra_pae, dtype=float)
    pae[:rec_n, rec_n:] = inter_pae
    pae[rec_n:, :rec_n] = inter_pae
    np.savez(tmp / f"pae_{stem}_model_{k}.npz", pae=pae)
    plddt = np.concatenate([np.full(rec_n, 0.9), np.full(pep_n, 0.5)])
    np.savez(tmp / f"plddt_{stem}_model_{k}.npz", plddt=plddt)
    conf = {"iptm": 0.8, "ptm": 0.95, "complex_plddt": 0.8,
            "complex_iplddt": 0.7, "complex_ipde": 1.5,
            "chains_ptm": {"0": 0.95, "1": 0.6},
            # chain index 0 = A = receptor, 1 = B = peptide (file order)
            "pair_chains_iptm": {"0": {"0": 0.95, "1": iptm_rp},
                                 "1": {"0": iptm_pr, "1": 0.6}}}
    (tmp / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(conf))
    return pdb


def test_chain_lengths_counts_ca_in_file_order(tmp_path):
    p = tmp_path / "x.pdb"
    _pdb(p, [("A", 5), ("B", 2)])
    assert C.chain_lengths(p) == [("A", 5), ("B", 2)]


def test_model_metrics_splits_the_pae_at_the_right_chain_boundary(tmp_path):
    pdb = _fold(tmp_path, "s", 0, rec_n=6, pep_n=3, inter_pae=2.0, intra_pae=1.0)
    m = C.model_metrics(tmp_path / "confidence_s_model_0.json",
                        tmp_path / "pae_s_model_0.npz",
                        tmp_path / "plddt_s_model_0.npz", pdb, pep_chain="B")
    assert m["pae_usable"] is True and m["token_total"] == 9
    # every inter-chain cell is 2.0 by construction -> both directions agree
    assert m["ipae_mean"] == pytest.approx(2.0)
    assert m["ipae_min_dir"] == pytest.approx(2.0)
    # LIS = (12-2)/12 over all inter pairs
    assert m["lis"] == pytest.approx((12.0 - 2.0) / 12.0)
    # peptide pLDDT is the B-chain slice only, NOT the complex mean
    assert m["pep_plddt"] == pytest.approx(0.5)


def test_iptm_min_takes_the_weakest_direction_for_the_peptide(tmp_path):
    pdb = _fold(tmp_path, "s", 0, iptm_pr=0.4, iptm_rp=0.9)
    m = C.model_metrics(tmp_path / "confidence_s_model_0.json",
                        tmp_path / "pae_s_model_0.npz",
                        tmp_path / "plddt_s_model_0.npz", pdb, pep_chain="B")
    assert m["iptm_pep_to_rec"] == pytest.approx(0.4)
    assert m["iptm_rec_to_pep"] == pytest.approx(0.9)
    assert m["iptm_min"] == pytest.approx(0.4)


def test_asymmetric_pae_blocks_give_a_smaller_min_direction(tmp_path):
    """ipae_min_dir must be the smaller DIRECTIONAL block mean, not the overall."""
    pdb = _fold(tmp_path, "s", 0, rec_n=4, pep_n=2)
    pae = np.load(tmp_path / "pae_s_model_0.npz")["pae"].copy()
    pae[:4, 4:] = 1.0        # receptor -> peptide: confident
    pae[4:, :4] = 9.0        # peptide -> receptor: not
    np.savez(tmp_path / "pae_s_model_0.npz", pae=pae)
    m = C.model_metrics(tmp_path / "confidence_s_model_0.json",
                        tmp_path / "pae_s_model_0.npz",
                        tmp_path / "plddt_s_model_0.npz", pdb, pep_chain="B")
    assert m["ipae_mean"] == pytest.approx(5.0)
    assert m["ipae_min_dir"] == pytest.approx(1.0)


def test_token_mismatch_refuses_to_guess_a_split(tmp_path):
    """A PAE whose size disagrees with the residue count must be marked unusable
    rather than silently sliced at the wrong boundary."""
    pdb = _fold(tmp_path, "s", 0, rec_n=4, pep_n=2)
    np.savez(tmp_path / "pae_s_model_0.npz", pae=np.ones((9, 9)))
    m = C.model_metrics(tmp_path / "confidence_s_model_0.json",
                        tmp_path / "pae_s_model_0.npz",
                        tmp_path / "plddt_s_model_0.npz", pdb, pep_chain="B")
    assert m["pae_usable"] is False
    assert "ipae_mean" not in m and m["iptm_min"] == pytest.approx(0.4)


def test_extract_joins_the_cache_and_is_resumable(tmp_path):
    work = tmp_path / "work"
    pred = work / "heldout_t1_0_9_abc" / "boltz_results_x" / "predictions" / "held_x"
    pred.mkdir(parents=True)
    for k in range(3):
        _fold(pred, "held_x", k, iptm_pr=0.4 + 0.05 * k)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(
        {"cache_key": "t1_0_9_abc", "variant_id": "t1_0_9_abc", "target_id": "t1",
         "subs_key": "2:W", "draw": 0, "dG_refined": -30.0, "dG_refined_std": 4.0,
         "n_valid_folds": 3, "held_rmsd": 5.0}) + "\n")
    out = tmp_path / "conf.jsonl"
    assert C.extract(work, cache, out) == 1
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["target_id"] == "t1" and r["subs_key"] == "2:W"
    assert r["dG_refined"] == -30.0 and r["n_models"] == 3
    assert r["iptm_min"] == pytest.approx(0.45)      # median of .40/.45/.50
    assert r["iptm_min_sd"] > 0 and r["iptm_min_n"] == 3
    # resumable: a second pass adds nothing
    assert C.extract(work, cache, out) == 0
    assert len(out.read_text().splitlines()) == 1


def test_oriented_flips_only_the_confidence_metrics():
    assert C._oriented("iptm_min", 0.8) == -0.8     # higher conf -> more negative
    assert C._oriented("ipae_mean", 5.0) == 5.0     # error-like, unchanged
    assert C._oriented("lis", 0.3) == -0.3
    assert C._oriented("iptm_min", None) is None


def test_spearman_matches_known_values():
    assert C._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert C._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert C._spearman([1, 2], [1, 2]) is None       # too few points


def test_extract_skips_folds_the_search_has_not_finished(tmp_path):
    """The running search writes into the same dir. A fold with no cache row is
    still being written: recording it would capture fewer diffusion samples and
    then be skipped forever on resume."""
    work = tmp_path / "work"
    for name in ("heldout_t1_0_9_done", "heldout_t1_0_9_inflight"):
        pred = work / name / "boltz_results_x" / "predictions" / "held_x"
        pred.mkdir(parents=True)
        for k in range(2 if "inflight" in name else 3):
            _fold(pred, "held_x", k)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(
        {"cache_key": "t1_0_9_done", "variant_id": "t1_0_9_done", "target_id": "t1",
         "subs_key": "", "draw": 0, "dG_refined": -30.0, "dG_refined_std": 4.0,
         "n_valid_folds": 3, "held_rmsd": 5.0}) + "\n")
    out = tmp_path / "conf.jsonl"
    assert C.extract(work, cache, out) == 1          # only the finished one
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["cache_key"] for r in rows] == ["t1_0_9_done"]
    # once the in-flight fold lands in the cache, a re-run picks it up
    with cache.open("a") as fh:
        fh.write(json.dumps(
            {"cache_key": "t1_0_9_inflight", "variant_id": "t1_0_9_inflight",
             "target_id": "t1", "subs_key": "1:W", "draw": 0, "dG_refined": -28.0,
             "dG_refined_std": 3.0, "n_valid_folds": 2, "held_rmsd": 6.0}) + "\n")
    assert C.extract(work, cache, out) == 1
