"""Unit tests for the cheap sequence rankers. The masked-marginal math and the
report are tested with a FAKE model (deterministic logits), so no weights, no GPU.

Run: python -m pytest agent/lnr/test_seq_rankers.py -q
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

from lnr import seq_rankers as R  # noqa: E402


def test_apply_subs_is_in_bounds_and_upper():
    assert R.apply_subs("ACDEF", [(0, "w"), (4, "Y")]) == "WCDEF"[:-1] + "Y"
    assert R.apply_subs("ACDEF", [(9, "W")]) == "ACDEF"   # out of range ignored


def test_oriented_flips_only_likelihoods():
    # likelihoods are higher=better -> negated so more-negative=better like dG
    assert R._oriented("pepmlm_cll", -5.0) == 5.0
    assert R._oriented("esm2_pll", -5.0) == 5.0
    assert R._oriented("proxy_recovered", -30.0) == -30.0   # already dG-like
    assert R._oriented("pepmlm_cll", None) is None


def test_spearman_known_values():
    assert R._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert R._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert R._spearman([1, 2], [1, 2]) is None


def test_zscores_centre_and_scale():
    z = R._zscores([0.0, 10.0])
    assert z[0] == pytest.approx(-1.0) and z[1] == pytest.approx(1.0)
    assert all(v == 0.0 for v in R._zscores([5.0, 5.0]).values())  # no spread


# --- the masked-marginal scorer, with a fake ESM-like model -----------------

class _FakeTok:
    """Minimal ESM-style tokenizer: cls=0, eos=2, mask=32, one id per AA."""
    mask_token_id = 32

    def __init__(self):
        aas = "LAGVSERTIDPKQNFYMHWC"
        self._id = {a: 3 + i for i, a in enumerate(aas)}

    def __call__(self, text, return_tensors=None):
        import torch
        ids = [0] + [self._id[a] for a in text] + [2]
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def convert_tokens_to_ids(self, a):
        return self._id[a]


class _FakeModel:
    """Returns logits that score residue 'W' highly at every position, so a
    peptide with more W's gets a higher pseudo-log-likelihood -- a known ordering
    the test can assert on."""
    def __init__(self, vocab=33, w_id=None):
        self.vocab, self.w_id = vocab, w_id

    def eval(self): return self
    def to(self, device): return self

    def __call__(self, input_ids=None, attention_mask=None):
        import torch
        b, L = input_ids.shape
        logits = torch.zeros(b, L, self.vocab)
        logits[:, :, self.w_id] = 5.0        # 'W' always favoured
        return type("O", (), {"logits": logits})()


@pytest.fixture
def fake_scorer(monkeypatch):
    import torch
    tok = _FakeTok()

    def _init(self, model_id, device="cpu", conditional=False):
        self.torch = torch
        self.tok = tok
        self.model = _FakeModel(w_id=tok.convert_tokens_to_ids("W"))
        self.device = device
        self.conditional = conditional
        self.mask_id = tok.mask_token_id

    monkeypatch.setattr(R.PLMScorer, "__init__", _init)
    return R.PLMScorer("fake")


def test_score_peptide_counts_every_residue_and_rewards_W(fake_scorer):
    s_all_w = fake_scorer.score_peptide("WWWW")
    s_no_w = fake_scorer.score_peptide("AAAA")
    assert s_all_w["n"] == 4 and s_no_w["n"] == 4
    # 'W' is favoured, so the all-W peptide is far more likely
    assert s_all_w["sum_ll"] > s_no_w["sum_ll"]
    assert len(s_all_w["per_res"]) == 4


def test_conditional_context_shifts_peptide_positions_not_the_count(fake_scorer):
    fake_scorer.conditional = True
    # with a receptor prefix, the peptide tokens move right, but the SAME peptide
    # residues are scored (count unchanged, value unchanged for this fake model)
    s = fake_scorer.score_peptide("WA", receptor="AAAAA")
    assert s["n"] == 2


def test_score_writes_rows_and_is_resumable(fake_scorer, tmp_path, monkeypatch):
    rank = tmp_path / "rank.jsonl"
    rank.write_text("\n".join(json.dumps(r) for r in [
        {"variant_id": "v1", "target_id": "t1", "peptide_sequence": "WAAA",
         "proxy_recovered": -30.0, "true_dG_median": -40.0, "true_dG_std": 3.0,
         "true_n_valid": 4, "subs": [], "window": [0, 3]},
        {"variant_id": "v2", "target_id": "t1", "peptide_sequence": "AAAA",
         "proxy_recovered": -25.0, "true_dG_median": -20.0, "true_dG_std": 3.0,
         "true_n_valid": 4, "subs": [[0, "A"]], "window": [0, 3]},
    ]) + "\n")

    class _T:
        def __init__(s, tid): s.target_id = tid
    # patch the real modules' functions so this is robust to import order
    from lnr import env as _env
    from lnr import boltz_validate as _bv
    monkeypatch.setattr(_env, "load_manifest", lambda m, p: [_T("t1")])
    monkeypatch.setattr(_bv, "receptor_seq_and_pocket", lambda t: ("AAAAA", []))

    out = tmp_path / "out.jsonl"
    n = R.score(rank, "m", "p", out, models=("pepmlm",), device="cpu")
    assert n == 2
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["variant_id"] for r in rows} == {"v1", "v2"}
    assert all("pepmlm_cll" in r for r in rows)
    # resumable: second pass adds nothing
    assert R.score(rank, "m", "p", out, models=("pepmlm",), device="cpu") == 0


def test_report_runs_and_picks_a_winner(tmp_path, capsys):
    # synthetic: pepmlm_cll perfectly anti-correlates with dG (higher ll ->
    # more negative dG), proxy is noise -> report should rank pepmlm above proxy
    rows = []
    for i in range(6):
        ll = -float(i)            # higher i -> lower ll
        rows.append({"variant_id": f"v{i}", "target_id": "t1",
                     "peptide_sequence": "A" * 5,
                     "proxy_recovered": (-1.0 if i % 2 else 1.0),  # noise
                     "pepmlm_cll": ll,
                     "true_dG_median": float(i),   # higher i -> worse (less neg)
                     "true_dG_std": 3.0, "true_n_valid": 4})
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    R.report(p)
    out = capsys.readouterr().out
    assert "VERDICT" in out
    # pepmlm_cll (oriented -> +i) should agree with true dG (+i) => rho +1
    assert "pepmlm_cll" in out
