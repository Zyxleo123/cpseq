"""Tests for the candidate binding metric and its referee.

These are the PURE parts -- geometry, orientation, the statistics, the verdict
rule. The claims that can only be checked against the real checkpoint (exact
determinism, the upstream flag inversion, side-chain invisibility) are asserted
by `bindscore.selftest`, which the job runs as a GATE before scoring anything.

The most important test here is `test_sign_convention_end_to_end`: this project
has repeatedly shipped sign slips, so a perfect metric is pushed through the
real orientation + grading path and asserted to come out PASS with a NEGATIVE
mean rho, the same convention the Rosetta dG gate uses.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import bindscore as bs  # noqa: E402
from lnr import metric_gate as mg  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_virtual_cb_has_physical_geometry():
    """IDEAL backbone geometry -> CB at a real CA-CB bond length, off the plane.

    Uses the real N-CA-C angle (111.2 deg), not a right angle: the formula's
    coefficients are fitted to true backbone geometry, so a square-cornered
    test residue puts CB at ~1.75 A and the test would be asserting the wrong
    thing about a correct implementation.
    """
    ang = math.radians(111.2)
    n = np.array([[1.458 * math.cos(ang), 1.458 * math.sin(ang), 0.0]])
    ca = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[1.525, 0.0, 0.0]])
    cb = bs.virtual_cb(n, ca, c)
    d = float(np.linalg.norm(cb - ca))
    assert 1.45 < d < 1.65, f"CB-CA distance {d} is not a CA-CB bond length"
    # It must not be coplanar with N, CA, C, or every residue would be a glycine
    # stub and the interface mask would be systematically wrong.
    normal = np.cross((n - ca)[0], (c - ca)[0])
    normal /= np.linalg.norm(normal)
    assert abs(float(np.dot((cb - ca)[0], normal))) > 0.5


def test_virtual_cb_is_sequence_independent_by_construction():
    """It takes only N, CA, C -- so threading a variant cannot move it, which is
    what lets every variant be scored over the SAME interface position set."""
    import inspect
    sig = inspect.signature(bs.virtual_cb)
    assert list(sig.parameters) == ["n_xyz", "ca_xyz", "c_xyz"]


def test_net_charge():
    assert bs.net_charge("GDEETGE") == -4
    assert bs.net_charge("GAEETGE") == -3      # lost the D2 carboxylate
    assert bs.net_charge("GDEEAGE") == -4      # T5->A changes no charge
    assert bs.net_charge("KKRR") == 4
    assert bs.net_charge("GHHG") == 0          # His counted neutral


def test_interface_mask_selects_only_nearby_positions():
    cb = np.array([
        [0.0, 0.0, 0.0],      # pep 0, close to receptor
        [50.0, 0.0, 0.0],     # pep 1, far
        [3.0, 0.0, 0.0],      # receptor
    ])
    m = bs.interface_mask(cb, np.array([0, 1]), np.array([2]), cutoff=8.0)
    assert m.tolist() == [True, False]


def test_interface_mask_without_receptor_keeps_everything():
    cb = np.zeros((2, 3))
    m = bs.interface_mask(cb, np.array([0, 1]), np.array([], dtype=int))
    assert m.all()


# ---------------------------------------------------------------------------
# The metric registry and orientation
# ---------------------------------------------------------------------------
def test_every_metric_declares_orientation_and_family():
    for name, spec in bs.METRICS.items():
        assert "lower_is_better" in spec, f"{name} does not declare orientation"
        assert isinstance(spec["lower_is_better"], bool)
        assert spec.get("family") in {
            "primary", "secondary", "control", "null", "baseline"}, name
        assert spec.get("doc"), f"{name} has no doc"


def test_primary_metric_is_icg_and_is_higher_is_better():
    assert bs.METRICS["icg"]["family"] == "primary"
    assert bs.METRICS["icg"]["lower_is_better"] is False
    assert bs.METRICS["dg_rosetta"]["lower_is_better"] is True


def test_orient_flips_only_higher_is_better_metrics():
    vals = {"A": 1.0, "B": -2.0}
    assert mg.orient(vals, "dg_rosetta") == vals             # already lower=better
    assert mg.orient(vals, "icg") == {"A": -1.0, "B": 2.0}   # flipped


def test_nulls_are_present_and_include_the_size_null():
    """The size null is mandatory: -peptide_length beats Rosetta dG
    cross-complex on this very benchmark (-0.376 vs -0.225)."""
    nulls = {k for k, v in bs.METRICS.items() if v["family"] in ("null", "baseline")}
    assert "null_neglen" in nulls
    assert "null_random" in nulls
    assert "dg_rosetta" in nulls


def test_control_family_contains_the_receptor_free_control():
    """If the peptide scored WITHOUT the receptor ranks as well as ICG, the
    metric is not about binding. That control has to exist."""
    assert bs.METRICS["pll_pep"]["family"] == "control"
    assert bs.METRICS["icg_sum"]["family"] == "control"


# ---------------------------------------------------------------------------
# Aggregation: the leakage firewall
# ---------------------------------------------------------------------------
def _rows(metric="icg"):
    out = []
    for tmpl in ("P1", "P2"):
        for pdb in ("P1", "P2"):
            out.append({
                "model": "m", "series": "S", "tier": "BB",
                "template": tmpl, "pdb": pdb, "is_self": tmpl == pdb,
                "metrics": {metric: 1.0 if tmpl == pdb else 0.0},
            })
    return out


def test_xtemplate_excludes_the_variants_own_crystal():
    means, per = mg.aggregate(_rows(), mode="xtemplate")
    assert means["icg"] == {"P1": 0.0, "P2": 0.0}
    assert all(len(v) == 1 for v in per["icg"].values())


def test_self_mode_uses_only_the_variants_own_crystal():
    means, _ = mg.aggregate(_rows(), mode="self")
    assert means["icg"] == {"P1": 1.0, "P2": 1.0}


def test_private_fields_are_not_graded_as_metrics():
    rows = _rows()
    rows[0]["metrics"]["_per_pos_gain"] = [1, 2, 3]
    means, _ = mg.aggregate(rows, mode="xtemplate")
    assert "_per_pos_gain" not in means


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_partial_spearman_removes_a_pure_confound():
    """x and y correlate ONLY through z -> partial correlation ~ 0.

    This is the dG situation exactly: rho(dG, pKd) = -0.225 but
    rho(dG, pKd | length) = -0.002, because both track peptide size.
    """
    z = [1, 2, 3, 4, 5, 6, 7, 8]
    x = [v * 2.0 for v in z]
    y = [v * 3.0 for v in z]
    assert abs(mg.partial_spearman(x, y, z)) < 1e-6


def test_partial_spearman_keeps_a_real_signal():
    z = [1, 2, 3, 4, 5, 6, 7, 8]
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [8, 7, 6, 5, 4, 3, 2, 1]
    # y is anti-correlated with x and with z; controlling z must not erase it
    assert mg.partial_spearman(x, y, z) is not None


def _truth(n=6, series="S03_keap1"):
    # pKd spread like the real KEAP1 series, so |dpKd| >= 1.0 pairs exist
    pkds = [6.85, 6.82, 6.33, 4.89, 4.00, 3.55][:n]
    return {f"P{i}": {"pdb": f"P{i}", "pkd": p, "series": series,
                      "pep_len": 7, "chain_type": "BB"}
            for i, p in enumerate(pkds)}


def _multi_truth(n_series=5, n=4):
    """Several independent series. The verdict now requires evidence across
    series, not a lucky ordering inside one -- so the honest fixture has to
    supply more than one."""
    truth = {}
    for s in range(n_series):
        for i, p in enumerate([7.0, 5.5, 4.5, 3.0][:n]):
            truth[f"S{s}P{i}"] = {"pdb": f"S{s}P{i}", "pkd": p,
                                  "series": f"X{s}", "pep_len": 7,
                                  "chain_type": "BB"}
    return truth


def test_sign_convention_end_to_end_perfect_metric_passes():
    """A higher-is-better metric that perfectly tracks pKd must come out of the
    real orientation + grading path as PASS with NEGATIVE mean rho.

    This is the test that would have caught the project's recurring sign slips:
    it exercises orient() -> pair_agreement() -> verdict() together rather than
    any one of them in isolation.
    """
    truth = _multi_truth()
    values = {p: truth[p]["pkd"] for p in truth}        # perfect, higher = better
    per = {p: [v] for p, v in values.items()}
    g = mg.grade(values, per, truth, metric="icg", n_perm=2000)
    assert g["headline"]["agreement"] == 1.0
    assert g["mean_rho"] is not None and g["mean_rho"] < -0.9, \
        "a working metric must have NEGATIVE rho after orientation"
    assert g["sign_test"]["n_right"] == g["sign_test"]["n"] == 5
    assert g["perm"]["p"] < 0.05
    v, _ = mg.verdict(g)
    assert v == "PASS"


def test_inverted_metric_fails():
    truth = _truth()
    values = {p: -truth[p]["pkd"] for p in truth}       # perfectly wrong
    per = {p: [v] for p, v in values.items()}
    g = mg.grade(values, per, truth, metric="icg", n_perm=500)
    assert g["headline"]["agreement"] == 0.0
    assert mg.verdict(g)[0] == "FAIL"


def test_one_lucky_series_cannot_pass_alone():
    """The correction forced by the first run: 13 pooled "pairs" drawn from two
    series are not 13 independent observations, and a single random draw scored
    0.846 on that tier. A perfect ordering within ONE series must now come out
    WEAK, because one series is one observation."""
    truth = _truth()
    values = {p: truth[p]["pkd"] for p in truth}
    per = {p: [v] for p, v in values.items()}
    g = mg.grade(values, per, truth, metric="icg", n_perm=2000)
    assert g["headline"]["agreement"] == 1.0          # looks perfect
    assert g["sign_test"]["n"] == 1                   # ...on one series
    assert mg.verdict(g)[0] == "WEAK"


def test_permutation_null_is_not_half_and_is_respected():
    """The permutation null's mean need not be 0.5 (ties and series sizes move
    it), and a metric must be scored against THAT, not against a coin."""
    truth = _multi_truth()
    values = {p: truth[p]["pkd"] for p in truth}
    perm = mg.permutation_test(values, truth, metric="icg", n_perm=2000)
    assert perm["n_series"] == 5
    assert perm["observed"] == 1.0
    assert 0.0 < perm["null_mean"] < 1.0
    assert perm["p"] < 0.05
    assert perm["null_p95"] < 1.0


def test_a_metric_that_only_ties_the_best_null_does_not_pass():
    """The clause the dG gate did not need: beating a coin flip is not enough,
    the metric must beat the best NULL. A metric that ties a residue count has
    not worked, whatever its CI says."""
    truth = _multi_truth()
    values = {p: truth[p]["pkd"] for p in truth}
    per = {p: [v] for p, v in values.items()}
    g = mg.grade(values, per, truth, metric="icg", n_perm=2000)
    assert mg.verdict(g, best_null_lo=None)[0] == "PASS"
    # same metric, but a null did just as well -> not a PASS
    lo = g["headline"]["ci95"][0]
    assert mg.verdict(g, best_null_lo=lo)[0] == "WEAK"


def test_discriminability_rewards_low_within_variant_spread():
    truth = _truth(n=3)
    tight = {"P0": [1.0, 1.01], "P1": [2.0, 2.01], "P2": [3.0, 3.01]}
    loose = {"P0": [1.0, 5.0], "P1": [2.0, 6.0], "P2": [3.0, 7.0]}
    dt = mg.discriminability(tight, {}, truth)["S03_keap1"]["ratio"]
    dl = mg.discriminability(loose, {}, truth)["S03_keap1"]["ratio"]
    assert dt > dl


def test_series_rho_is_never_pooled_across_series():
    """Pooling would import the cross-target pKd range (3.55-10.29) and inflate
    it into fake within-target skill -- the exact inflation this gate exists to
    prevent."""
    truth = {}
    truth.update(_truth(n=3, series="A"))
    truth.update({f"Q{i}": {"pdb": f"Q{i}", "pkd": p, "series": "B",
                            "pep_len": 7, "chain_type": "BB"}
                  for i, p in enumerate([10.0, 9.5, 9.0])})
    values = {k: truth[k]["pkd"] for k in truth}
    out = mg.series_rhos(values, truth)
    assert set(out) == {"A", "B"}
    assert all(v["n"] == 3 for v in out.values())


def test_grade_reports_the_mandatory_size_control():
    truth = _truth()
    values = {p: truth[p]["pkd"] for p in truth}
    per = {p: [v] for p, v in values.items()}
    g = mg.grade(values, per, truth, metric="icg")
    assert "cross_rho_partial_len" in g
    assert "cross_rho_len" in g


# ---------------------------------------------------------------------------
# Threading contract
# ---------------------------------------------------------------------------
def test_marginals_rejects_a_length_mismatch():
    """Threading onto a template requires equal peptide length; a silent
    truncation would score a different molecule than the one named in the row."""
    class FakePose:
        pep_idx = np.array([0, 1, 2])
        protein_dict = {"S": None}

    class FakeScorer(bs.MPNNScorer):
        def __init__(self):
            self.aa_to_int = {"A": 0}

    with pytest.raises(ValueError, match="equal length"):
        bs.MPNNScorer.marginals(FakeScorer(), FakePose(), seq="AA")


def test_log_background_is_normalised():
    total = sum(math.exp(v) for v in bs.LOG_BG.values())
    assert abs(total - 1.0) < 1e-9
    assert len(bs.LOG_BG) == 20


# ---------------------------------------------------------------------------
# TargetPSSM: the deployable artefact
# ---------------------------------------------------------------------------
ALPH = list("ACDEFGHIKLMNPQRSTVWYX")


def _pssm(n=3):
    rng = np.random.default_rng(0)
    M = rng.normal(size=(n, 21))
    return bs.TargetPSSM(M, ALPH, list(range(n)), "t.pdb", "P",
                         "A" * n, "m", iface=[True] * n)


def test_pssm_score_is_mean_of_lookups():
    """pll_cplx is the raw mean of lookups -- asked for EXPLICITLY, because it is
    no longer the default."""
    t = _pssm(3)
    seq = "DEG"
    want = np.mean([t.M[i, ALPH.index(a)] for i, a in enumerate(seq)])
    assert abs(t.score(seq, metric="pll_cplx") - want) < 1e-12


def test_pssm_score_defaults_to_the_metric_that_was_GRADED():
    """The deployed metric is pll_bg (13/13 on the CPSet head-to-tail pairs,
    within-complex median rho +0.34 on SKEMPI). pll_cplx was never graded, so a
    silent pll_cplx default would deploy an ungraded metric under a graded
    metric's evidence. The default is therefore pll_bg."""
    t = _pssm(3)
    seq = "DEG"
    want = np.mean([t.M[i, ALPH.index(a)] - bs.LOG_BG[a] for i, a in enumerate(seq)])
    assert t.DEFAULT_METRIC == "pll_bg"
    assert abs(t.score(seq) - want) < 1e-12
    assert abs(t.score(seq) - t.score(seq, metric="pll_cplx")) > 1e-9


def test_the_two_metrics_do_not_share_an_optimum():
    """The background subtraction is a per-COLUMN offset, so it is NOT a monotone
    relabelling: it reorders the per-position argmax. This is why the metric has
    to be carried explicitly through score/best_sequence/best_subs rather than
    inferred -- optimising one while reporting the other is a silent swap."""
    rng = np.random.default_rng(7)
    t = bs.TargetPSSM(rng.normal(size=(12, 21)), ALPH, list(range(12)), "t.pdb",
                      "P", "A" * 12, "m")
    assert t.best_sequence(metric="pll_bg") != t.best_sequence(metric="pll_cplx")


def test_pssm_score_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        _pssm(3).score("AA")


@pytest.mark.parametrize("metric", ["pll_bg", "pll_cplx"])
def test_pssm_best_sequence_is_the_exact_argmax(metric):
    """Available only because the metric is a per-position sum. With a noisy
    oracle the optimum can only be searched for; here it is read off -- and it is
    read off the MATRIX THE METRIC IS A MEAN OF, not always the raw one."""
    t = _pssm(4)
    E = t.eff(metric)
    best = t.best_sequence(metric=metric)
    assert len(best) == 4
    for i, a in enumerate(best):
        col = [c for c, x in enumerate(ALPH) if x in set("ACDEFGHIKLMNPQRSTVWY")]
        assert E[i, ALPH.index(a)] == max(E[i, c] for c in col)


def test_pssm_best_subs_is_the_brute_force_optimum():
    """The claim that makes this metric a different kind of in-loop fitness: the
    best <=k substitution set is READ OFF, not searched for. Checked against
    exhaustive enumeration, because the whole regret measurement rests on it."""
    import itertools
    rng = np.random.default_rng(11)
    t = bs.TargetPSSM(rng.normal(size=(5, 21)), ALPH, list(range(5)), "t.pdb",
                      "P", "ACDEF", "m")
    seq = "ACDEF"
    aas = "ACDEFGHIKLMNPQRSTVWY"
    for k in (1, 2, 3):
        subs, got = t.best_subs(seq, k)
        assert len(subs) <= k
        brute = -1e18
        for npos in range(k + 1):
            for pos in itertools.combinations(range(5), npos):
                for combo in itertools.product(aas, repeat=npos):
                    m = list(seq)
                    for p, a in zip(pos, combo):
                        m[p] = a
                    brute = max(brute, t.score("".join(m)))
        assert abs(got - brute) < 1e-12, f"k={k}: read-off {got} != brute {brute}"


def test_pssm_best_subs_never_proposes_a_losing_substitution():
    """'at most k' -- a position whose best residue is already native must not be
    spent, or the reported optimum is not an optimum."""
    t = _pssm(4)
    t.M[:] = -5.0
    for p in range(4):
        t.M[p, ALPH.index(t.pep_seq[p])] = 10.0   # native already best everywhere
    subs, score = t.best_subs(t.pep_seq, 3)
    assert subs == [] and abs(score - t.score(t.pep_seq)) < 1e-12


def test_pssm_matches_catches_the_silent_misalignment():
    """env.LNRTarget and LigandMPNN parse PDBs differently (env DROPS residues
    missing a backbone atom). If they disagree but return the same length, every
    lookup reads the wrong row and NOTHING raises. So the table refuses a caller
    sequence that is not the same string as its own template."""
    t = bs.TargetPSSM(np.zeros((5, 21)), ALPH, list(range(5)), "t.pdb", "P",
                      "ACDEF", "m")
    assert t.matches("ACDEF")[0] is True
    assert t.matches("ACDE")[0] is False                  # length
    off_by_one, detail = t.matches("CDEFG")               # same length, shifted
    assert off_by_one is False and "differ" in detail
    # One parser resolved a modified residue and the other did not: tolerated,
    # but REPORTED, because it is the one disagreement that is not a misalignment.
    t2 = bs.TargetPSSM(np.zeros((5, 21)), ALPH, list(range(5)), "t.pdb", "P",
                       "ACXEF", "m")
    assert t2.matches("ACXEF") == (True, "aligned on 5 positions")
    ok, detail = t2.matches("ACAEF")
    assert ok is True and "1 unparsable-residue position(s) ignored" in detail


def test_pssm_position_subset_and_scorable_set_are_sequence_independent():
    """Two variants must be averaged over the SAME positions or their scores are
    not comparable -- so the position set is fixed by the template, never by the
    sequence being scored."""
    t = bs.TargetPSSM(np.arange(5 * 21, dtype=float).reshape(5, 21), ALPH,
                      list(range(5)), "t.pdb", "P", "ACXEF", "m",
                      iface=[True, False, False, True, False])
    assert t.scorable == [0, 1, 3, 4]        # the 'X' template position is dropped
    assert t.scorable == bs.TargetPSSM(
        t.M, ALPH, list(range(5)), "t.pdb", "P", "ACXEF", "m").scorable
    win = t.score("ACDEF", positions=[0, 1])
    assert abs(win - np.mean([t.eff()[0, ALPH.index("A")],
                              t.eff()[1, ALPH.index("C")]])) < 1e-12
    # an interface restriction uses the iface mask intersected with scorable
    assert abs(t.score("ACDEF", interface_only=True)
               - np.mean([t.eff()[0, ALPH.index("A")],
                          t.eff()[3, ALPH.index("E")]])) < 1e-12
    with pytest.raises(ValueError, match="no scorable position"):
        t.score("ACDEF", positions=[2])      # only the dropped position


def test_pssm_best_sequence_never_returns_X():
    """'X' is in the model's 21-letter alphabet but is not an amino acid; a
    designed sequence containing it would be unsynthesisable."""
    t = _pssm(6)
    t.M[:, ALPH.index("X")] = 99.0
    assert "X" not in t.best_sequence()


def test_pssm_roundtrips_through_disk(tmp_path):
    t = _pssm(5)
    p = tmp_path / "t.json"
    t.save(p)
    back = bs.TargetPSSM.load(p)
    assert back.pep_seq == t.pep_seq and back.n == t.n
    assert abs(back.score("DEGAC") - t.score("DEGAC")) < 1e-12


def test_pssm_is_deterministic_across_calls():
    t = _pssm(4)
    assert t.score("DEGA") == t.score("DEGA")


# ---------------------------------------------------------------------------
# SKEMPI arm: the sign convention, which must match the prior Rosetta gate
# ---------------------------------------------------------------------------
def test_skempi_sign_convention_matches_the_prior_gate():
    """measured ddG > 0 = WEAKER. `ddg_pred` is negated at emission so that a
    favoured mutation (higher log-prob for the mutant) comes out NEGATIVE, i.e.
    stronger. A working metric therefore has POSITIVE correlation -- the same
    orientation as the rho +0.22 this arm is compared against."""
    from lnr import bindscore_skempi as bk
    src = Path(bk.__file__).read_text()
    assert '"ddg_pred": -d_c' in src, \
        "ddg_pred must be negated so positive = predicted weaker"
    assert '"ddg_pred_icg": -(d_c - d_m)' in src
    assert bk.REF["random expectation"] == 0.0


# ---------------------------------------------------------------------------
# The memorisation null
# ---------------------------------------------------------------------------
def test_consensus_null_is_leave_one_out():
    """A variant must not vote for its own consensus -- the same firewall the
    xtemplate mode applies to structures."""
    seqs = {"a": "AAA", "b": "AAA", "c": "AAA", "d": "CCC"}
    out = bs._consensus_excluding(seqs)
    assert out["d"] == -3.0          # differs from the other three everywhere
    assert out["a"] == 0.0           # matches consensus of b, c, d


def test_consensus_null_skips_groups_too_small_to_have_one():
    """A consensus over one other member is just that member, which would make
    the null a distance-to-an-arbitrary-sibling rather than to a consensus."""
    assert bs._consensus_excluding({"a": "AA", "b": "AC"}) == {}


def test_consensus_null_groups_by_length():
    seqs = {"a": "AAA", "b": "AAC", "c": "AAG", "x": "AAAA", "y": "AAAC"}
    out = bs._consensus_excluding(seqs)
    assert set(out) == {"a", "b", "c"}     # the 4-mers are too small a group


def test_consensus_null_is_declared_a_null():
    assert bs.METRICS["null_consensus"]["family"] == "null"
    assert bs.METRICS["null_consensus"]["lower_is_better"] is True
