"""Fast unit tests for the iterative-search mutation arms. No Rosetta, no API.

The real proxy scan is a subprocess PyRosetta call; here `S.mutant_scan` is
monkeypatched to a deterministic fake that rewards tryptophan ('W') substitutions,
so the evolutionary loop, the matched budget, the diagnostics and the finalist-row
wiring are all exercised with no GPU and no spend. The LLM arm uses the project's
ScriptedClient.

Run: python -m pytest agent/lnr/test_seqmut_iter.py -q
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

from lnr import seqmut as S  # noqa: E402
from lnr import seqmut_iter as I  # noqa: E402


class _T:
    """Minimal env.LNRTarget stand-in -- the fields the loop reads."""
    def __init__(self, tid, seq, per_res):
        self.target_id = tid
        self.seq = seq
        self.length = len(seq)
        self.per_res_contacts = tuple(per_res)
        self.n_contacts = sum(per_res)
        self.pdb_path = ""
        self.pep_chain = "B"
        self.rec_chain = "A"


def _fake_scan_factory():
    """Deterministic proxy: native_dG rewards 'W' (-5 each), penalises any other
    substitution (+2 each); per-residue ddG uniform so retention is constant and
    recovered is monotone in native. Populates the cache like the real scan."""
    def fake(target, subs, cache, relax_repeats=1, timeout=900.0):
        hit = cache.get(target.target_id, subs)
        if hit is not None:
            return hit
        n_w = sum(1 for _, a in subs if str(a).upper() == "W")
        n_other = len(subs) - n_w
        row = {"target_id": target.target_id, "subs_key": S.subs_key(subs),
               "subs": [[int(i), str(a).upper()] for i, a in subs],
               "native_dG": -50.0 - 5.0 * n_w + 2.0 * n_other,
               "per_residue_ddG_ala": {i: 1.0 for i in range(target.length)}}
        cache.put(row)
        return row
    return fake


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(S, "mutant_scan", _fake_scan_factory())


def _scripted(replies):
    from lp2cp.llm.anthropic_api import ScriptedClient
    return ScriptedClient(replies, model_id="scripted")


# --- genome helpers ---------------------------------------------------------

def test_norm_canonicalises_and_dedupes_positions():
    assert I._norm([(3, "a"), (1, "W")]) == ((1, "W"), (3, "A"))
    # one entry per position (last wins), invalid AA dropped, sorted
    assert I._norm([(2, "W"), (2, "F"), (0, "Z")]) == ((2, "F"),)
    assert I._norm([]) == ()


def test_iface_positions_prefers_contacts_then_falls_back():
    t = _T("x", "ACDEFG", [0, 3, 0, 5, 0, 1])
    assert I._iface_positions(t) == [1, 3, 5]
    assert I._iface_positions(_T("y", "AC", [0, 0])) == [0, 1]   # no contacts -> all


# --- Population: proxy fitness + matched budget -----------------------------

def test_population_scores_caches_and_enforces_budget(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 2, "b": 5, "retention": 0.4, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    pop = I.Population(t, win, cache, k_mut=2, relax_repeats=1, timeout=1.0, max_evals=3)
    # retention over [2,5] with uniform ddG = 4/10 = 0.4; recovered = 0.4*native
    r0 = pop.evaluate([], gen=0)
    assert abs(r0 - 0.4 * -50.0) < 1e-6
    rw = pop.evaluate([(0, "W")], gen=1)              # one W -> native -55
    assert abs(rw - 0.4 * -55.0) < 1e-6 and rw < r0   # W is better (more negative)
    pop.evaluate([(1, "F")], gen=1)                   # 3rd distinct eval -> budget full
    assert pop.evals == 3 and not pop.budget_left()
    # a NEW genome past budget returns None without scoring; a cached one is free
    assert pop.evaluate([(5, "Y")], gen=2) is None and pop.evals == 3
    assert pop.evaluate([(0, "W")], gen=2) == rw and pop.evals == 3


def test_evaluate_many_parallel_matches_serial_and_charges_budget_per_new(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    batch = [[], [(0, "W")], [(1, "W")], [(0, "W")], [(2, "F")]]   # one dup (0:W)
    # serial reference
    ref = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, max_evals=20)
    r_serial = {k: ref.evaluate(g, 0) for g, k in
                [(g, S.subs_key(I._norm(g))) for g in batch]}
    # parallel (4 workers): identical recovered values, 4 distinct new genomes
    par = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, max_evals=20)
    r_par = par.evaluate_many(batch, gen=0, n_workers=4)
    assert r_par == {k: r_serial[k] for k in r_par}
    assert par.evals == 4                               # dup (0:W) not double-charged
    # budget cap is honoured by the batch: only (cap - evals) new genomes scanned
    cap = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, max_evals=2)
    cap.evaluate_many([[(3, "W")], [(4, "W")], [(5, "W")]], gen=0, n_workers=4)
    assert cap.evals == 2 and not cap.budget_left()


def test_population_elite_and_best_ignore_unscored(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    pop = I.Population(t, win, cache, 2, 1, 1.0, max_evals=10)
    for g in ([], [(0, "W")], [(0, "W"), (1, "W")], [(0, "F")]):
        pop.evaluate(g, gen=0)
    best = pop.best()
    assert best["subs"] == ((0, "W"), (1, "W"))        # two W = strongest
    elite = pop.elite(frac=0.3, min_n=2)
    assert len(elite) == 2 and elite[0]["recovered"] <= elite[1]["recovered"]


# --- random-EA operator -----------------------------------------------------

def test_crossover_respects_budget_and_inherits_positions():
    import random
    rng = random.Random(0)
    d = I._crossover(((1, "W"), (3, "F")), ((3, "Y"), (5, "L")), k_mut=2, rng=rng)
    assert len(d) <= 2
    assert set(d).issubset({1, 3, 5})
    for i, a in d.items():
        assert a in I.AA20


def test_point_mutate_keeps_within_budget_and_valid():
    import random
    t = _T("x", "ACDEFGHIKL", [1] * 10)
    iface = I._iface_positions(t)
    rng = random.Random(1)
    for _ in range(50):
        d = I._point_mutate({0: "W"}, t, k_mut=2, iface=iface, rng=rng, p=1.0)
        assert len(d) <= 2
        for i, a in d.items():
            assert 0 <= i < t.length and a in I.AA20


def test_ea_offspring_are_valid_and_carry_parent_fitness(patched):
    import random
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    pop = I.Population(t, win, cache, 2, 1, 1.0, max_evals=20)
    for g in ([], [(0, "W")], [(1, "W")]):
        pop.evaluate(g, gen=0)
    kids = I.ea_offspring(pop.elite(0.5), t, 2, I._iface_positions(t),
                          random.Random(0), m=5)
    assert len(kids) == 5
    for g, prec in kids:
        assert len(g) <= 2 and prec is not None


# --- LLM parsing (multi-candidate) ------------------------------------------

def test_parse_many_reads_all_tags_dedupes_and_clamps():
    txt = ("reasoning...\n<mut>3:W,5:L</mut>\n<mut>3:W, 5:L</mut>\n"
           "<mut>2:F</mut>\n<mut>none</mut>\n<mut>7:Y</mut>")
    got = I.parse_many(txt, L=10, k_mut=2, m=3)
    # dup (3:W,5:L) collapsed; clamped to m=3; order preserved
    assert got == [[(3, "W"), (5, "L")], [(2, "F")], []]


def test_parse_many_drops_bad_tokens():
    got = I.parse_many("<mut>99:W, 2:B, 3:L</mut>", L=5, k_mut=5, m=5)
    assert got == [[(3, "L")]]                          # 99 oob, B invalid


def test_parse_many_empty_when_no_tags():
    assert I.parse_many("no tags here", L=10, k_mut=2, m=3) == []


# --- the evolutionary loop --------------------------------------------------

def test_evolve_ea_never_regresses_frontier_and_respects_budget(patched):
    t = _T("x1", "ACDEFGHIKLMN", [1] * 12)
    win = {"a": 0, "b": 11, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    pop, traj, meta = I.evolve("iter-ea", t, win, cache, k_mut=3, n_seed=6,
                               n_gen=6, m_off=5, max_evals=30, elite_frac=0.3,
                               relax_repeats=1, timeout=1.0)
    assert pop.evals <= 30                              # matched budget honoured
    seed_best = traj[0]["best_recovered"]
    assert pop.best_recovered() <= seed_best + 1e-9     # elitism: never worse
    # trajectory carries the 2604.19440 diagnostics on every generation row
    g1 = [d for d in traj if d.get("gen") == 1][0]
    for key in ("breakthrough", "beat_frontier_rate", "refine_rate", "n_scored"):
        assert key in g1


def test_evolve_llm_drives_toward_W_and_accounts_cost(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    # the model keeps proposing strong (W-rich) sets -> loop should adopt them
    llm = _scripted(["<mut>0:W,1:W</mut>\n<mut>2:W</mut>\n<mut>0:W,3:W</mut>"])
    pop, traj, meta = I.evolve("iter-llm", t, win, cache, k_mut=2, n_seed=4,
                               n_gen=4, m_off=3, max_evals=25, elite_frac=0.4,
                               relax_repeats=1, timeout=1.0, llm=llm)
    best = pop.best()
    assert any(a == "W" for _, a in best["subs"])       # found a W substitution
    assert meta["llm_calls"] >= 1 and meta["model_id"] == "scripted"
    assert meta["status_counts"].get("ok", 0) >= 1


def test_evolve_llm_falls_back_when_model_malforms(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    llm = _scripted(["I won't use the tag"])            # never parseable
    pop, traj, meta = I.evolve("iter-llm", t, win, cache, k_mut=2, n_seed=4,
                               n_gen=3, m_off=3, max_evals=20, elite_frac=0.4,
                               relax_repeats=1, timeout=1.0, llm=llm, max_retries=1)
    assert meta.get("fallback_gens", 0) >= 1            # random child kept it moving
    assert meta["status_counts"].get("malformed", 0) >= 1
    assert pop.best_recovered() is not None             # still produced a result


# --- driver plumbing --------------------------------------------------------

def test_load_windows_reads_committed_window_per_target(tmp_path):
    f = tmp_path / "seqmut_rows.jsonl"
    f.write_text(
        json.dumps({"target_id": "a", "mut_arm": "nomut", "committed_window": [2, 7],
                    "contact_retention": 0.5, "window_clean": True}) + "\n" +
        json.dumps({"target_id": "a", "mut_arm": "llm", "committed_window": [9, 9]}) + "\n" +
        json.dumps({"target_id": "b", "mut_arm": "nomut", "committed_window": None}) + "\n")
    w = I._load_windows(f)
    assert w["a"] == {"a": 2, "b": 7, "retention": 0.5, "clean": True}  # first wins
    assert "b" not in w                                 # null window skipped


def test_finalist_row_is_boltz_addressable_and_carries_iter_block(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    llm = _scripted(["<mut>0:W,1:W</mut>\n<mut>2:W</mut>"])
    pop, traj, meta = I.evolve("iter-llm", t, win, cache, k_mut=2, n_seed=4,
                               n_gen=3, m_off=3, max_evals=20, elite_frac=0.4,
                               relax_repeats=1, timeout=1.0, llm=llm)
    row = I._finalist_row(t, "iter-llm", win, cache, pop, traj, meta, 2, 1, 1.0,
                          "placement_search")
    # the row is a standard seqmut row: boltz_validate keys on (target_id, _arm_label)
    assert row["committed_window"] == [0, 9] and row["subs"]
    assert row["recovered_binding_energy"] == pop.best_recovered()
    # model carried into the label so two models never collide
    assert S._arm_label(row) == "iter-llm:scripted:k2"
    assert row["iter"]["best_recovered"] == pop.best_recovered()
    assert "breakthrough_rate" in row["iter"] and "mean_beat_frontier" in row["iter"]


def test_ea_finalist_label_has_no_model(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    cache = S.ScanCache(Path("/dev/null"))
    pop, traj, meta = I.evolve("iter-ea", t, win, cache, k_mut=2, n_seed=4,
                               n_gen=3, m_off=3, max_evals=20, elite_frac=0.4,
                               relax_repeats=1, timeout=1.0)
    row = I._finalist_row(t, "iter-ea", win, cache, pop, traj, meta, 2, 1, 1.0,
                          "placement_search")
    assert S._arm_label(row) == "iter-ea:k2"            # no model segment


class _FakeBoltz:
    """A GPU-like scorer with no GPU: dG rewards 'W' (-5 each). Exercises the
    real-score-in-loop plumbing (serial eval, Boltz finalist row) without Boltz."""
    is_gpu = True

    def __init__(self):
        self.calls = 0

    def score(self, t, subs, window):
        self.calls += 1
        n_w = sum(1 for _, a in subs if str(a).upper() == "W")
        dG = -20.0 - 5.0 * n_w + 1.0 * (len(subs) - n_w)
        return dG, {"dG_refined": dG, "n_valid_folds": 4, "variant_id": "x"}

    def close(self):
        pass


def test_evolve_with_boltz_scorer_uses_real_dG_and_serial(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    sc = _FakeBoltz()
    pop, traj, meta = I.evolve("iter-llm", t, win, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=4, n_gen=3, m_off=3, max_evals=12,
                               elite_frac=0.4, relax_repeats=1, timeout=1.0,
                               llm=_scripted(["<mut>0:W,1:W</mut>\n<mut>2:W</mut>"]),
                               n_workers=4, scorer=sc)
    # the in-loop fitness is the scorer's dG (more negative = better)
    assert pop.best_recovered() <= -20.0
    assert sc.calls == pop.evals                     # one fold per distinct eval, serial
    # finalist row is a Boltz row: no proxy native_dG, carries inloop_evaluator
    row = I._finalist_row(t, "iter-llm", win, S.ScanCache(Path("/dev/null")), pop,
                          traj, meta, 2, 1, 1.0, "placement_search")
    assert row["inloop_evaluator"] == "boltz" and row["native_dG"] is None
    assert row["recovered_binding_energy"] == pop.best_recovered()
    assert row["committed_window"] == [0, 9] and row["subs"]   # foldable by the report


def test_run_end_to_end_ea_only_writes_rows_traj_and_nomut(patched, tmp_path, monkeypatch):
    # a seqmut rows file that supplies the committed window (so run_campaign is
    # never called) and lets us run with no manifest machinery
    seqrows = tmp_path / "seqmut_rows.jsonl"
    seqrows.write_text(json.dumps({"target_id": "x1", "mut_arm": "nomut",
                                   "committed_window": [0, 9],
                                   "contact_retention": 1.0, "window_clean": True}) + "\n")
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    from lnr import env as E
    monkeypatch.setattr(E, "load_manifest", lambda *a, **k: [t])

    out = tmp_path / "seqmut_iter_rows.jsonl"
    traj = tmp_path / "seqmut_iter_traj.jsonl"
    args = types.SimpleNamespace(
        manifest="m", pdb_dir="p", out=str(out), traj=str(traj),
        scan_cache=str(tmp_path / "scans.jsonl"), seed_wt="", seed_scans="",
        seed_windows=str(seqrows), arms="iter-ea", closure_arm="placement_search",
        budget=6, k_mut=2, n_seed=4, n_gen=3, m_off=3, max_evals=15,
        elite_frac=0.3, n_workers=2, relax_repeats=1, max_seconds=8.0,
        per_target_timeout=1.0, limit=0, backend="", model="", effort="low",
        max_spend_usd=0.0, max_tokens=6000, max_retries=1, save_raw=False)
    I.run(args)

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    labels = {S._arm_label(r) for r in rows}
    assert "nomut:k2" in labels and "iter-ea:k2" in labels   # base + finalist both written
    assert all(r["target_id"] == "x1" for r in rows)
    tr = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
    assert tr and all(d["arm"] == "iter-ea" for d in tr)     # trajectory persisted


def test_single_shot_seqmut_rejects_iter_arms(tmp_path, monkeypatch):
    # the single-shot driver must refuse an iter-* arm with a pointer to this module
    monkeypatch.setattr(sys, "argv",
                        ["seqmut", "--manifest", "m", "--pdb-dir", "p",
                         "--out", str(tmp_path / "o.jsonl"), "--mut-arms", "iter-llm"])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert "seqmut_iter.py" in str(e.value)
