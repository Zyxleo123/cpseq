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


def test_two_effort_levels_split_the_spend_cap_instead_of_doubling_it(
        tmp_path, monkeypatch):
    """The cap is PER CLIENT, and the hi-effort arm needs a second client.

    Handing both clients `--max-spend-usd` would make the run's real ceiling 2x
    what was asked for, against a convention (docs/seqmut_notes.md) where every
    spend figure is per-run and a cap that IS reached silently turns the arm into
    a random-EA control for the rest of the run. So the cap is divided.
    """
    from lnr import run as R
    seen = []
    monkeypatch.setattr(R, "make_llm",
                        lambda backend, model, effort, cap, max_tokens:
                        seen.append((effort, cap)) or types.SimpleNamespace(
                            model_id=model, effort=effort))
    efforts, clients = I.make_llm_clients(
        ["iter-llm", "iter-llm-hi"], backend="openai", model="m", effort="low",
        hi_effort="high", max_spend_usd=3.0, max_tokens=6000)
    assert sorted(seen) == [("high", 1.5), ("low", 1.5)]
    assert sum(c for _, c in seen) == 3.0       # the asked-for ceiling, not 2x
    assert efforts == {"low": ["iter-llm"], "high": ["iter-llm-hi"]}
    assert set(clients) == {"low", "high"}

    # Arms at the SAME effort share one client, so one ledger and the whole cap.
    # Dividing per ARM instead would under-fund a run that asked for no second
    # effort level, which is the opposite failure and just as quiet.
    seen.clear()
    I.make_llm_clients(["iter-llm", "iter-llm-ala"], backend="openai", model="m",
                       effort="low", hi_effort="high", max_spend_usd=3.0,
                       max_tokens=6000)
    assert seen == [("low", 3.0)]

    # And the EA-only path builds no client at all, so it must not demand a
    # --backend: the control arm reads no brief and makes no call.
    seen.clear()
    assert I.make_llm_clients([], backend="", model="", effort="low",
                              hi_effort="high", max_spend_usd=0.0,
                              max_tokens=6000) == ({}, {})
    assert seen == []


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


# --- noisy-fitness defences (each test pins one MEASURED failure mode) --------

class _NoisyBoltz:
    """Stochastic evaluator stand-in. Every genome's TRUE fitness is -20 - 5*n_W,
    but a measurement adds a per-draw offset, so draw 0 of the WT is a lucky -30
    while its truth is -20 -- exactly the optimizer's curse seen on 1bjr, where a
    lucky in-loop draw of -8.01 REU vs wild-type realised -0.57 on a redraw."""

    is_gpu = True
    folds = True
    stochastic = True

    def __init__(self, lucky_key="", lucky_offset=-10.0):
        self.calls = 0
        self.draws = []
        self.lucky_key = lucky_key
        self.lucky_offset = lucky_offset

    def truth(self, subs):
        n_w = sum(1 for _, a in subs if str(a).upper() == "W")
        return -20.0 - 5.0 * n_w + 1.0 * (len(subs) - n_w)

    def sem(self, row):
        return row.get("dG_refined_std", 0.0) / 2.0

    def score(self, t, subs, window, draw=0):
        self.calls += 1
        self.draws.append((S.subs_key(subs), draw))
        dG = self.truth(subs)
        if draw == 0 and S.subs_key(subs) == self.lucky_key:
            dG += self.lucky_offset              # the lucky first draw
        return dG, {"dG_refined": dG, "dG_refined_std": 4.0, "n_valid_folds": 4,
                    "variant_id": "x", "draw": draw}

    def close(self):
        pass


def test_arms_share_an_identical_seed_population(patched):
    """The matched-budget control was confounded at the seed: evolve() keyed the
    RNG on the arm, so each arm drew a DIFFERENT starting pool (measured mean gap
    7.66 REU between arms, 19.7 on 1jrr -- larger than the effect being read)."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    seeds = {}
    for arm, llm in (("iter-ea", None),
                     ("iter-llm", _scripted(["<mut>0:W</mut>"])),
                     ("iter-llm-ala", _scripted(["<mut>0:W</mut>"]))):
        pop, traj, _ = I.evolve(arm, t, win, S.ScanCache(Path("/dev/null")),
                                k_mut=2, n_seed=6, n_gen=1, m_off=2, max_evals=10,
                                elite_frac=0.4, relax_repeats=1, timeout=1.0,
                                llm=llm, n_workers=1)
        gen0 = {S.subs_key(m["subs"]) for m in pop.members.values() if m["gen"] == 0}
        seeds[arm] = (gen0, traj[0]["best_recovered"])
    pools = [v[0] for v in seeds.values()]
    assert pools[0] == pools[1] == pools[2], "arms must start from the same pool"
    assert len({v[1] for v in seeds.values()}) == 1, "and so from the same frontier"


def test_seed_pool_is_stable_across_arms_but_operator_draws_are_not(patched):
    """The seed RNG is arm-independent; the OPERATOR rng must still be per-arm so
    the EA's draws are not coupled to the LLM arm's fallbacks."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    out = {}
    for arm in ("iter-ea", "iter-llm"):
        pop, _, _ = I.evolve(arm, t, win, S.ScanCache(Path("/dev/null")),
                             k_mut=2, n_seed=4, n_gen=3, m_off=3, max_evals=20,
                             elite_frac=0.4, relax_repeats=1, timeout=1.0,
                             llm=(None if arm == "iter-ea" else
                                  _scripted(["<mut>7:W</mut>"])), n_workers=1)
        out[arm] = {S.subs_key(m["subs"]) for m in pop.members.values() if m["gen"] > 0}
    assert out["iter-ea"] != out["iter-llm"]      # operators diverge after seeding


def test_reassess_corrects_a_lucky_incumbent_and_costs_budget():
    """Without re-measurement one lucky draw owns the frontier for the whole run."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    sc = _NoisyBoltz(lucky_key="", lucky_offset=-10.0)   # WT measured lucky at -30
    pop = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, 20,
                       scorer=sc)
    pop.evaluate([], gen=0)
    assert pop.members[""]["recovered"] == -30.0        # the lucky draw
    before_evals = pop.evals
    shift = pop.reassess(pop.members[""], gen=1)
    assert pop.evals == before_evals + 1                # honestly charged
    assert pop.reassessments == 1
    assert sc.draws[-1] == ("", 1)                      # an INDEPENDENT draw
    assert pop.members[""]["recovered"] == -25.0        # mean of -30 and -20
    assert shift == 5.0                                 # the curse, measured live
    assert len(pop.members[""]["obs"]) == 2 and pop.members[""]["sem"] > 0


def test_reassess_is_a_noop_and_free_on_a_deterministic_evaluator(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    pop = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, 20)
    pop.evaluate([], gen=0)
    n = pop.evals
    assert pop.reassess(pop.best(), gen=1) is None
    assert pop.evals == n and pop.reassessments == 0


def test_commit_is_pessimistic_and_exploration_optimistic():
    """A noisy -31 must not outrank a well-measured -30 when we COMMIT, but it
    must still be explored."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    pop = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, 20,
                       z_explore=1.0, z_commit=1.0)
    pop.members["noisy"] = {"subs": ((0, "W"),), "obs": [-36.0, -26.0],
                            "recovered": -31.0, "sem": 5.0, "gen": 1}
    pop.members["solid"] = {"subs": ((1, "W"),), "obs": [-30.0, -30.0],
                            "recovered": -30.0, "sem": 0.1, "gen": 1}
    assert S.subs_key(pop.best()["subs"]) == "1:W"      # commit to the solid one
    assert S.subs_key(pop.leader()["subs"]) == "0:W"    # explore the noisy one
    assert S.subs_key(pop.elite(1.0)[0]["subs"]) == "0:W"


def test_brief_shows_noise_tried_set_and_an_exploration_floor():
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    elite = [{"subs": ((2, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 2.0,
              "gen": 1}]
    tried = [{"subs": ((5, "D"),), "obs": [-10.0], "recovered": -10.0, "sem": 1.0,
              "gen": 0}]
    b = I.iter_brief(t, win, 2, elite, 4, 2, folded=True, tried=tried, noise_sd=2.0)
    assert "+/-SE" in b and "NOISY MEASUREMENTS" in b
    assert "5:D" in b and "not on the frontier" in b    # the failure memory
    assert "NOT on the frontier" in b                   # the exploration floor
    assert "Positions already on the frontier: 2" in b
    # the deterministic-proxy brief keeps its old shape (no noise language)
    plain = I.iter_brief(t, win, 2, elite, 4, 2, folded=False)
    assert "NOISY MEASUREMENTS" not in plain and "score(REU)" in plain


def test_brief_lists_every_measured_genome_not_just_the_shown_rows():
    """The duplicate-avoidance instruction has to be SATISFIABLE.

    The brief tells the model each proposal must be absent from what it has
    already measured. While the brief listed only 12 elite + 14 tried rows, a
    pool of hundreds made that uncheckable -- and the measured duplicate rate
    was 34-45%. Every measured key must appear.
    """
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    elite = [{"subs": ((2, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 0.0,
              "gen": 1}]
    measured = ["2:W", "", "4:Q", "7:L,8:M", "3:E"]
    b = I.iter_brief(t, win, 2, elite, 4, 5, measured=measured,
                     mutable=list(range(10)))
    for k in ("2:W", "4:Q", "7:L,8:M", "3:E"):
        assert k in b, f"{k} missing from the brief's measured list"
    assert "(none)" in b                  # the wild-type, rendered readably
    assert "COMPLETE" in b                # and the instruction says so


def _loaded_pop(scored):
    """A Population whose members are set directly: {subs_key: recovered}."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    pop = I.Population(t, win, S.ScanCache(Path("/dev/null")), 2, 1, 1.0, 500)
    for key, rec in scored.items():
        subs = tuple((int(tok.split(":")[0]), tok.split(":")[1])
                     for tok in key.split(",")) if key else ()
        pop.members[key] = {"subs": subs, "obs": [rec], "recovered": rec,
                            "sem": 0.0, "gen": 1}
    return pop


def test_measured_keys_are_raw_so_the_wild_type_is_not_exempt_from_the_filter():
    """`measured_keys` feeds both the display and the duplicate filter.

    The wild-type's key is the empty string. Prettifying it to "(none)" inside
    `measured_keys` would make `"" in set(keys)` false, quietly exempting the
    wild-type from the repeat filter so it could be re-proposed forever.
    """
    pop = _loaded_pop({"": -20.0, "2:W": -30.0})
    keys = pop.measured_keys()
    assert "" in keys and "(none)" not in keys
    assert keys[0] == "2:W"                  # best-first
    assert set(keys) == set(pop.members)     # exactly the filter's domain


def test_coverage_block_reports_untouched_positions_and_no_scores():
    """Coverage, deliberately WITHOUT credit.

    Telling the model where it has not looked is honest search bookkeeping. A
    per-position *gain* table would instead hand it the separable objective's
    solution -- a different, deliberately unpulled lever -- so no fitness value
    may leak into this block.
    """
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    blk = I._coverage_block(["2:W", "2:F", "5:D"], t, mutable=[2, 5, 8])
    assert "NEVER EDITED" in blk and "8" in blk
    left = dict(seg.strip().split(":") for seg in
                blk.split("\n")[1].strip().split("; "))
    # reported as what is LEFT: W and F are spent at position 2, so absent --
    # and so is 'D', which is position 2's own wild-type residue
    assert set("WFD").isdisjoint(left["2"]) and "A" in left["2"]
    # position 5 ('G') had D measured; both D and the wild-type G are gone
    assert set("DG").isdisjoint(left["5"]) and "A" in left["5"]
    # no fitness value may leak into this block -- that is the unpulled lever
    assert "nats" not in blk and "REU" not in blk and "+" not in blk


def test_coverage_block_marks_a_position_exhausted_for_single_edits():
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    allk = [f"2:{a}" for a in I.AA20 if a != "D"]      # position 2 is 'D'
    blk = I._coverage_block(allk, t, mutable=[2])
    assert "EXHAUSTED" in blk and "only combinations remain" in blk


def test_tried_block_leads_with_near_misses_not_the_worst_genomes():
    """The 14 rows should be spent on the informative end of the pool.

    Worst-first (the old order) filled the block with obviously-bad edits,
    which localise nothing; the sets just behind the frontier are the ones whose
    near-success says where the next gain is.
    """
    pop = _loaded_pop({f"{p}:W": -float(p) for p in range(10)})
    rows = pop.tried(limit=3)
    assert S.subs_key(rows[0]["subs"]) == "9:W"      # best first
    assert S.subs_key(rows[-1]["subs"]) == "0:W"     # worst kept for contrast
    # and nothing is dropped when the pool fits in the limit
    assert len(pop.tried(limit=99)) == 10


def test_residue_table_names_the_receptor_residues_each_position_faces():
    """The objective asks the model to match the chemistry across from each
    position; a contact COUNT cannot express it."""
    t = _T("x1", "ACD", [3, 1, 0])
    tbl = I.S._residue_table(t, None, rec_ctx=[["F86", "A85"], ["K12"], []])
    assert "faces" in tbl and "F86 A85" in tbl and "K12" in tbl
    assert "(no contacts)" in tbl                 # position 2 touches nothing
    # and without the column the table is byte-identical to the old one
    assert "faces" not in I.S._residue_table(t, None)


def test_llm_offspring_drops_repeats_and_reasks_once(patched):
    """34-45% of LLM proposals re-proposed a genome already in the pool. A repeat
    now costs a re-ask, not a proposal slot."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    elite = [{"subs": ((2, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 0.0,
              "gen": 1}]
    meta = {"llm_calls": 0, "llm_cost_usd": 0.0, "status_counts": {}}
    llm = _scripted(["<mut>2:W</mut>\n<mut>3:D</mut>",     # 2:W and 3:D both known
                     "<mut>7:W</mut>\n<mut>8:Y</mut>"])    # the re-ask delivers
    kids = I.llm_offspring(llm, t, win, 2, elite, 2, 1, None, meta,
                           __import__("random").Random(0), list(range(10)),
                           known={"2:W", "3:D"}, tried=())
    keys = {S.subs_key(g) for g, _ in kids}
    assert keys == {"7:W", "8:Y"}            # nothing already measured is returned
    assert meta["n_dup_proposed"] == 2
    assert meta["llm_calls"] == 2            # one re-ask, then done
    assert meta.get("fallback_gens") is None  # the EA fallback was NOT needed


def test_llm_offspring_falls_back_only_when_the_reask_also_fails(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    elite = [{"subs": ((2, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 0.0,
              "gen": 1}]
    meta = {"llm_calls": 0, "llm_cost_usd": 0.0, "status_counts": {}}
    llm = _scripted(["<mut>2:W</mut>", "<mut>2:W</mut>"])   # only ever a repeat
    kids = I.llm_offspring(llm, t, win, 2, elite, 2, 1, None, meta,
                           __import__("random").Random(0), list(range(10)),
                           known={"2:W"}, tried=())
    assert kids and meta["fallback_gens"] == 1
    assert meta["status_counts"].get("all_duplicates") == 1


def test_evolve_logs_the_noise_accounting_in_traj_and_finalist(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    sc = _NoisyBoltz(lucky_key="", lucky_offset=-10.0)
    pop, traj, meta = I.evolve("iter-ea", t, win, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=4, n_gen=3, m_off=2, max_evals=20,
                               elite_frac=0.5, relax_repeats=1, timeout=1.0,
                               n_workers=1, scorer=sc)
    gens = [d for d in traj if d.get("phase") != "seed"]
    assert gens and all("incumbent_shift" in d for d in gens)
    assert pop.reassessments >= 1
    row = I._finalist_row(t, "iter-ea", win, S.ScanCache(Path("/dev/null")), pop,
                          traj, meta, 2, 1, 1.0, "placement_search")
    it = row["iter"]
    assert it["n_reassess"] >= 1 and it["best_n_obs"] >= 1
    assert "mean_incumbent_shift" in it and "best_sem" in it


def test_no_reassess_restores_the_single_draw_frontier(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    sc = _NoisyBoltz()
    pop, _, _ = I.evolve("iter-ea", t, win, S.ScanCache(Path("/dev/null")),
                         k_mut=2, n_seed=4, n_gen=3, m_off=2, max_evals=20,
                         elite_frac=0.5, relax_repeats=1, timeout=1.0,
                         n_workers=1, scorer=sc, reassess=False)
    assert pop.reassessments == 0
    assert all(len(m["obs"]) <= 1 for m in pop.scored())


def test_ea_offspring_resamples_known_genomes_so_both_arms_spend_the_budget():
    """The LLM arm re-asks on a repeat; the EA must resample on one, or it quietly
    makes fewer DISTINCT evaluations than its own control (measured 18 vs 24 of a
    24-eval budget) and the matched-budget comparison is void."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    elite = [{"subs": ((0, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 0.0,
              "gen": 0},
             {"subs": ((0, "W"), (1, "Y")), "obs": [-28.0], "recovered": -28.0,
              "sem": 0.0, "gen": 0}]
    rng = __import__("random").Random(5)
    # every genome the two elites can produce by plain crossover is "known"
    known = {"0:W", "1:Y", "0:W,1:Y", ""}
    kids = I.ea_offspring(elite, t, 2, list(range(10)), rng, 6, known=known)
    assert len(kids) == 6
    keys = [S.subs_key(g) for g, _ in kids]
    assert not (set(keys) & known), f"EA returned already-known genomes: {keys}"
    assert len(set(keys)) == 6, f"EA returned duplicates within the batch: {keys}"


def test_ea_offspring_without_known_is_unchanged():
    """Default call site behaviour (no `known`) must be the old one."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    elite = [{"subs": ((0, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 0.0,
              "gen": 0}]
    kids = I.ea_offspring(elite, t, 2, list(range(10)),
                          __import__("random").Random(5), 3)
    assert len(kids) == 3
    assert all(len(g) <= 2 for g, _ in kids)
    assert all(pr == -30.0 for _, pr in kids)


def test_brief_states_the_wild_type_bar_even_when_wt_is_not_elite():
    """Every headline number is paired against nomut, so the WT score is THE bar.
    It is usually not in the elite, and the old text pointed at a '(none)' row
    that was not on screen."""
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    elite = [{"subs": ((2, "W"),), "obs": [-30.0], "recovered": -30.0, "sem": 1.0,
              "gen": 1}]
    b = I.iter_brief(t, win, 2, elite, 4, 2, folded=True, wt_score=-20.85)
    assert "wild-type peptide, unmodified, scores -20.85 REU" in b
    # with no WT score known, fall back to naming the (none) row
    b2 = I.iter_brief(t, win, 2, elite, 4, 2, folded=True)
    assert "'(none)' below is the wild-type sequence" in b2


def test_finalist_row_persists_llm_raw_when_save_raw_is_set(patched):
    t = _T("x1", "ACDEFGHIKL", [1] * 10)
    win = {"a": 0, "b": 9, "retention": 1.0, "clean": True}
    pop, traj, meta = I.evolve("iter-llm", t, win, S.ScanCache(Path("/dev/null")),
                               k_mut=2, n_seed=3, n_gen=2, m_off=2, max_evals=10,
                               elite_frac=0.5, relax_repeats=1, timeout=1.0,
                               llm=_scripted(["<mut>0:W</mut>\n<mut>1:W</mut>"]),
                               save_raw=True, n_workers=1)
    row = I._finalist_row(t, "iter-llm", win, S.ScanCache(Path("/dev/null")), pop,
                          traj, meta, 2, 1, 1.0, "placement_search")
    assert row.get("llm_raw"), "--save-raw collected replies but dropped them"
    assert any("<mut>" in r for r in row["llm_raw"])
