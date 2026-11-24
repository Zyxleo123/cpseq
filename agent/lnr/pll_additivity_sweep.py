r"""Designed fold sweep: does the additivity gap GROW with k?

The pilot (`pll_additivity.py`, jobs 120901/120905) found the frozen pll table
over-charges mutations -- 17/22 gaps positive, sign test p = 0.0169, 7 of them
bootstrap-resolved. It could not say whether the effect grows with the number of
substitutions, because it reused folds made for another experiment: k was only
{1, 2} and the mutants were an unbalanced convenience sample.

A k-slope is what turns "the table is biased" into "the table degrades
predictably", and only the second licenses a claim about where `best_subs` stops
being usable. So this job FOLDS A DESIGNED SET and writes rows in exactly the
shape `pll_additivity.report` already consumes -- the statistics are not
reimplemented here, and the two experiments stay comparable row for row.

THE DESIGN, AND WHY EACH PART IS THERE

  k = 1..4, capped by the released window. Substitutions are drawn ONLY from
  inside the window, because the cached folds pinned the backbone everywhere
  else: an edit in the held region cannot move the structure, so pooling it with
  a window edit would dilute exactly the effect being measured. Windows here run
  2-9 residues, so not every target reaches k=4; the per-target cap is recorded.

  Three mutants per (target, k), each answering a different question:
    `opt`     the table's own exact argmax under the window restriction. If the
              gap is large HERE, the optimum `best_subs` returns is the thing
              being mispredicted -- the most decision-relevant case.
    `pos`     a random subset of positive-gain substitutions. Tests whether the
              bias is a property of the optimum or of edits in general.
    `any`     a random subset of any substitutions, including losing ones. The
              table predicts a loss; does the pose absorb that too?
  Without `pos`/`any`, a gap at the optimum could be a selection effect: the
  argmax is where the table is most confident and therefore most exposed.

  NATIVE REPLICATES. The native is folded at several independent draws, not one.
  Every gap is a difference against the native's own-pose score, so the native's
  error enters every single comparison -- it is the one quantity worth paying to
  pin down, and the pilot's floor (median sd 0.228 nats/res) came from single
  draws. More folds per sequence was the stated lever; this is it.

WHAT THIS STILL CANNOT SAY. It reuses `boltz_inloop.fold_score_one` verbatim,
which folds against a HELD TEMPLATE: backbone outside the window stays native.
That is deliberate -- it is the project's actual oracle and the proven code path,
so this job cannot fail for a new reason -- but it means the measured gap REMAINS
A LOWER BOUND. A free co-fold arm, which would turn the bound into an estimate,
needs a yaml path without `_held_templates_block` and is the follow-up, not this.
Do not let the designed-sweep label launder that caveat away.

RESUMABLE, because `debug` caps at two hours and a full sweep does not fit.
Rows are appended and completed variants are skipped by key, so hitting the wall
costs a resubmission rather than a restart. The submitter chains several jobs
with `--dependency=afterany` for exactly that reason.

Run:
  python agent/lnr/pll_additivity_sweep.py --selftest
  python agent/lnr/pll_additivity_sweep.py --plan        # no GPU, prints the set
  python agent/lnr/pll_additivity_sweep.py --run --out sweep.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

K_VALUES = (1, 2, 3, 4)
ARMS = ("opt", "pos", "any")


def load_windows(rows_path):
    """{target_id: (a, b)} from the committed windows of the seqmut rows.

    `committed_window` is a LIST [a, b]. An earlier read of this file treated it
    as a dict and silently found zero windows, so the shape is asserted rather
    than assumed."""
    out = {}
    for line in Path(rows_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        w = r.get("committed_window")
        if not (isinstance(w, (list, tuple)) and len(w) == 2):
            continue
        a, b = int(w[0]), int(w[1])
        if a <= b:
            out[r["target_id"]] = (a, b)
    return out


def pick_subs(pssm, seq, k, arm, a, b, rng, metric=None):
    """`(subs, reason)` -- `k` substitutions inside [a, b] per the arm's question.

    `subs` is None when k cannot be supplied, and `reason` says WHICH of two
    quite different causes applies. The distinction is not cosmetic: a window
    with 4 positions that yields no k=4 `opt` is telling you the TABLE ran out
    of positive-gain positions, not that the window was small -- which is a fact
    about how quickly `best_subs` saturates, and exactly the kind of thing that
    gets misread as a geometry limit. (It did: the first version of this file
    reported "window too small" for 1elr k=1 on a 2-position window.)

    Never returns fewer than k: a short return would make the `k` column a lie."""
    win = [p for p in range(a, b + 1) if p < len(seq)]
    if len(win) < k:
        return None, f"window spans {len(win)} position(s), k={k}"
    if arm == "opt":
        subs, _ = pssm.best_subs(seq, k, metric=metric, mutable=win)
        if len(subs) == k:
            return list(subs), ""
        return None, (f"table offers only {len(subs)} positive-gain "
                      f"position(s) in the window, k={k}")
    gains = pssm.position_gains(seq, metric=metric, mutable=win)
    if arm == "pos":
        pool = [(p, aa) for p, aa, g in gains if g > 0]
        if len(pool) < k:
            return None, (f"only {len(pool)} positive-gain substitution(s) "
                          f"in the window, k={k}")
        return sorted(rng.sample(pool, k)), ""
    # `any`: any position in the window, any residue other than the incumbent.
    alpha = [c for c in pssm.alphabet if c in set("ACDEFGHIKLMNPQRSTVWY")]
    pos = rng.sample(win, k)
    return sorted((p, rng.choice([c for c in alpha if c != seq[p]]))
                  for p in pos), ""


def plan(manifest, pdb_dir, seqmut_rows, pssm_dir, k_values=K_VALUES,
         n_native_draws=3, seed=7, metric=None, targets=(), verbose=True):
    """The variant set, with no GPU and no folding. Printing the plan before
    spending 5 GPU-hours is how a window that cannot support k=4, or a target
    with no cached PSSM, gets caught for free."""
    from lnr import bindscore as bs
    from lnr import env as E

    wins = load_windows(seqmut_rows)
    ts = {t.target_id: t for t in E.load_manifest(str(manifest), str(pdb_dir))}
    want = set(targets) if targets else set(wins)
    items, skipped = [], []
    for tid in sorted(want & set(wins) & set(ts)):
        a, b = wins[tid]
        t = ts[tid]
        hits = sorted(Path(pssm_dir).glob(f"pssm_{tid}_*.json"))
        if not hits:
            skipped.append((tid, "no cached PSSM"))
            continue
        p = bs.TargetPSSM.load(hits[0])
        if p.pep_seq != t.seq:
            skipped.append((tid, f"PSSM/manifest sequence mismatch "
                                 f"({p.pep_seq} vs {t.seq})"))
            continue
        rng = random.Random(f"{seed}:{tid}")
        for d in range(n_native_draws):
            items.append({"target_id": tid, "window": [a, b], "arm": "native",
                          "k": 0, "draw": d, "subs": []})
        span = len([q for q in range(a, b + 1) if q < len(t.seq)])
        for k in k_values:
            if span < k:
                continue
            for arm in ARMS:
                subs, why = pick_subs(p, t.seq, k, arm, a, b, rng,
                                      metric=metric)
                if subs is None:
                    skipped.append((tid, f"k={k} arm={arm}: {why}"))
                    continue
                items.append({"target_id": tid, "window": [a, b], "arm": arm,
                              "k": k, "draw": 0,
                              "subs": [[int(i), aa] for i, aa in subs]})
    if verbose:
        print(f"targets planned : {len({i['target_id'] for i in items})}")
        print(f"fold calls       : {len(items)}  "
              f"(native {sum(1 for i in items if i['arm'] == 'native')}, "
              f"mutant {sum(1 for i in items if i['arm'] != 'native')})")
        by_k = {}
        for i in items:
            by_k[i["k"]] = by_k.get(i["k"], 0) + 1
        print(f"by k             : {dict(sorted(by_k.items()))}")
        if skipped:
            print(f"skipped          : {len(skipped)}")
            for s in skipped[:10]:
                print(f"    {s[0]}: {s[1]}")
    return items


def item_key(it):
    return (f"{it['target_id']}|{it['window'][0]}-{it['window'][1]}"
            f"|{it['arm']}|k{it['k']}|d{it['draw']}"
            f"|{''.join(f'{i}{a}' for i, a in it['subs'])}")


def done_keys(out_path):
    p = Path(out_path)
    if not p.exists():
        return set()
    ks = set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ks.add(json.loads(line)["sweep_key"])
        except (ValueError, KeyError):
            continue
    return ks


def run(manifest, pdb_dir, seqmut_rows, pssm_dir, out_path, work,
        replicas=4, boltz_seed=1234, k_values=K_VALUES, n_native_draws=3,
        seed=7, metric=None, scope="all", device="cpu", targets=(),
        limit=0, verbose=True):
    from lnr import bindscore as bs
    from lnr import boltz_inloop as BI
    from lnr import env as E
    from lnr import pll_additivity as PA
    from proteinfoundation.cpseq.adapters.boltz2 import Boltz2Config
    from lnr import boltz_validate as BV

    items = plan(manifest, pdb_dir, seqmut_rows, pssm_dir, k_values=k_values,
                 n_native_draws=n_native_draws, seed=seed, metric=metric,
                 targets=targets, verbose=verbose)
    have = done_keys(out_path)
    todo = [i for i in items if item_key(i) not in have]
    if limit:
        todo = todo[:limit]
    print(f"\nalready done: {len(have)}   to fold now: {len(todo)}"
          f"{f' (limit {limit})' if limit else ''}")
    if not todo:
        print("nothing to fold -- the sweep is complete for this plan.")
        return []

    cfg = Boltz2Config()
    cfg.use_potentials = True
    cfg.seed = int(boltz_seed)
    BV.configure_replicas(cfg, replicas)

    ts = {t.target_id: t for t in E.load_manifest(str(manifest), str(pdb_dir))}
    interface_only = (scope == "iface")
    scorer = bs.MPNNScorer(model_name=bs.PRIMARY_MODEL, device=device)
    frozen = {}
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for n, it in enumerate(todo, 1):
        tid = it["target_id"]
        t = ts[tid]
        if tid not in frozen:
            frozen[tid] = bs.TargetPSSM.load(
                sorted(Path(pssm_dir).glob(f"pssm_{tid}_*.json"))[0])
        fz = frozen[tid]
        subs = [(int(i), aa) for i, aa in it["subs"]]
        a, b = it["window"]
        print(f"\n[{n}/{len(todo)}] {tid} {it['arm']} k={it['k']} "
              f"draw={it['draw']} subs={it['subs']}", flush=True)
        rec = BI.fold_score_one(t, subs, {"a": a, "b": b}, cfg, work,
                               replicas=replicas, draw=it["draw"])
        row = {"sweep_key": item_key(it), "target_id": tid,
               "window": [a, b], "arm": it["arm"], "k": it["k"],
               "draw": it["draw"], "hash": rec.get("variant_id"),
               "native_seq": fz.pep_seq,
               "subs": it["subs"], "in_window": True,
               "predict_status": rec.get("predict_status"),
               "fold_error": rec.get("error")}
        pdbs = []
        if rec.get("boltz_pdb"):
            pdbs = sorted(Path(rec["boltz_pdb"]).parent.glob("*_model_*.pdb"))
        per_pdb = []
        for pdb in pdbs:
            try:
                sq = PA.chain_seq(pdb, PA.BINDER_CHAIN)
                if len(sq) != fz.n or "X" in sq:
                    per_pdb.append({"pdb": str(pdb),
                                    "error": f"len {len(sq)} != {fz.n} or X"})
                    continue
                own = bs.TargetPSSM.from_complex(
                    pdb, pep_chain=PA.BINDER_CHAIN,
                    rec_chains=PA.RECEPTOR_CHAINS,
                    model_name=bs.PRIMARY_MODEL, device=device, scorer=scorer)
                per_pdb.append({"pdb": str(pdb), "seq": sq,
                                "pll_own": own.score(
                                    sq, metric=metric,
                                    interface_only=interface_only)})
            except Exception as exc:                       # noqa: BLE001
                per_pdb.append({"pdb": str(pdb),
                                "error": f"{type(exc).__name__}: {exc}"[:300]})
        ok = [r for r in per_pdb if "pll_own" in r]
        row["per_pdb"] = per_pdb
        row["n_pdb"] = len(pdbs)
        row["n_ok"] = len(ok)
        if ok:
            sq = ok[0]["seq"]
            row["seq"] = sq
            row["is_native"] = (sq == fz.pep_seq)
            vals = [r["pll_own"] for r in ok]
            row["pll_own_median"] = st.median(vals)
            row["pll_own_sd"] = st.stdev(vals) if len(vals) > 1 else None
            row["pll_frozen"] = fz.score(sq, metric=metric,
                                         interface_only=interface_only)
        # APPEND, one row at a time and flushed: a two-hour wall must cost the
        # variant in flight and nothing else.
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        written.append(row)
        print(f"    n_ok={row['n_ok']}/{row['n_pdb']} "
              f"own={row.get('pll_own_median')} "
              f"frozen={row.get('pll_frozen')} "
              f"{'' if not row.get('fold_error') else row['fold_error']}",
              flush=True)

    print(f"\nwrote {len(written)} rows -> {out}")
    return written


def selftest():
    ok = True

    def eq(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {label}: {got!r}"
              + ("" if good else f" != {want!r}"))

    print("window loading asserts the LIST shape")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rows.jsonl"
        p.write_text("\n".join([
            json.dumps({"target_id": "1bjr", "committed_window": [6, 9]}),
            json.dumps({"target_id": "bad1", "committed_window": {"a": 1,
                                                                  "b": 2}}),
            json.dumps({"target_id": "bad2", "committed_window": None}),
            json.dumps({"target_id": "bad3"}),
            json.dumps({"target_id": "1elr", "committed_window": [0, 1]}),
        ]) + "\n")
        w = load_windows(p)
        eq("list windows load", w, {"1bjr": (6, 9), "1elr": (0, 1)})
        eq("a dict window is NOT silently accepted", "bad1" in w, False)

    print("keys and resumability")
    it = {"target_id": "1bjr", "window": [6, 9], "arm": "opt", "k": 2,
          "draw": 0, "subs": [[6, "W"], [7, "K"]]}
    eq("key is stable", item_key(it), "1bjr|6-9|opt|k2|d0|6W7K")
    it2 = dict(it, draw=1)
    eq("draw changes the key", item_key(it2) != item_key(it), True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jsonl"
        p.write_text(json.dumps({"sweep_key": item_key(it)}) + "\n"
                     + "not json\n"
                     + json.dumps({"no_key": 1}) + "\n")
        ks = done_keys(p)
        eq("completed key is found", item_key(it) in ks, True)
        eq("malformed lines do not crash resume", len(ks), 1)
        eq("missing file is empty, not an error",
           done_keys(Path(d) / "nope.jsonl"), set())

    print("subs selection")

    class FakePSSM:
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")

        def best_subs(self, seq, k, metric=None, mutable=None):
            m = sorted(mutable)[:k]
            return [(p, "W") for p in m], 0.0

        def position_gains(self, seq, metric=None, mutable=None):
            return [(p, "K", 0.1) for p in sorted(mutable)]

    fp, rng = FakePSSM(), random.Random(0)
    eq("opt takes k from inside the window",
       pick_subs(fp, "A" * 10, 2, "opt", 3, 7, rng)[0], [(3, "W"), (4, "W")])
    short = pick_subs(fp, "A" * 10, 4, "opt", 3, 4, rng)
    eq("too few positions refuses rather than shrinking k", short[0], None)
    eq("and the reason names the WINDOW, not the table",
       "window spans 2" in short[1], True)
    # the distinction that was mislabelled: enough positions, too few gains
    class NoGain(FakePSSM):
        def best_subs(self, seq, k, metric=None, mutable=None):
            return [(sorted(mutable)[0], "W")], 0.0
    starved = pick_subs(NoGain(), "A" * 10, 3, "opt", 3, 7, rng)
    eq("a gain-starved table is NOT reported as a small window",
       starved[0] is None and "positive-gain" in starved[1], True)
    s = pick_subs(fp, "A" * 10, 2, "any", 3, 7, rng)[0]
    eq("any stays inside the window", all(3 <= p <= 7 for p, _ in s), True)
    eq("any never proposes the incumbent",
       all(aa != "A" for _, aa in s), True)
    eq("any returns exactly k", len(s), 2)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--manifest", default="/zfsauton/scratch/yixiz/LNR/test.txt")
    ap.add_argument("--pdb-dir", default="/zfsauton/scratch/yixiz/LNR/pdbs")
    ap.add_argument("--seqmut-rows",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/seqmut_rows.jsonl")
    ap.add_argument("--pssm-dir",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/pssm")
    ap.add_argument("--out", default="pll_additivity_sweep.jsonl")
    ap.add_argument("--work",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/boltz_sweep")
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--boltz-seed", type=int, default=1234)
    ap.add_argument("--native-draws", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scope", default="all", choices=("all", "iface"))
    ap.add_argument("--metric", default="")
    ap.add_argument("--targets", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="fold at most N variants this invocation (smoke test)")
    a = ap.parse_args()
    ks = tuple(k for k in K_VALUES if k <= a.k_max)
    tg = tuple(x for x in a.targets.replace(",", " ").split() if x)
    if a.selftest:
        return selftest()
    if a.plan:
        plan(a.manifest, a.pdb_dir, a.seqmut_rows, a.pssm_dir, k_values=ks,
             n_native_draws=a.native_draws, seed=a.seed,
             metric=(a.metric or None), targets=tg)
        return 0
    if a.run:
        run(a.manifest, a.pdb_dir, a.seqmut_rows, a.pssm_dir, a.out, a.work,
            replicas=a.replicas, boltz_seed=a.boltz_seed, k_values=ks,
            n_native_draws=a.native_draws, seed=a.seed,
            metric=(a.metric or None), scope=a.scope, device=a.device,
            targets=tg, limit=a.limit)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
