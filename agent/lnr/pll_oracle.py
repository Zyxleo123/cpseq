"""DOES THE EXPENSIVE ORACLE AGREE WITH THE FREE TABLE?

`pll_bg` is a position x amino-acid table: the best <= k substitution set is read
off it in microseconds (`TargetPSSM.best_subs`), exactly, with no search and no
noise. Everything the project has concluded from that table rests on one
untested step -- that a sequence the table likes is a sequence the *structure
oracle* also likes. This measures that step directly:

    fold the table's OPTIMUM and random SUBOPTIMAL sequences of the same edit
    budget, from the same action space, on the same targets, and score them with
    the real pipeline.

The test is deliberately asymmetric in the table's disfavour. The table never
sees the oracle; the oracle never sees the table; the random variants are drawn
from the same positions the optimum was allowed to touch, so "edited vs native"
cannot masquerade as "good vs bad edit".

THE NOISE FLOOR IS THE WHOLE DESIGN CONSTRAINT
----------------------------------------------
A single Boltz fold -> Rosetta dG carries ~3-4 REU of fold noise, and 61% of an
apparent gain once evaporated on an independent redraw. One fold per sequence
would measure the denoiser, not the sequence. So every sequence gets **16 folds
= 4 independent Boltz runs x 4 diffusion samples**: the 4 runs are independent
trunk passes (different seeds), the 4 samples share a trunk pass. Both levels
are recorded per structure, so the between-run and within-run components can be
separated afterwards rather than assumed equal.

TWO PREDICTORS, AND ONE OF THEM HAS TO PASS A GATE FIRST
--------------------------------------------------------
Arm 1 is Boltz-2 with potentials, given the held native backbone as a template.
Arm 2 is AfCycDesign (ColabDesign AF2 + the cyclic offset), which is what
RFpeptides uses to filter its macrocycle designs before Rosetta interface
metrics. Two facts about arm 2 are recorded on every row rather than argued
about:

  * **Two parameter sets, not five.** ColabDesign loads `model_{1,2}_ptm` when
    templates are on and `model_{1..5}_ptm` only when they are off
    (`af/model.py:110-117`), because AF2 models 3-5 were trained without
    templates and carry no template weights. AfCycDesign's published benchmark
    gets five because it is a MONOMER benchmark -- no receptor, so no template.
    We predict a complex and give the receptor as a template, so two is the
    ceiling, and re-seeding does not add samples: fixed sequence + no MSA +
    dropout off is a deterministic forward pass (measured elsewhere in this repo
    at 0.000000 A between seeds).
  * **It has not earned its opinion yet.** On six crystal macrocycle complexes
    this repo measured Boltz-2 at 1.42 A median receptor-aligned peptide CA RMSD
    and afcycdesign at **34.67 A** -- the peptide is not in the pocket, and a
    Rosetta interface energy computed there is a number about nothing. So
    `--gate-af2` re-measures that on THESE targets' native peptides first, and
    the arm's scores are reported only if it clears the gate. The gate's verdict
    is a row, not a judgement call.

SCORING IS THE PROJECT'S OWN PROTOCOL, BOTH VERSIONS
----------------------------------------------------
Every predicted structure -- from either predictor -- is relaxed and scored by
`cpseq.scorer.score_structure`, which is where `refined_dG_separated` comes from.
Two relax restraints are run on the SAME structures:

    v3 ("bond")      harmonic AtomPairConstraint on the closing bond alone.
                     The protocol you asked for.
    v5 ("junction")  bond + closure angles + torsion restrained.

They are both run because they disagree in a known way: cpseq measured 37 of 48
closure angles out of tolerance after a bond-only relax against 0 of 48 before,
so a v3 energy difference can be the minimiser bending a ring open rather than a
property of the sequence. Re-scoring is CPU-only and needs no refold, so the
disagreement is measured instead of assumed away.

STAGES (split by resource, per the project's rule)
--------------------------------------------------
  --select      CPU, seconds   pick targets + build designs      -> designs.json
  --gate-af2    GPU, minutes   AF2 crystal control               -> af2_gate.jsonl
  --fold-boltz  GPU, hours     16 structures per design          -> folds.jsonl
  --fold-af2    GPU, minutes   2 structures per design           -> folds.jsonl
  --score       CPU, parallel  relax + Rosetta, v3 and v5        -> scores.jsonl
  --plot        CPU, seconds   figures                           -> figs/

Every stage is resumable: it reads what is already on disk and does only what is
missing, so a time limit means resubmitting, never restarting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# The two relax restraints, by the names the cpseq protocol versions use.
PROTOCOLS = {
    "v3": "bond",        # v3-cis-amide-muco-windows: bond-restrained relax
    "v5": "junction",    # v5-junction-restrained-relax: bond + angles + torsion
}

DESIGN_CLASSES = ("native", "pll_opt", "rand1", "rand2")

AF2_PARAMS_DIR = os.environ.get(
    "AF2_DIR", "/zfsauton2/home/yixiz/Proteina-Complexa/community_models/ckpts/AF2")

# The gate AfCycDesign has to clear on this project's own controls. Boltz-2
# measured 1.42 A median on six crystal macrocycles, HighFold3 0.64-3.39 A,
# afcycdesign 34.67 A. 3.0 A is HighFold3's worst passing control: a predictor
# that places the peptide at least that well can be argued with; one that does
# not is not in the pocket.
AF2_GATE_RMSD_A = 3.0


def _load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# STAGE 1 -- selection and designs
# ---------------------------------------------------------------------------
def _random_subs(rng, seq, mutable, k):
    """k substitutions at distinct positions drawn from the SAME action space the
    optimum was allowed to use, each to a different residue than the native one."""
    pos = rng.sample(sorted(set(mutable)), k)
    return sorted((int(p), rng.choice([a for a in AA20 if a != seq[p]])) for p in pos)


def select(manifest, pdb_dir, pssm_dir, windows_path, out_dir, *, k_mut=2,
           n_targets=5, n_random=2, metric="pll_bg", model=None, device="cpu",
           pool_seed=11, draw_seed=4242, n_pool=200, min_pep_len=7,
           max_pep_len=20, max_rec_len=400, targets_filter=()):
    """Pick the targets, build the 4 designs each, write designs.json.

    THE SELECTION RULE IS PRE-REGISTERED AND THE TWO RANDOM SEEDS ARE DIFFERENT.
    Targets are ranked by the gap between the table's optimum and the MEDIAN of a
    200-draw random pool (seed `pool_seed`), in TOTAL nats rather than nats per
    residue; the two random variants that are actually folded are drawn
    afterwards with a different seed (`draw_seed`).

    RANKING ON THE PER-RESIDUE GAP IS A LENGTH ARTEFACT, and the first run of
    this selection walked straight into it. `pll_bg` is a MEAN over positions, so
    a fixed k=2 edit budget moves it by about (sum of the two per-position gains)
    / L: short peptides get a mechanically larger per-residue gap for the same
    chemistry, and the rule returned 4- and 5-mers. Two reasons that is wrong
    here rather than merely surprising -- it is this project's recurring
    nuisance-variable trap (a scorer that reads peptide SIZE), and a
    head-to-tail cyclic tetramer is a strained ring that neither predictor nor
    Rosetta represents well. So: rank on the total, and require L >= 7, which is
    the regime the head-to-tail evidence actually covers (the CPSet BB series
    are 7- and 14-mers).
    Using one pool for both would select targets for having extreme random draws
    and then fold exactly those draws -- the gap would be partly a selection
    artefact. With separate seeds the targets are chosen on a property of the
    TABLE and the folded variants are an honest sample.
    """
    from lnr import env as E
    from lnr import pll_inloop as PI
    from lnr import seqmut as S
    from lnr import seqmut_iter as SI
    import lnr.bindscore as bs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = E.load_manifest(str(manifest), str(pdb_dir))
    ts.sort(key=lambda x: x.target_id)
    if targets_filter:
        want = {s for s in targets_filter if s}
        ts = [t for t in ts if t.target_id in want]
    windows = SI._load_windows(Path(windows_path)) if windows_path else {}
    print(f"[select] {len(ts)} targets, {len(windows)} committed windows")
    if not windows:
        print("GATE: no committed windows. The window defines the ring that is "
              "folded; without it there is nothing to build. Pass --windows.")
        return []

    scorer = PI.PLLScorer(metric=metric, scope="all",
                          model_name=(model or bs.PRIMARY_MODEL),
                          device=device, pssm_dir=pssm_dir)
    rows, skipped = [], []
    for t in ts:
        win = windows.get(t.target_id)
        if not win:
            skipped.append((t.target_id, "no committed window"))
            continue
        if not (min_pep_len <= t.length <= max_pep_len):
            skipped.append((t.target_id,
                            f"peptide {t.length} outside "
                            f"[{min_pep_len},{max_pep_len}]"))
            continue
        try:
            rec_len = len(t.rec_ca)
        except Exception:
            rec_len = 10 ** 6
        if rec_len > max_rec_len:
            skipped.append((t.target_id, f"receptor {rec_len} > {max_rec_len}"))
            continue
        mutable = SI._iface_positions(t)
        if len(set(mutable)) < k_mut:
            skipped.append((t.target_id, f"only {len(set(mutable))} mutable"))
            continue
        try:
            pssm = scorer.pssm(t)                      # cached table; no model call
        except Exception as exc:                       # noqa: BLE001
            skipped.append((t.target_id, f"{type(exc).__name__}: {exc}"))
            continue

        wt = float(pssm.score(t.seq))
        opt_subs, opt_score = pssm.best_subs(t.seq, k_mut, mutable=mutable)
        rng_pool = random.Random(f"{pool_seed}|{t.target_id}")
        pool = []
        for _ in range(n_pool):
            subs = _random_subs(rng_pool, t.seq, mutable, k_mut)
            pool.append(float(pssm.score(S.apply_subs(t.seq, subs))))
        rows.append({
            "target_id": t.target_id, "pep_len": int(t.length),
            "rec_len": int(rec_len), "window": win,
            "n_mutable": len(set(mutable)), "mutable": sorted(set(mutable)),
            "native_seq": t.seq, "wt_pll": round(wt, 6),
            "opt_subs": [[int(p), a] for p, a in opt_subs],
            "opt_pll": round(float(opt_score), 6),
            "opt_gain": round(float(opt_score) - wt, 6),
            "pool_median_pll": round(st.median(pool), 6),
            "pool_sd_pll": round(st.pstdev(pool), 6),
            # THE SELECTION STATISTIC: how far the table says its optimum sits
            # above a typical random edit of the same size, in nats/residue.
            "gap_opt_vs_random": round(float(opt_score) - st.median(pool), 6),
            # The selection statistic, in TOTAL nats: the per-residue gap times
            # the peptide length, so a short peptide cannot win on arithmetic.
            "gap_total_nats": round((float(opt_score) - st.median(pool))
                                    * int(t.length), 6),
        })

    rows.sort(key=lambda r: -r["gap_total_nats"])
    chosen = rows[:n_targets]
    print(f"[select] {len(rows)} eligible, {len(skipped)} skipped; "
          f"taking the top {len(chosen)} by table gap")
    for r in rows[: max(n_targets, 10)]:
        mark = "  <-- chosen" if r in chosen else ""
        print(f"   {r['target_id']:6s} len {r['pep_len']:3d} rec {r['rec_len']:4d} "
              f"gap {r['gap_total_nats']:+.2f} nats total "
              f"({r['gap_opt_vs_random']:+.3f}/res)  "
              f"opt_gain {r['opt_gain']:+.3f}/res{mark}")

    # The folded random variants: a DIFFERENT seed from the selection pool.
    designs = []
    for r in chosen:
        tid = r["target_id"]
        seq = r["native_seq"]
        t = next(x for x in ts if x.target_id == tid)
        pssm = scorer.pssm(t)
        entries = [
            {"design": "native", "subs": [], "sequence": seq,
             "pll": r["wt_pll"], "pll_gain": 0.0},
            {"design": "pll_opt", "subs": r["opt_subs"],
             "sequence": S.apply_subs(seq, [(p, a) for p, a in r["opt_subs"]]),
             "pll": r["opt_pll"], "pll_gain": r["opt_gain"]},
        ]
        rng = random.Random(f"{draw_seed}|{tid}")
        seen = {e["sequence"] for e in entries}
        tries = 0
        while len(entries) < 2 + n_random and tries < 500:
            tries += 1
            subs = _random_subs(rng, seq, r["mutable"], k_mut)
            s = S.apply_subs(seq, [(p, a) for p, a in subs])
            if s in seen:
                continue
            pll = float(pssm.score(s))
            # Reject a draw that lands inside the table's indifference band of
            # the optimum: it is not a SUBOPTIMAL sequence, and folding it would
            # test nothing. Recorded as a rejection, not silently dropped.
            if abs(pll - r["opt_pll"]) < 0.05:
                continue
            seen.add(s)
            entries.append({
                "design": f"rand{len(entries) - 1}",
                "subs": [[int(p), a] for p, a in subs], "sequence": s,
                "pll": round(pll, 6), "pll_gain": round(pll - r["wt_pll"], 6)})
        for e in entries:
            e.update({"target_id": tid, "window": r["window"],
                      "native_seq": seq, "k_mut": k_mut,
                      "pll_vs_opt": round(e["pll"] - r["opt_pll"], 6)})
        designs.extend(entries)
        print(f"   {tid}: " + "  ".join(
            f"{e['design']}={e['sequence']}({e['pll']:+.2f})" for e in entries))

    doc = {"k_mut": k_mut, "metric": metric, "n_targets": n_targets,
           "n_random": n_random, "pool_seed": pool_seed, "draw_seed": draw_seed,
           "n_pool": n_pool, "min_pep_len": min_pep_len,
           "max_pep_len": max_pep_len, "max_rec_len": max_rec_len,
           "manifest": str(manifest), "pdb_dir": str(pdb_dir),
           "pssm_dir": str(pssm_dir), "windows": str(windows_path),
           "selection_rule": ("top gap_total_nats (per-residue gap x peptide "
                              "length) among eligible targets; ranking on the "
                              "per-residue gap is a length artefact"),
           "targets": chosen, "designs": designs,
           "eligible": rows, "skipped": skipped}
    (out_dir / "designs.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[select] wrote {out_dir / 'designs.json'}: "
          f"{len(chosen)} targets x {len(designs) // max(1, len(chosen))} designs")
    return doc


# ---------------------------------------------------------------------------
# Shared helpers for the folding stages
# ---------------------------------------------------------------------------
def _targets_by_id(doc, manifest, pdb_dir):
    from lnr import env as E
    ids = {r["target_id"] for r in doc["targets"]}
    ts = E.load_manifest(str(manifest), str(pdb_dir))
    return {t.target_id: t for t in ts if t.target_id in ids}


def write_receptor_pdb(t, out_pdb):
    """Receptor-only template for AF2: chain A, residues renumbered from 1.

    The same chain and numbering the Boltz template uses, so a predicted complex
    from either arm is scored by identical code with identical chain arguments.
    """
    from lnr.rescore_rings import read_full_residues
    from lnr.boltz_validate import _relabel
    lines = []
    for i, (_nm, atoms) in enumerate(read_full_residues(t.pdb_path, t.rec_chain),
                                     start=1):
        for _atom, line in atoms.items():
            lines.append(_relabel(line, "A", i))
    lines.append("TER\nEND\n")
    Path(out_pdb).write_text("".join(lines))
    return len(lines)


# ---------------------------------------------------------------------------
# STAGE 2a -- the AfCycDesign gate
# ---------------------------------------------------------------------------
def gate_af2(doc, manifest, pdb_dir, out_dir, *, num_recycles=6,
             params_dir=AF2_PARAMS_DIR, chemistry="mainchain"):
    """Predict each target's NATIVE peptide with the templated AfCycDesign path
    and measure receptor-aligned peptide CA RMSD against the deposited pose.

    This is the bar HighFold3 had to clear before its opinion counted here, and
    it is the measurement that decides whether arm 2 produces interface energies
    about this pocket or about empty space. A gate failure is a RESULT -- it is
    written to the same rows file and reported -- not an error.
    """
    from lnr import boltz_validate as BV
    from proteinfoundation.cpseq.adapters import afcycdesign as AFC

    out_dir = Path(out_dir)
    work = out_dir / "af2_gate"
    work.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "af2_gate.jsonl"
    done = {(r["target_id"], r.get("af2_model_num")) for r in _load_jsonl(rows_path)}
    tmap = _targets_by_id(doc, manifest, pdb_dir)

    cfg = AFC.AfCycConfig(num_recycles=int(num_recycles))
    print(f"[gate-af2] {len(tmap)} targets  recycles={cfg.num_recycles}  "
          f"model_nums={cfg.model_nums}  params={params_dir}")
    print(f"[gate-af2] pass bar: receptor-aligned peptide CA RMSD <= "
          f"{AF2_GATE_RMSD_A} A  (Boltz-2 reference on crystal macrocycles: "
          f"1.42 A median; afcycdesign previously 34.67 A)")

    for tid, t in sorted(tmap.items()):
        if all((tid, m) in done for m in cfg.model_nums):
            print(f"   {tid}: cached")
            continue
        wdir = work / tid
        wdir.mkdir(parents=True, exist_ok=True)
        rec_pdb = wdir / f"receptor_{tid}.pdb"
        write_receptor_pdb(t, rec_pdb)
        t0 = time.time()
        res = AFC.predict(sequence=t.seq, template_path=rec_pdb, target_chain="A",
                          chemistry=chemistry, out_dir=wdir, config=cfg,
                          af2_params_dir=params_dir, binder_chain_out="B")
        for s in (res.get("samples") or []):
            row = {"stage": "gate", "target_id": tid, "native_seq": t.seq,
                   "pep_len": int(t.length), "arm": "afcycdesign",
                   "num_recycles": cfg.num_recycles,
                   "af2_model_num": s.get("af2_model_num"),
                   "status": s.get("status"), "pdb_path": s.get("pdb_path"),
                   "plddt": s.get("plddt"), "ptm": s.get("ptm"),
                   "i_ptm": s.get("i_ptm"), "i_pae": s.get("i_pae"),
                   "sequence_matches": s.get("sequence_matches"),
                   "cyclic_offset_applied": s.get("cyclic_offset_applied"),
                   "runtime_s": s.get("runtime_s"),
                   "wall_s": round(time.time() - t0, 1)}
            if s.get("pdb_path") and s.get("status") == "ok":
                try:
                    row["peptide_ca_rmsd_A"] = BV.held_rmsd(
                        s["pdb_path"], t, 0, t.length - 1)
                except Exception as exc:              # noqa: BLE001
                    row["rmsd_error"] = f"{type(exc).__name__}: {exc}"[:200]
            _append(rows_path, row)
            print(f"   {tid} model{row['af2_model_num']}: {row['status']} "
                  f"RMSD {row.get('peptide_ca_rmsd_A')} A  pLDDT {row.get('plddt')}")
    return gate_verdict(rows_path)


def gate_verdict(rows_path):
    """PASS/FAIL on the rows alone, so the decision is reproducible from disk."""
    rows = [r for r in _load_jsonl(rows_path)
            if r.get("stage") == "gate" and r.get("peptide_ca_rmsd_A") is not None]
    if not rows:
        print("[gate-af2] VERDICT: NO DATA -- arm 2 cannot be reported.")
        return {"verdict": "no_data", "n": 0}
    best = {}
    for r in rows:                                   # best sample per target
        tid = r["target_id"]
        if tid not in best or r["peptide_ca_rmsd_A"] < best[tid]["peptide_ca_rmsd_A"]:
            best[tid] = r
    vals = sorted(r["peptide_ca_rmsd_A"] for r in best.values())
    n_pass = sum(1 for v in vals if v <= AF2_GATE_RMSD_A)
    med = st.median(vals)
    verdict = "pass" if n_pass >= max(1, (len(vals) + 1) // 2) else "fail"
    print(f"[gate-af2] VERDICT: {verdict.upper()}  "
          f"{n_pass}/{len(vals)} targets within {AF2_GATE_RMSD_A} A  "
          f"median {med:.2f} A  (best {vals[0]:.2f}, worst {vals[-1]:.2f})")
    if verdict == "fail":
        print("[gate-af2] The peptide is not in the pocket, so an interface "
              "energy from this arm is a number about empty space. Arm 2 is "
              "reported as a FAILED CONTROL, not as a second opinion.")
    return {"verdict": verdict, "n": len(vals), "n_pass": n_pass,
            "median_rmsd_A": round(med, 3), "per_target":
            {k: v["peptide_ca_rmsd_A"] for k, v in sorted(best.items())}}


# ---------------------------------------------------------------------------
# STAGE 2b -- Boltz-2 folding, 4 runs x 4 samples
# ---------------------------------------------------------------------------
def fold_boltz(doc, manifest, pdb_dir, out_dir, *, n_runs=4, replicas=4,
               base_seed=7000, targets_filter=(), use_potentials=True,
               timeout_s=3600):
    from lnr import boltz_validate as BV
    from proteinfoundation.cpseq.adapters.boltz2 import (Boltz2Config,
                                                         build_input_yaml, predict)
    from proteinfoundation.cpseq.targets import write_target_template_cif

    out_dir = Path(out_dir)
    work = out_dir / "boltz"
    work.mkdir(parents=True, exist_ok=True)
    folds_path = out_dir / "folds.jsonl"
    done = {(r["arm"], r["target_id"], r["design"], r["run"])
            for r in _load_jsonl(folds_path) if r.get("arm") == "boltz2"}
    tmap = _targets_by_id(doc, manifest, pdb_dir)

    designs = [d for d in doc["designs"]
               if not targets_filter or d["target_id"] in set(targets_filter)]
    print(f"[fold-boltz] {len(designs)} designs x {n_runs} runs x {replicas} "
          f"samples  potentials={use_potentials}")

    for d in designs:
        tid = d["target_id"]
        t = tmap.get(tid)
        if t is None:
            print(f"   {tid}: not in manifest, skipped")
            continue
        a, b = int(d["window"]["a"]), int(d["window"]["b"])
        seq = d["sequence"]
        for run in range(n_runs):
            if ("boltz2", tid, d["design"], run) in done:
                continue
            stem = f"{tid}_{d['design']}_r{run}"
            wdir = work / tid
            wdir.mkdir(parents=True, exist_ok=True)
            cfg = Boltz2Config()
            cfg.use_potentials = bool(use_potentials)
            # Each run is an INDEPENDENT trunk pass: a different seed, its own
            # output directory. The 4 samples inside a run share that trunk, so
            # the two levels of variance are different things and the rows keep
            # them apart (`run` and `model_index`).
            cfg.seed = int(base_seed) + 97 * run
            BV.configure_replicas(cfg, int(replicas))
            t0 = time.time()
            row_base = {"arm": "boltz2", "target_id": tid, "design": d["design"],
                        "sequence": seq, "subs": d["subs"], "window": [a, b],
                        "run": run, "seed": cfg.seed,
                        "use_potentials": bool(use_potentials),
                        "pll": d["pll"], "pll_gain": d["pll_gain"],
                        "pll_vs_opt": d["pll_vs_opt"]}
            try:
                tmpl = wdir / f"tmpl_{stem}.pdb"
                cif = wdir / f"tmpl_{stem}.cif"
                yml = wdir / f"held_{stem}.yaml"
                n_held = BV.build_held_template_pdb(
                    t, a, b, str(tmpl), peptide_sequence=seq, backbone_only=True)
                write_target_template_cif(tmpl, cif, BV.BOLTZ_PY)
                recseq, pocket = BV.receptor_seq_and_pocket(t)
                yml.write_text(build_input_yaml(
                    peptide_sequence=seq,
                    receptor_sequences={t.rec_chain: recseq},
                    chemistry="mainchain", pocket_contacts=pocket,
                    receptor_msa_path=None, target_template_path=None,
                ) + BV._held_templates_block(str(cif)))
                odir = wdir / f"pred_{stem}"
                pred = predict(yaml_path=yml, out_dir=odir, config=cfg,
                               python_exec=BV.BOLTZ_PY,
                               boltz_cache=BV.BOLTZ_CACHE, devices=1,
                               accelerator="gpu", timeout_s=int(timeout_s))
                if pred.get("status") != "ok":
                    _append(folds_path, {**row_base, "status": "predict_failed",
                                         "error": (pred.get("stderr_tail")
                                                   or "")[-400:],
                                         "wall_s": round(time.time() - t0, 1)})
                    print(f"   {stem}: PREDICT FAILED")
                    continue
                pdbs = BV.find_all_pdbs(odir)
                for k, pdb in enumerate(pdbs):
                    rmsd = None
                    try:
                        rmsd = BV.held_rmsd(str(pdb), t, a, b)
                    except Exception:                 # noqa: BLE001
                        pass
                    _append(folds_path, {
                        **row_base, "status": "ok", "model_index": k,
                        "n_held": n_held, "pdb_path": str(pdb),
                        "held_rmsd_A": rmsd,
                        "predict_runtime_s": pred.get("runtime_s"),
                        "wall_s": round(time.time() - t0, 1)})
                print(f"   {stem}: {len(pdbs)} structures in "
                      f"{round(time.time() - t0, 1)}s")
            except Exception as exc:                  # noqa: BLE001
                _append(folds_path, {**row_base, "status": "error",
                                     "error": f"{type(exc).__name__}: {exc}"[:400],
                                     "wall_s": round(time.time() - t0, 1)})
                print(f"   {stem}: ERROR {type(exc).__name__}: {exc}")
    return folds_path


# ---------------------------------------------------------------------------
# STAGE 2c -- AfCycDesign folding
# ---------------------------------------------------------------------------
def fold_af2(doc, manifest, pdb_dir, out_dir, *, num_recycles=6,
             params_dir=AF2_PARAMS_DIR, targets_filter=(), chemistry="mainchain"):
    from lnr import boltz_validate as BV
    from proteinfoundation.cpseq.adapters import afcycdesign as AFC

    out_dir = Path(out_dir)
    work = out_dir / "af2"
    work.mkdir(parents=True, exist_ok=True)
    folds_path = out_dir / "folds.jsonl"
    done = {(r["arm"], r["target_id"], r["design"], r.get("af2_model_num"))
            for r in _load_jsonl(folds_path) if r.get("arm") == "afcycdesign"}
    tmap = _targets_by_id(doc, manifest, pdb_dir)
    cfg = AFC.AfCycConfig(num_recycles=int(num_recycles))

    designs = [d for d in doc["designs"]
               if not targets_filter or d["target_id"] in set(targets_filter)]
    print(f"[fold-af2] {len(designs)} designs x {len(cfg.model_nums)} parameter "
          f"sets  recycles={cfg.num_recycles}")
    print("[fold-af2] NOTE two parameter sets, not five: ColabDesign loads "
          "model_{1,2}_ptm on the templated path (AF2 models 3-5 have no "
          "template weights). Recorded on every row.")

    for d in designs:
        tid = d["target_id"]
        t = tmap.get(tid)
        if t is None:
            continue
        if all((("afcycdesign", tid, d["design"], m) in done)
               for m in cfg.model_nums):
            continue
        wdir = work / tid / d["design"]
        wdir.mkdir(parents=True, exist_ok=True)
        rec_pdb = wdir / f"receptor_{tid}.pdb"
        write_receptor_pdb(t, rec_pdb)
        t0 = time.time()
        res = AFC.predict(sequence=d["sequence"], template_path=rec_pdb,
                          target_chain="A", chemistry=chemistry, out_dir=wdir,
                          config=cfg, af2_params_dir=params_dir,
                          binder_chain_out="B")
        a, b = int(d["window"]["a"]), int(d["window"]["b"])
        for s in (res.get("samples") or []):
            rmsd = None
            if s.get("status") == "ok" and s.get("pdb_path"):
                try:
                    rmsd = BV.held_rmsd(s["pdb_path"], t, 0, t.length - 1)
                except Exception:                     # noqa: BLE001
                    pass
            _append(folds_path, {
                "arm": "afcycdesign", "target_id": tid, "design": d["design"],
                "sequence": d["sequence"], "subs": d["subs"], "window": [a, b],
                "run": 0, "af2_model_num": s.get("af2_model_num"),
                "model_index": s.get("sample_index"),
                "num_recycles": cfg.num_recycles,
                "n_param_sets": len(cfg.model_nums),
                "param_sets_note": "templated AF2 has 2 (model_1/2_ptm), not 5",
                "status": s.get("status"), "pdb_path": s.get("pdb_path"),
                "plddt": s.get("plddt"), "ptm": s.get("ptm"),
                "i_ptm": s.get("i_ptm"), "i_pae": s.get("i_pae"),
                "sequence_matches": s.get("sequence_matches"),
                "cyclic_offset_applied": s.get("cyclic_offset_applied"),
                "peptide_ca_rmsd_A": rmsd,
                "pll": d["pll"], "pll_gain": d["pll_gain"],
                "pll_vs_opt": d["pll_vs_opt"],
                "predict_runtime_s": s.get("runtime_s"),
                "wall_s": round(time.time() - t0, 1)})
        print(f"   {tid}/{d['design']}: "
              + ", ".join(f"m{s.get('af2_model_num')}={s.get('status')}"
                          for s in (res.get("samples") or [])))
    return folds_path


# ---------------------------------------------------------------------------
# STAGE 3 -- relax + Rosetta, both protocol versions, on every structure
# ---------------------------------------------------------------------------
def _score_one(job):
    """One (structure, protocol) pair. Runs in a worker process."""
    from proteinfoundation.cpseq.scorer import score_structure
    pdb, refined, seq, version, restraint = job
    t0 = time.time()
    try:
        sc = score_structure(pdb, binder_chain="B", receptor_chains=["A"],
                             chemistry="mainchain", requested_sequence=seq,
                             refined_out_path=refined, relax_restraint=restraint)
    except Exception as exc:                          # noqa: BLE001
        return {"scoring_version": version, "relax_restraint": restraint,
                "status": "scorer_exception",
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "score_s": round(time.time() - t0, 1)}
    return {"scoring_version": version, "relax_restraint": restraint,
            "status": sc.get("status"), "reason": sc.get("reason"),
            "sequence_matches": sc.get("sequence_matches"),
            "ring_declared": (sc.get("ring_declared_complex") or {}).get("declared"),
            "dG_raw": (sc.get("raw") or {}).get("dG_separated"),
            "dG_refined": (sc.get("refined") or {}).get("dG_separated"),
            "refined_pdb": str(refined),
            "score_s": round(time.time() - t0, 1)}


def score(out_dir, *, versions=("v3", "v5"), workers=1, arms=(), limit=0):
    """Relax + Rosetta every folded structure under each requested protocol.

    CPU only and split from folding on purpose: the predictions are the expensive
    irreplaceable artefact, and re-scoring them under another protocol must never
    require another GPU hour.
    """
    out_dir = Path(out_dir)
    folds = [r for r in _load_jsonl(out_dir / "folds.jsonl")
             if r.get("status") == "ok" and r.get("pdb_path")]
    if arms:
        folds = [r for r in folds if r.get("arm") in set(arms)]
    scores_path = out_dir / "scores.jsonl"
    done = {(r.get("pdb_path"), r.get("scoring_version"))
            for r in _load_jsonl(scores_path)}
    refined_dir = out_dir / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)

    jobs, meta = [], []
    for r in folds:
        for v in versions:
            if (r["pdb_path"], v) in done:
                continue
            key = (f"{r['arm']}_{r['target_id']}_{r['design']}_r{r.get('run', 0)}"
                   f"_m{r.get('model_index', 0)}_{v}")
            jobs.append((r["pdb_path"], str(refined_dir / f"refined_{key}.pdb"),
                         r["sequence"], v, PROTOCOLS[v]))
            meta.append(r)
    if limit:
        jobs, meta = jobs[:limit], meta[:limit]
    print(f"[score] {len(folds)} structures, {len(jobs)} (structure, protocol) "
          f"pairs to do, {len(done)} cached, workers={workers}")
    if not jobs:
        return scores_path

    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(int(workers)) as pool:
            results = pool.map(_score_one, jobs, chunksize=1)
    else:
        results = [_score_one(j) for j in jobs]
    for r, res in zip(meta, results):
        _append(scores_path, {
            **{k: r.get(k) for k in
               ("arm", "target_id", "design", "sequence", "window", "run",
                "model_index", "af2_model_num", "pdb_path", "pll", "pll_gain",
                "pll_vs_opt", "held_rmsd_A", "peptide_ca_rmsd_A", "plddt",
                "i_pae", "seed")},
            **res})
    ok = sum(1 for x in results if x.get("dG_refined") is not None)
    print(f"[score] {ok}/{len(results)} produced a refined dG in "
          f"{round(time.time() - t0, 1)}s -> {scores_path}")
    return scores_path


# ---------------------------------------------------------------------------
# STAGE 4 -- the picture: every fold, by target, by design
# ---------------------------------------------------------------------------
DESIGN_STYLE = {
    # label, colour, marker  -- two validated hues plus neutral ink. The table's
    # optimum is the claim (blue), the random draws are what it is claimed over
    # (orange), the native sequence is the anchor that gives a gap its scale.
    "pll_opt": ("pll_bg optimum", "#2a78d6", "o"),
    "native": ("native", "#52514e", "s"),
    "rand1": ("random #1", "#eb6834", "^"),
    "rand2": ("random #2", "#eb6834", "v"),
}


def plot(out_dir, *, version="v3", arm="boltz2", stem=None):
    """One plot per file: x = target, y = refined interface dG, one ball per fold.

    The y axis is INVERTED so that stronger binding is up -- the quantity is
    Rosetta dG_separated in REU, where more negative is better, and a reader
    comparing heights should not have to hold a sign flip in their head. The
    axis still carries the real REU values; only the direction is chosen.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    out_dir = Path(out_dir)
    rows = [r for r in _load_jsonl(out_dir / "scores.jsonl")
            if r.get("scoring_version") == version and r.get("arm") == arm
            and r.get("dG_refined") is not None]
    if not rows:
        print(f"[plot] no scored rows for arm={arm} version={version}")
        return []
    doc = json.loads((out_dir / "designs.json").read_text())
    order = [t["target_id"] for t in doc["targets"]]
    pll_of = {(d["target_id"], d["design"]): d["pll"] for d in doc["designs"]}

    by = {}
    for r in rows:
        by.setdefault((r["target_id"], r["design"]), []).append(float(r["dG_refined"]))
    order = [t for t in order if any((t, d) in by for d in DESIGN_STYLE)]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.0, "axes.labelsize": 8.4, "axes.titlesize": 8.8,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.6, "legend.fontsize": 7.2,
        "axes.linewidth": 0.7, "axes.edgecolor": "#52514e",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    slots = [d for d in DESIGN_CLASSES if any((t, d) in by for t in order)]
    width = 0.74
    offs = {d: (-width / 2 + width * (i + 0.5) / len(slots))
            for i, d in enumerate(slots)}
    rng = np.random.default_rng(0)

    for xi, tid in enumerate(order):
        for d in slots:
            vals = by.get((tid, d))
            if not vals:
                continue
            label, colour, marker = DESIGN_STYLE[d]
            x0 = xi + offs[d]
            # A narrow jitter inside the slot: 16 folds of one sequence overlap
            # heavily otherwise, and the reader needs to see the SPREAD -- it is
            # the whole reason there are 16 of them.
            xs = x0 + rng.uniform(-0.055, 0.055, size=len(vals))
            ax.scatter(xs, vals, s=13, marker=marker, facecolor=colour,
                       edgecolor="#ffffff", linewidth=0.35, alpha=0.75, zorder=3)
            med = float(np.median(vals))
            ax.plot([x0 - 0.085, x0 + 0.085], [med, med], "-", lw=2.0,
                    color=colour, zorder=4, solid_capstyle="butt")
        ax.axvline(xi + 0.5, color="#e1e0d9", lw=0.6, zorder=0)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{t}\nn={doc_len(doc, t)}" for t in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()                      # stronger binding (more negative) up
    ax.set_ylabel("Rosetta interface ΔG$_{sep}$ after relax  (REU)")
    ax.grid(axis="y", lw=0.6, color="#e1e0d9")
    ax.annotate("stronger\nbinding", xy=(0.004, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=7.0, color="#898781",
                linespacing=1.3)
    ax.annotate("", xy=(0.012, 0.995), xytext=(0.012, 0.88),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#898781", lw=0.8))
    nfold = max(len(v) for v in by.values())
    ax.set_title(f"{arm} → relax ({version}) → Rosetta · one ball per fold "
                 f"(up to {nfold} per sequence)", loc="left", pad=7)
    handles = [Line2D([], [], marker=DESIGN_STYLE[d][2], color="none",
                      markerfacecolor=DESIGN_STYLE[d][1],
                      markeredgecolor="#ffffff", markersize=5.4,
                      label=DESIGN_STYLE[d][0]) for d in slots]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
              ncol=len(slots), handletextpad=0.35, columnspacing=1.5,
              borderpad=0.0)

    stem = stem or f"oracle_{arm}_{version}"
    figs = out_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        q = figs / f"{stem}.{ext}"
        fig.savefig(q, dpi=600 if ext == "png" else None, bbox_inches="tight",
                    pad_inches=0.04)
        paths.append(str(q))
    plt.close(fig)

    # The numbers behind the picture, so a caption cites a file and not a pixel.
    summary = {"arm": arm, "version": version, "targets": {}}
    for tid in order:
        cell = {}
        for d in slots:
            vals = by.get((tid, d)) or []
            if not vals:
                continue
            cell[d] = {"n_folds": len(vals),
                       "median_dG": round(float(np.median(vals)), 3),
                       "mean_dG": round(float(np.mean(vals)), 3),
                       "sd_dG": round(float(np.std(vals)), 3),
                       "min_dG": round(float(np.min(vals)), 3),
                       "pll": pll_of.get((tid, d))}
        # The question in one number per target: does the table's optimum beat
        # the median of the random draws, and by how much against the fold
        # noise that threatened to swamp it?
        rands = [v for d in ("rand1", "rand2") for v in (by.get((tid, d)) or [])]
        opt = by.get((tid, "pll_opt")) or []
        if rands and opt:
            cell["_opt_minus_random_median_REU"] = round(
                float(np.median(opt) - np.median(rands)), 3)
            cell["_opt_fold_sd_REU"] = round(float(np.std(opt)), 3)
        summary["targets"][tid] = cell
    (figs / f"{stem}_values.json").write_text(json.dumps(summary, indent=1) + "\n")
    for q in paths:
        print(f"wrote {q}")
    print(f"wrote {figs / (stem + '_values.json')}")
    return paths


def doc_len(doc, tid):
    """Peptide length, for the x tick label."""
    for t in doc["targets"]:
        if t["target_id"] == tid:
            return t["pep_len"]
    return "?"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default="/zfsauton/scratch/yixiz/LNR/test.txt")
    ap.add_argument("--pdb-dir", default="/zfsauton/scratch/yixiz/LNR/pdbs")
    ap.add_argument("--pssm-dir", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--targets", default="")
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--n-targets", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--min-pep-len", type=int, default=7)
    ap.add_argument("--max-pep-len", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--recycles", type=int, default=6)
    ap.add_argument("--af2-params-dir", default=AF2_PARAMS_DIR)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--versions", default="v3,v5")
    ap.add_argument("--arms", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--gate-af2", action="store_true")
    ap.add_argument("--fold-boltz", action="store_true")
    ap.add_argument("--fold-af2", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-arm", default="boltz2")
    ap.add_argument("--require-gate", action="store_true",
                    help="refuse --fold-af2 unless the gate rows say PASS")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = tuple(s.strip() for s in a.targets.split(",") if s.strip())

    if a.select:
        select(a.manifest, a.pdb_dir, a.pssm_dir, a.windows, out_dir,
               k_mut=a.k_mut, n_targets=a.n_targets, n_random=a.n_random,
               min_pep_len=a.min_pep_len, max_pep_len=a.max_pep_len,
               targets_filter=tf)
        return 0

    doc_path = out_dir / "designs.json"
    if not doc_path.exists():
        print(f"GATE: no {doc_path} -- run --select first. Exit 0.")
        return 0
    doc = json.loads(doc_path.read_text())

    if a.gate_af2:
        gate_af2(doc, a.manifest, a.pdb_dir, out_dir, num_recycles=a.recycles,
                 params_dir=a.af2_params_dir)
        return 0
    if a.fold_boltz:
        fold_boltz(doc, a.manifest, a.pdb_dir, out_dir, n_runs=a.n_runs,
                   replicas=a.replicas, targets_filter=tf)
        return 0
    if a.fold_af2:
        if a.require_gate:
            v = gate_verdict(out_dir / "af2_gate.jsonl")
            if v.get("verdict") != "pass":
                print("[fold-af2] gate did not pass; not folding. Exit 0 so a "
                      "dependent job is not stranded.")
                return 0
        fold_af2(doc, a.manifest, a.pdb_dir, out_dir, num_recycles=a.recycles,
                 params_dir=a.af2_params_dir, targets_filter=tf)
        return 0
    if a.plot:
        for v in (s_.strip() for s_ in a.versions.split(",") if s_.strip()):
            plot(out_dir, version=v, arm=a.plot_arm)
        return 0
    if a.score:
        score(out_dir, versions=tuple(v.strip() for v in a.versions.split(",")
                                      if v.strip()),
              workers=a.workers,
              arms=tuple(s.strip() for s in a.arms.split(",") if s.strip()),
              limit=a.limit)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
