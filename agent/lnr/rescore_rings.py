"""Rung-3 rescore: FastRelax + InterfaceAnalyzer on the agent's CLOSED rings.

The LNR binding metric is rung-2: `retention x native_dG` -- the share of native
interface energy the HELD residues carry, with zero credit for repack/recovery.
This measures rung-3 on the SAME protocol that produced `native_dG`
(rosetta_interface.py): build a full-atom closed ring, FastRelax it (restrained
to start), score Rosetta InterfaceAnalyzer `dG_separated`. The gap between
rung-2 and rung-3 is exactly "how much of the apparent binding loss was just the
missing side-chain repack".

Full-atom build (no fold, no sequence change -- LNR keeps the native sequence):
  * solve the closed backbone for the committed window (env._solve_closed);
  * for each residue, rigid-transform its NATIVE atoms (backbone + side chain)
    from the native (N,CA,C) frame onto the closed (N,CA,C) frame -- a per-
    residue graft of the native rotamer onto the new backbone. Held residues keep
    native coords exactly (the solver leaves [a,b] fixed), so their transform is
    the identity; released residues carry their side chain to the closed pose.
  * combine with the native receptor (unchanged) into one PDB.
Clashes introduced by the graft are what FastRelax is there to resolve -- which
is the whole point of the measurement.

CAVEAT: the closing peptide bond is NOT declared to Rosetta here; the ring is
scored as a coordinate-restrained linear peptide (termini held near the closed
position by `constrain_relax_to_start_coords`). The interface dG is dominated by
the peptide-receptor contacts, not the closing bond, so this is a sound first
estimate; a chemistry-correct pass (cpseq.scorer.score_structure, which declares
the mainchain bond) is the follow-up.

Subprocess-isolated per target (a PyRosetta segfault is uncatchable in-process).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}


def _set_xyz(line: str, xyz) -> str:
    return f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"


def read_full_residues(path: str, chain: str):
    """Residues of `chain` with all of N/CA/C, in file order (matches env indexing),
    each as (resname, {atom_name: pdb_line}). No AA filter here, so the index set
    equals env.load_target's."""
    res: dict = {}
    order: list = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        atom = line[12:16].strip()
        key = (line[22:26].strip(), line[26])
        if key not in res:
            res[key] = {"name": line[17:20].strip(), "lines": {}}
            order.append(key)
        # first altloc wins (blank or A)
        res[key]["lines"].setdefault(atom, line)
    out = []
    for key in order:
        r = res[key]
        if all(a in r["lines"] for a in ("N", "CA", "C")):
            out.append((r["name"], r["lines"]))
    return out


def read_receptor_lines(path: str, chain: str):
    """Standard-AA ATOM lines of the receptor (altloc blank/A), coords unchanged."""
    out = []
    for line in open(path):
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        if line[16] not in (" ", "A") or line[17:20].strip() not in AA3:
            continue
        out.append(f"{line[:16]} {line[17:]}" if line[16] != " " else line)
    return out


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform mapping rows of P onto rows of Q: x' = R@(x-cP)+cQ."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cP, cQ


def build_ring_pdb(t, a, b, Nn, CAn, Cn, out_pdb: str) -> int:
    """Graft native side chains onto the closed backbone and write ring+receptor PDB.
    Returns number of peptide residues written."""
    pep = read_full_residues(t.pdb_path, t.pep_chain)
    if len(pep) != t.length:
        raise RuntimeError(f"peptide residue count {len(pep)} != env length {t.length}")
    # alignment sanity: residue 0 N must match env.N[0]
    n0 = np.array([float(pep[0][1]["N"][30:38]), float(pep[0][1]["N"][38:46]),
                   float(pep[0][1]["N"][46:54])])
    if np.linalg.norm(n0 - t.N[0]) > 1e-2:
        raise RuntimeError("index misalignment between full-atom read and env backbone")

    pep_lines = []
    ri = 0
    for i, (resname, atoms) in enumerate(pep):
        P = np.array([t.N[i], t.CA[i], t.C[i]])
        Q = np.array([Nn[i], CAn[i], Cn[i]])
        R, cP, cQ = kabsch(P, Q)
        if resname not in AA3:        # skip unscoreable residue, keep ring contiguous
            continue
        ri += 1
        for atom, line in atoms.items():
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            new = R @ (xyz - cP) + cQ
            clean = f"{line[:16]} {line[17:]}" if line[16] != " " else line
            pep_lines.append(_set_xyz(clean, new))

    rec_lines = read_receptor_lines(t.pdb_path, t.rec_chain)
    with open(out_pdb, "w") as fo:
        fo.writelines(rec_lines)
        fo.write("TER\n")
        fo.writelines(pep_lines)
        fo.write("TER\nEND\n")
    return ri


def _score_one(pdb_path, pep_chain, rec_chain, a, b, relax_repeats, q):
    """Score the ring under four protocols that differ ONLY in what may move, and
    report how far the HELD window drifts under each -- because the held window IS
    the LLM's commitment, and a protocol that moves it is no longer scoring that
    ring:

      dG_raw        nothing moves (grafted rotamers, clashes and all) -- reference.
      dG_repack     side chains repacked, ALL backbone fixed. Held window stays
                    exactly where the LLM put it. This is "score, don't relax".
      dG_heldfixed  FastRelax with the held window's backbone FIXED; released
                    backbone + receptor + all side chains free. Lets the released
                    loop settle WITHOUT breaking the hold.
      dG_fullrelax  ordinary FastRelax -- the held window is free to drift. Kept
                    only to measure how much binding is "recovered" by silently
                    re-docking the residues the LLM committed to holding.
    """
    try:
        import numpy as np
        import pyrosetta
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            InitializeFromCommandline, RestrictToRepacking)
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pyrosetta.init("-mute all -ignore_unrecognized_res "
                       "-ignore_zero_occupancy false")
        sf = pyrosetta.get_fa_scorefxn()
        pose0 = pyrosetta.pose_from_pdb(pdb_path)
        chains = f"{rec_chain}_{pep_chain}"
        info = pose0.pdb_info()
        pep_pose = [i for i in range(1, pose0.total_residue() + 1)
                    if info.chain(i) == pep_chain]
        # env window index k == k-th peptide residue in pose order (PDB kept env order)
        held = pep_pose[a:b + 1] if b < len(pep_pose) else []

        def held_ca(p):
            return np.array([[p.residue(i).xyz("CA").x, p.residue(i).xyz("CA").y,
                              p.residue(i).xyz("CA").z] for i in held])
        ref = held_ca(pose0) if held else None

        def rmsd(p):
            if not held:
                return None
            return round(float(np.sqrt(((held_ca(p) - ref) ** 2).sum(axis=1).mean())), 3)

        def idg(p):
            ia = InterfaceAnalyzerMover(chains)
            ia.set_pack_separated(True)
            ia.set_compute_packstat(False)
            ia.apply(p)
            return round(ia.get_interface_dG(), 3)

        out = {"n_held": len(held)}
        out["dG_raw"] = idg(pose0.clone())

        # repack only -- all backbone fixed
        tf = TaskFactory()
        tf.push_back(InitializeFromCommandline())
        tf.push_back(RestrictToRepacking())
        pr = pose0.clone()
        PackRotamersMover(sf, tf.create_task_and_apply_taskoperations(pr)).apply(pr)
        out["dG_repack"] = idg(pr)
        out["held_rmsd_repack"] = rmsd(pr)

        # held-fixed relax -- held window bb frozen, everything else free
        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        for i in held:
            mm.set_bb(i, False)
        hf = pose0.clone()
        fr = FastRelax(sf, relax_repeats)
        fr.set_movemap(mm)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(hf)
        out["dG_heldfixed"] = idg(hf)
        out["held_rmsd_heldfixed"] = rmsd(hf)

        # full relax -- hold NOT enforced (contrast)
        full = pose0.clone()
        fr2 = FastRelax(sf, relax_repeats)
        fr2.constrain_relax_to_start_coords(True)
        fr2.apply(full)
        out["dG_fullrelax"] = idg(full)
        out["held_rmsd_fullrelax"] = rmsd(full)
        q.put(out)
    except Exception as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}"[:200]})


def best_clean_window(row, clash):
    best = (-1.0, None)
    for h in row.get("history", []):
        if (h.get("closes")
                and clash.get((row["target_id"], h["a"], h["b"]), False)
                and h["retention"] > best[0]):
            best = (h["retention"], (h["a"], h["b"]))
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default="5l9b,1d4t,4w50,5njx")
    ap.add_argument("--arm", default="gpt-6-luna|llm-mask")
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--per-target-timeout", type=float, default=1200.0)
    args = ap.parse_args()

    _HERE = Path(__file__).resolve().parent
    if str(_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_HERE.parent))
    from lnr import env as E

    d = Path(args.out_dir)
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}

    def load(p):
        p = Path(p)
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    clash = {(c["target_id"], c["a"], c["b"]): c.get("clash_free", False)
             for c in load(d / "clashes.jsonl")}
    bind = {}
    for bbb in load(d / "rosetta_interface.jsonl"):
        if "per_residue_ddG_ala" in bbb:
            pr = {int(k): (v or 0.0) for k, v in bbb["per_residue_ddG_ala"].items()}
            tot = bbb.get("total_positive_ddG") or 1.0
            bind[bbb["target_id"]] = (pr, tot, bbb.get("native_dG"))

    model, armname = args.arm.split("|")
    rows = {}
    for f in sorted(d.glob("llm_*.jsonl")):
        for r in load(f):
            if r["arm"] == armname and r.get("model_id") == model:
                rows[r["target_id"]] = r

    def rung2(tid, w):
        if not w or tid not in bind:
            return None
        pr, tot, ndg = bind[tid]
        ret = sum(pr[k] for k in range(w[0], w[1] + 1) if k in pr and pr[k] > 0) / tot
        return round(ret * ndg, 3)

    out = d / "rescore_rings.jsonl"
    ctx = mp.get_context("spawn")
    results = []
    with out.open("w") as fh:
        for tid in [x.strip() for x in args.targets.split(",") if x.strip()]:
            t0 = time.time()
            rec = {"target_id": tid, "arm": args.arm}
            if tid not in targets or tid not in rows:
                rec["error"] = "missing target or arm row"
                fh.write(json.dumps(rec) + "\n"); results.append(rec); continue
            t = targets[tid]
            w = best_clean_window(rows[tid], clash)
            rec["window"] = list(w) if w else None
            rec["native_dG"] = bind.get(tid, (None, None, None))[2]
            rec["rung2_proxy_dG"] = rung2(tid, w)
            if not w:
                rec["error"] = "no clean closer"
                fh.write(json.dumps(rec) + "\n"); results.append(rec); continue
            ok, resid, Nn, CAn, Cn = E._solve_closed(t, w[0], w[1], n_restarts=12,
                                                     seed=0, max_seconds=15.0)
            if not ok:
                rec["error"] = f"re-solve failed (resid={resid:.2f})"
                fh.write(json.dumps(rec) + "\n"); results.append(rec); continue
            pdb = str(d / f"ring_{tid}.pdb")
            try:
                nres = build_ring_pdb(t, w[0], w[1], Nn, CAn, Cn, pdb)
                rec["ring_residues"] = nres
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"build: {type(exc).__name__}: {exc}"[:150]
                fh.write(json.dumps(rec) + "\n"); results.append(rec); continue

            q = ctx.Queue()
            p = ctx.Process(target=_score_one,
                            args=(pdb, t.pep_chain, t.rec_chain, w[0], w[1],
                                  args.relax_repeats, q))
            p.start(); p.join(args.per_target_timeout)
            if p.is_alive():
                p.terminate(); p.join(); rec["error"] = "timeout"
            elif p.exitcode != 0:
                rec["error"] = f"crashed_signal_{-p.exitcode}"
            else:
                try:
                    rec.update(q.get_nowait())
                except Exception:
                    rec["error"] = "no_result"
            rec["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            results.append(rec)
            print(f"[rescore] {tid:6s} win={rec.get('window')} "
                  f"native={rec.get('native_dG')} rung2={rec.get('rung2_proxy_dG')} "
                  f"repack={rec.get('dG_repack')} heldfixed={rec.get('dG_heldfixed')} "
                  f"fullrelax={rec.get('dG_fullrelax')} "
                  f"heldRMSD(repack/heldfix/full)="
                  f"{rec.get('held_rmsd_repack')}/{rec.get('held_rmsd_heldfixed')}/"
                  f"{rec.get('held_rmsd_fullrelax')} {rec.get('error','')}", flush=True)

    # summary: dG under each protocol + how far the held window moved under each
    print("\n==== committed-ring binding dG by protocol (REU), + held-window drift ====")
    print("rung2 = retention x native_dG (proxy) · repack = side chains only, bb FIXED")
    print("heldfixed = relax w/ held bb frozen · fullrelax = ordinary relax (hold lost)\n")
    print(f"{'target':7s} {'win':8s} {'native':>7s} {'rung2':>7s} {'raw':>7s} "
          f"{'repack':>7s} {'heldfix':>7s} {'full':>7s} | held-CA RMSD Å "
          f"{'rpk':>5s} {'hfx':>5s} {'full':>5s}")
    for r in results:
        if r.get("error"):
            print(f"{r['target_id']:7s} {str(r.get('window','')):8s} "
                  f"ERROR: {r['error']}")
            continue
        print(f"{r['target_id']:7s} {str(r.get('window','')):8s} "
              f"{str(r.get('native_dG','')):>7s} {str(r.get('rung2_proxy_dG','')):>7s} "
              f"{str(r.get('dG_raw','')):>7s} {str(r.get('dG_repack','')):>7s} "
              f"{str(r.get('dG_heldfixed','')):>7s} {str(r.get('dG_fullrelax','')):>7s} | "
              f"{'':12s}{str(r.get('held_rmsd_repack','')):>5s} "
              f"{str(r.get('held_rmsd_heldfixed','')):>5s} "
              f"{str(r.get('held_rmsd_fullrelax','')):>5s}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
