"""Real Rosetta interface energy per LNR target, and per-residue binding share.

Binding quality the field's way: Rosetta interface dG_separated (the
RFpeptides/AfCycDesign acceptance metric, the sibling project's native
calibration), plus a per-residue alanine scan so a held window's binding
quality can be scored as the fraction of native interface ENERGY its residues
carry -- energy-weighted retention against the native complex.

Run as a VALIDATION SUBSET by default: the pass is expensive (a full-protein
FastRelax is ~1-2 min, the alanine scan is L more InterfaceAnalyzer calls) and
the point is to certify that the cheap CA-contact retention tracks real
binding-energy retention, then report contact retention for all targets with
that correlation attached. `--limit` bounds the subset.

Two robustness fixes, both learned from a segfault on 1bjr:
  * CLEAN POSE. The deposited PDBs carry non-protein residues (ligands, ions,
    waters); InterfaceAnalyzer's separated-pack segfaults on them even after
    relax. Each target is reduced to the two PROTEIN chains before scoring.
  * SUBPROCESS ISOLATION. A PyRosetta segfault cannot be caught in-process, so
    each target runs in its own child process with a wall-clock timeout; a
    crashed or timed-out target is recorded as failed and the batch continues.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path


def _score_one(path, pep_chain, rec_chain, relax_repeats, q):
    """Child process: clean -> relax -> native dG -> per-residue alanine scan.
    Result (or an 'error' dict) is put on the queue. A segfault here never
    reaches the parent except as a nonzero exitcode, which the parent handles.
    """
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.toolbox import mutate_residue

        pyrosetta.init("-mute all -ignore_unrecognized_res "
                       "-ignore_zero_occupancy false")
        sf = pyrosetta.get_fa_scorefxn()

        # Filter to a clean two-chain PDB on disk and let pose_from_pdb handle
        # connectivity/termini/disulfides. Manual residue-by-residue pose
        # surgery broke disulfide detection and segfaulted; a filtered PDB
        # does not. Keep only ATOM records (drop HETATM ligands/ions/waters)
        # of the two chains, standard amino acids only.
        import os
        import tempfile
        AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
               "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}
        lines = []
        for ln in open(path):
            if ln.startswith("ATOM") and ln[21] in (pep_chain, rec_chain) \
                    and ln[17:20].strip() in AA3 and ln[16] in (" ", "A"):
                lines.append(ln)
            elif ln.startswith("TER"):
                lines.append(ln)
        fd, tmp = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, "w") as fo:
            fo.writelines(lines)
            fo.write("END\n")
        clean = pyrosetta.pose_from_pdb(tmp)
        os.unlink(tmp)
        cpdbi = clean.pdb_info()
        pep_pose = [i for i in range(1, clean.total_residue() + 1)
                    if cpdbi.chain(i) == pep_chain]
        chains = f"{rec_chain}_{pep_chain}"

        fr = FastRelax(relax_repeats)
        fr.set_scorefxn(sf)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(clean)

        def idg(pose):
            ia = InterfaceAnalyzerMover(chains)
            ia.set_pack_separated(True)
            ia.set_compute_packstat(False)
            ia.apply(pose)
            return ia.get_interface_dG()

        dg0 = idg(clean)
        per_res = {}
        for local, ri in enumerate(pep_pose):
            if clean.residue(ri).name1() == "A":
                per_res[local] = 0.0
                continue
            m = clean.clone()
            mutate_residue(m, ri, "A", pack_radius=8.0, pack_scorefxn=sf)
            per_res[local] = round(idg(m) - dg0, 3)
        q.put({"native_dG": round(dg0, 3),
               "per_residue_ddG_ala": per_res})
    except Exception as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}"[:200]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--per-target-timeout", type=float, default=900.0)
    args = ap.parse_args()

    _HERE = Path(__file__).resolve().parent
    if str(_HERE.parent) not in sys.path:
        sys.path.insert(0, str(_HERE.parent))
    from lnr import env as E

    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}
    ids = [r.split()[0] for r in Path(args.manifest).read_text().splitlines()
           if r.split() and r.split()[0] in targets]
    if args.limit:
        ids = ids[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["target_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[rosif] {len(ids)} targets (subset), {len(done)} done", flush=True)

    ctx = mp.get_context("spawn")
    with out.open("a") as fh:
        for tid in ids:
            if tid in done:
                continue
            t = targets[tid]
            t0 = time.time()
            q = ctx.Queue()
            p = ctx.Process(target=_score_one,
                            args=(t.pdb_path, t.pep_chain, t.rec_chain,
                                  args.relax_repeats, q))
            p.start()
            p.join(args.per_target_timeout)
            row = {"target_id": tid, "pep_chain": t.pep_chain,
                   "rec_chain": t.rec_chain, "length": t.length,
                   "per_res_contacts": list(t.per_res_contacts)}
            if p.is_alive():
                p.terminate(); p.join()
                row["error"] = "timeout"
            elif p.exitcode != 0:
                row["error"] = f"crashed_signal_{-p.exitcode}"
            else:
                try:
                    res = q.get_nowait()
                except Exception:
                    res = {"error": "no_result"}
                row.update(res)
            if "per_residue_ddG_ala" in row:
                pr = row["per_residue_ddG_ala"]
                row["total_positive_ddG"] = round(
                    sum(v for v in pr.values() if v and v > 0) or 1.0, 3)
            row["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            tag = row.get("error", f"nat_dG={row.get('native_dG')}")
            print(f"[rosif] {tid:8s} {tag}  {row['seconds']:.0f}s", flush=True)
    print("[rosif] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
