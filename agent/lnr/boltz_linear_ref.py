"""Linear-peptide Boltz->Rosetta reference: the real-pipeline dG CEILING.

The arms' binding dG is Boltz(head-to-tail CYCLIC) -> Rosetta. The NATIVE row in
the scorecard is Rosetta on the DEPOSITED LINEAR coordinates -- a DIFFERENT
pipeline -- so native-vs-arm conflates the cyclisation loss with a Boltz-vs-
deposited scoring shift, and is not a fair ceiling.

This folds the WT peptide LINEAR through the SAME pipeline as the arms: same
Boltz-2, same receptor + pocket, the whole peptide templated at its native bound
pose, but the head-to-tail `cyclic: true` flag REMOVED (mainchain adds only that
one line -- see adapters/boltz2.py), and scored with a plain two-chain Rosetta
interface dG (no ring declaration). The ONLY difference from the nomut cyclic
fold is the closing bond, so:

    linear_ref - nomut_cyclic  = the pure cyclisation cost (real pipeline)
    arm - linear_ref           = how close a cyclic design gets to the linear ceiling

Caveat: the cyclic arms are scored by cpseq.score_structure (ring-aware); this
reference uses a plain InterfaceAnalyzer dG_separated (the same core metric,
without the cyclic ring declaration), since no "linear" chemistry exists. The
whole-peptide RMSD to native is reported so a fold that drifted off the pose can
be excluded, exactly as in the cyclic pipeline.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PC = "/zfsauton2/home/yixiz/Proteina-Complexa"
for _p in (f"{PC}/src", PC, "/zfsauton2/home/yixiz/lp2cp-agent/agent"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import env as E  # noqa: E402
# stable helpers (coordinate/template/discovery) -- read-only reuse, not edits
from lnr.boltz_validate import (  # noqa: E402
    BOLTZ_CACHE, BOLTZ_PY, build_held_template_pdb, find_pdb, held_rmsd,
    receptor_seq_and_pocket)


def _strip_cyclic(yaml_text: str) -> str:
    """Remove the single `cyclic: true` line -> a LINEAR peptide fold. mainchain
    adds nothing else cyclic (no bond block), so this is the whole change."""
    keep = [ln for ln in yaml_text.splitlines() if ln.strip() != "cyclic: true"]
    return "\n".join(keep) + "\n"


def _templates_block(cif: str) -> str:
    # self-contained (not importing the contended _held_templates_block): force
    # the WHOLE peptide to its native pose so this is the native-pose linear
    # binding -- the best-case the cyclisation is trying to preserve.
    return ("templates:\n"
            f"  - cif: {cif}\n"
            "    chain_id: [A, B]\n"
            "    force: true\n"
            "    threshold: 2.0\n")


def _score_linear(pdb, pep_chain, rec_chain, relax_repeats, q):
    """Child: relax the Boltz linear complex and take InterfaceAnalyzer
    dG_separated. No ring, no mutation -- a plain two-chain interface score, the
    same core metric as the cyclic dG_separated."""
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pyrosetta.init("-mute all -ignore_unrecognized_res "
                       "-ignore_zero_occupancy false")
        sf = pyrosetta.get_fa_scorefxn()
        pose = pyrosetta.pose_from_pdb(pdb)
        fr = FastRelax(relax_repeats)
        fr.set_scorefxn(sf)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)
        ia = InterfaceAnalyzerMover(f"{rec_chain}_{pep_chain}")
        ia.set_pack_separated(True)
        ia.set_compute_packstat(False)
        ia.apply(pose)
        q.put({"linear_dG": round(ia.get_interface_dG(), 3)})
    except Exception as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}"[:200]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", required=True, help="comma list or 'all'")
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--per-target-timeout", type=float, default=1800.0)
    ap.add_argument("--work", default="")
    args = ap.parse_args()

    from proteinfoundation.cpseq.adapters.boltz2 import (Boltz2Config,
                                                         build_input_yaml, predict)
    from proteinfoundation.cpseq.targets import write_target_template_cif

    d = Path(args.out_dir)
    work = Path(args.work) if args.work else d / "boltz_linref"
    work.mkdir(parents=True, exist_ok=True)
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}
    want = (sorted(targets) if args.targets == "all"
            else [x.strip() for x in args.targets.split(",") if x.strip()])

    out = d / "boltz_linear_ref.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["target_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[linref] {len(want)} targets, {len(done)} done", flush=True)

    cfg = Boltz2Config()
    cfg.use_potentials = True   # force-template steering of the native pose
    ctx = mp.get_context("spawn")
    with out.open("a") as fh:
        for tid in want:
            if tid in done or tid not in targets:
                continue
            t = targets[tid]
            rec = {"target_id": tid, "peptide_sequence": t.seq,
                   "native_dG_deposited": None}
            try:
                recseq, pocket = receptor_seq_and_pocket(t)
                tmpl = work / f"lin_{tid}.pdb"
                # whole peptide [0, L-1] templated at native, backbone only
                build_held_template_pdb(t, 0, t.length - 1, str(tmpl),
                                        backbone_only=True)
                cif = work / f"lin_{tid}.cif"
                write_target_template_cif(tmpl, cif, BOLTZ_PY)
                yaml_text = build_input_yaml(
                    peptide_sequence=t.seq,
                    receptor_sequences={t.rec_chain: recseq},
                    chemistry="mainchain", pocket_contacts=pocket,
                    receptor_msa_path=None, target_template_path=None,
                ) + _templates_block(str(cif))
                yaml_text = _strip_cyclic(yaml_text)          # <-- LINEAR
                yml = work / f"lin_{tid}.yaml"
                yml.write_text(yaml_text)
                odir = work / f"out_{tid}"
                t0 = time.time()
                st = predict(yaml_path=yml, out_dir=odir, config=cfg,
                             python_exec=BOLTZ_PY, boltz_cache=BOLTZ_CACHE,
                             devices=1, accelerator="gpu", timeout_s=3600)
                rec["predict_status"] = st.get("status") if isinstance(st, dict) else str(st)
                pdb = find_pdb(odir)
                if pdb is None:
                    rec["error"] = "no pdb"
                else:
                    rec["boltz_pdb"] = str(pdb)
                    rec["pep_rmsd"] = held_rmsd(pdb, t, 0, t.length - 1)
                    q = ctx.Queue()
                    p = ctx.Process(target=_score_linear,
                                    args=(str(pdb), "B", "A", args.relax_repeats, q))
                    p.start(); p.join(args.per_target_timeout)
                    if p.is_alive():
                        p.terminate(); p.join(); rec["error"] = "score timeout"
                    elif p.exitcode != 0:
                        rec["error"] = f"score_crash_{-p.exitcode}"
                    else:
                        try:
                            rec.update(q.get_nowait())
                        except Exception:
                            rec["error"] = "no score"
                rec["seconds"] = round(time.time() - t0, 1)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[linref] {tid:8s} linear_dG={rec.get('linear_dG')} "
                  f"pep_rmsd={rec.get('pep_rmsd')} {rec.get('error','')}", flush=True)
    print("[linref] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
