#!/usr/bin/env python3
"""Does the DECLARED RESPONSE MODEL agree with real Rosetta on these structures?

Everything this project has measured so far is a controller comparison inside
`lp2cp/envbuild/surrogate.py`: the parents' geometry is measured from deposited
coordinates, but how an objective RESPONDS to an edit is a closed-form model.
That instrument was the right one for the controller question -- it is the only
way to know the exact optimum of the action space, which is what makes regret
computable -- and it licenses nothing about biophysics.

This script asks the separate question the surrogate cannot answer: of the
specific claims the Stage 3 result rests on, which ones does a real energy
function agree with? It runs PyRosetta on the parents' own PDB files, which are
on disk, and writes one JSONL row per parent so the comparison is redrawable.

Three claims are tested, in increasing order of how much they are MODELLED
rather than measured:

  1. LEVEL (mostly measured, expected to agree). Surrogate `bsa` against
     Rosetta `dSASA_int`. The surrogate's BSA is side-chain area buried on the
     peptide side; Rosetta's dSASA_int counts both sides, so the expected
     relationship is dSASA_int ~ 2 x bsa. A disagreement here would mean the
     measured inputs are wrong, which would be far more serious than a wrong
     response model.

  2. THE SYMPTOM RANKING (modelled). `interface_mass` -- contacts, polar pairs
     and buried area combined into a per-residue ddG contribution -- is what
     `rim_first_closure` and `greedy_interface` rank positions by, so the bar
     itself depends on it. Tested against a real per-residue alanine scan:
     mutate each peptide position to ALA, repack the 8 A neighbourhood,
     re-score the interface. Spearman between the surrogate's ranking and
     Rosetta's.

  3. THE LEVER CLAIM (most modelled, and the one Stage 3 rests on). The v2
     surrogate asserts that a redesign at a PARTIALLY EXPOSED position 3-6.5 A
     from the partner GAINS interface, while the same edit at a buried
     high-contact hotspot can only lose it -- `interface_headroom` versus
     `interface_mass`. That asymmetry is why `repack_headroom` and
     `substitute_headroom` exist, so if Rosetta does not reproduce it, the
     Stage 3 tools are exploiting a modelling artifact. Tested by actually
     substituting a larger side chain (TRP/PHE/LEU/ARG) at the top headroom
     positions AND at the top interface-carrying positions, and comparing the
     real change in dG_separated and dSASA_int between the two groups.

What it deliberately does NOT do: no backbone rebuild, no ring closure, no
folding. REBUILD_SEGMENT and NARROW_PIVOTS need GenKIC and a cyclic-peptide
structure predictor, and `edit_cost(REBUILD_SEGMENT, L) = ceil(L/2)` is a
declared budget model rather than a physical quantity, so neither is checkable
this way. Those remain unvalidated and must be reported as such.

Relaxation. Unrelaxed, these deposited structures score dG_separated ~ +222 in
ref2015 -- clashes dominate and the number is meaningless. The sibling
project's own calibration shows the same thing (norelax +42 against relaxed
-36), so every pose here is relaxed with coordinate constraints to the starting
coordinates first. Constraints are on because the question is about the
DEPOSITED geometry, not about where a free relax wanders to.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, average ranks for ties. No scipy dependency."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", required=True, help="parents_full.json")
    ap.add_argument("--pdb-root", required=True, help="CPSea processed/ root")
    ap.add_argument("--out", required=True, help="output jsonl (resumable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--candidates", default="W,F,L,R")
    ap.add_argument("--n-probe", type=int, default=2,
                    help="positions per group (headroom / interface-carrying)")
    ap.add_argument("--skip-relax", action="store_true",
                    help="diagnostic only -- scores are meaningless unrelaxed")
    args = ap.parse_args()

    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.toolbox import mutate_residue

    pyrosetta.init("-mute all -ignore_unrecognized_res "
                   "-ignore_zero_occupancy false")
    sf = pyrosetta.get_fa_scorefxn()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lp2cp import tools as T
    from lp2cp.envbuild import surrogate

    parents = json.loads(Path(args.parents).read_text())
    if args.limit:
        parents = parents[: args.limit]

    out = Path(args.out)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["parent_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[val] {len(parents)} parents, {len(done)} already done", flush=True)

    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    root = Path(args.pdb_root)

    def interface(pose, chains: str) -> tuple[float, float]:
        ia = InterfaceAnalyzerMover(chains)
        ia.set_pack_separated(True)
        ia.set_compute_packstat(False)
        ia.apply(pose)
        return ia.get_interface_dG(), ia.get_interface_delta_sasa()

    with out.open("a") as fh:
        for p in parents:
            pid = p["parent_id"]
            if pid in done:
                continue
            t_all = time.time()
            hits = list(root.glob(f"*/{pid}.pdb"))
            if not hits:
                print(f"[val] {pid}: no PDB found, skipping", flush=True)
                continue
            pose = pyrosetta.pose_from_pdb(str(hits[0]))
            pdbi = pose.pdb_info()
            chain_of: dict[str, list[int]] = {}
            for i in range(1, pose.total_residue() + 1):
                chain_of.setdefault(pdbi.chain(i), []).append(i)
            n_pos = len(p["residues"])
            pep = [c for c, v in chain_of.items() if len(v) == n_pos]
            if not pep:
                print(f"[val] {pid}: no chain of length {n_pos} "
                      f"(have {[(c, len(v)) for c, v in chain_of.items()]})",
                      flush=True)
                continue
            pep_chain = pep[0]
            rec_chain = [c for c in chain_of if c != pep_chain][0]
            chains = f"{rec_chain}_{pep_chain}"
            # peptide position (1-based in the parent record) -> pose index
            pose_of = {k + 1: r for k, r in enumerate(chain_of[pep_chain])}

            dg_raw, sasa_raw = interface(pose, chains)

            t_rel = time.time()
            if not args.skip_relax:
                fr = FastRelax(args.relax_repeats)
                fr.set_scorefxn(sf)
                fr.constrain_relax_to_start_coords(True)
                fr.apply(pose)
            t_rel = time.time() - t_rel
            dg0, sasa0 = interface(pose, chains)

            # Real per-residue backbone strain. The surrogate's `strain_rel` is
            # a Ramachandran-deviation proxy; the Rosetta equivalent is the
            # backbone-torsion part of ref2015 at that residue. Needed here
            # because the real environment's strain threshold has to be set
            # from a measured distribution rather than carried over from the
            # surrogate, where it was explicitly TUNED (stage2_notes section 5).
            sf(pose)
            st = pyrosetta.rosetta.core.scoring.ScoreType
            bb_terms = []
            for nm in ("rama_prepro", "omega", "p_aa_pp", "fa_dun"):
                try:
                    bb_terms.append(getattr(st, nm))
                except AttributeError:
                    continue
            emap = pose.energies()
            strain = {}
            for pos, ri in pose_of.items():
                e = emap.residue_total_energies(ri)
                strain[pos] = round(sum(float(e[t]) for t in bb_terms), 3)

            res = {r["pos"]: r for r in p["residues"]}

            class _R:
                """Minimal duck type for tools.headroom / interface_mass."""

                def __init__(self, d):
                    self.min_dist = d["min_dist"]
                    self.buried_frac = d["buried_frac"]
                    self.contacts_5a = d["contacts_5a"]
                    self.polar_3p5a = d["polar_3p5a"]
                    self.dsasa = d["dsasa"]

            rr = {k: _R(v) for k, v in res.items()}
            hr = {k: T.headroom(v) for k, v in rr.items()}
            im = {k: surrogate.interface_mass(v) for k, v in rr.items()}

            # --- claim 2: real alanine scan vs the surrogate's symptom ranking
            ala = {}
            for pos, ri in pose_of.items():
                if pose.residue(ri).name1() == "A":
                    continue
                q = pose.clone()
                mutate_residue(q, ri, "A", pack_radius=8.0, pack_scorefxn=sf)
                dg, sasa = interface(q, chains)
                ala[pos] = {"ddG_on_ala": dg - dg0, "dsasa_after": sasa}
            common = sorted(ala)
            rho_mass = spearman(
                [-im[k] for k in common],                     # more carried
                [ala[k]["ddG_on_ala"] for k in common],       # worse on Ala
            )
            rho_cont = spearman(
                [float(res[k]["contacts_5a"]) for k in common],
                [ala[k]["ddG_on_ala"] for k in common],
            )

            # --- claim 3: the lever claim, in real energy ---------------------
            top_hr = [k for k in sorted(hr, key=lambda k: -hr[k])
                      if hr[k] > 0.05][: args.n_probe]
            top_im = [k for k in sorted(im, key=lambda k: im[k])][: args.n_probe]
            probe = {}
            for group, positions in (("headroom", top_hr), ("carrying", top_im)):
                rows = []
                for pos in positions:
                    best = None
                    for aa in cands:
                        q = pose.clone()
                        mutate_residue(q, pose_of[pos], aa, pack_radius=8.0,
                                       pack_scorefxn=sf)
                        dg, sasa = interface(q, chains)
                        row = {"aa": aa, "d_dG": dg - dg0,
                               "d_dsasa": sasa - sasa0}
                        if best is None or row["d_dG"] < best["d_dG"]:
                            best = row
                    rows.append({
                        "pos": pos, "wt": res[pos]["wt"],
                        "burf": res[pos]["buried_frac"],
                        "mind": res[pos]["min_dist"],
                        "cont5A": res[pos]["contacts_5a"],
                        "headroom": round(hr[pos], 3),
                        "interface_mass": round(im[pos], 3),
                        "best": best,
                    })
                probe[group] = rows

            row = {
                "parent_id": pid,
                "pdb": str(hits[0]),
                "chains": chains,
                "n_pos": n_pos,
                "relax_seconds": round(t_rel, 1),
                "seconds": round(time.time() - t_all, 1),
                "rosetta": {
                    "dG_separated_norelax": round(dg_raw, 2),
                    "dSASA_int_norelax": round(sasa_raw, 1),
                    "dG_separated": round(dg0, 2),
                    "dSASA_int": round(sasa0, 1),
                },
                "surrogate": {
                    "ddg_bind": p["scorecard0"]["ddg_bind"],
                    "bsa": p["scorecard0"]["bsa"],
                    "native_bsa": p["native_bsa"],
                },
                "real_strain_per_pos": strain,
                "surrogate_strain_per_pos": {
                    r["pos"]: r["strain_rel"] for r in p["residues"]},
                "rho_strain": spearman(
                    [strain[k] for k in sorted(strain)],
                    [res[k]["strain_rel"] for k in sorted(strain)]),
                "alanine_scan": ala,
                "rho_interface_mass_vs_real_ala": rho_mass,
                "rho_contacts_vs_real_ala": rho_cont,
                "lever_probe": probe,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[val] {pid:30s} dG {dg0:8.2f}  dSASA {sasa0:7.1f}  "
                  f"rho(mass,ala)={rho_mass if rho_mass is None else round(rho_mass,2)}  "
                  f"{row['seconds']:.0f}s (relax {row['relax_seconds']:.0f}s)",
                  flush=True)
    print("[val] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
