"""Per-position structural diagnostics for every candidate in the Stage 0 set.

This is the evidence the agent reads. It must contain nothing derived from a
ddG measurement -- the whole question is whether structure alone is enough to
decide where to spend an edit.

Indexed by ORIGINAL PDB numbering. SKEMPI ships two numbering schemes and they
disagree: on a 129-site check across 1JTG/1A22/3HFM, `Mutation(s)_PDB` matched
the RCSB coordinates 129/129 while `Mutation(s)_cleaned` matched 66/129.
Indexing by the cleaned field annotates the wrong residue roughly half the time
and nothing downstream would notice.

Caveats recorded rather than hidden:
  - Waters and non-standard residues are stripped before SASA. A cofactor
    sitting in the interface therefore does not bury its neighbours, so dSASA
    for those positions is an underestimate.
  - The asymmetric unit is used, not the biological assembly. SKEMPI's chain
    pairs name the two sides explicitly, so the relevant contact is present,
    but a crystallographic neighbour that would bury a residue in vivo is not.
"""

from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, MMCIFParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

try:  # Biopython moved this symbol around across versions
    from Bio.PDB.Polypeptide import protein_letters_3to1 as _T31
except ImportError:  # pragma: no cover
    from Bio.Data.IUPACData import protein_letters_3to1_extended as _T31

# Tien et al. 2013, theoretical maximum residue SASA (A^2), used to turn an
# absolute SASA into a comparable relative burial.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0,
    "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0,
    "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0,
    "Y": 263.0, "V": 174.0,
}
STANDARD = set(MAX_ASA)

HYDROPHOBIC = set("AVLIMFWCP")
AROMATIC = set("FWYH")
CHARGED = set("DEKR")
POLAR = set("STNQHY")

CONTACT_CUTOFF = 5.0      # heavy-atom contact across the interface
HBOND_CUTOFF = 3.5        # N/O pair distance, a geometry-free H-bond proxy


def _res_key(res) -> str | None:
    """'<chain>:<resnum><icode>' in original PDB numbering, or None."""
    het, num, icode = res.get_id()
    if het.strip():
        return None
    code = icode.strip()
    return f"{res.get_parent().id}:{num}{code}"


def fetch_structure(pdb_id: str, cache: Path) -> Path | None:
    pdb_id = pdb_id.lower()
    for ext, url in (
        (".pdb", f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"),
        (".cif", f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"),
    ):
        dest = cache / f"{pdb_id}{ext}"
        if dest.exists() and dest.stat().st_size > 2000:
            return dest
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                blob = fh.read()
            if len(blob) > 2000:
                dest.write_bytes(blob)
                return dest
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return None


def _load_model(path: Path, pdb_id: str):
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, str(path))[0]


def _standard_only(model, chains: set[str]):
    """A deep copy holding only standard amino acids on `chains`, no hydrogens."""
    out = copy.deepcopy(model)
    for chain in list(out):
        if chain.id not in chains:
            out.detach_child(chain.id)
            continue
        for res in list(chain):
            het, _, _ = res.get_id()
            name = res.get_resname().strip().capitalize()
            one = _T31.get(res.get_resname().strip(), _T31.get(name, "X"))
            if het.strip() or one not in STANDARD:
                chain.detach_child(res.get_id())
                continue
            for atom in list(res):
                if atom.element == "H" or atom.get_id().startswith("H"):
                    res.detach_child(atom.get_id())
    for chain in list(out):
        if len(chain) == 0:
            out.detach_child(chain.id)
    return out


def _heavy_coords(model, chains: set[str]) -> np.ndarray:
    pts = [
        atom.coord
        for chain in model
        if chain.id in chains
        for res in chain
        for atom in res
    ]
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)


def _polar_coords(model, chains: set[str]) -> np.ndarray:
    pts = [
        atom.coord
        for chain in model
        if chain.id in chains
        for res in chain
        for atom in res
        if atom.element in ("N", "O")
    ]
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)


def complex_features(model, chains1: set[str], chains2: set[str]) -> dict:
    """Per-residue interface diagnostics for both sides of one complex."""
    sr = ShrakeRupley()

    both = _standard_only(model, chains1 | chains2)
    side_models = {
        "1": _standard_only(model, chains1),
        "2": _standard_only(model, chains2),
    }
    sr.compute(both, level="R")
    for sm in side_models.values():
        if len(list(sm)) and sum(len(c) for c in sm):
            sr.compute(sm, level="R")

    sasa_complex: dict[str, float] = {}
    resname: dict[str, str] = {}
    for chain in both:
        for res in chain:
            key = _res_key(res)
            if key is None:
                continue
            sasa_complex[key] = float(res.sasa)
            resname[key] = _T31.get(res.get_resname().strip(), "X")

    sasa_iso: dict[str, float] = {}
    side_of: dict[str, str] = {}
    for tag, sm in side_models.items():
        for chain in sm:
            for res in chain:
                key = _res_key(res)
                if key is None:
                    continue
                sasa_iso[key] = float(res.sasa)
                side_of[key] = tag

    heavy = {"1": _heavy_coords(both, chains1), "2": _heavy_coords(both, chains2)}
    polar = {"1": _polar_coords(both, chains1), "2": _polar_coords(both, chains2)}

    feats: dict[str, dict] = {}
    for chain in both:
        for res in chain:
            key = _res_key(res)
            if key is None or key not in side_of:
                continue
            side = side_of[key]
            other = "2" if side == "1" else "1"
            coords = np.asarray([a.coord for a in res], dtype=np.float32)
            opp, opp_polar = heavy[other], polar[other]

            if len(opp) and len(coords):
                dmat = np.linalg.norm(coords[:, None, :] - opp[None, :, :], axis=-1)
                n_contacts = int((dmat < CONTACT_CUTOFF).sum())
                min_dist = float(dmat.min())
            else:
                n_contacts, min_dist = 0, 99.0

            res_polar = np.asarray(
                [a.coord for a in res if a.element in ("N", "O")], dtype=np.float32
            )
            if len(opp_polar) and len(res_polar):
                pm = np.linalg.norm(
                    res_polar[:, None, :] - opp_polar[None, :, :], axis=-1
                )
                n_hbond = int((pm < HBOND_CUTOFF).sum())
            else:
                n_hbond = 0

            one = resname[key]
            iso = sasa_iso.get(key, 0.0)
            cpx = sasa_complex.get(key, 0.0)
            maxasa = MAX_ASA.get(one, 200.0)
            feats[key] = {
                "chain": key.split(":")[0],
                "resnum": key.split(":")[1],
                "aa": one,
                "side": side,
                "sasa_complex": round(cpx, 1),
                "sasa_isolated": round(iso, 1),
                "dsasa": round(max(iso - cpx, 0.0), 1),
                "rel_sasa_isolated": round(min(iso / maxasa, 1.5), 3),
                "rel_sasa_complex": round(min(cpx / maxasa, 1.5), 3),
                "buried_frac_on_binding": round(
                    (max(iso - cpx, 0.0) / iso) if iso > 1.0 else 0.0, 3
                ),
                "n_contacts_5A": n_contacts,
                "min_dist_to_partner": round(min_dist, 2),
                "n_polar_pairs_3p5A": n_hbond,
                "is_hydrophobic": one in HYDROPHOBIC,
                "is_aromatic": one in AROMATIC,
                "is_charged": one in CHARGED,
                "is_polar": one in POLAR,
            }
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pdb-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: featurise at most N complexes (0 = all). "
                         "Bounds the work actually done, not just the report.")
    args = ap.parse_args()

    bench = Path(args.bench)
    cache = Path(args.pdb_cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    decisions = json.loads((bench / "decisions_public.json").read_text())
    # One complex can appear in several slices; features are per complex.
    wanted: dict[str, dict] = {}
    for d in decisions:
        wanted.setdefault(
            d["complex_id"],
            {"pdb_id": d["pdb_id"], "chains1": d["chains1"], "chains2": d["chains2"]},
        )
    print(f"[features] {len(wanted)} complexes over {len(decisions)} decisions")
    if args.limit:
        keep = sorted(wanted)[: args.limit]
        wanted = {k: wanted[k] for k in keep}
        print(f"[features] --limit {args.limit}: featurising {len(wanted)} complexes")

    # Resumable: an existing output is extended, never recomputed. A time limit
    # then costs a resubmission and not the whole pass.
    dest = out / "features.json"
    store: dict[str, dict] = (
        json.loads(dest.read_text()) if dest.exists() else {}
    )

    failures: dict[str, str] = {}
    for i, (complex_id, spec) in enumerate(sorted(wanted.items()), 1):
        if complex_id in store:
            continue
        path = fetch_structure(spec["pdb_id"], cache)
        if path is None:
            failures[complex_id] = "download_failed"
            print(f"[features] {i:3d}/{len(wanted)} {complex_id:14s} DOWNLOAD FAILED")
            continue
        try:
            model = _load_model(path, spec["pdb_id"])
            feats = complex_features(
                model, set(spec["chains1"]), set(spec["chains2"])
            )
        except Exception as exc:  # noqa: BLE001 - one bad entry must not kill the pass
            failures[complex_id] = f"{type(exc).__name__}: {exc}"
            print(f"[features] {i:3d}/{len(wanted)} {complex_id:14s} FAILED {exc}")
            continue
        store[complex_id] = feats
        if i % 10 == 0 or i == len(wanted):
            dest.write_text(json.dumps(store, sort_keys=True))
        print(f"[features] {i:3d}/{len(wanted)} {complex_id:14s} {len(feats):4d} residues")
    dest.write_text(json.dumps(store, sort_keys=True))

    # The verification gate: every candidate site must resolve to a residue
    # whose identity matches the wild type SKEMPI recorded. A silent mismatch
    # means the agent is reading diagnostics for the wrong residue.
    checked = matched = missing = mismatched = 0
    bad_examples: list[str] = []
    covered_decisions = 0
    for d in decisions:
        feats = store.get(d["complex_id"])
        if not feats:
            continue
        ok_all = True
        for cand in d["candidates"]:
            for (chain, pos), wt in zip(cand["pdb_sites"], cand["wt"]):
                checked += 1
                entry = feats.get(f"{chain}:{pos}")
                if entry is None:
                    missing += 1
                    ok_all = False
                    if len(bad_examples) < 8:
                        bad_examples.append(f"{d['complex_id']} {chain}:{pos} missing")
                elif entry["aa"] != wt:
                    mismatched += 1
                    ok_all = False
                    if len(bad_examples) < 8:
                        bad_examples.append(
                            f"{d['complex_id']} {chain}:{pos} is {entry['aa']} not {wt}"
                        )
                else:
                    matched += 1
        covered_decisions += int(ok_all)

    report = {
        "n_complexes_requested": len(wanted),
        "n_complexes_featurised": len(store),
        "failures": failures,
        "sites_checked": checked,
        "sites_matched": matched,
        "sites_missing_from_structure": missing,
        "sites_identity_mismatch": mismatched,
        "site_match_rate": round(matched / checked, 4) if checked else 0.0,
        "decisions_fully_covered": covered_decisions,
        "decisions_total": len(decisions),
        "mismatch_examples": bad_examples,
        "notes": [
            "indexed by Mutation(s)_PDB numbering",
            "waters and non-standard residues stripped before SASA",
            "asymmetric unit, not biological assembly",
        ],
    }
    (out / "features_report.json").write_text(json.dumps(report, indent=1))
    print(f"[features] site match rate {report['site_match_rate']:.4f} "
          f"({matched}/{checked}); decisions fully covered "
          f"{covered_decisions}/{len(decisions)}")
    if bad_examples:
        print("[features] examples:", *bad_examples, sep="\n[features]   ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
