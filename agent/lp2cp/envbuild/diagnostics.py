"""Measured structural diagnostics for one cyclic peptide-receptor complex.

EVERY NUMBER THIS MODULE RETURNS IS COMPUTED FROM REAL COORDINATES. Nothing
here is a prediction and nothing is sampled. That matters because these are
the diagnostics the agent reads: the difficulty of a parent, where its
closure-interface conflict actually sits, and how much interface is at stake
all come from an experimental structure. Only the RESPONSE to a hypothetical
edit is modelled, and that lives in `surrogate.py`, separately and loudly.

Source: CPSea_PDB preprocessed by the sibling project -- chain A is the
receptor, chain B the cyclic peptide, caps stripped, cyclisation bonds kept as
CONECT records with remapped serials.

The one derived quantity is `strain_rel`: a Ramachandran-basin deviation in
REU-like units, standing in for Rosetta's per-residue backbone strain term. It
is a geometric measurement of the deposited backbone, not an energy, and the
docstring on `_strain` says exactly what it computes.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..types import ResidueDiag

# numpy and biotite are imported INSIDE `measure`, not at module scope. The
# campaign runner imports this module transitively (via build_parents, for
# `load_parents`) but never measures a structure: an API-only job reads JSON and
# calls an HTTP endpoint, so it has no reason to need the scientific stack.
# With a top-level numpy import, the dedicated API venv -- which has
# `anthropic` and nothing else -- could not import the package at all.

# Backbone dihedral basins, degrees. Standard Ramachandran minima: right-handed
# alpha, beta sheet, polyproline II, left-handed alpha. A residue sitting in
# none of them is strained, and how far it sits from the nearest is the measure.
BASINS: tuple[tuple[float, float], ...] = (
    (-63.0, -43.0),    # alpha_R
    (-120.0, 130.0),   # beta
    (-75.0, 145.0),    # PPII
    (57.0, 40.0),      # alpha_L
)
# Glycine has no side chain and reaches basins the others cannot; proline's phi
# is locked by the ring. Declared multipliers rather than per-residue tables:
# the quantity is a proxy and a bigger table would imply a precision it has not
# got.
STRAIN_SCALE: dict[str, float] = {"G": 0.45, "P": 1.35}
STRAIN_PER_DEG2 = 4.0 / (60.0 ** 2)   # 60 degrees off-basin ~ 4.0 REU, the limit
# Soft saturation ceiling. A quadratic in off-basin angle grows without bound,
# which is wrong in two ways: a real Ramachandran term plateaus once the
# backbone is simply in a disallowed region rather than continuing to climb,
# and the unbounded tail made the strain objective's NORMALISED margin (see
# qualification._margins) a multiple of every other objective's, so the
# tie-break rule never reached any other defect class and the parent set came
# out 87% BACKBONE_STRAIN_AT_CLOSURE. Measured over 150 CPSea parents: raw
# deviations run 1.0-30.6 with a median of 7.2; saturating at 12.0 maps that to
# 1.0-8.6 with a median of 4.5, which straddles the design doc's 4.0 limit and
# puts strain on the same margin scale as BSA and the conformer population.
STRAIN_SATURATION = 12.0

POLAR = {"N", "O"}
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _ang_delta(a: float, b: float) -> float:
    """Signed difference between two angles in degrees, wrapped to [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _strain(phi: float | None, psi: float | None, wt: str) -> float:
    """Ramachandran-basin deviation of one residue, in REU-like units.

    Defined as: angular distance (degrees) from (phi,psi) to the nearest
    canonical basin, squared, scaled so that 60 degrees off-basin reaches 4.0 --
    the design doc's per-residue strain limit -- then soft-saturated at
    STRAIN_SATURATION. Residue-type multipliers for glycine and proline.

    This is a measurement of the deposited backbone's geometry. It is NOT a
    Rosetta score and is not calibrated against one; it is used because it is
    the quantity a real strain term would be highly correlated with, it is
    computable from coordinates alone, and the thing being tested is whether an
    agent can read a strain column and act on it.
    """
    if phi is None or psi is None or not math.isfinite(phi) or not math.isfinite(psi):
        # Chain termini have one dihedral undefined. In a HEAD-TO-TAIL cyclic
        # peptide there are no termini, but the deposited file is still a linear
        # residue list, so position 1's phi and position n's psi come back NaN.
        # Returning 0.0 would make the closure positions look strain-free --
        # exactly the positions the whole LP->CP premise is about -- so they are
        # marked unknown and the caller fills them from the ring geometry.
        return float("nan")
    d2 = min(
        _ang_delta(phi, bphi) ** 2 + _ang_delta(psi, bpsi) ** 2
        for bphi, bpsi in BASINS
    )
    raw = STRAIN_PER_DEG2 * d2 * STRAIN_SCALE.get(wt, 1.0)
    return float(raw / (1.0 + raw / STRAIN_SATURATION))


def _read_conect_cyclization(path: Path, binder_chain: str = "B"
                             ) -> list[tuple[int, int]]:
    """Cyclisation bonds from the file's own CONECT records.

    The preprocessor kept these with remapped serials, so this is the deposited
    bond topology rather than a guess from the `cyclization_type` string. A
    CONECT between two binder residues more than one apart in sequence is a
    cyclisation bond; adjacent pairs are ordinary peptide bonds and are skipped.

    The serial -> residue map is built from the file's own ATOM/HETATM columns
    rather than through the parsed structure. biotite's reader does not carry
    the deposited serial through a filtered AtomArray, so matching CONECT
    serials against post-filter indices silently found nothing and every parent
    fell back to an ASSUMED head-to-tail topology -- which for a disulphide- or
    sidechain-cyclised peptide would put the ring-distance metric, and with it
    the whole closure-conflict geometry, in the wrong place.
    """
    bonds: set[tuple[int, int]] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []

    serial_to_res: dict[int, tuple[str, int]] = {}
    order: dict[str, list[int]] = {}
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            serial = int(line[6:11])
            chain = line[21].strip() or " "
            resid = int(line[22:26])
        except (ValueError, IndexError):
            continue
        serial_to_res[serial] = (chain, resid)
        seen = order.setdefault(chain, [])
        if not seen or seen[-1] != resid:
            if resid not in seen:
                seen.append(resid)
    idx_of = {rid: i + 1 for i, rid in enumerate(order.get(binder_chain, []))}

    for line in lines:
        if not line.startswith("CONECT"):
            continue
        try:
            serials = [int(line[i:i + 5]) for i in range(6, min(len(line), 31), 5)
                       if line[i:i + 5].strip()]
        except ValueError:
            continue
        if len(serials) < 2:
            continue
        a = serials[0]
        for b in serials[1:]:
            ra, rb = serial_to_res.get(a), serial_to_res.get(b)
            if not ra or not rb:
                continue
            (ca, rida), (cb, ridb) = ra, rb
            if ca != binder_chain or cb != binder_chain:
                continue
            ia, ib = idx_of.get(rida), idx_of.get(ridb)
            if not ia or not ib or abs(ia - ib) <= 1:
                continue
            bonds.add((min(ia, ib), max(ia, ib)))
    return sorted(bonds)


def measure(pdb_path: str | Path, binder_chain: str = "B") -> dict:
    """Measure one complex. Returns per-residue diagnostics plus interface totals.

    Raises on a structure the pipeline cannot use (missing chain, no peptide,
    non-standard residues) rather than returning partial rows: a parent with
    half its diagnostics present would give the agent a table of dashes and the
    resulting decision would be scored as if it had evidence.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdbio
    import numpy as np

    path = Path(pdb_path)
    pdbf = pdbio.PDBFile.read(str(path))
    st = pdbf.get_structure(model=1)
    st = st[(st.element != "H") & (st.hetero == False)]  # noqa: E712

    # Normalise the binder to chain B so everything downstream has one
    # convention, whatever the metadata said.
    if binder_chain != "B":
        cid = st.chain_id.copy()
        cid[st.chain_id == binder_chain] = "B"
        cid[st.chain_id == "B"] = binder_chain
        st.chain_id = cid

    pep = st[st.chain_id == "B"]
    rec = st[st.chain_id != "B"]
    if pep.array_length() == 0 or rec.array_length() == 0:
        raise ValueError(f"{path.name}: need both a binder chain and a receptor")

    pep_res = struc.get_residues(pep)[0]           # res_id, ordered
    n_pos = len(pep_res)
    if n_pos < 4:
        raise ValueError(f"{path.name}: peptide of {n_pos} residues is too short")

    names = [
        str(pep.res_name[pep.res_id == rid][0]) for rid in pep_res
    ]
    unknown = [nm for nm in names if nm not in AA3TO1]
    if unknown:
        raise ValueError(f"{path.name}: non-standard residues {sorted(set(unknown))}")
    seq = "".join(AA3TO1[nm] for nm in names)
    # 1-based peptide index <-> deposited res_id
    idx_of = {int(rid): i + 1 for i, rid in enumerate(pep_res)}

    # --- SASA: bound complex vs free peptide. Shrake-Rupley, single radii. ----
    sasa_bound_all = struc.sasa(st, vdw_radii="Single")
    sasa_free_pep = struc.sasa(pep, vdw_radii="Single")
    sasa_free_rec = struc.sasa(rec, vdw_radii="Single")
    in_b = st.chain_id == "B"
    bsa_total = float(
        np.nansum(sasa_free_pep) - np.nansum(sasa_bound_all[in_b])
        + np.nansum(sasa_free_rec) - np.nansum(sasa_bound_all[~in_b])
    ) / 2.0

    # --- dihedrals -----------------------------------------------------------
    phi, psi, _omega = struc.dihedral_backbone(pep)
    phi = np.degrees(np.asarray(phi, dtype=float))
    psi = np.degrees(np.asarray(psi, dtype=float))

    # --- cyclisation bond ----------------------------------------------------
    # Read against the ORIGINAL chain id: the swap above renamed the binder to
    # B in the parsed array, but the file on disk still says whatever the
    # metadata said, and CONECT is read from the file.
    bonds = _read_conect_cyclization(path, binder_chain)
    if bonds:
        anchors = tuple(sorted({p for b in bonds for p in b if p > 0}))
    else:
        # No usable CONECT: fall back to head-to-tail, which is what the
        # dataset's dominant `cyclization_type` is. Recorded in the return so a
        # parent whose topology was assumed can be excluded later.
        anchors = (1, n_pos)
    anchors = tuple(p for p in anchors if 1 <= p <= n_pos) or (1, n_pos)

    def ring_distance(p: int) -> int:
        """Positions along the ring from p to the nearest anchor.

        Measured around the CYCLE, not along the sequence: in a head-to-tail
        12-mer, position 2 and position 11 are both one step from an anchor,
        and a linear |p - anchor| would call position 11 nine steps away. The
        whole closure-vs-interface conflict is about proximity to the closure
        bond, so getting this wrong would mis-site the conflict.
        """
        best = n_pos
        for a in anchors:
            d = abs(p - a)
            best = min(best, min(d, n_pos - d))
        return int(best)

    # --- per-residue interface terms ----------------------------------------
    rec_coord = rec.coord
    rec_is_polar = np.isin(rec.element, list(POLAR))
    rec_labels = [
        f"{AA3TO1.get(str(n), 'X')}{c}{int(r)}"
        for n, c, r in zip(rec.res_name, rec.chain_id, rec.res_id)
    ]

    out: list[ResidueDiag] = []
    strains: list[float] = []
    for i, rid in enumerate(pep_res):
        mask = pep.res_id == rid
        sc = mask & ~np.isin(pep.atom_name, list(BACKBONE))
        # Side-chain dSASA, with a glycine fallback to whole-residue: glycine
        # has no side chain, and a 0.0 there would read as "nothing buried"
        # rather than "nothing to bury".
        sel = sc if sc.sum() else mask
        bsel = np.zeros(st.array_length(), dtype=bool)
        # Map the peptide-local selection back into the complex array.
        bsel[np.where(in_b)[0][sel]] = True
        dsasa = float(np.nansum(sasa_free_pep[sel]) - np.nansum(sasa_bound_all[bsel]))
        free = float(np.nansum(sasa_free_pep[mask]))
        bmask_res = np.zeros(st.array_length(), dtype=bool)
        bmask_res[np.where(in_b)[0][mask]] = True
        bound = float(np.nansum(sasa_bound_all[bmask_res]))
        buried_frac = (free - bound) / free if free > 1e-6 else 0.0

        d = np.linalg.norm(
            pep.coord[mask][:, None, :] - rec_coord[None, :, :], axis=-1
        )
        contacts = int((d < 5.0).sum())
        min_dist = float(d.min()) if d.size else 99.0
        pol_pep = np.isin(pep.element[mask], list(POLAR))
        polar_pairs = int(((d < 3.5) & pol_pep[:, None] & rec_is_polar[None, :]).sum())
        partner = None
        if polar_pairs:
            jj = np.where((d < 3.5) & pol_pep[:, None] & rec_is_polar[None, :])
            best = int(jj[1][np.argmin(d[jj])])
            partner = rec_labels[best]

        p = i + 1
        s = _strain(
            float(phi[i]) if i < len(phi) else None,
            float(psi[i]) if i < len(psi) else None,
            AA3TO1[names[i]],
        )
        strains.append(s)
        out.append(
            ResidueDiag(
                pos=p,
                wt=AA3TO1[names[i]],
                dsasa=round(dsasa, 1),
                buried_frac=round(float(min(max(buried_frac, 0.0), 1.0)), 3),
                contacts_5a=contacts,
                polar_3p5a=polar_pairs,
                min_dist=round(min_dist, 2),
                phi=None if i >= len(phi) or not np.isfinite(phi[i]) else round(float(phi[i]), 1),
                psi=None if i >= len(psi) or not np.isfinite(psi[i]) else round(float(psi[i]), 1),
                strain_rel=0.0,          # filled below once the median is known
                ring_dist=ring_distance(p),
                is_anchor=p in anchors,
                hbond_partner=partner,
            )
        )

    # Termini with an undefined dihedral (positions 1 and n of the deposited
    # residue list) take the median of the defined ones rather than 0.0 or NaN.
    # In a head-to-tail cycle those are the closure residues; giving them 0.0
    # would mark the closure as the least strained place in the ring.
    known = [s for s in strains if np.isfinite(s)]
    fill = float(np.median(known)) if known else 1.0
    out = [
        ResidueDiag(**{**r.__dict__, "strain_rel": round(
            fill if not np.isfinite(strains[i]) else strains[i], 3)})
        for i, r in enumerate(out)
    ]

    return {
        "sequence": seq,
        "n_pos": n_pos,
        "anchors": anchors,
        "cyclization_bonds": bonds,
        "anchors_assumed": not bonds,
        "native_bsa": round(bsa_total, 1),
        "residues": tuple(out),
        "receptor_n_res": int(len(struc.get_residues(rec)[0])),
    }
