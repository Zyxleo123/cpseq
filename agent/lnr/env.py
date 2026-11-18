"""The LNR environment: linear peptide-protein complexes, REAL ring closure.

Why this exists, and why it is not the surrogate
------------------------------------------------
Every number the Stage 2/3 agent produced came from a DECLARED RESPONSE MODEL
(lp2cp/envbuild/surrogate.py): the parents' geometry was measured, but how an
objective responds to an edit was modelled. That was the right instrument for
the controller question -- it is the only way to know the exact optimum of the
action space -- but it licenses nothing biophysical, and a Rosetta check of it
(agent/validate_real_scorer.py) found that one of its load-bearing claims (a
substitution at a partially exposed position GAINS interface) does not hold in
real energy.

Two things were wrong with testing on the CP parents at all, and this module
fixes both:

  1. The parent was a DEPOSITED CYCLIC complex -- already a near-optimal ring.
     "Repair it" has little real headroom, and the cyclisation damage the
     benchmark repaired was itself modelled. Here the parent is the LINEAR
     peptide in its bound pose, and the damage -- the ring does not close
     without moving residues off their bound positions -- is REAL and measured.

  2. The response to an edit was modelled. Here nothing is modelled. The two
     quantities that decide a move are both computed from coordinates by the
     sibling project's calibrated closure solver (Proteina-Complexa,
     script_utils/kinematic_ceiling.py):

       CLOSES    -- feasible_torsion(): solve for free phi/psi that bring the
                    head-to-tail bond within tolerance while a window [a,b] is
                    HELD at its bound conformation. Real torsion space, ideal
                    bond geometry, native omega.
       RETENTION -- retention_ceiling(): the fraction of the bound interface's
                    CA-CA contacts that survive when only [a,b] is held. A real
                    coordinate quantity, the project's own sdedit_cyclize
                    contact definition.

The decision, and why it is the LP->CP conflict in one number
-------------------------------------------------------------
A move is a held window [a,b]: the residues inside it keep their bound
conformation, everything outside is released to let the ring close. Releasing
MORE residues makes closure easier and throws away interface; releasing FEWER
keeps interface but may not close. So the controller must spend a RELEASE
BUDGET where it costs the least binding -- which is the allocation question the
whole project is about, now with closure and interface both real.

    utility(a,b) = retention_strict(a,b)   if the ring closes with [a,b] held
                 = 0                        otherwise

What is and is not modelled here, stated plainly
------------------------------------------------
  REAL: the bound coordinates, the contact set, closure feasibility in torsion
        space, interface retention. No LLM, no learned scorer, no response
        model anywhere in the objective.
  IDEALISED (declared, from the solver): ideal bond lengths/angles and native
        omega in the closure test; the CA-CA 10 A contact definition; strict
        retention counts only contacts made by held residues (the conservative
        reading). These are the solver's idealisations, shared byte-for-byte
        with the sibling project's ceiling audit so the numbers are comparable.
  NOT MODELLED AT ALL: side-chain repacking, ddG, folding. A closing window is
        a GEOMETRIC claim that a backbone exists, not a binding prediction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The closure solver is the sibling project's, imported rather than copied: a
# second copy of feasible_torsion would drift from the one the ceiling audit
# used, and then an LNR retention could not be divided into that audit's.
_PC = "/zfsauton2/home/yixiz/Proteina-Complexa"
for _p in (_PC, f"{_PC}/script_utils"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kinematic_ceiling as kc  # noqa: E402

CA_CONTACT_CUTOFF_A = kc.CA_CONTACT_CUTOFF_A
BACKBONE = ("N", "CA", "C")


@dataclass(frozen=True)
class LNRTarget:
    """One linear peptide-receptor complex, backbone and contacts measured."""

    target_id: str
    pdb_path: str
    pep_chain: str
    rec_chain: str
    seq: str                      # one-letter, peptide, N->C
    N: np.ndarray                 # (L,3) backbone N
    CA: np.ndarray                # (L,3)
    C: np.ndarray                 # (L,3)
    rec_ca: np.ndarray            # (M,3) receptor CA
    contacts: frozenset           # {(pep_idx, rec_idx)} CA-CA within cutoff
    per_res_contacts: tuple       # contacts[i] = # receptor CA within cutoff of residue i

    @property
    def length(self) -> int:
        return len(self.CA)

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def _read_chain(path: str, chain: str):
    """Backbone N/CA/C and sequence for one chain, in residue order.

    Robust to alt locs (keeps blank or 'A'), insertion codes (keyed into the
    residue identity), and residues missing a backbone atom (dropped, with the
    sequence kept consistent so an index is always a real residue triple).
    """
    res: dict = {}
    order: list = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[21] != chain:
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE:
            continue
        resname = line[17:20].strip()
        key = (line[22:26].strip(), line[26])      # resseq + icode
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if key not in res:
            res[key] = {"name": resname}
            order.append(key)
        res[key][atom] = xyz
    N, CA, C, seq = [], [], [], []
    for key in order:
        r = res[key]
        if all(a in r for a in BACKBONE):
            N.append(r["N"]); CA.append(r["CA"]); C.append(r["C"])
            seq.append(_THREE_TO_ONE.get(r["name"], "X"))
    return (np.asarray(N, float), np.asarray(CA, float),
            np.asarray(C, float), "".join(seq))


def _rec_ca(path: str, chain: str) -> np.ndarray:
    out = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain:
            continue
        if line[16] not in (" ", "A") or line[12:16].strip() != "CA":
            continue
        out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(out, float)


def load_target(target_id: str, pdb_path: str, c1: str, c2: str) -> LNRTarget | None:
    """Load a target. The peptide is the shorter chain (5-19 vs 100-400 here)."""
    Na, CAa, Ca, seqa = _read_chain(pdb_path, c1)
    Nb, CAb, Cb, seqb = _read_chain(pdb_path, c2)
    if len(CAa) == 0 or len(CAb) == 0:
        return None
    if len(CAa) <= len(CAb):
        pep_c, rec_c = c1, c2
        N, CA, C, seq = Na, CAa, Ca, seqa
    else:
        pep_c, rec_c = c2, c1
        N, CA, C, seq = Nb, CAb, Cb, seqb
    rec = _rec_ca(pdb_path, rec_c)
    if len(CA) < 4 or len(rec) == 0:
        return None
    contacts = kc.ca_contact_set(CA, rec)
    per_res = tuple(sum(1 for (p, _) in contacts if p == i) for i in range(len(CA)))
    return LNRTarget(
        target_id=target_id, pdb_path=pdb_path, pep_chain=pep_c, rec_chain=rec_c,
        seq=seq, N=N, CA=CA, C=C, rec_ca=rec,
        contacts=frozenset(contacts), per_res_contacts=per_res,
    )


def load_manifest(manifest: str, pdb_dir: str) -> list[LNRTarget]:
    out = []
    for line in Path(manifest).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pdb, c1, c2 = parts[0], parts[1], parts[2]
        path = str(Path(pdb_dir) / f"{pdb}.pdb")
        if not Path(path).exists():
            continue
        t = load_target(pdb, path, c1, c2)
        if t is not None:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# The real scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowScore:
    a: int
    b: int
    window_len: int
    release_count: int
    analytic_feasible: bool
    closes: bool
    residual_A: float
    retention: float          # strict, in [0,1]
    utility: float            # retention if closes else 0
    seconds: float


def _spec(t: LNRTarget):
    return kc.mainchain_spec(t.length)


def retention_of(t: LNRTarget, a: int, b: int) -> float:
    strict, _ = kc.retention_ceiling(
        set(t.contacts), t.rec_ca, t.CA, a, b, cutoff_A=CA_CONTACT_CUTOFF_A)
    return 0.0 if strict != strict else float(strict)   # nan -> 0


def analytic_feasible(t: LNRTarget, a: int, b: int) -> bool:
    ok, _, _ = kc.feasible_analytic(t.CA, _spec(t), a, b)
    return bool(ok)


def score_window(t: LNRTarget, a: int, b: int,
                 n_restarts: int = 6, max_seconds: float = 12.0,
                 skip_torsion_if_analytic_infeasible: bool = True) -> WindowScore:
    """The REAL score of holding [a,b]. Clamped to the peptide, a<=b.

    The analytic pass is a genuine upper bound -- anything it rejects is truly
    infeasible -- so by default a window it rejects is scored closes=False
    without paying the torsion solver, which is what keeps a full campaign
    affordable. The torsion solver is the real confirmation and is only spent
    on windows that could actually close.
    """
    import time

    L = t.length
    a = max(0, min(int(a), L - 1))
    b = max(a, min(int(b), L - 1))
    t0 = time.time()
    retention = retention_of(t, a, b)
    afeas = analytic_feasible(t, a, b)
    if skip_torsion_if_analytic_infeasible and not afeas:
        closes, resid = False, float("nan")
    else:
        closes, resid = kc.feasible_torsion(
            t.N, t.CA, t.C, _spec(t), a, b,
            n_restarts=n_restarts, max_seconds=max_seconds)
    return WindowScore(
        a=a, b=b, window_len=b - a + 1, release_count=L - (b - a + 1),
        analytic_feasible=afeas, closes=bool(closes),
        residual_A=float(resid), retention=retention,
        utility=retention if closes else 0.0,
        seconds=round(time.time() - t0, 2),
    )


def frontier(t: LNRTarget, confirm_top: int = 12,
             n_restarts: int = 6, max_seconds: float = 10.0) -> dict:
    """The achievable retention ceiling, torsion-confirmed. NOT an oracle table.

    The user's framing is that the controllers are compared RELATIVE to each
    other, not against a per-cell regret oracle, so this does not enumerate and
    confirm every window. It walks analytic-feasible windows in descending
    retention and torsion-confirms until one closes -- that first confirmed
    window is the best real retention attainable, and its release count is the
    cheapest closure that keeps the most interface. Confirming at most
    `confirm_top` windows bounds the cost; the analytic bound guarantees
    nothing better than the top of that list exists.
    """
    L = t.length
    cands = []
    for a in range(L):
        for b in range(a, L):
            if analytic_feasible(t, a, b):
                cands.append((retention_of(t, a, b), a, b))
    cands.sort(key=lambda x: (-x[0], (x[2] - x[1])))   # retention, then smaller window
    best = None
    tried = 0
    for ret, a, b in cands:
        if tried >= confirm_top:
            break
        tried += 1
        closes, resid = kc.feasible_torsion(
            t.N, t.CA, t.C, _spec(t), a, b,
            n_restarts=n_restarts, max_seconds=max_seconds)
        if closes:
            best = {"retention": ret, "a": a, "b": b,
                    "window_len": b - a + 1, "release_count": L - (b - a + 1)}
            break
    return {
        "target_id": t.target_id, "length": L, "n_contacts": t.n_contacts,
        "n_analytic_feasible": len(cands), "confirmed": tried,
        "ceiling": best,                    # None if nothing in top-N closed
    }


# ---------------------------------------------------------------------------
# Physical hardening: does the closed backbone exist WITHOUT steric clashes?
# ---------------------------------------------------------------------------
#
# `score_window` reports geometric closure -- closing torsions exist within
# tolerance. That is a necessary condition, not a sufficient one: a backbone
# can close on ideal bonds yet pass through itself or the receptor. This pass
# rebuilds the closed backbone from the solved torsions and checks CA-level
# sterics, upgrading "closes" to "closes without the chain overlapping itself
# or the partner". It is a steric sanity gate, deterministic, no learned
# scorer and no side chains -- a coarse but honest physical check, not a fold.


def _solve_closed(t: LNRTarget, a: int, b: int, n_restarts: int = 8,
                  seed: int = 0, tol_A: float = 0.1, max_seconds: float = 12.0,
                  max_nfev: int = 800):
    """Re-solve closure for [a,b] and return the closed backbone coordinates.

    Mirrors the sibling solver's feasible_torsion loop (same ideal-bond,
    native-omega idealisation) but returns the rebuilt N/CA/C of the first
    restart that closes, so a steric check can run on it. Returns
    (closed, residual, Nn, CAn, Cn) or (False, residual, None, None, None).
    """
    import time as _time

    import numpy as np
    from scipy.optimize import least_squares

    N, CA, C = t.N, t.CA, t.C
    spec = _spec(t)
    length = len(CA)
    tors = kc.backbone_torsions(N, CA, C)
    phi0 = np.nan_to_num(tors["phi"], nan=-120.0)
    psi0 = np.nan_to_num(tors["psi"], nan=130.0)
    omega = np.nan_to_num(tors["omega"], nan=180.0)
    free = [k for k in range(length) if k < a or k > b]
    if not free:
        resid0 = abs(kc._spec_residual(N, CA, C, spec))
        return (resid0 <= tol_A, resid0, N.copy(), CA.copy(), C.copy())

    def unpack(x):
        phi, psi = phi0.copy(), psi0.copy()
        for idx, k in enumerate(free):
            phi[k] = x[2 * idx]
            psi[k] = x[2 * idx + 1]
        return phi, psi

    def resid(x):
        phi, psi = unpack(x)
        Nn, CAn, Cn = kc.rebuild_from_window(N, CA, C, a, b, phi, psi, omega)
        return np.array([kc._spec_residual(Nn, CAn, Cn, spec)])

    rng = np.random.default_rng(seed)
    x0 = np.concatenate([[phi0[k], psi0[k]] for k in free])
    t_start = _time.perf_counter()
    best = float("inf")
    for attempt in range(max(1, n_restarts)):
        if attempt and _time.perf_counter() - t_start > max_seconds:
            break
        start = x0 if attempt == 0 else x0 + rng.normal(0.0, 60.0, size=x0.shape)
        try:
            sol = least_squares(resid, start, method="trf", max_nfev=max_nfev)
        except Exception:
            continue
        r = abs(float(sol.fun[0]))
        if r < best:
            best = r
        if r <= tol_A:
            phi, psi = unpack(sol.x)
            Nn, CAn, Cn = kc.rebuild_from_window(N, CA, C, a, b, phi, psi, omega)
            return True, r, Nn, CAn, Cn
    return False, best, None, None, None


def _ca_clashes(CAn, rec_ca, a, b, self_cut_A, rec_cut_A):
    """CA-level steric counts. Held residues [a,b] keep native coords, so they
    are excluded from the receptor count (their distances ARE the deposited
    binding geometry, not a clash closure introduced)."""
    import numpy as np

    L = len(CAn)
    d = np.linalg.norm(CAn[:, None, :] - CAn[None, :, :], axis=-1)
    self_cl = 0
    for i in range(L):
        for j in range(i + 2, L):            # |i-j| >= 2
            if d[i, j] < self_cut_A:
                self_cl += 1
    released = [k for k in range(L) if k < a or k > b]
    rec_cl = 0
    if released:
        drec = np.linalg.norm(CAn[released][:, None, :] - rec_ca[None, :, :], axis=-1)
        rec_cl = int((drec < rec_cut_A).sum())
    return self_cl, rec_cl


def clash_report(t: LNRTarget, a: int, b: int,
                 self_cut_A: float = 3.0, rec_cut_A: float = 3.0,
                 **solve_kw) -> dict:
    """CA-level steric check on the closed backbone of held window [a,b].

    self clash   -- peptide CA(i),CA(j) with |i-j|>=2 closer than `self_cut_A`:
                    the chain passing through itself (real CAs never get within
                    ~4 A non-sequentially, so <3 is overlap).
    receptor clash -- a RELEASED residue's CA within `rec_cut_A` of a receptor
                    CA: the closed ring driving a freed residue into the
                    partner. Held residues are native, so excluded.

    CA-only (the receptor is loaded at CA resolution) -- a coarse but honest
    steric gate that catches gross backbone overlap, not side-chain bumps. No
    fold, no learned scorer.
    """
    closed, resid, Nn, CAn, Cn = _solve_closed(t, a, b, **solve_kw)
    if not closed:
        return {"closed": False, "residual": resid, "clash_free": False,
                "self_clashes": None, "receptor_clashes": None}
    self_cl, rec_cl = _ca_clashes(CAn, t.rec_ca, a, b, self_cut_A, rec_cut_A)
    return {"closed": True, "residual": resid,
            "self_clashes": self_cl, "receptor_clashes": rec_cl,
            "clash_free": (self_cl == 0 and rec_cl == 0)}


# ---------------------------------------------------------------------------
# Finer gate: backbone-triplet (N/CA/C) sterics, not CA-only
# ---------------------------------------------------------------------------
#
# The CA gate above catches gross overlap but is coarse -- two CAs can sit 3.2 A
# apart with their carbonyls interpenetrating. This gate checks all three
# backbone heavy atoms per residue on BOTH sides (receptor read at backbone
# resolution), a 3x finer steric test, still side-chain-free so it stays an
# honest backbone claim rather than a fold. The 2.5 A cutoff is calibrated
# against the deposited natives: the smallest non-adjacent backbone-atom
# distance in any deposited peptide is 2.85 A (self) / 2.68 A (pep-vs-receptor),
# so 2.5 A keeps every native clash-free while flagging real interpenetration.

_REC_BB_CACHE: dict = {}


def _rec_backbone(path: str, chain: str) -> np.ndarray:
    """Receptor backbone N/CA/C atoms as (K,3). Cached per (path, chain)."""
    key = (path, chain)
    if key in _REC_BB_CACHE:
        return _REC_BB_CACHE[key]
    pts = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain:
            continue
        if line[16] not in (" ", "A") or line[12:16].strip() not in BACKBONE:
            continue
        pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    arr = np.asarray(pts, float)
    _REC_BB_CACHE[key] = arr
    return arr


def _bb_clashes(Nn, CAn, Cn, rec_bb, a, b, self_cut_A, rec_cut_A):
    """Backbone-triplet steric counts. Mirrors `_ca_clashes` but over N/CA/C of
    every residue; same held/released accounting (held residues are native, so
    excluded from the receptor count)."""
    L = len(CAn)
    pep = np.stack([Nn, CAn, Cn], axis=1).reshape(-1, 3)   # (3L, 3), res = idx//3
    res_of = np.repeat(np.arange(L), 3)
    d = np.linalg.norm(pep[:, None, :] - pep[None, :, :], axis=-1)
    # exclude same and sequence-adjacent residues (|i-j| <= 1): those carry the
    # bonded C(i)-N(i+1) peptide bond and legitimately short backbone distances.
    # The ring is HEAD-TO-TAIL CYCLIC, so residues 0 and L-1 are also bonded
    # (the closing C(L-1)-N(0) peptide bond ~1.33 A) and must be treated as
    # adjacent too -- otherwise every closed ring shows a phantom self-clash at
    # its own closure junction.
    di = np.abs(res_of[:, None] - res_of[None, :])
    far = (di >= 2) & (di != L - 1)      # di==L-1 is the 0<->L-1 wrap pair
    self_cl = int(((d < self_cut_A) & far).sum() // 2)
    self_min = float(d[far].min()) if far.any() else float("inf")
    released = [k for k in range(L) if k < a or k > b]
    rec_cl = 0
    rec_min = float("inf")
    if released and len(rec_bb):
        rel_mask = np.isin(res_of, np.asarray(released))
        drec = np.linalg.norm(pep[rel_mask][:, None, :] - rec_bb[None, :, :], axis=-1)
        rec_cl = int((drec < rec_cut_A).sum())
        rec_min = float(drec.min())
    # min distances let the report sweep the cutoff (2.3/2.5/2.7) as a pure
    # threshold with no re-solve: clash_free_bb at cut c == (self_min>=c & rec_min>=c).
    return self_cl, rec_cl, self_min, rec_min


def clash_report_fine(t: LNRTarget, a: int, b: int,
                      self_cut_A: float = 2.5, rec_cut_A: float = 2.5,
                      **solve_kw) -> dict:
    """Re-solve closure for [a,b] and grade it under BOTH the CA gate (3.0 A,
    identical to `clash_report`, for cross-check) and the finer backbone-triplet
    gate (2.5 A over N/CA/C, receptor at backbone resolution).

    Emits `clash_free` (CA) and `clash_free_bb` (backbone) from the SAME solved
    coordinates, so a report can switch gate with `--clash-field` and the two
    verdicts are perfectly matched per window.
    """
    closed, resid, Nn, CAn, Cn = _solve_closed(t, a, b, **solve_kw)
    if not closed:
        return {"closed": False, "residual": resid,
                "clash_free": False, "clash_free_bb": False,
                "self_clashes": None, "receptor_clashes": None,
                "self_clashes_bb": None, "receptor_clashes_bb": None}
    ca_self, ca_rec = _ca_clashes(CAn, t.rec_ca, a, b, 3.0, 3.0)
    rec_bb = _rec_backbone(t.pdb_path, t.rec_chain)
    bb_self, bb_rec, bb_self_min, bb_rec_min = _bb_clashes(
        Nn, CAn, Cn, rec_bb, a, b, self_cut_A, rec_cut_A)
    return {"closed": True, "residual": resid,
            "self_clashes": ca_self, "receptor_clashes": ca_rec,
            "clash_free": (ca_self == 0 and ca_rec == 0),
            "self_clashes_bb": bb_self, "receptor_clashes_bb": bb_rec,
            "self_min_bb": bb_self_min, "rec_min_bb": bb_rec_min,
            "clash_free_bb": (bb_self == 0 and bb_rec == 0)}


def native_clashes(t: LNRTarget, self_cut_A: float = 3.0,
                   rec_cut_A: float = 3.0) -> dict:
    """Clash counts of the DEPOSITED (open) peptide, as a metric sanity check:
    a real structure should be self-clash-free, so a nonzero self count here
    means the threshold is mis-set, not that closure broke anything."""
    self_cl, rec_cl = _ca_clashes(t.CA, t.rec_ca, 0, t.length - 1,
                                  self_cut_A, rec_cut_A)
    return {"self_clashes": self_cl, "receptor_clashes": rec_cl}
