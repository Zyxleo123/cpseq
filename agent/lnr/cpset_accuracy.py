"""THE CYCLIC-PEPTIDE ACCURACY GATE: does a scorer rank MEASURED binding of
cyclic peptides against the SAME receptor?

Why this exists
---------------
`skempi_accuracy.py` measured our scorer's accuracy on protein-PROTEIN interface
mutations and got rho +0.22 / sign 59% -- too weak to rank affinity, which ended
the "which score" thread and forced the native-mode pivot. But that gate had two
confounds it could not remove:

  * WRONG CHEMISTRY. SKEMPI is linear protein-protein interfaces. Our target is a
    head-to-tail cyclic peptide. The gate's own caveat said so.
  * EFFECTS INSIDE THE NOISE. SKEMPI single mutations are typically ~1 kcal/mol,
    while our oracle's fold noise is ~4 REU. A scorer cannot be shown to rank
    what the ruler cannot resolve, so a null result there is ambiguous: bad
    scorer, or effects too small to see?

This gate removes both. Ground truth is CPSet Dataset I (Zhao et al.,
Brief Bioinform 2025;26(6):bbaf632, doi 10.1093/bib/bbaf632; data at
https://github.com/huifengzhao/MM_PBGBSA-CP): 50 protein-CYCLIC PEPTIDE crystal
complexes with experimental Kd. Grouping those 50 by receptor sequence identity
(see `regroup`) finds that 38 of them fall into 10 CONGENERIC SERIES -- same
receptor, cyclic peptide variants, measured Kd -- which the paper itself never
used (its validation is entirely cross-complex; see "What the paper did NOT
show" below).

That is the within-target instrument this project has never had. The headline
series is 7K2*: one receptor (KEAP1 Kelch domain) and six all-canonical
HEAD-TO-TAIL (BB) cyclic 7-mers differing by 1-2 substitutions, spanning
pKd 3.55-6.85 = 3.3 log units ~ 4.5 kcal/mol. Single substitutions worth
2.3-2.8 log units each. That is 3-4x our noise floor, not inside it -- so for
the first time a null result here is informative rather than ambiguous.

What this gate measures
-----------------------
Scores the DEPOSITED crystal coordinates -- no Boltz fold -- so it isolates the
SCORING function from fold noise, exactly as the SKEMPI gate did. Three numbers:

  1. HEADLINE -- within-series pair sign agreement. For every pair of complexes
     in the same series, does the scorer put the measured-stronger peptide at the
     lower (more negative) dG? Pooled over all 10 series this is 70 ordered
     pairs, stratified by |dpKd| so that pairs too close to call (e.g. 6O21 vs
     4K1E, |dpKd| = 0.02) do not dilute the signal. Reported with a Wilson 95%
     CI, because "better than a coin flip" is the actual question.
  2. Per-series Spearman rho(dG, pKd), with the SIGN CONVENTION stated below.
     Per-series, never pooled across series: pooling would let the easy
     cross-target dynamic range (pKd 3.55-10.29 over 9 kcal/mol) masquerade as
     within-target ranking skill, which is precisely the inflation this gate
     exists to avoid.
  3. Cross-complex correlation over all 50, reported ONLY as the
     paper-comparable number, with the paper's own values printed beside it.

TIERS, because cyclization chemistry differs. `Chain type` in the CSV is BB
(head-to-tail backbone, OUR chemistry), BB+SS (head-to-tail plus a disulfide),
or SS*n (disulfide-only -- a cyclic peptide, but NOT head-to-tail). A series is
tiered BB if every member is backbone-cyclized, MIX if some are, SS if none are.
Any claim about the LNR pipeline needs the BB/MIX tier (28 of the 70 pairs); the
SS tier still tests the scoring function on a cyclic-peptide interface.

SIGN CONVENTION (the project's recurring bug class -- stated once, asserted in
tests): pKd = -log10(Kd), so HIGHER pKd = STRONGER binding. Rosetta
dG_separated is MORE NEGATIVE = stronger. Therefore a WORKING scorer gives a
NEGATIVE rho(dG, pKd). This matches the paper, which also reports negative Rp.
Pair agreement is defined sign-free ("stronger measured => lower dG"), so it
cannot be fooled by a sign slip.

PRE-REGISTERED verdict (fixed before the first run, so it cannot be tuned to the
result -- the `seq_rankers` rule had to be rewritten after it crowned esm2 on a
+0.08-vs--0.14 difference, and `confidence_metrics` had to be rewritten after
`complex_plddt`'s invariant-bought discriminability fooled it):

  PASS  pair agreement at |dpKd| >= 1.0 has Wilson 95% CI lower bound >= 0.65
        AND mean per-series rho <= -0.40
        AND >= 60% of series with n >= 3 have negative rho
  WEAK  CI lower bound > 0.50 but PASS not met
  FAIL  CI lower bound <= 0.50 (indistinguishable from a coin flip)

A PASS means affinity ranking is achievable on our chemistry and the project's
pivot away from it was driven by SKEMPI's wrong chemistry + sub-noise effects,
not by the scorer. A FAIL closes the affinity objective with ground truth that
is finally the right chemistry AND well above the noise floor -- a much stronger
negative result than the SKEMPI gate could give, and it justifies the
native-mode pivot properly.

What the paper did NOT show (do not import its headline)
--------------------------------------------------------
The paper's tuned MM/PBSA reaches Rp = -0.732, but that is Pearson across 24
HETEROGENEOUS complexes spanning pKd 3.55-10.29. It contains no within-target
experiment at all: no mutation series, no congeneric series on a fixed receptor,
no per-target metric, and no sensitivity analysis to small sequence changes. Its
only stratifications are interface polarity and peptide length. So -0.732 is
evidence about absolute affinity across targets, NOT about ranking substitutions
of one peptide, and it must not be banked as the latter. Reference values the
report prints for calibration:

    MM/PBSA ff03/implicit/eps_in=2, 50 complexes, cross-complex   Rp -0.545
    same, energy-minimized CRYSTAL complexes, 24-subset (ceiling) Rp -0.813
    AutoDock CrankPep baseline, 24-subset                         Rp -0.316
    best of 16 docking scoring functions (their 2024 JCIM study)   Rp  0.378
    our Rosetta dG vs measured ddG on SKEMPI (wrong chemistry)     rho +0.22

Scorer choice
-------------
Plain Rosetta InterfaceAnalyzer `dG_separated` on the relaxed complex -- the SAME
core quantity `boltz_linear_ref.py` uses, with NO ring declaration, applied
uniformly to all three cyclization chemistries. Uniformity is deliberate: a
head-to-tail bond must not be declared on an SS-only peptide (wrong chemistry),
and the project already established that dG_separated is the same core number
with or without the ring declared. This isolates the scoring question from the
ring-chemistry question. The ring-aware `cpseq.scorer.score_structure` on the
BB/MIX tier is the follow-up, and MM/PBSA per the paper's tuned protocol is the
other contender; `--scorer` is the hook for both, and every row records which
scorer produced it so the two can never be pooled by accident.

Structure preparation (recorded per row, never silent)
------------------------------------------------------
The repo ships each complex as `{PDB}_protein.pdb` + `{PDB}_CP.pdb`, which this
merges into one two-chain PDB. Three things are NOT taken on trust:

  * CHAINS ARE READ FROM THE FILES, never from the CSV's `Chain id`. They
    disagree for 1SLE (CSV says `BD:M`, the files hold `D` and `P`); trusting the
    CSV would hand InterfaceAnalyzer a chain spec matching nothing and yield a
    meaningless dG with no error.
  * CYX -> CYS, so Rosetta's geometric disulfide detection applies (43 CYX
    residues across the set).
  * ACE/NHE caps are DROPPED (10 ACE, 14 NHE, all in the 1VW*/1SL* series),
    because Rosetta has no residue type for them here. This changes those
    peptides' terminal chemistry, so the count lands in each row's `prep` block
    and the report flags any series affected.

Usage
-----
  # one-time: fetch the benchmark (also fine to pre-stage by hand)
  python -m lnr.cpset_accuracy --fetch --data-dir <dir>
  # verify the frozen series table against the structures
  python -m lnr.cpset_accuracy --regroup --data-dir <dir>
  # score and report
  python -m lnr.cpset_accuracy --score --data-dir <dir> --out cpset_dg.jsonl \
      --replicas 3 --workers 16
  python -m lnr.cpset_accuracy --report cpset_dg.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Correlation helpers are IMPORTED, not reimplemented: a second copy of Spearman
# would silently diverge from the one that produced the SKEMPI number, and these
# two gates are meant to be read side by side.
from lnr.skempi_accuracy import _pearson, _spearman  # noqa: E402

DATASET1_CSV_URL = (
    "https://raw.githubusercontent.com/huifengzhao/MM_PBGBSA-CP/main/dataset1.csv"
)
DATASET1_TAR_URL = (
    "https://raw.githubusercontent.com/huifengzhao/MM_PBGBSA-CP/main/dataset1.tar.bz2"
)

# Reference values from the paper, printed beside our own so no number is ever
# read without its comparator.
PAPER_REF = {
    "MM/PBSA ff03/implicit/eps2, 50 complexes": -0.545,
    "MM/PBSA on minimized CRYSTAL, 24-subset (their ceiling)": -0.813,
    "two-step workflow, 24-subset (their headline)": -0.732,
    "AutoDock CrankPep, 24-subset": -0.316,
}

STD_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
# Residues renamed on the way in, with the reason. CYX is a disulfide-bonded
# cysteine in Amber naming; Rosetta wants CYS and detects the bond from geometry.
RENAME_AA3 = {"CYX": "CYS", "MSE": "MET"}
# Caps with no Rosetta residue type in this pipeline. Dropped, and counted.
DROP_AA3 = {"ACE", "NHE", "NME"}

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "CYX": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# ---------------------------------------------------------------------------
# The FROZEN series table.
#
# Derived by `regroup()` (union-find over longest-receptor-chain sequence
# identity >= 0.95) and then written down here on purpose. The headline
# statistic is a within-SERIES quantity, so if the grouping could shift with a
# parser tweak the headline could shift with it and nobody would notice. `
# --regroup` recomputes and DIFFS against this table, so a change has to be
# acknowledged rather than absorbed.
#
# The grouping is unambiguous: every same-receptor identity came out >= 0.95 and
# every different-receptor identity <= 0.55, with NOTHING in between. Two near
# misses the 0.95-on-longest-chain rule gets right and a whole-file comparison
# does not: 6D3X ships a receptor DIMER (492 residues vs 247 for its three
# siblings) yet is the same target; and 1SFI's group vs 6O21's are both ~223
# residues and share the peptide SIPPI motif but are only ~0.67 identical, so
# they are correctly NOT merged -- merging them would silently turn a
# cross-target comparison into a "within-target" one, the exact failure mode
# this gate is built to avoid.
# ---------------------------------------------------------------------------
SERIES: tuple[dict, ...] = (
    {"id": "S01_strep", "tier": "SS", "pdbs": (
        "1VWB", "1SLD", "1SLE", "1VWN", "1VWD", "1VWO", "1VWP")},
    {"id": "S02_qfdlstrrlk", "tier": "SS", "pdbs": (
        "4GW1", "4GW5", "5FF6", "5TH2", "5EUK", "5F88")},
    {"id": "S03_keap1", "tier": "BB", "pdbs": (
        "7K2H", "7K2M", "7K2E", "7K2I", "7K2F", "7K2G")},
    {"id": "S04_sfti", "tier": "MIX", "pdbs": ("1SFI", "6BVH", "1G9I", "1SMF")},
    {"id": "S05_6d3", "tier": "BB", "pdbs": ("6D3X", "6D40", "6D3Z", "6D3Y")},
    {"id": "S06_swrglenh", "tier": "SS", "pdbs": ("2NWN", "3OY5", "4GLY")},
    {"id": "S07_srpp", "tier": "SS", "pdbs": ("2UZ6", "2BR8")},
    {"id": "S08_fcqrs", "tier": "MIX", "pdbs": ("6O21", "4K1E")},
    {"id": "S09_paysry", "tier": "SS", "pdbs": ("4ZHL", "4X1Q")},
    {"id": "S10_h5", "tier": "SS", "pdbs": ("5H5R", "5H5Q")},
)
SERIES_OF = {p: s["id"] for s in SERIES for p in s["pdbs"]}
TIER_OF = {s["id"]: s["tier"] for s in SERIES}
# Thresholds the pooled pair statistic is stratified by. 0.0 keeps every pair
# (including coin-flips like |dpKd| 0.02); 1.0 is "can it call a 10x affinity
# difference", which is the question that matters for design.
DPKD_BINS = (0.0, 0.3, 0.5, 1.0)
HEADLINE_BIN = 1.0

# ---------------------------------------------------------------------------
# Ground truth + structures
# ---------------------------------------------------------------------------


def load_dataset1(csv_path) -> dict:
    """CPSet Dataset I ground truth, keyed by PDB code.

    `chain_id` is read but NEVER used to locate chains -- see `derive_chains`.
    It is carried only so the report can flag the 1SLE disagreement.
    """
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            pdb = (row.get("PDB code") or "").strip().upper()
            if not pdb:
                continue
            out[pdb] = {
                "pdb": pdb,
                "chain_id_csv": (row.get("Chain id") or "").strip(),
                "pep_len": int(float(row["Peptide length"])),
                "polar_pct": float(row["Polar interface (%)"]),
                "kd_m": float(row["Kd (M)"]),
                "pkd": float(row["pKd"]),
                "chain_type": (row.get("Chain type") or "").strip(),
                "series": SERIES_OF.get(pdb),
            }
    return out


def chain_seqs(pdb_path) -> dict:
    """{chain: one-letter sequence} in file order, over ATOM and HETATM."""
    per = {}
    order = {}
    for ln in _read_lines(pdb_path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]
        key = ln[22:27]
        d = per.setdefault(ch, {})
        if key not in d:
            d[key] = ln[17:20].strip()
            order.setdefault(ch, []).append(key)
    return {ch: "".join(AA3to1.get(per[ch][k], "X") for k in order[ch])
            for ch in per}


def _read_lines(path):
    with open(path, errors="replace") as fh:
        return fh.readlines()


def complex_paths(data_dir, pdb):
    """(protein_pdb, cp_pdb) for one complex, or None if either is missing."""
    base = Path(data_dir) / "dataset1" / pdb
    if not base.is_dir():
        base = Path(data_dir) / pdb
    pro = sorted(base.glob("*protein*.pdb"))
    cp = sorted(base.glob("*CP*.pdb"))
    if not pro or not cp:
        return None
    return str(pro[0]), str(cp[0])


def derive_chains(protein_pdb, cp_pdb):
    """(receptor_chains, peptide_chain) READ FROM THE STRUCTURES.

    The CSV's `Chain id` is wrong for 1SLE, and a wrong chain spec makes
    InterfaceAnalyzer return a meaningless number without raising, so the files
    are the only trustworthy source. Returns None if the peptide is not a single
    chain or if receptor and peptide chains collide (neither occurs in Dataset I,
    but a silent assumption here would be a silent wrong answer).
    """
    rec = sorted(chain_seqs(protein_pdb))
    pep = sorted(chain_seqs(cp_pdb))
    if len(pep) != 1 or not rec or set(rec) & set(pep):
        return None
    return "".join(rec), pep[0]


def prepare_pdb(protein_pdb, cp_pdb, out_path):
    """Merge receptor + cyclic peptide into one clean two-chain PDB.

    Returns a `prep` dict that goes into every row: what was renamed, what was
    dropped, and the residue counts. Nothing about preparation is silent.
    """
    prep = {"renamed": {}, "dropped_caps": {}, "dropped_other": {},
            "n_res_rec": 0, "n_res_pep": 0, "altloc_dropped": 0}
    chains = derive_chains(protein_pdb, cp_pdb)
    if chains is None:
        return None, {"error": "could not derive a single peptide chain"}
    rec_chains, pep_chain = chains

    def emit(path, keep, which):
        lines = []
        seen = set()
        for ln in _read_lines(path):
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            if ln[21] not in keep:
                continue
            # Keep only the primary altloc; a second conformer duplicates atoms
            # and Rosetta reads it as a clash.
            if ln[16] not in (" ", "A"):
                prep["altloc_dropped"] += 1
                continue
            res = ln[17:20].strip()
            if res in DROP_AA3:
                prep["dropped_caps"][res] = prep["dropped_caps"].get(res, 0) + 1
                continue
            if res in RENAME_AA3:
                new = RENAME_AA3[res]
                prep["renamed"][res] = prep["renamed"].get(res, 0) + 1
                ln = ln[:17] + f"{new:>3s}" + ln[20:]
                res = new
            if res not in STD_AA3:
                prep["dropped_other"][res] = prep["dropped_other"].get(res, 0) + 1
                continue
            # ATOM, so Rosetta does not treat a renamed HETATM as a ligand.
            lines.append("ATOM  " + ln[6:])
            seen.add((ln[21], ln[22:27]))
        prep[which] = len(seen)
        return lines

    body = emit(protein_pdb, set(rec_chains), "n_res_rec")
    body.append("TER\n")
    body += emit(cp_pdb, {pep_chain}, "n_res_pep")
    body.append("TER\nEND\n")
    if not prep["n_res_rec"] or not prep["n_res_pep"]:
        return None, {**prep, "error": "receptor or peptide empty after cleaning"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(body))
    prep["interface"] = f"{rec_chains}_{pep_chain}"
    prep["rec_chains"] = rec_chains
    prep["pep_chain"] = pep_chain
    return str(out_path), prep


# ---------------------------------------------------------------------------
# Series grouping (pure; verifies the frozen table)
# ---------------------------------------------------------------------------


def _identity(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def regroup(signatures, threshold=0.95):
    """Union-find over receptor signature identity -> {group_key: [pdb, ...]}.

    `signatures` is {pdb: longest-receptor-chain sequence}. Pure, so the frozen
    SERIES table can be re-derived and diffed without touching any structure.
    """
    ids = sorted(signatures)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    borderline = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = _identity(signatures[a], signatures[b])
            if r >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            elif r > 0.55:
                borderline.append((a, b, round(r, 3)))
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return groups, borderline


def series_tier(chain_types):
    """BB if every member is backbone-cyclized, MIX if some are, else SS."""
    bb = [t.startswith("BB") for t in chain_types]
    if all(bb):
        return "BB"
    return "MIX" if any(bb) else "SS"


# ---------------------------------------------------------------------------
# The headline statistic (pure, unit-tested)
# ---------------------------------------------------------------------------


def within_series_pairs(truth, min_dpkd=0.0, pdbs=None):
    """Every ordered within-series pair with |dpKd| >= min_dpkd.

    Yields dicts with `strong`/`weak` = the measured-stronger/weaker PDB, so the
    agreement test below needs no sign convention of its own.
    """
    pool = set(pdbs) if pdbs is not None else set(truth)
    by_series = {}
    for pdb, rec in truth.items():
        if pdb in pool and rec.get("series"):
            by_series.setdefault(rec["series"], []).append(pdb)
    out = []
    for sid, members in sorted(by_series.items()):
        members.sort()
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                d = truth[a]["pkd"] - truth[b]["pkd"]
                if abs(d) < min_dpkd:
                    continue
                strong, weak = (a, b) if d > 0 else (b, a)
                out.append({"series": sid, "tier": TIER_OF.get(sid, "?"),
                            "strong": strong, "weak": weak,
                            "dpkd": round(abs(d), 3)})
    return out


def wilson95(k, n):
    """Wilson 95% CI for a proportion. The question is 'better than a coin
    flip', so the interval -- not the point estimate -- is what decides."""
    if n <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def pair_agreement(pairs, score_by_pdb):
    """Fraction of pairs where the measured-stronger peptide got the LOWER
    (more negative) score. Ties count as failures, not as successes: a scorer
    that cannot separate two peptides has not ranked them."""
    n = ok = 0
    skipped = 0
    for pr in pairs:
        s, w = score_by_pdb.get(pr["strong"]), score_by_pdb.get(pr["weak"])
        if s is None or w is None:
            skipped += 1
            continue
        n += 1
        if s < w:
            ok += 1
    lo, hi = wilson95(ok, n)
    return {"n_pairs": n, "n_correct": ok, "skipped": skipped,
            "agreement": round(ok / n, 4) if n else None,
            "ci95": (round(lo, 4), round(hi, 4))}


def verdict(stats):
    """The PRE-REGISTERED rule from the module docstring. Takes the computed
    stats and returns (label, reasons) -- no thresholds live anywhere else."""
    head = stats.get("headline") or {}
    lo = (head.get("ci95") or (0.0, 1.0))[0]
    rhos = [r for r in stats.get("series_rho", []) if r is not None]
    n3 = stats.get("series_rho_n3") or []
    mean_rho = st.mean(rhos) if rhos else None
    frac_neg = (sum(1 for r in n3 if r < 0) / len(n3)) if n3 else None
    reasons = [
        f"pair agreement |dpKd|>={HEADLINE_BIN} CI95 lower bound = {lo:.3f} "
        f"(PASS needs >=0.65, FAIL at <=0.50)",
        f"mean per-series rho = {mean_rho if mean_rho is None else round(mean_rho, 3)} "
        f"(PASS needs <=-0.40)",
        f"fraction of n>=3 series with negative rho = "
        f"{frac_neg if frac_neg is None else round(frac_neg, 3)} (PASS needs >=0.60)",
    ]
    if lo <= 0.50:
        return "FAIL", reasons
    if (lo >= 0.65 and mean_rho is not None and mean_rho <= -0.40
            and frac_neg is not None and frac_neg >= 0.60):
        return "PASS", reasons
    return "WEAK", reasons


# ---------------------------------------------------------------------------
# Scoring worker (PyRosetta; not unit-tested)
# ---------------------------------------------------------------------------


def _fail(task, msg):
    return {"pdb": task["pdb"], "replica": task["replica"],
            "scorer": task["scorer"], "valid": False, "error": msg[:200]}


def score_one(task):
    """One independent relax + InterfaceAnalyzer dG_separated of one prepared
    complex. A fresh process per task means a fresh `pyrosetta.init` with no
    `-constant_seed`, so each replica is an INDEPENDENT relax draw."""
    rec = {"pdb": task["pdb"], "replica": task["replica"],
           "scorer": task["scorer"], "pkd": task.get("pkd"),
           "series": task.get("series"), "tier": task.get("tier"),
           "chain_type": task.get("chain_type"),
           "interface": task.get("interface"), "prep": task.get("prep")}
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
        pyrosetta.init(
            "-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
            "-load_PDB_components false -relax:default_repeats 2 "
            "-no_fconfig -use_input_sc", silent=True)
        pose = pyrosetta.pose_from_pdb(task["prepared_pdb"])
        sf = pyrosetta.get_fa_scorefxn()
        fr = FastRelax(sf, 1)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)
        ia = InterfaceAnalyzerMover(task["interface"])
        ia.set_pack_separated(True)
        ia.set_compute_packstat(False)
        ia.apply(pose)
        rec["dG"] = float(ia.get_interface_dG())
        rec["dsasa"] = float(ia.get_interface_delta_sasa())
        rec["total_score"] = float(sf(pose))
        rec["n_res"] = int(pose.total_residue())
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["valid"] = False
    return rec


def _worker(task, q):
    try:
        q.put(score_one(task))
    except Exception as exc:  # noqa: BLE001
        q.put(_fail(task, f"{type(exc).__name__}: {exc}"))


def _run_isolated(tasks, workers, timeout):
    """Each task in its OWN short-lived process.

    Deliberately a local copy of the `skempi_accuracy` runner rather than an
    import: that module produced a published number and is left untouched. The
    isolation is not optional -- the SKEMPI run lost 102 of 363 tasks to
    PyRosetta SEGFAULTS (exitcode -11) on specific structures, which cannot be
    caught in-process; under a shared pool one crash is a BrokenProcessPool that
    discards every completed result.
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pending, running = list(tasks), {}
    while pending or running:
        while pending and len(running) < workers:
            t = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(t, q), daemon=True)
            p.start()
            running[p] = (t, q, time.time())
        time.sleep(0.5)
        for p in list(running):
            t, q, start = running[p]
            if not p.is_alive():
                p.join()
                try:
                    out = q.get(timeout=10)
                except Exception:  # noqa: BLE001
                    out = _fail(t, f"worker died (exitcode {p.exitcode}; "
                                   "likely OOM/segfault)")
                running.pop(p, None)
                yield out
            elif time.time() - start > timeout:
                p.terminate()
                p.join()
                running.pop(p, None)
                yield _fail(t, f"timeout after {timeout}s")


def build_tasks(truth, data_dir, work_dir, replicas, scorer, only=None):
    """Prepare every structure once, then one task per (pdb, replica)."""
    tasks, prep_rows, skipped = [], [], []
    for pdb in sorted(truth):
        if only and pdb not in only:
            continue
        paths = complex_paths(data_dir, pdb)
        if paths is None:
            skipped.append((pdb, "structures not found"))
            continue
        out = Path(work_dir) / f"{pdb}_prepared.pdb"
        prepared, prep = prepare_pdb(paths[0], paths[1], out)
        if prepared is None:
            skipped.append((pdb, prep.get("error", "prep failed")))
            continue
        prep_rows.append({"pdb": pdb, **prep})
        rec = truth[pdb]
        # A compact prep summary rides along in EVERY row, so the report can
        # flag a disturbed series without re-reading any structure -- the
        # project's "figures must be redrawable without recomputation" rule.
        prep_tag = {
            "caps": dict(prep["dropped_caps"]),
            "cyx": prep["renamed"].get("CYX", 0),
            "altloc": prep["altloc_dropped"],
            "n_res_pep": prep["n_res_pep"],
            "pep_len_csv": rec["pep_len"],
        }
        for r in range(replicas):
            tasks.append({
                "pdb": pdb, "replica": r, "scorer": scorer,
                "prepared_pdb": prepared, "interface": prep["interface"],
                "pkd": rec["pkd"], "series": rec["series"],
                "tier": TIER_OF.get(rec["series"] or "", "-"),
                "chain_type": rec["chain_type"], "prep": prep_tag,
            })
    return tasks, prep_rows, skipped


def score(data_dir, out_path, replicas=3, workers=8, timeout=1800,
          scorer="rosetta", only=None, work_dir=None):
    csv_path = Path(data_dir) / "dataset1.csv"
    if not csv_path.exists():
        print(f"[cpset] GATE FAILED: no ground truth at {csv_path}")
        return 0
    truth = load_dataset1(csv_path)
    work = Path(work_dir or (Path(out_path).parent / "cpset_prepared"))
    work.mkdir(parents=True, exist_ok=True)
    tasks, prep_rows, skipped = build_tasks(
        truth, data_dir, work, replicas, scorer, only=only)
    print(f"[cpset] {len(truth)} complexes in ground truth; "
          f"{len(prep_rows)} prepared; {len(skipped)} skipped", flush=True)
    for pdb, why in skipped:
        print(f"[cpset]   SKIP {pdb}: {why}")
    (work / "prep_notes.json").write_text(json.dumps(prep_rows, indent=1))

    done = set()
    if Path(out_path).exists():
        for ln in open(out_path):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("valid"):
                done.add((r.get("pdb"), r.get("replica"), r.get("scorer")))
    tasks = [t for t in tasks
             if (t["pdb"], t["replica"], t["scorer"]) not in done]
    print(f"[cpset] {len(done)} rows already done; {len(tasks)} tasks to run "
          f"({workers} workers)", flush=True)

    n_ok = n_bad = 0
    with open(out_path, "a") as fh:
        for rec in _run_isolated(tasks, workers, timeout):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if rec.get("valid"):
                n_ok += 1
            else:
                n_bad += 1
                print(f"[cpset] FAIL {rec.get('pdb')} r{rec.get('replica')}: "
                      f"{rec.get('error')}", flush=True)
            if (n_ok + n_bad) % 10 == 0:
                print(f"[cpset] {n_ok} ok / {n_bad} failed", flush=True)
    print(f"[cpset] done: {n_ok} ok, {n_bad} failed -> {out_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def aggregate(rows):
    """Per-PDB mean dG over replicas, plus the within-PDB spread (the ruler's
    own noise, which bounds what any ranking claim can mean)."""
    per = {}
    for r in rows:
        if not r.get("valid") or not isinstance(r.get("dG"), (int, float)):
            continue
        per.setdefault(r["pdb"], {"dgs": [], "meta": r})["dgs"].append(r["dG"])
    out = {}
    for pdb, d in per.items():
        dgs = d["dgs"]
        sd = st.stdev(dgs) if len(dgs) > 1 else 0.0
        out[pdb] = {
            "pdb": pdb, "dG": st.mean(dgs), "sd": sd, "n": len(dgs),
            "sem": sd / math.sqrt(len(dgs)) if len(dgs) > 1 else None,
            "pkd": d["meta"].get("pkd"), "series": d["meta"].get("series"),
            "tier": d["meta"].get("tier"),
            "chain_type": d["meta"].get("chain_type"),
            "prep": d["meta"].get("prep") or {},
        }
    return out


def prep_flags(agg):
    """Preparation caveats that touch the headline, surfaced per SERIES.

    Dropping a terminal cap changes that peptide's chemistry, so a series where
    it happened carries a caveat its rho must be read with. Silence here would
    be the `confidence_metrics` trap again: a clean-looking number whose
    cleanliness came from something nobody printed.
    """
    by_series = {}
    for pdb, a in agg.items():
        sid = a.get("series")
        if not sid:
            continue
        p = a.get("prep") or {}
        d = by_series.setdefault(sid, {"capped": [], "len_mismatch": []})
        if p.get("caps"):
            d["capped"].append(pdb)
        if (p.get("n_res_pep") is not None and p.get("pep_len_csv") is not None
                and p["n_res_pep"] != p["pep_len_csv"]):
            d["len_mismatch"].append(
                f"{pdb}({p['n_res_pep']}vs{p['pep_len_csv']})")
    return {k: v for k, v in by_series.items()
            if v["capped"] or v["len_mismatch"]}


def compute_stats(agg, truth):
    """Everything the verdict needs, as data (so it is testable without I/O)."""
    score_by_pdb = {p: a["dG"] for p, a in agg.items()}
    stats = {"n_complexes": len(agg)}

    # 1. pooled within-series pair agreement, stratified
    stats["bins"] = {}
    for b in DPKD_BINS:
        pairs = within_series_pairs(truth, min_dpkd=b, pdbs=set(agg))
        stats["bins"][b] = {
            "all": pair_agreement(pairs, score_by_pdb),
            "bb": pair_agreement([p for p in pairs if p["tier"] in ("BB", "MIX")],
                                 score_by_pdb),
        }
    stats["headline"] = stats["bins"][HEADLINE_BIN]["all"]

    # 2. per-series Spearman, NEVER pooled across series
    per_series = []
    rhos, rhos_n3 = [], []
    for s in SERIES:
        mem = [p for p in s["pdbs"] if p in agg]
        if len(mem) < 2:
            continue
        xs = [agg[p]["dG"] for p in mem]
        ys = [truth[p]["pkd"] for p in mem]
        rho = _spearman(xs, ys) if len(mem) >= 3 else None
        pear = _pearson(xs, ys) if len(mem) >= 3 else None
        pks = [truth[p]["pkd"] for p in mem]
        per_series.append({
            "id": s["id"], "tier": s["tier"], "n": len(mem),
            "rho": None if rho is None else round(rho, 3),
            "pearson": None if pear is None else round(pear, 3),
            "dpkd_range": round(max(pks) - min(pks), 2),
            "dg_spread": round(max(xs) - min(xs), 2),
        })
        if rho is not None:
            rhos.append(rho)
            rhos_n3.append(rho)
    stats["per_series"] = per_series
    stats["series_rho"] = rhos
    stats["series_rho_n3"] = rhos_n3

    # 3. cross-complex, the paper-comparable number ONLY
    pdbs = sorted(agg)
    xs = [agg[p]["dG"] for p in pdbs]
    ys = [truth[p]["pkd"] for p in pdbs]
    stats["cross_complex"] = {
        "n": len(pdbs),
        "pearson": None if len(pdbs) < 3 else round(_pearson(xs, ys) or 0, 3),
        "spearman": None if len(pdbs) < 3 else round(_spearman(xs, ys) or 0, 3),
    }

    # 4. the ruler's own noise, and resolution.
    #
    # Resolution must use the WITHIN-SERIES signal, not the spread over all 50.
    # The cross-target spread carries the pKd 3.55-10.29 dynamic range, so
    # dividing it by the within-complex SEM flatters the ruler with a contrast
    # this gate exists to exclude -- the same mistake `complex_plddt` got away
    # with until the verdict rule was fixed. The honest question is "can it
    # resolve variants of the same peptide on the same receptor", so the
    # numerator is the dG spread left AFTER removing each series' own mean.
    sems = [a["sem"] for a in agg.values() if a["sem"] is not None]
    within = st.mean(sems) if sems else None
    ss, dof = 0.0, 0
    for s in SERIES:
        mem = [p for p in s["pdbs"] if p in agg]
        if len(mem) < 2:
            continue
        dgs = [agg[p]["dG"] for p in mem]
        mu = st.mean(dgs)
        ss += sum((d - mu) ** 2 for d in dgs)
        dof += len(mem) - 1
    pooled = math.sqrt(ss / dof) if dof else None
    stats["noise"] = {
        "mean_within_pdb_sem": None if within is None else round(within, 3),
        "within_series_sd": None if pooled is None else round(pooled, 3),
        "cross_target_sd": (None if len(xs) < 2 else round(st.stdev(xs), 3)),
        "resolution": (None if not within or not pooled
                       else round(pooled / within, 2)),
        "resolution_cross_target_INFLATED": (
            None if not within or len(xs) < 2
            else round(st.stdev(xs) / within, 2)),
    }
    return stats


def report(out_path):
    rows = []
    with open(out_path) as fh:
        for ln in fh:
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
    if not rows:
        print("no rows")
        return 1
    agg = aggregate(rows)
    data_csv = Path(out_path).parent / "dataset1.csv"
    if not data_csv.exists():
        for cand in (Path(out_path).parent.parent / "cpset_raw" / "dataset1.csv",
                     Path("/zfsauton/scratch/yixiz/lp2cp_stage2/cpset_raw/dataset1.csv")):
            if cand.exists():
                data_csv = cand
                break
    truth = load_dataset1(data_csv)
    stats = compute_stats(agg, truth)
    scorers = sorted({r.get("scorer") for r in rows if r.get("valid")})

    print("=" * 78)
    print("CPSet WITHIN-TARGET cyclic-peptide accuracy gate")
    print(f"scorer(s): {', '.join(str(s) for s in scorers)}   "
          f"complexes scored: {stats['n_complexes']}/50")
    n_bad = sum(1 for r in rows if not r.get("valid"))
    print(f"failed rows: {n_bad}")
    print("sign convention: higher pKd = stronger; dG more negative = stronger;")
    print("                 a WORKING scorer gives NEGATIVE rho(dG, pKd).")
    print("=" * 78)

    print("\n--- 1. HEADLINE: within-series pair sign agreement -------------")
    print("    'does the scorer put the measured-stronger peptide at lower dG?'")
    print(f"{'|dpKd|>=':>9} {'n':>4} {'correct':>8} {'agree':>7} {'95% CI':>16}"
          f"   {'BB/MIX tier':>22}")
    for b in DPKD_BINS:
        a = stats["bins"][b]["all"]
        t = stats["bins"][b]["bb"]
        ci = (f"[{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]" if a["n_pairs"] else "-")
        bb = (f"{t['agreement']:.3f} (n={t['n_pairs']})"
              if t["n_pairs"] else "-")
        mark = "  <== headline" if b == HEADLINE_BIN else ""
        print(f"{b:>9.1f} {a['n_pairs']:>4} {a['n_correct']:>8} "
              f"{(a['agreement'] if a['agreement'] is not None else 0):>7.3f} "
              f"{ci:>16}   {bb:>22}{mark}")
    print("    (ties count as failures; a coin flip is 0.500)")

    print("\n--- 2. Per-series Spearman rho(dG, pKd) ------------------------")
    print("    negative = working. Never pooled across series: the cross-target")
    print("    dynamic range (pKd 3.55-10.29) would inflate it.")
    print(f"{'series':<16} {'tier':>5} {'n':>3} {'rho':>7} {'pearson':>8} "
          f"{'dpKd rng':>9} {'dG spread':>10}")
    for s in stats["per_series"]:
        print(f"{s['id']:<16} {s['tier']:>5} {s['n']:>3} "
              f"{(s['rho'] if s['rho'] is not None else float('nan')):>7.3f} "
              f"{(s['pearson'] if s['pearson'] is not None else float('nan')):>8.3f} "
              f"{s['dpkd_range']:>9.2f} {s['dg_spread']:>10.2f}")
    if stats["series_rho"]:
        print(f"{'MEAN':<16} {'':>5} {'':>3} "
              f"{st.mean(stats['series_rho']):>7.3f}")

    print("\n--- 3. Cross-complex (the PAPER-COMPARABLE number only) -------")
    cc = stats["cross_complex"]
    print(f"    ours      n={cc['n']:>3}  Pearson {cc['pearson']}  "
          f"Spearman {cc['spearman']}")
    for name, val in PAPER_REF.items():
        print(f"    paper     {name:<54} {val:+.3f}")
    print("    NB: the paper reports NO within-target result, so its numbers")
    print("        calibrate row 3 only -- never rows 1-2.")

    print("\n--- 4. The ruler's own noise ----------------------------------")
    nz = stats["noise"]
    print(f"    mean within-complex SEM over replicas    : "
          f"{nz['mean_within_pdb_sem']} REU")
    print(f"    WITHIN-SERIES dG SD (series means removed): "
          f"{nz['within_series_sd']} REU")
    print(f"    RESOLUTION (within-series SD / SEM)      : {nz['resolution']}"
          "   <-- below ~1 nothing is rankable")
    print(f"    [cross-target SD {nz['cross_target_sd']} REU would give "
          f"{nz['resolution_cross_target_INFLATED']} -- INFLATED by the")
    print("     pKd 3.55-10.29 range across different targets; not the question]")

    flags = prep_flags(agg)
    print("\n--- 5. Preparation caveats by series --------------------------")
    if not flags:
        print("    none: no caps dropped, no peptide-length disagreements.")
    for sid, f in sorted(flags.items()):
        bits = []
        if f["capped"]:
            bits.append(f"caps dropped on {len(f['capped'])} member(s): "
                        f"{','.join(sorted(f['capped']))}")
        if f["len_mismatch"]:
            bits.append("structure/CSV peptide length differs: "
                        f"{','.join(f['len_mismatch'])}")
        print(f"    {sid} [{TIER_OF.get(sid, '?')}]: {'; '.join(bits)}")
    bb_clean = all(TIER_OF.get(s) not in ("BB", "MIX") for s in flags)
    print(f"    BB/MIX tier (the head-to-tail claim) free of caveats: "
          f"{'YES' if bb_clean else 'NO'}")

    label, reasons = verdict(stats)
    print("\n" + "=" * 78)
    print(f"PRE-REGISTERED VERDICT: {label}")
    for r in reasons:
        print(f"  - {r}")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# Fetch / regroup
# ---------------------------------------------------------------------------


def fetch(data_dir):
    """Download the benchmark. Kept as a subcommand so the provenance of the
    ground truth is in the code, not in somebody's shell history."""
    import subprocess
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for url, name in ((DATASET1_CSV_URL, "dataset1.csv"),
                      (DATASET1_TAR_URL, "dataset1.tar.bz2")):
        dest = d / name
        if dest.exists() and dest.stat().st_size:
            print(f"[cpset] have {dest}")
            continue
        print(f"[cpset] fetching {url}")
        subprocess.run(["curl", "-sfL", "-o", str(dest), url], check=True)
    tar = d / "dataset1.tar.bz2"
    if tar.exists() and not (d / "dataset1").is_dir():
        print(f"[cpset] extracting {tar}")
        subprocess.run(["tar", "-xjf", str(tar), "-C", str(d)], check=True)
    n = len(list((d / "dataset1").glob("*"))) if (d / "dataset1").is_dir() else 0
    print(f"[cpset] {n} complex directories under {d / 'dataset1'}")
    return 0


def regroup_cmd(data_dir, threshold=0.95):
    """Re-derive the series grouping from the structures and DIFF it against the
    frozen SERIES table. A silent regrouping would silently move the headline."""
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    sigs = {}
    for pdb in sorted(truth):
        paths = complex_paths(data_dir, pdb)
        if paths is None:
            print(f"[cpset] missing structures for {pdb}")
            continue
        cs = chain_seqs(paths[0])
        if cs:
            sigs[pdb] = max(cs.values(), key=len)
    groups, borderline = regroup(sigs, threshold)
    derived = {}
    for mem in groups.values():
        if len(mem) < 2:
            continue
        key = frozenset(mem)
        derived[key] = sorted(mem)
    frozen = {frozenset(s["pdbs"]): sorted(s["pdbs"]) for s in SERIES}
    print(f"[cpset] {len(sigs)} receptors -> {len(groups)} groups; "
          f"{len(derived)} series of n>=2 (frozen table has {len(frozen)})")
    only_new = [v for k, v in derived.items() if k not in frozen]
    only_old = [v for k, v in frozen.items() if k not in derived]
    if not only_new and not only_old:
        print("[cpset] OK: derived grouping MATCHES the frozen SERIES table.")
    else:
        print("[cpset] MISMATCH -- the frozen table must be updated deliberately:")
        for v in only_new:
            print(f"    derived but not frozen: {v}")
        for v in only_old:
            print(f"    frozen but not derived: {v}")
    print(f"[cpset] borderline identities in (0.55, {threshold}) -- "
          f"{len(borderline)} (0 means the grouping is unambiguous):")
    for a, b, r in sorted(borderline, key=lambda x: -x[2])[:10]:
        print(f"    {a} ~ {b}  {r}")
    tot = sum(len(v) * (len(v) - 1) // 2 for v in derived.values())
    print(f"[cpset] total within-series pairs = {tot}")
    return 0 if (not only_new and not only_old) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--regroup", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", metavar="JSONL")
    ap.add_argument("--data-dir",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/cpset_raw")
    ap.add_argument("--out", default="cpset_dg.jsonl")
    ap.add_argument("--work-dir")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--scorer", default="rosetta", choices=("rosetta",),
                    help="MM/PBSA lands here once the Amber gate opens")
    ap.add_argument("--only", help="comma-separated PDB codes (debug)")
    ap.add_argument("--identity-threshold", type=float, default=0.95)
    args = ap.parse_args()

    if args.fetch:
        return fetch(args.data_dir)
    if args.regroup:
        return regroup_cmd(args.data_dir, args.identity_threshold)
    if args.report:
        return report(args.report)
    if args.score:
        only = ({s.strip().upper() for s in args.only.split(",")}
                if args.only else None)
        return score(args.data_dir, args.out, replicas=args.replicas,
                     workers=args.workers, timeout=args.timeout,
                     scorer=args.scorer, only=only, work_dir=args.work_dir)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
