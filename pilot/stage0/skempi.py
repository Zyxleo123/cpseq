"""SKEMPI 2.0 -> tidy mutation records.

One job: turn the semicolon-delimited release CSV into rows that carry a signed
binding-energy change and an explicit list of mutated positions. Everything
downstream (decision construction, scoring) reads only what this module emits.

Sign convention
---------------
ddG = dG_mut - dG_wt = RT * ln(Kd_mut / Kd_wt)

So ddG < 0 means the mutant binds TIGHTER than wild type, and ddG > 0 means the
mutation costs binding energy. Stage 0 is a damage-minimisation task, so the
oracle MINIMISES ddG.

Averaging
---------
A (complex, mutation) pair measured by two different references is averaged in
ddG space, not in Kd space. ddG is already a log of the dissociation constant,
so the mean of two ddG values is the log of the geometric mean of the two Kd
ratios -- the standard combination for energies. Averaging Kd first and taking
the log afterwards would be averaging a nonlinear transform, which biases the
result toward whichever measurement had the larger Kd.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

R_KCAL = 1.9872036e-3  # kcal / (mol K)
DEFAULT_T = 298.0

# SKEMPI's interface-location classes. COR/SUP/RIM sit in the interface;
# INT (protein interior) and SUR (non-interface surface) do not.
INTERFACE_CLASSES = ("COR", "SUP", "RIM")

# "LI38G" -> wt=L, chain=I, position=38, mut=G.  The position may carry an
# insertion code ("52A"), and SKEMPI occasionally uses a negative number.
_MUT_RE = re.compile(r"^([A-Za-z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Za-z])$")

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


@dataclass(frozen=True)
class Mutation:
    chain: str
    position: str          # string: may carry an insertion code
    wt: str                # one-letter wild-type residue
    mut: str               # one-letter mutant residue
    pdb_chain: str = ""    # same fields in the ORIGINAL pdb numbering
    pdb_position: str = ""

    @property
    def site(self) -> tuple[str, str]:
        """Identity of the position, independent of what it was mutated to."""
        return (self.chain, self.position)

    @property
    def pdb_site(self) -> tuple[str, str]:
        return (self.pdb_chain or self.chain, self.pdb_position or self.position)

    def __str__(self) -> str:
        return f"{self.wt}{self.chain}{self.position}{self.mut}"


@dataclass
class Record:
    complex_id: str                 # SKEMPI "#Pdb", e.g. "1JTG_A_B"
    pdb_id: str
    chains1: str
    chains2: str
    mutations: list[Mutation]
    ddg: float
    locations: tuple[str, ...]      # per-mutation COR/SUP/RIM/INT/SUR
    hold_out_type: str
    protein1: str
    protein2: str
    method: str
    reference: str
    temperature: float
    n_sources: int = 1
    extras: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.mutations)

    @property
    def site_set(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({m.site for m in self.mutations}))

    @property
    def at_interface(self) -> bool:
        return any(loc in INTERFACE_CLASSES for loc in self.locations)

    @property
    def all_at_interface(self) -> bool:
        return bool(self.locations) and all(
            loc in INTERFACE_CLASSES for loc in self.locations
        )


def _pos_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def _parse_mutations(cleaned: str, pdb: str) -> list[Mutation] | None:
    """Parse the paired `Mutation(s)_cleaned` / `Mutation(s)_PDB` fields.

    Both numbering schemes are kept. `cleaned` is the canonical key used for
    grouping (it is what SKEMPI's own structures use); `pdb` is what a file
    downloaded from RCSB is numbered with, so it is the one the structural
    feature pass must index by. They differ -- 1CSE_E_I's LI38G in cleaned
    numbering is LI45G in PDB numbering -- and mixing them up silently
    annotates the wrong residue.
    """
    cleaned_parts = [p.strip() for p in str(cleaned).split(",") if p.strip()]
    pdb_parts = [p.strip() for p in str(pdb).split(",") if p.strip()]
    if not cleaned_parts:
        return None
    if len(pdb_parts) != len(cleaned_parts):
        pdb_parts = cleaned_parts  # fall back rather than mis-pair

    out: list[Mutation] = []
    for c_tok, p_tok in zip(cleaned_parts, pdb_parts):
        cm = _MUT_RE.match(c_tok)
        pm = _MUT_RE.match(p_tok)
        if cm is None:
            return None
        wt, chain, position, mut = cm.groups()
        if pm is not None and pm.group(1) == wt and pm.group(4) == mut:
            pdb_chain, pdb_position = pm.group(2), pm.group(3)
        else:
            pdb_chain, pdb_position = chain, position
        # Insertion codes: SKEMPI writes them lower case ("100b"), the PDB
        # files upper case ("100B"). Measured on the first full pass, that cost
        # 48 of 3832 candidate sites across 17 decisions -- they resolved to no
        # residue at all, so those candidates reached the agent with every
        # structural column blank. Normalised here, at the one place both
        # numbering schemes are parsed, rather than at each of the five
        # consumers that index a feature table by position.
        position = position.upper()
        pdb_position = pdb_position.upper()
        out.append(
            Mutation(
                chain=chain, position=position, wt=wt.upper(), mut=mut.upper(),
                pdb_chain=pdb_chain, pdb_position=pdb_position,
            )
        )
    return out


def load_records(csv_path: str | Path) -> tuple[list[Record], dict]:
    """Return parsed records plus a provenance/rejection report.

    The report is part of the output, not a log line: a silently shrinking
    denominator is how a benchmark ends up measuring its own filters.
    """
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    report = {
        "csv_rows": int(len(df)),
        "rejected_no_affinity": 0,
        "rejected_unparsable_mutation": 0,
        "rejected_repeat_site": 0,
    }

    staged: dict[tuple, list[Record]] = {}
    for row in df.to_dict("records"):
        kd_mut = _pos_float(row.get("Affinity_mut_parsed"))
        kd_wt = _pos_float(row.get("Affinity_wt_parsed"))
        if kd_mut is None or kd_wt is None:
            report["rejected_no_affinity"] += 1
            continue

        muts = _parse_mutations(
            row.get("Mutation(s)_cleaned", ""), row.get("Mutation(s)_PDB", "")
        )
        if muts is None:
            report["rejected_unparsable_mutation"] += 1
            continue
        if len({m.site for m in muts}) != len(muts):
            # Two mutations listed at one position: the position set would not
            # have size k and the budget bookkeeping stops meaning anything.
            report["rejected_repeat_site"] += 1
            continue

        temperature = _pos_float(row.get("Temperature")) or DEFAULT_T
        ddg = R_KCAL * temperature * math.log(kd_mut / kd_wt)

        complex_id = str(row["#Pdb"])
        bits = complex_id.split("_")
        pdb_id = bits[0]
        chains1 = bits[1] if len(bits) > 1 else ""
        chains2 = bits[2] if len(bits) > 2 else ""

        rec = Record(
            complex_id=complex_id,
            pdb_id=pdb_id,
            chains1=chains1,
            chains2=chains2,
            mutations=muts,
            ddg=ddg,
            locations=tuple(
                p.strip() for p in str(row.get("iMutation_Location(s)", "")).split(",")
            ),
            hold_out_type=str(row.get("Hold_out_type", "")),
            protein1=str(row.get("Protein 1", "")),
            protein2=str(row.get("Protein 2", "")),
            method=str(row.get("Method", "")),
            reference=str(row.get("Reference", "")),
            temperature=temperature,
        )
        key = (complex_id, tuple(str(m) for m in muts))
        staged.setdefault(key, []).append(rec)

    # Collapse repeat measurements of the same mutation (see module docstring).
    records: list[Record] = []
    for key, group in staged.items():
        head = group[0]
        if len(group) > 1:
            head.ddg = sum(g.ddg for g in group) / len(group)
            head.n_sources = len(group)
            head.reference = ";".join(sorted({g.reference for g in group}))
        records.append(head)

    report["records"] = len(records)
    report["records_collapsed_from_duplicates"] = sum(
        1 for r in records if r.n_sources > 1
    )
    report["complexes"] = len({r.complex_id for r in records})
    return records, report
