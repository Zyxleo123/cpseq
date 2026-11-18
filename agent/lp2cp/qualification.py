"""The frozen qualification function, and the defect-class assignment it implies.

Invariant 2 of the design doc: the qualification function is frozen, hashed and
versioned before the run. The agent may change what it proposes and what it
accepts locally; it may never change what qualifies.

That is made mechanical here rather than promised. `QUAL_HASH` is a digest over
the threshold table AND over the source text of the predicate, so editing
either one changes the hash; every campaign record carries the hash it ran
under, and `assert_frozen()` refuses to start a campaign whose on-disk
qualification manifest does not match the code. A run whose thresholds moved
mid-campaign is therefore not merely discouraged, it does not execute.

Thresholds are the design doc's worked example, taken literally, with one
deliberate change: the BSA criterion is a fraction of the native complex's
MEASURED buried area rather than the doc's absolute 550 A^2. The doc's number
came from one 12-mer; this benchmark spans peptides of 5-16 residues whose
native BSA ranges over roughly an order of magnitude, so an absolute floor
would make short peptides unqualifiable by construction and long ones
qualify for free. The fraction is declared here, before the run, and is part
of the hash.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from .enums import DEFECT_PRECEDENCE, DefectClass
from .types import ScoreCard

QUAL_VERSION = "qual-v1"

# ---------------------------------------------------------------------------
# The thresholds. Frozen. Part of the hash.
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    # ipTM >= 0.80 AND >= 3/5 seed mode-consensus
    "iptm_min": 0.80,
    "consensus_frac_min": 0.60,
    # BSA floor as a fraction of the native complex's measured buried area
    "bsa_frac_of_native_min": 0.80,
    # Rosetta ddG_bind <= -30 REU, expressed per the doc
    "ddg_bind_max": -30.0,
    # max per-residue backbone strain <= 4.0 REU
    "max_strain_max": 4.0,
    # free peptide: bound conformer >= 5% of the top-10 kcal/mol window
    "free_conformer_frac_min": 0.05,
}

# Which objective maps to which defect class. One-to-one, so a candidate's
# failing metrics determine its TRUE class with no judgement call -- that is
# what makes "did the LLM recover the class" a measurable question.
OBJECTIVE_TO_DEFECT: dict[str, DefectClass] = {
    "max_strain": DefectClass.BACKBONE_STRAIN_AT_CLOSURE,
    "ddg_bind": DefectClass.INTERFACE_CONTACT_LOSS,
    "bsa": DefectClass.BURIAL_DEFICIT,
    "free_conformer_frac": DefectClass.FREE_CONFORMER_MISMATCH,
    "iptm": DefectClass.FOLD_CONFIDENCE_LOW,
}


@dataclass(frozen=True)
class Verdict:
    """Why a candidate did or did not qualify, with the margins that decided it."""

    qualifies: bool
    failed: tuple[str, ...]
    margins: dict[str, float]          # normalised shortfall per objective, >0 = failing
    defect_class: DefectClass
    defect_locus: tuple[int, ...]

    def to_row(self) -> dict:
        return {
            "qualifies": self.qualifies,
            "failed": list(self.failed),
            "margins": {k: round(v, 4) for k, v in self.margins.items()},
            "defect_class": self.defect_class.value,
            "defect_locus": list(self.defect_locus),
        }


def _margins(card: ScoreCard, native_bsa: float, k_budget: int) -> dict[str, float]:
    """Normalised shortfall per objective. >0 means failing, by that fraction.

    Normalising by the threshold's own magnitude is what lets "strain 11.2
    against a limit of 4.0" be compared with "BSA 495 against a floor of 550"
    when two objectives fail at once. Without it the tie-break would be decided
    by whichever metric happens to have larger units.
    """
    t = THRESHOLDS
    bsa_floor = t["bsa_frac_of_native_min"] * max(native_bsa, 1.0)
    cons_frac = card.seed_consensus / max(card.n_seeds, 1)
    return {
        "iptm": (t["iptm_min"] - card.iptm) / t["iptm_min"],
        "consensus": (t["consensus_frac_min"] - cons_frac) / t["consensus_frac_min"],
        "bsa": (bsa_floor - card.bsa) / bsa_floor,
        # ddg is negative-is-better; a shortfall is being less negative than
        # the ceiling. Normalised by |ceiling| so the sign convention does not
        # leak into the comparison.
        "ddg_bind": (card.ddg_bind - t["ddg_bind_max"]) / abs(t["ddg_bind_max"]),
        "max_strain": (card.max_strain - t["max_strain_max"]) / t["max_strain_max"],
        "free_conformer_frac": (
            (t["free_conformer_frac_min"] - card.free_conformer_frac)
            / t["free_conformer_frac_min"]
        ),
        # The edit-distance clause. Hard, not scored: a design over budget is
        # not a worse design, it is not a design.
        "edit_distance": float(card.edit_distance - k_budget) / max(k_budget, 1),
    }


def qualify(card: ScoreCard, native_bsa: float, k_budget: int) -> Verdict:
    """The frozen predicate. Every arm, every controller, every method.

    Returns the full verdict rather than a bool: the margins are what the
    defect-class assignment and the tie-break rule read, and the scorer needs
    them to report WHY a campaign failed, not only that it did.
    """
    m = _margins(card, native_bsa, k_budget)
    failed = tuple(k for k, v in m.items() if v > 1e-9)
    qualifies = not failed

    # Defect class. Consensus is folded into FOLD_CONFIDENCE_LOW (it is the
    # same objective measured a second way) and edit_distance has no defect
    # class at all -- an over-budget candidate is a compile failure, which the
    # execution service catches before scoring, so it cannot reach here from a
    # legitimate path.
    scored = {
        "max_strain": m["max_strain"],
        "ddg_bind": m["ddg_bind"],
        "bsa": m["bsa"],
        "free_conformer_frac": m["free_conformer_frac"],
        "iptm": max(m["iptm"], m["consensus"]),
    }
    worst = [(k, v) for k, v in scored.items() if v > 1e-9]
    if not worst:
        dc = DefectClass.NO_DEFECT
    else:
        top = max(v for _, v in worst)
        # Level 1: largest normalised margin. Level 2: declared precedence.
        # `ties` is everything within 1% of the top margin, so floating-point
        # noise cannot decide a class.
        ties = [OBJECTIVE_TO_DEFECT[k] for k, v in worst if v >= top - 0.01 * max(top, 1e-9)]
        dc = min(ties, key=DEFECT_PRECEDENCE.index)

    # Locus. Only the strain defect has a locus the scorecard knows directly;
    # for the others the locus is the interface, which the controller fills in
    # from the parent's diagnostics rather than being invented here.
    locus = card.strain_locus if dc == DefectClass.BACKBONE_STRAIN_AT_CLOSURE else ()
    return Verdict(qualifies, failed, m, dc, tuple(locus))


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def _predicate_source() -> str:
    """Source text of everything that decides a verdict.

    Hashing the source and not only the numbers is the point: a change to how
    `_margins` normalises, or to the tie-break, alters what qualifies just as
    much as moving a threshold, and a hash over the threshold dict alone would
    not notice.
    """
    return "\n".join(
        inspect.getsource(fn) for fn in (_margins, qualify)
    ) + "\n".join(f"{d.value}" for d in DEFECT_PRECEDENCE)


def manifest() -> dict:
    src = _predicate_source()
    body = {
        "version": QUAL_VERSION,
        "thresholds": THRESHOLDS,
        "objective_to_defect": {k: v.value for k, v in OBJECTIVE_TO_DEFECT.items()},
        "defect_precedence": [d.value for d in DEFECT_PRECEDENCE],
        "predicate_sha256": hashlib.sha256(src.encode()).hexdigest(),
    }
    body["qualification_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return body


QUAL_HASH: str = manifest()["qualification_hash"]


def freeze_to(path: str | Path) -> dict:
    """Write the manifest. Called once, by the benchmark build job."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    man = manifest()
    p.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return man


def assert_frozen(path: str | Path) -> dict:
    """Refuse to run if the on-disk manifest and the code disagree.

    This is the mechanism behind invariant 2. A campaign that started under one
    qualification function and finished under another produces rows that cannot
    be pooled, and the failure is invisible in the results -- so it is a hard
    stop at startup rather than a warning in a log nobody reads.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no frozen qualification manifest at {p}. Run the benchmark build "
            f"job first; it calls freeze_to() before anything is scored."
        )
    on_disk = json.loads(p.read_text())
    live = manifest()
    if on_disk.get("qualification_hash") != live["qualification_hash"]:
        raise RuntimeError(
            "QUALIFICATION FUNCTION CHANGED since the benchmark was frozen.\n"
            f"  on disk: {on_disk.get('qualification_hash')} "
            f"({on_disk.get('version')})\n"
            f"  in code: {live['qualification_hash']} ({live['version']})\n"
            "Rows produced under the two are not poolable. Either revert the "
            "change, or rebuild the benchmark under a new version string and "
            "score the two separately."
        )
    return live
