"""The typed objects that travel between the agent, the service and the backend.

The design doc is explicit that the diagnosis is a typed object and not prose,
for a reason that is about attributability rather than tidiness: free text
means the LLM re-derives the mask on every call, so a failure cannot be
attributed to bad diagnosis versus bad compilation -- and the "aggregate
scores only" arm stops being a well-defined difference in fields present on
one object.

Everything here is frozen and JSON-round-trippable. A campaign trace is a list
of these, so the whole run is replayable from the trace alone: the scorer never
re-runs the agent, and a figure can be redrawn with no GPU and no API key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .enums import (
    Arm,
    DefectClass,
    Operator,
    RepairOutcome,
    StepStatus,
    parse_enum,
)


def _canon(obj: Any) -> str:
    """Canonical JSON for hashing. Sorted keys, no whitespace drift."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()[:n]


# ---------------------------------------------------------------------------
# The parent problem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidueDiag:
    """Measured per-residue diagnostics for one peptide position.

    Every field here is computed from real coordinates (see
    `envbuild/diagnostics.py`) -- no field is a model prediction. `strain_rel`
    is the one derived quantity: a Ramachandran-basin deviation, in REU-like
    units, which stands in for Rosetta's per-residue backbone strain term.
    """

    pos: int                      # 1-based index along the peptide
    wt: str                       # one-letter code of the residue there
    dsasa: float                  # A^2 of side-chain surface buried on binding
    buried_frac: float            # fraction of this residue's surface buried
    contacts_5a: int              # heavy-atom contacts to the receptor < 5 A
    polar_3p5a: int               # N/O pairs to the receptor < 3.5 A
    min_dist: float               # closest approach to the receptor, A
    phi: float | None             # backbone dihedral, degrees
    psi: float | None
    strain_rel: float             # Ramachandran-basin deviation, REU-like
    ring_dist: int                # positions along the ring to the closure bond
    is_anchor: bool               # participates in the cyclisation bond
    hbond_partner: str | None     # receptor residue of the best polar pair


@dataclass(frozen=True)
class ParentProblem:
    """One failed cyclic parent: the thing an iteration starts from.

    `lp_sequence` is the linear parent the whole campaign is measured against
    (`edit distance to LP <= k`), and `cp_sequence` is the current cyclic
    candidate. At iteration 0 they differ by however many substitutions the
    unguided generation run already spent.
    """

    parent_id: str
    pdb_id: str
    receptor_name: str
    cluster_id: str               # by-TARGET grouping; the held-out split key
    split: str                    # dev / test, declared before any tuning
    lp_sequence: str
    cp_sequence: str
    cyclization: str              # head_tail, disulfide, ...
    k_budget: int                 # substitutions allowed relative to LP
    k_spent: int                  # already spent by iteration 0
    anchors: tuple[int, ...]      # positions forming the cyclisation bond
    residues: tuple[ResidueDiag, ...]
    native_bsa: float             # measured BSA of the real complex, A^2
    true_defect_class: DefectClass  # LABEL -- never shown to the agent
    scorecard0: "ScoreCard"       # the parent's own evaluation

    @property
    def n_pos(self) -> int:
        return len(self.cp_sequence)

    @property
    def k_remaining(self) -> int:
        return max(0, self.k_budget - self.k_spent)

    def public(self) -> dict:
        """Everything the agent may see. `true_defect_class` is NOT in it.

        Label separation is enforced here rather than by convention: the
        campaign runner only ever serialises this, so a prompt cannot leak the
        answer even if a prompt template asks for a field by name.
        """
        d = asdict(self)
        d.pop("true_defect_class", None)
        return d


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreCard:
    """Every metric the frozen qualification function reads.

    `provenance` names what produced these numbers and is carried into every
    result row. In this build it is always a `surrogate_v1:<hash>` string --
    the numbers come from the declared response model in `envbuild/surrogate.py`
    applied to real measured structure, NOT from Boltz-2 or Rosetta. A reader
    who sees a ddG in REU must be able to tell from the row itself whether a
    physics engine computed it, so the field is required and never defaulted.
    """

    iptm: float
    seed_consensus: int           # seeds agreeing with the modal pose
    n_seeds: int
    bsa: float                    # A^2
    ddg_bind: float               # REU-like; more negative is better
    max_strain: float             # REU-like, worst per-residue
    strain_locus: tuple[int, ...] # where that worst strain sits
    free_conformer_frac: float    # bound-like population in the free ensemble
    edit_distance: int            # to the LINEAR parent
    hbonds_retained: tuple[int, ...]  # parent H-bond positions still present
    provenance: str

    def to_row(self) -> dict:
        d = asdict(self)
        d["strain_locus"] = list(self.strain_locus)
        d["hbonds_retained"] = list(self.hbonds_retained)
        return d


# ---------------------------------------------------------------------------
# The three LLM-owned objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosisRecord:
    """Step 2's output. The doc's schema, field for field."""

    candidate: str
    defect_class: DefectClass
    defect_locus: tuple[int, ...]
    preserved_features: tuple[str, ...]  # e.g. "IDENTITY:4", "HBOND:7", "BSA_FLOOR"
    evidence_refs: tuple[str, ...]       # which diagnostic rows were used
    confidence: float

    @staticmethod
    def parse(obj: dict, n_pos: int, candidate: str) -> "DiagnosisRecord":
        """Strict parse with a closed enum and a bounded locus.

        A locus position outside the peptide is a parse failure, not a clamp: a
        silently clamped locus would let a model that misread the numbering
        score as if it had read them.
        """
        dc = parse_enum(DefectClass, obj.get("defect_class", ""))
        locus = tuple(int(p) for p in (obj.get("defect_locus") or []))
        bad = [p for p in locus if not 1 <= p <= n_pos]
        if bad:
            raise ValueError(f"defect_locus out of range 1..{n_pos}: {bad}")
        conf = obj.get("confidence", 0.5)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            raise ValueError(f"confidence not a number: {conf!r}") from None
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence outside [0,1]: {conf}")
        return DiagnosisRecord(
            candidate=candidate,
            defect_class=dc,
            defect_locus=locus,
            preserved_features=tuple(str(x) for x in (obj.get("preserved_features") or [])),
            evidence_refs=tuple(str(x) for x in (obj.get("evidence_refs") or [])),
            confidence=conf,
        )


@dataclass(frozen=True)
class PreservationContract:
    """What a repair is not allowed to spend. Arm 3's whole content.

    A contract is per-lineage and agent-declared, which is what distinguishes
    it from the frozen qualification function: the agent may change what it
    proposes and what it accepts locally, and may never change what qualifies.
    """

    fixed_identities: tuple[int, ...] = ()
    required_hbonds: tuple[int, ...] = ()
    bsa_floor: float = 0.0
    min_hbonds: int = 0

    def holds(self, card: ScoreCard, seq: str, parent_seq: str) -> tuple[bool, tuple[str, ...]]:
        """Does this candidate honour the contract? Returns (ok, violations)."""
        bad: list[str] = []
        for p in self.fixed_identities:
            i = p - 1
            if not (0 <= i < len(seq) and 0 <= i < len(parent_seq)):
                bad.append(f"IDENTITY:{p}:out_of_range")
            elif seq[i] != parent_seq[i]:
                bad.append(f"IDENTITY:{p}:{parent_seq[i]}->{seq[i]}")
        kept = set(card.hbonds_retained)
        missing = [p for p in self.required_hbonds if p not in kept]
        if missing:
            bad.append("HBOND_LOST:" + ",".join(str(p) for p in missing))
        if len(kept) < self.min_hbonds:
            bad.append(f"MIN_HBONDS:{len(kept)}<{self.min_hbonds}")
        if self.bsa_floor and card.bsa < self.bsa_floor:
            bad.append(f"BSA_FLOOR:{card.bsa:.0f}<{self.bsa_floor:.0f}")
        return (not bad), tuple(bad)

    @staticmethod
    def parse(obj: dict, n_pos: int) -> "PreservationContract":
        def poslist(key):
            out = []
            for p in obj.get(key) or []:
                p = int(p)
                if not 1 <= p <= n_pos:
                    raise ValueError(f"{key} position out of range 1..{n_pos}: {p}")
                out.append(p)
            return tuple(sorted(set(out)))

        return PreservationContract(
            fixed_identities=poslist("fixed_identities"),
            required_hbonds=poslist("required_hbonds"),
            bsa_floor=float(obj.get("bsa_floor") or 0.0),
            min_hbonds=int(obj.get("min_hbonds") or 0),
        )


@dataclass(frozen=True)
class DesignSpec:
    """Step 3's output: a constraint set, not a molecule.

    `open_positions` is where identities may change; `rebuild_segment` is the
    contiguous span whose backbone may move; `fixed_positions` is what must not
    move at all. The budget split is implicit in how many positions are opened,
    and step 4 enforces it at the filter stage.
    """

    operator: Operator
    open_positions: tuple[int, ...] = ()
    fixed_positions: tuple[int, ...] = ()
    rebuild_segment: tuple[int, int] | None = None   # inclusive 1-based (lo, hi)
    n_eval_seeds: int = 5
    contract: PreservationContract | None = None
    rationale: str = ""
    relax_clause: str = ""      # only meaningful for RELAX_CONTRACT
    relax_to: float | None = None

    @property
    def spec_hash(self) -> str:
        return sha(
            {
                "op": self.operator.value,
                "open": sorted(self.open_positions),
                "fixed": sorted(self.fixed_positions),
                "seg": list(self.rebuild_segment) if self.rebuild_segment else None,
                "seeds": self.n_eval_seeds,
            }
        )

    @staticmethod
    def parse(obj: dict, n_pos: int, max_seeds: int) -> "DesignSpec":
        op = parse_enum(Operator, obj.get("operator", ""))

        def poslist(key):
            out = []
            for p in obj.get(key) or []:
                p = int(p)
                if not 1 <= p <= n_pos:
                    raise ValueError(f"{key} position out of range 1..{n_pos}: {p}")
                out.append(p)
            return tuple(sorted(set(out)))

        seg = obj.get("rebuild_segment")
        segment = None
        if seg:
            if isinstance(seg, dict):
                lo, hi = int(seg.get("start", 0)), int(seg.get("end", 0))
            else:
                lo, hi = int(seg[0]), int(seg[-1])
            if not (1 <= lo <= hi <= n_pos):
                raise ValueError(f"rebuild_segment {lo}-{hi} outside 1..{n_pos}")
            segment = (lo, hi)

        seeds = int(obj.get("n_eval_seeds") or 5)
        if not 1 <= seeds <= max_seeds:
            raise ValueError(f"n_eval_seeds {seeds} outside 1..{max_seeds}")

        contract = None
        if obj.get("contract"):
            contract = PreservationContract.parse(obj["contract"], n_pos)

        relax_to = obj.get("relax_to")
        return DesignSpec(
            operator=op,
            open_positions=poslist("open_positions"),
            fixed_positions=poslist("fixed_positions"),
            rebuild_segment=segment,
            n_eval_seeds=seeds,
            contract=contract,
            rationale=str(obj.get("rationale") or "")[:600],
            relax_clause=str(obj.get("relax_clause") or "")[:200],
            relax_to=None if relax_to is None else float(relax_to),
        )


@dataclass(frozen=True)
class AllocationDecision:
    """Step 8's output: what the next block of budget buys.

    `EDIT` spends a substitution, `ESCALATE` spends evaluation depth instead
    (the doc's iteration 3 -- the non-greedy decision with the weakest
    prior-art support), `RELAX` records a RelaxationEvent, `STOP` terminates.
    """

    action: str                   # EDIT | ESCALATE | RELAX | STOP
    rationale: str = ""
    n_eval_seeds: int = 5
    ACTIONS = ("EDIT", "ESCALATE", "RELAX", "STOP")

    @staticmethod
    def parse(obj: dict, max_seeds: int) -> "AllocationDecision":
        a = str(obj.get("action", "")).strip().upper()
        if a not in AllocationDecision.ACTIONS:
            raise ValueError(
                f"action {a!r} not one of {AllocationDecision.ACTIONS}"
            )
        seeds = int(obj.get("n_eval_seeds") or 5)
        if not 1 <= seeds <= max_seeds:
            raise ValueError(f"n_eval_seeds {seeds} outside 1..{max_seeds}")
        return AllocationDecision(
            action=a, rationale=str(obj.get("rationale") or "")[:600], n_eval_seeds=seeds
        )


# ---------------------------------------------------------------------------
# Results of acting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One design the backend returned. `edits` is relative to the LINEAR parent."""

    cand_id: str
    parent_id: str
    sequence: str
    edits: tuple[tuple[int, str, str], ...]   # (pos, from, to)
    operator: Operator
    locus: tuple[int, ...]
    backend: str
    intervention_id: str = ""

    @property
    def n_edits(self) -> int:
        return len(self.edits)


@dataclass(frozen=True)
class RelaxationEvent:
    """An explicit, justified weakening of a preservation contract.

    The doc: relaxation events are a result, not plumbing. Their rate against k
    is direct evidence for the closure-interface conflict the whole design
    presumes exists, and is probably the cleanest quantitative support for that
    premise. So every one is recorded with what changed, from what to what, and
    which evidence justified it -- and the lineage BRANCHES rather than the
    contract being edited in place, so a qualifying design always names the
    contract it qualified under.
    """

    iteration: int
    clause: str
    from_value: str
    to_value: str
    justification: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepRecord:
    """One LLM-owned step: what it cost, what it returned, how it ended."""

    step: str                     # DIAGNOSE | SPECIFY | ALLOCATE
    status: StepStatus
    n_calls: int = 0
    n_retries: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_used: bool = False
    detail: str = ""
    raw: str = ""


@dataclass
class IterationRecord:
    """The full trace of one pass through the eight steps."""

    iteration: int
    candidate_in: str
    steps: list[StepRecord] = field(default_factory=list)
    diagnosis: DiagnosisRecord | None = None
    spec: DesignSpec | None = None
    allocation: AllocationDecision | None = None
    compile_ok: bool = True
    compile_reason: str = ""
    overwide_mask: bool = False
    n_proposed: int = 0
    n_budget_filtered: int = 0
    n_scope_leaks: int = 0
    n_scored: int = 0
    best_cand: str | None = None
    best_card: ScoreCard | None = None
    outcome: RepairOutcome = RepairOutcome.NO_CHANGE
    contract_violations: tuple[str, ...] = ()
    relaxation: RelaxationEvent | None = None
    k_spent_after: int = 0
    seeds_spent: int = 0
    notes: str = ""

    def to_row(self) -> dict:
        return {
            "iteration": self.iteration,
            "candidate_in": self.candidate_in,
            "steps": [asdict(s) | {"status": s.status.value} for s in self.steps],
            "diagnosis": (
                asdict(self.diagnosis) | {"defect_class": self.diagnosis.defect_class.value}
                if self.diagnosis else None
            ),
            "spec": (
                {
                    **asdict(self.spec),
                    "operator": self.spec.operator.value,
                    "spec_hash": self.spec.spec_hash,
                }
                if self.spec else None
            ),
            "allocation": asdict(self.allocation) if self.allocation else None,
            "compile_ok": self.compile_ok,
            "compile_reason": self.compile_reason,
            "overwide_mask": self.overwide_mask,
            "n_proposed": self.n_proposed,
            "n_budget_filtered": self.n_budget_filtered,
            "n_scope_leaks": self.n_scope_leaks,
            "n_scored": self.n_scored,
            "best_cand": self.best_cand,
            "best_card": self.best_card.to_row() if self.best_card else None,
            "outcome": self.outcome.value,
            "contract_violations": list(self.contract_violations),
            "relaxation": asdict(self.relaxation) if self.relaxation else None,
            "k_spent_after": self.k_spent_after,
            "seeds_spent": self.seeds_spent,
            "notes": self.notes,
        }


@dataclass
class CampaignRecord:
    """One (parent, arm, controller) campaign, start to termination.

    This object is the unit the scorer reads. It carries enough to recompute
    every metric without the agent, the backend, a GPU or an API key -- which
    is the global rule that figures must be redrawable without recomputation.
    """

    campaign_id: str
    parent_id: str
    cluster_id: str
    split: str
    arm: Arm
    controller: str
    model_id: str
    qualification_hash: str
    env_hash: str
    k_budget: int
    iterations: list[IterationRecord] = field(default_factory=list)
    qualified: list[str] = field(default_factory=list)
    best_cand: str | None = None
    best_card: ScoreCard | None = None
    best_outcome: RepairOutcome = RepairOutcome.NO_CHANGE
    termination: str = ""
    n_declines: int = 0
    n_malformed: int = 0
    # Transport / rate-limit / BILLING failures. Kept apart from n_malformed
    # because an arm whose calls all 400'd is not a measurement of the model.
    n_errors: int = 0
    total_cost_usd: float = 0.0
    total_seconds: float = 0.0
    total_llm_calls: int = 0
    seeds_spent: int = 0
    relaxations: list[RelaxationEvent] = field(default_factory=list)
    # Filled by the controller from the execution service and the LLM usage
    # counter. Declared here rather than attached ad hoc so that `to_row`
    # carries them: the service's catch counts (scope leaks, budget filters,
    # over-wide masks) are reportable results, not debug output.
    service_ledger: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    # RELAX chosen by an arm that has no contract to relax. Counted rather
    # than recorded as a RelaxationEvent, so the relaxation rate stays a
    # measurement of contract-bound arms only -- but counted, because an agent
    # asking to relax a commitment it was never given is itself informative.
    n_relax_without_contract: int = 0

    def to_row(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "parent_id": self.parent_id,
            "cluster_id": self.cluster_id,
            "split": self.split,
            "arm": self.arm.value,
            "controller": self.controller,
            "model_id": self.model_id,
            "qualification_hash": self.qualification_hash,
            "env_hash": self.env_hash,
            "k_budget": self.k_budget,
            "n_iterations": len(self.iterations),
            "qualified": list(self.qualified),
            "n_qualified": len(self.qualified),
            "best_cand": self.best_cand,
            "best_card": self.best_card.to_row() if self.best_card else None,
            "best_outcome": self.best_outcome.value,
            "termination": self.termination,
            "n_declines": self.n_declines,
            "n_malformed": self.n_malformed,
            "n_errors": self.n_errors,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_seconds": round(self.total_seconds, 2),
            "total_llm_calls": self.total_llm_calls,
            "seeds_spent": self.seeds_spent,
            "n_relaxations": len(self.relaxations),
            "n_relax_without_contract": self.n_relax_without_contract,
            "relaxations": [asdict(r) for r in self.relaxations],
            "service_ledger": dict(self.service_ledger),
            "usage": dict(self.usage),
            "iterations": [it.to_row() for it in self.iterations],
        }


__all__ = [
    "AllocationDecision", "Candidate", "CampaignRecord", "DesignSpec",
    "DiagnosisRecord", "IterationRecord", "ParentProblem", "PreservationContract",
    "RelaxationEvent", "ResidueDiag", "ScoreCard", "StepRecord", "sha", "replace",
]
