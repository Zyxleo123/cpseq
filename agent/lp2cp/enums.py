"""Closed enumerations. Nothing here may be extended at run time.

Invariant 4 of the design doc: `defect_class` is a closed enum fixed in
advance. If the agent can mint categories mid-campaign the repair matrix
becomes unanalyzable after the fact -- you cannot tabulate "repair rate by
defect class" over categories that were invented by the thing being measured.

So every enum in this module is validated at parse time and an unrecognised
value is a parse FAILURE, never a new member. `DEFECT_PRECEDENCE` below is the
design doc's third "settle before starting" item, decided here rather than
post hoc.
"""

from __future__ import annotations

from enum import Enum


class DefectClass(str, Enum):
    """Why a cyclic candidate failed qualification.

    One class per failing objective in the frozen qualification function, plus
    NO_DEFECT for a candidate that qualifies. The mapping objective -> class is
    one-to-one by construction so that a candidate's failing metrics determine
    its true class without a judgement call; what the LLM is asked to do is
    recover that class from the diagnostics, which is the step the capability
    survey finds untested anywhere.
    """

    BACKBONE_STRAIN_AT_CLOSURE = "BACKBONE_STRAIN_AT_CLOSURE"
    INTERFACE_CONTACT_LOSS = "INTERFACE_CONTACT_LOSS"
    BURIAL_DEFICIT = "BURIAL_DEFICIT"
    FREE_CONFORMER_MISMATCH = "FREE_CONFORMER_MISMATCH"
    FOLD_CONFIDENCE_LOW = "FOLD_CONFIDENCE_LOW"
    NO_DEFECT = "NO_DEFECT"


# The tie-break rule for two simultaneous defects, DECLARED BEFORE THE RUN.
#
# The design doc lists this as one of three things to settle first, for a
# concrete reason: without it, a candidate failing both strain and BSA has its
# row in the repair matrix assigned by whoever tabulated it, and "repair rate
# for BACKBONE_STRAIN_AT_CLOSURE" stops being a measurement.
#
# The rule has two levels:
#   1. Rank by NORMALISED MARGIN -- how far each metric falls below its own
#      threshold, in units of that threshold's own scale. This makes "strain
#      11.2 against a limit of 4.0" (2.8x over) comparable with "BSA 495
#      against a floor of 550" (0.10 short) without either metric's units
#      dominating by accident.
#   2. Ties in the margin break on this fixed precedence order. Closure comes
#      first because it is the defect the LP->CP premise is about; fold
#      confidence comes last because a low ipTM is usually downstream of one of
#      the others rather than an independent defect.
DEFECT_PRECEDENCE: tuple[DefectClass, ...] = (
    DefectClass.BACKBONE_STRAIN_AT_CLOSURE,
    DefectClass.INTERFACE_CONTACT_LOSS,
    DefectClass.BURIAL_DEFICIT,
    DefectClass.FREE_CONFORMER_MISMATCH,
    DefectClass.FOLD_CONFIDENCE_LOW,
)


class Operator(str, Enum):
    """The intervention families the agent may specify.

    These are design-problem operators, not molecules: every one of them names
    a region and a mode of change, and a downstream solver fills in residues
    and coordinates. That is the line the design doc forbids crossing -- if the
    agent could emit an identity, a win could not be attributed to allocation
    quality rather than luck on chemistry.

    EVIDENCE_ESCALATION and RELAX_CONTRACT are operators rather than control
    flow on purpose: both are decisions the doc wants measured (the
    evidence-allocation row of the capability survey, and the relaxation rate
    against k), and an operator is a thing that appears in a trace and can be
    counted.
    """

    SUBSTITUTE = "SUBSTITUTE"                  # side-chain identity on an open mask
    REBUILD_SEGMENT = "REBUILD_SEGMENT"        # backbone pivots over a contiguous span
    NARROW_PIVOTS = "NARROW_PIVOTS"            # tighten an existing rebuild
    ANCHOR_SHIFT = "ANCHOR_SHIFT"              # move the cyclisation anchor residues
    RESTRAIN_AND_REPACK = "RESTRAIN_AND_REPACK"  # restrain preserved atoms, repack
    EVIDENCE_ESCALATION = "EVIDENCE_ESCALATION"  # spend on evaluation, not on an edit
    RELAX_CONTRACT = "RELAX_CONTRACT"          # explicit RelaxationEvent, branches lineage
    STOP = "STOP"                              # terminate the campaign


# Operators that consume mutation budget when they land. EVIDENCE_ESCALATION
# deliberately does not: the whole point of the design doc's iteration 3 is
# that the agent can spend evaluation depth INSTEAD of an edit, and if
# escalation also cost an edit that trade-off would not exist.
BUDGET_CONSUMING: frozenset[Operator] = frozenset(
    {
        Operator.SUBSTITUTE,
        Operator.REBUILD_SEGMENT,
        Operator.NARROW_PIVOTS,
        Operator.ANCHOR_SHIFT,
    }
)


class RepairOutcome(str, Enum):
    """The label every enumerated cell carries. Stage 1 step 3 of the doc.

    DEFECT_RESOLVED_CONTRACT_BROKEN is the load-bearing one: it is the cell
    that makes the 2->3 ablation demonstrable rather than asserted. Arm 2 --
    mask but no preservation contract -- counts such a cell as a repair because
    the defect went away; arm 3 does not. If backbone sampling can always buy
    closure by spending the interface, arm 2's repair rate measures nothing.
    """

    CONTRACT_HOLDING_REPAIR = "CONTRACT_HOLDING_REPAIR"
    DEFECT_RESOLVED_CONTRACT_BROKEN = "DEFECT_RESOLVED_CONTRACT_BROKEN"
    NO_CHANGE = "NO_CHANGE"
    INFEASIBLE = "INFEASIBLE"


class StepStatus(str, Enum):
    """How one LLM-owned step ended.

    DECLINED exists because Stage 0 measured it: Haiku 5.5 refused 67% of
    benign protein-interface allocation questions as biosecurity-sensitive, and
    Opus/Sonnet refused 3-6 of 73. The design doc's control loop has no
    provision for a step that simply declines, and an agent making hundreds of
    allocation calls will hit it. A decline is therefore a first-class step
    outcome with a declared deterministic fallback, counted per arm -- not an
    exception, and not silently retried against a different model (which is
    what the API's server-side refusal fallback would do, and which is correct
    in production and wrong in an experiment).
    """

    OK = "OK"
    MALFORMED = "MALFORMED"      # unparseable after the error-correction reprompt
    DECLINED = "DECLINED"        # model refused on safety grounds
    ERROR = "ERROR"              # transport / runtime failure
    BUDGET_CAP = "BUDGET_CAP"    # spend ledger stopped the call before it was made


class Arm(str, Enum):
    """The five arms of the doc's ablation. Tools and budget matched throughout.

    A1  diagnosis flows nowhere -- fresh proposals each iteration
    A2  + conditioning (mask, pivots, fixed identities)
    A3  + preservation contract
    A4  + history-derived proposal prior
    A5  + distilled reflections

    2->3 tests whether protecting a partial solution is what makes repair work.
    3->4 tests whether broader history adds anything.
    4->5 tests whether distilling experience beats retrieving the evidence.
    """

    A1_FRESH = "A1_FRESH"
    A2_CONDITIONING = "A2_CONDITIONING"
    A3_CONTRACT = "A3_CONTRACT"
    A4_PRIOR = "A4_PRIOR"
    A5_REFLECTION = "A5_REFLECTION"


ARM_ORDER: tuple[Arm, ...] = (
    Arm.A1_FRESH,
    Arm.A2_CONDITIONING,
    Arm.A3_CONTRACT,
    Arm.A4_PRIOR,
    Arm.A5_REFLECTION,
)


def arm_flags(arm: Arm) -> dict[str, bool]:
    """What each arm is allowed to carry between iterations.

    Nested, as the doc's table is: arm n gets everything arm n-1 gets. Written
    as explicit booleans rather than an index comparison so that a reader can
    see at a glance which mechanism is on in which arm, and so that adding a
    mechanism cannot silently switch itself on in every later arm.
    """
    i = ARM_ORDER.index(arm)
    return {
        # Arm 1 gets a diagnosis prompt too -- the doc's "nothing" means the
        # diagnosis does not flow into the NEXT proposal, not that the agent is
        # never asked. Keeping the call makes the compute matched across arms,
        # which the evidence-allocation confound requires.
        "conditioning": i >= 1,   # masks, pivots, fixed identities
        "contract": i >= 2,       # preservation contract enforced and prompted
        "history_prior": i >= 3,  # retrieved outcomes as a soft proposal prior
        "reflections": i >= 4,    # distilled lessons
    }


def parse_enum(cls, value: str):
    """Strict parse. An unrecognised value raises -- it never becomes a member.

    `str`-valued enums accept their own members and exact value strings only.
    Case and surrounding whitespace are normalised because models emit
    `backbone_strain_at_closure` and ` BACKBONE_STRAIN_AT_CLOSURE ` about as
    often as the canonical form, and treating those as new categories would
    violate the closed-enum invariant for no reason.
    """
    if isinstance(value, cls):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{cls.__name__}: expected a string, got {type(value).__name__}")
    key = value.strip().upper().replace("-", "_").replace(" ", "_")
    for member in cls:
        if member.value.upper() == key or member.name == key:
            return member
    raise ValueError(
        f"{cls.__name__}: {value!r} is not a member. "
        f"Allowed: {', '.join(m.value for m in cls)}"
    )
