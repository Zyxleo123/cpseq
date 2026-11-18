"""Prompt scaffold and action schema for the three LLM-owned steps.

Shape taken from Agent Rosetta, which the design doc picks because it is "the
only [harness] published at template granularity and the only one whose LLM
emits constraints rather than molecules". Its five elements, all present here:

  * a system prompt with a `{reasoning_formatting}` slot, normalising reasoning
    and non-reasoning models to one output shape -- needed to compare a local
    8B against a thinking API model without the format being the difference
  * a design brief templating `{prompt}`, `{state}` and
    `{action_formatting_instructions}`
  * action history as a tabular `{summary}`, NOT a transcript -- which is
    directly the shape the retrieved-memory arm needs
  * a two-call action pattern: choose the action, then re-prompt with the
    relevant schema documentation appended to fill in its parameters
  * an explicit error-correction re-prompt on a malformed action

Tagged terminal output (`<diagnosis>...</diagnosis>`) follows ABLE's
convention, so a typed field is extracted without parsing prose.

Two things are load-bearing rather than stylistic.

FORCED DEPTH, IN EVERY ARM AND EVERY STEP. BioDesignBench's finding is that the
evidence-allocation deficit is behavioural, not capability-limited: a plain
instruction to evaluate thoroughly recovers most of the apparent gap (+9.3 and
+15.9 points under matched compute). If only some arms were forced-depth
prompted, any win would be confounded with that instruction. So the depth
instruction is part of the SHARED scaffold and the arms differ only in which
evidence and which history they receive. The doc's consequence: the escalation
rule has to beat UNIFORM forced depth at matched compute, not a shallow
default, which is why `uniform_depth` is a baseline controller.

THE LLM NEVER NAMES A RESIDUE. Every template here asks for positions,
operators, spans and contracts. Identities are the solver's. That is invariant
1, and it is what makes a win attributable to allocation quality rather than
luck on chemistry.
"""

from __future__ import annotations

import json
import re

from .enums import Arm, DefectClass, Operator, arm_flags
from .types import ParentProblem, ScoreCard

# Agent Rosetta's `{reasoning_formatting}` slot, used for exactly what it is
# for: normalising reasoning and non-reasoning models to the same OUTPUT shape
# without forcing them into the same reasoning MECHANISM.
#
# Measured, and the reason there are two variants. Asking an API model that
# already emits native thinking blocks to also wrap its reasoning in
# <reasoning>...</reasoning> made claude-sonnet-5-5 return
# stop_reason=refusal, category `reasoning_extraction` on 6 of 11 calls -- the
# two reasoning channels collide. Local instruct weights have no native
# thinking channel and DO need to be told to deliberate before answering,
# because the forced-depth instruction is the thing BioDesignBench showed
# recovers most of the evidence-allocation gap, and it has to reach every arm.
#
# So the depth instruction in the task brief is identical for both; only where
# the deliberation is written down differs.
REASONING_FORMATTING_TAGGED = (
    "Think briefly inside <reasoning>...</reasoning> (under 180 words), then "
    "give the single tagged answer block the step asks for. Emit nothing after "
    "the closing tag."
)
REASONING_FORMATTING_NATIVE = (
    "Reason through the evidence before answering, then reply with ONLY the "
    "single tagged answer block the step asks for. Do not wrap your reasoning "
    "in tags of your own and emit nothing after the closing tag."
)

# Set once per run by the campaign runner, from the model backend.
REASONING_FORMATTING = REASONING_FORMATTING_TAGGED


def set_reasoning_style(native_thinking: bool) -> None:
    """Pick the reasoning-format slot for the model about to run.

    Called once at campaign start. It changes only where the model is told to
    put its deliberation, never whether it is told to deliberate -- the
    forced-depth instruction lives in the task brief and is shared by every arm
    and every backend, which is what keeps a win from being confounded with
    that instruction.
    """
    global REASONING_FORMATTING
    REASONING_FORMATTING = (
        REASONING_FORMATTING_NATIVE if native_thinking
        else REASONING_FORMATTING_TAGGED
    )

# The framing below is deliberate and was measured, not stylistic.
#
# An earlier version opened "You are the controller of an automated
# cyclic-peptide design campaign. A linear peptide binder is being converted
# into a cyclic analog that must keep binding its target." Run against the API,
# claude-opus-5-5 declined 68 of 68 steps with stop_reason=refusal,
# category `bio`; claude-sonnet-5-5 did not. That phrasing reads as de novo
# binder engineering against an unnamed target, which is not what this is.
#
# What this actually is: an offline methods benchmark. The structures are
# published PDB entries, every number the model reads is a geometric
# measurement computed from those deposited coordinates, the model selects
# POSITIONS and never residue identities, and the objective is to PRESERVE an
# interaction the deposited structure already shows while fixing ring geometry.
# Saying so is a correction to an inaccurate prompt, not an evasion: the task
# the agent performs is unchanged, and the refusal rate is reported either way.
SYSTEM = """You are the controller in a computational benchmark for automated \
structural-biology workflows. This is a methods evaluation run offline on \
published structures from the Protein Data Bank; every quantity you are shown \
is a geometric measurement computed from those deposited coordinates.

The decision under study is BUDGET ALLOCATION. A published peptide-protein \
complex is being re-modelled as a head-to-tail cyclic analog of the same \
peptide, which introduces ring-closure strain. You decide which positions to \
open for a downstream solver to work on, so that the ring closes without \
giving up the interaction the deposited structure already shows.

You never choose amino-acid identities and you never emit a sequence. \
Calibrated solvers (GenKIC for backbone closure, ProteinMPNN for side chains) \
fill in every residue inside the mask you specify. Your output is a set of \
positions, an operator, and an evaluation depth.

The tension you are managing is specific. Closing the ring requires backbone \
geometry near the cyclisation bond; the existing interaction depends on side \
chains at the interface. In some complexes those are the same positions, and \
then relieving strain costs interface contacts and no move is free.

{reasoning_formatting}"""

# ---------------------------------------------------------------------------
# Evidence rendering -- the {state} block
# ---------------------------------------------------------------------------

DIAG_COLUMNS = [
    ("pos", "position in the peptide, 1-based"),
    ("aa", "residue currently there"),
    ("ringd", "steps around the ring to the cyclisation bond (0 = anchor)"),
    ("strain", "backbone strain, REU-like; the qualification limit is 4.0"),
    ("dSASA", "A^2 of side chain buried on binding"),
    ("burf", "fraction of this residue's surface buried by the target"),
    ("cont5A", "heavy-atom contacts to the target within 5 A"),
    ("pol3.5", "polar (N/O) pairs to the target within 3.5 A"),
    ("mind", "closest approach to the target, A"),
    ("hbond", "target residue of the best polar contact, if any"),
]


def render_residue_table(parent: ParentProblem) -> str:
    rows = [
        [
            str(r.pos), r.wt, str(r.ring_dist), f"{r.strain_rel:.2f}",
            f"{r.dsasa:.0f}", f"{r.buried_frac:.2f}", str(r.contacts_5a),
            str(r.polar_3p5a), f"{r.min_dist:.1f}", r.hbond_partner or "-",
        ]
        for r in parent.residues
    ]
    header = [h for h, _ in DIAG_COLUMNS]
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    out = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
           "  " + "  ".join("-" * widths[i] for i in range(len(header)))]
    for r in rows:
        out.append("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(header))))
    out.append("")
    out.append("Column meanings:")
    out += [f"  {h}: {d}" for h, d in DIAG_COLUMNS]
    return "\n".join(out)


def render_scorecard(card: ScoreCard, parent: ParentProblem) -> str:
    from .qualification import THRESHOLDS as T

    floor = T["bsa_frac_of_native_min"] * parent.native_bsa
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"
    lines = [
        f"  ipTM                 {card.iptm:.3f}   "
        f"limit >= {T['iptm_min']:.2f}            {mark(card.iptm >= T['iptm_min'])}",
        f"  seed consensus       {card.seed_consensus}/{card.n_seeds}     "
        f"limit >= {T['consensus_frac_min']:.0%} of seeds   "
        f"{mark(card.seed_consensus / max(card.n_seeds,1) >= T['consensus_frac_min'])}",
        f"  BSA                  {card.bsa:.0f} A^2  "
        f"limit >= {floor:.0f} A^2        {mark(card.bsa >= floor)}",
        f"  ddG_bind             {card.ddg_bind:.1f} REU "
        f"limit <= {T['ddg_bind_max']:.0f} REU       {mark(card.ddg_bind <= T['ddg_bind_max'])}",
        f"  max backbone strain  {card.max_strain:.2f} REU "
        f"limit <= {T['max_strain_max']:.1f} REU        "
        f"{mark(card.max_strain <= T['max_strain_max'])}"
        + (f"   at position(s) {','.join(map(str, card.strain_locus))}"
           if card.strain_locus else ""),
        f"  free-peptide bound-like population  {card.free_conformer_frac:.3f}  "
        f"limit >= {T['free_conformer_frac_min']:.2f}  "
        f"{mark(card.free_conformer_frac >= T['free_conformer_frac_min'])}",
        f"  edit distance to LP  {card.edit_distance}       "
        f"limit <= {parent.k_budget}               "
        f"{mark(card.edit_distance <= parent.k_budget)}",
    ]
    if card.hbonds_retained:
        lines.append(
            f"  parent polar contacts still present: positions "
            f"{','.join(map(str, card.hbonds_retained))}"
        )
    return "\n".join(lines)


def render_history(history: list[dict]) -> str:
    """Action history as a TABLE, not a transcript. Agent Rosetta's {summary}.

    A transcript would re-present every earlier deliberation and let the model
    re-derive its mask from prose on each call; a table gives outcomes only,
    which is the shape the doc's retrieved-memory arm needs and keeps the
    prompt length flat in the number of iterations.
    """
    if not history:
        return "  (no actions yet -- this is the first iteration)"
    header = ["iter", "operator", "locus", "cost", "dStrain", "dBSA", "dddG", "outcome"]
    rows = [
        [
            str(h["iteration"]), h["operator"][:17], h["locus"][:16],
            str(h["cost"]), h["d_strain"], h["d_bsa"], h["d_ddg"], h["outcome"][:26],
        ]
        for h in history
    ]
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    out = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
           "  " + "  ".join("-" * widths[i] for i in range(len(header)))]
    for r in rows:
        out.append("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(header))))
    return "\n".join(out)


def render_prior(prior: dict[str, dict]) -> str:
    """Arm 4's proposal prior: a soft down-weight, explicitly not a ban.

    The doc is specific about why: "Not a ban: a ban drawn from n=1 noise
    permanently deletes a region of design space and you never notice." So the
    prior is presented as counts and an advisory weight, with the count it is
    based on visible, and the gate (`min_n`) stated in the text.
    """
    if not prior:
        return ""
    lines = [
        "# Retrieved history (advisory -- a down-weight, not a prohibition)",
        "",
        "  Outcomes of this edit class on earlier attempts, scoped to where it",
        "  was observed. Low counts mean little; treat n=1 as noise.",
        "",
        "  edit class                 n   repairs  contract-broken  weight",
        "  -------------------------  --  -------  ---------------  ------",
    ]
    for key, v in sorted(prior.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(
            f"  {key[:25]:25s}  {v['n']:2d}  {v['repairs']:7d}  "
            f"{v['broken']:15d}  {v['weight']:.2f}"
        )
    return "\n".join(lines)


def render_reflections(reflections: list[str]) -> str:
    """Arm 5's distilled lessons. Hypotheses tagged to attempts, never rules.

    The doc's wording: a reflection is "a hypothesis tagged to its supporting
    attempts, never a biochemical rule". The framing in the prompt matters,
    because a distilled lesson presented as fact is how a wrong n=2 inference
    becomes a permanent constraint.
    """
    if not reflections:
        return ""
    return "\n".join(
        ["# Distilled lessons from this campaign (hypotheses, not rules)", ""]
        + [f"  - {r}" for r in reflections]
    )


# ---------------------------------------------------------------------------
# Step 2 -- DIAGNOSE
# ---------------------------------------------------------------------------

DEPTH_DIAGNOSE = """Work through the evidence deliberately before answering:
1. Read the scorecard and list EVERY clause that is failing, not just the first.
2. For each failing clause, find the positions in the table that explain it. \
Strain points at rows with a high `strain`; a BSA or ddG shortfall points at \
rows with large `dSASA` and `cont5A`.
3. If more than one clause fails, decide which is the PRIMARY defect: the one \
that is furthest past its limit relative to that limit's own scale.
4. Name what must be protected while repairing it -- the positions carrying \
most of the binding, and any polar contact you would not want to lose.
5. Check explicitly whether the primary defect's positions OVERLAP the \
positions carrying the binding. If they do, say so: that is a conflict and it \
changes what repair is possible."""

DIAGNOSE_BRIEF = """# Campaign state

Target: {receptor}
Linear parent (LP): {lp}
Current cyclic candidate {cand}: {cp}
Cyclisation: {cyc}, bond between positions {anchors}
Mutation budget: {k_budget} total substitutions from LP; {k_remaining} still unspent
Evidence budget: {seeds_left} evaluation seeds left of {seed_budget}

# Current evaluation (the frozen qualification function)

{scorecard}

# Per-residue diagnostics (measured from the complex)

{state}

{history_block}

# Your task

{depth}

{action_formatting}"""

DIAGNOSE_ACTION = """Emit exactly one block:

<diagnosis>
{{"defect_class": "<ONE OF: {classes}>",
 "defect_locus": [<peptide positions, 1-based, that carry the defect>],
 "preserved_features": [<e.g. "IDENTITY:4", "HBOND:7", "BSA_FLOOR">],
 "evidence_refs": [<which rows or clauses you used, e.g. "strain@11", "BSA">],
 "confidence": <0.0 to 1.0>}}
</diagnosis>

`defect_class` must be copied verbatim from the list -- it is a closed set and \
anything else is rejected. `defect_locus` must be positions of THIS peptide \
(1 to {n_pos})."""

# ---------------------------------------------------------------------------
# Step 3 -- SPECIFY (the two-call pattern's second call)
# ---------------------------------------------------------------------------

OPERATOR_DOC = {
    Operator.SUBSTITUTE: (
        "SUBSTITUTE   open `open_positions` for side-chain redesign. Costs one "
        "substitution per position. Changes chemistry at the interface; does "
        "not move the backbone, so it cannot relieve closure strain."
    ),
    Operator.REBUILD_SEGMENT: (
        "REBUILD_SEGMENT   open `rebuild_segment` (start,end) for backbone "
        "rebuilding with GenKIC pivots. Costs ceil(length/2) substitutions. "
        "This is the only operator that substantially relieves backbone "
        "strain. It gives up more interface at the positions it moves than a "
        "substitution would, and it can pull a polar contact off its partner."
    ),
    Operator.NARROW_PIVOTS: (
        "NARROW_PIVOTS   a 2-residue `rebuild_segment`, tighter than a full "
        "rebuild. Costs 1. Relieves less strain but gives up much less "
        "interface, and keeps polar contacts far more often. The move to reach "
        "for when a wider rebuild resolved the defect but broke the interface."
    ),
    Operator.ANCHOR_SHIFT: (
        "ANCHOR_SHIFT   move the cyclisation bond to a different pair of "
        "positions via `open_positions`. Costs 2. Relocates WHERE the closure "
        "strain sits rather than relieving it -- useful when the strained "
        "positions are also the binding positions and nothing local can win."
    ),
    Operator.RESTRAIN_AND_REPACK: (
        "RESTRAIN_AND_REPACK   add coordinate restraints on the positions in "
        "`open_positions` and repack. Costs NOTHING. Cannot relieve strain, "
        "but is the only operator that can RECOVER a polar contact an earlier "
        "rebuild lost, and it slightly rigidifies the free peptide."
    ),
}

SPECIFY_SELECT = """You have diagnosed: {defect_class} at position(s) {locus}.

Choose ONE operator for this iteration. Available:

{operator_docs}

Budget: {k_remaining} substitution(s) unspent of {k_budget}.
{contract_block}
Reply with exactly:
<operator>NAME</operator>"""

SPECIFY_FILL = """{lead} for {defect_class} at position(s) {locus}.

{operator_doc}

# Campaign state (unchanged)

Current candidate: {cp}   ({k_remaining} of {k_budget} substitutions unspent)
Cyclisation bond between positions {anchors}; peptide positions run 1..{n_pos}

{scorecard}

{state}

{contract_block}{history_block}
# Your task

{depth}
{contract_reminder}
{action_formatting}"""

DEPTH_SPECIFY = """Before answering:
1. Name the exact positions this operator will touch, and add up what they \
cost against the budget still unspent.
2. For each position you are about to touch, read its `cont5A` and `dSASA`. \
Those are the interface you are spending. If a position carries a lot of both, \
justify it or move the mask off it.
3. Say what you are NOT touching, and put those positions in `fixed_positions`.
4. If the budget cannot cover the mask you want, choose the narrower mask \
rather than hoping -- the service will discard every candidate that overspends, \
and an over-wide mask wastes the iteration."""

# The same instruction with the GAIN direction restored. Items 1, 3 and 4 are
# byte-identical to DEPTH_SPECIFY; only item 2 changes.
#
# Why this variant exists. DEPTH_SPECIFY item 2 names `cont5A` and `dSASA` --
# the two columns that measure what an edit at a position COSTS -- and never
# names `burf` or `mind`, which are what an edit there could GAIN. Every arm
# and every model in Stage 2 read that instruction byte-identically, and the
# failure those arms showed was substituting at buried high-contact hotspots
# where burf is near 1 and there is no gain to be had. An instruction that
# points attention only at the cost side is a plausible partial cause of that,
# and it is OUR prompt rather than a property of any model, so it has to be
# ruled in or out by experiment before the negative result can stand.
#
# The band constant (3-6.5 A) is deliberately NOT stated here. Saying "a
# partially exposed position near the target has room" is domain framing a
# protein engineer brings to the table; naming the surrogate's own gain window
# would be handing over the response model.
DEPTH_SPECIFY_SYMMETRIC = """Before answering:
1. Name the exact positions this operator will touch, and add up what they \
cost against the budget still unspent.
2. For each position you are about to touch, read all four interface columns. \
`cont5A` and `dSASA` are the interface you are SPENDING there. `burf` and \
`mind` are what an edit there could GAIN: a position already packed against \
the target (burf near 1) has no room to gain and can only give interface up, \
while a partially exposed position close to but not touching the target has \
room for a redesigned side chain to reach it. Weigh both directions before \
you place the mask, not the cost alone.
3. Say what you are NOT touching, and put those positions in `fixed_positions`.
4. If the budget cannot cover the mask you want, choose the narrower mask \
rather than hoping -- the service will discard every candidate that overspends, \
and an over-wide mask wastes the iteration."""

# Which depth instruction the SPECIFY step uses. Set once per run, before any
# prompt is built, and recorded in the scaffold tag so a row always says which
# instruction produced it.
DEPTH_SPECIFY_ACTIVE = DEPTH_SPECIFY


def set_depth_style(symmetric: bool) -> None:
    global DEPTH_SPECIFY_ACTIVE
    DEPTH_SPECIFY_ACTIVE = (
        DEPTH_SPECIFY_SYMMETRIC if symmetric else DEPTH_SPECIFY)


# ---------------------------------------------------------------------------
# The TOOL action space (lp2cp/tools.py)
# ---------------------------------------------------------------------------

DEPTH_TOOL = """Before answering:
1. Name which qualification objectives are currently FAILING, and by how much.
2. For each tool you are seriously considering, say what it would SPEND \
(substitutions, and interface at the positions it acts on) and what it could \
GAIN. A tool that acts where there is nothing to gain spends for nothing.
3. Say which tools you are not choosing, one line each.
4. If no tool can improve a failing objective within the budget still unspent, \
choose `stop` rather than spending -- a committed edit cannot be taken back."""

TOOL_SELECT = """Choose ONE tool to apply this iteration.

Current candidate: {cp}   ({k_remaining} of {k_budget} substitutions unspent)
Cyclisation bond between positions {anchors}; peptide positions run 1..{n_pos}

{scorecard}

{state}

# Tools

Each tool decides its own positions from the measured columns above, by a fixed
rule stated in its description. You choose WHICH tool runs this iteration. You
do not choose where it acts.

{menu}
{history_block}
# Your task

{depth}

Reply with exactly:
<tool>NAME</tool>"""


SPECIFY_ACTION = """Emit exactly one block:

<design_spec>
{{"operator": "{operator}",
 "open_positions": [<positions whose identity may change>],
 "fixed_positions": [<positions that must not move>],
 "rebuild_segment": {{"start": <pos>, "end": <pos>}},
 "n_eval_seeds": <1..{max_seeds}>,
 {contract_field}"rationale": "<one sentence>"}}
</design_spec>

Omit `rebuild_segment` unless the operator uses it. Positions are 1..{n_pos}. \
`n_eval_seeds` is how deeply to evaluate what comes back; it is drawn from the \
shared evidence budget ({seeds_left} seeds left)."""

CONTRACT_FIELD = """"contract": {"fixed_identities": [<positions whose residue must not change>],
              "required_hbonds": [<positions whose polar contact must survive>],
              "bsa_floor": <minimum BSA in A^2>},
 """

# Injected into the SPECIFY brief when `inject_contract` is on. The ordinary
# CONTRACT_BLOCK states the contract once, near the top; this restates the
# PROTECTED POSITIONS immediately before the action schema, where the model is
# choosing which positions to open.
#
# Measured, and the reason this exists: with the contract stated only once,
# Sonnet A3 opened a position it had itself named in `preserved_features` one
# step earlier in 4 of 35 iterations, and in 10 more it chose a mask that
# guaranteed every returned candidate would violate the contract it had just
# written -- 40% of its iterations produced nothing, against 0% for the arm
# with no contract at all. The failure was not that the contract was too
# strict; it was that the agent did not carry its own commitment into the next
# step.
CONTRACT_REMINDER = """
!! POSITIONS CLOSED BY YOUR OWN PRESERVATION CONTRACT: {protected}

You declared these protected one step ago. A design problem that opens any of
them is REJECTED by the execution service before the solver ever runs, and the
iteration is wasted. Put every one of them in `fixed_positions`, and choose
`open_positions` and `rebuild_segment` from the remaining positions only.
If repairing the defect genuinely requires one of them, do not open it here --
stop and RELAX the contract explicitly on the next allocation step instead.
"""

CONTRACT_BLOCK = """# Preservation contract (in force)

{contract}

A candidate that breaks this contract is REJECTED by the execution service \
even if it resolves the defect. The contract is yours: you declared it, and you \
may change it only through an explicit RELAX action, which is recorded.

"""

NO_CONTRACT_BLOCK = ""

# ---------------------------------------------------------------------------
# Step 8 -- ALLOCATE
# ---------------------------------------------------------------------------

DEPTH_ALLOCATE = """Think about what the next block of budget buys:
1. Is anything still failing? If nothing is, the campaign is done -- STOP.
2. How confident are you in the numbers you just read? They came from \
{last_seeds} evaluation seeds. A marginal pass at few seeds may not survive a \
deeper look, and a false pass is worse than a known failure because the \
campaign stops on it.
3. Compare the value of ONE more edit against the value of KNOWING whether the \
candidate you have actually holds. You have {k_remaining} substitution(s) and \
{seeds_left} seeds. Spending seeds costs no substitutions.
{relax_item}"""

# Item 4 is the ONLY part of the depth instruction that varies by arm, and it
# varies because it names a mechanism the arm may not have: telling an arm with
# no preservation contract to "relax the clause you are least willing to
# defend" instructs it to do something it cannot represent. Items 1-3 -- the
# forced-depth reasoning that BioDesignBench showed recovers most of the
# evidence-allocation gap -- are byte-identical in every arm, which is what
# keeps a win from being confounded with that instruction.
DEPTH_ALLOCATE_RELAX_ITEM = """
4. If the remaining budget cannot satisfy every clause of your contract at \
once, say so and RELAX the clause you are least willing to defend -- do not \
spend the last edit on an infeasible target."""

ALLOCATE_BRIEF = """# Where the campaign stands

Iteration {iteration} of at most {max_iters}.
Best candidate so far: {best_cand}
{scorecard}

{history_block}

Budget: {k_remaining} of {k_budget} substitutions unspent; \
{seeds_left} of {seed_budget} evaluation seeds left.
{contract_block}{prior_block}{reflect_block}

# Your task

{depth}

{action_formatting}"""

ALLOCATE_ACTION = """Emit exactly one block:

<allocation>
{{"action": "<{actions}>",
 "n_eval_seeds": <1..{max_seeds}, only meaningful for ESCALATE>,
 "rationale": "<one sentence>"}}
</allocation>

  EDIT      spend a substitution on another design problem
  ESCALATE  spend evaluation seeds re-checking the current best candidate more
            deeply instead of editing. Costs no substitutions.
{relax_line}  STOP      stop the campaign"""

# RELAX is offered ONLY to arms that have a preservation contract.
#
# Measured: arm 2 recorded 8 RelaxationEvents across 11 campaigns while having
# no contract at all -- `service.active_contract` is None there, so each one
# weakened an empty contract and changed nothing. Relaxation rate is the metric
# the design doc calls "probably the cleanest quantitative support" for the
# closure/interface conflict, so letting a contract-free arm report 0.545 would
# have put pure noise in a headline number, and would have made arm 2 look like
# it was managing a trade-off it cannot even represent.
RELAX_LINE = """  RELAX     the contract cannot be met with the budget left; weaken one clause
            explicitly and branch the lineage
"""

RELAX_BRIEF = """The preservation contract in force is:

{contract}

Budget left: {k_remaining} substitution(s). You judged the contract \
unsatisfiable within it.

{scorecard}

Weaken exactly ONE clause, by the smallest amount that makes progress \
possible, and justify it. This is recorded as a RelaxationEvent against the \
lineage: a design that qualifies afterwards is reported as qualifying under a \
RELAXED contract, never as a clean pass.

Emit exactly one block:

<relaxation>
{{"clause": "<one of: bsa_floor | required_hbonds | fixed_identities>",
 "relax_to": <new numeric floor, or the position to drop>,
 "justification": "<why this clause and not another>",
 "evidence_refs": [<what supports it>]}}
</relaxation>"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _seeds_line(service) -> tuple[int, int]:
    return (
        max(0, service.seed_budget - service.seeds_spent),
        service.seed_budget,
    )


def build_diagnose(
    parent: ParentProblem, card: ScoreCard, cand_id: str, service,
    arm: Arm, history: list[dict],
) -> tuple[str, str]:
    flags = arm_flags(arm)
    seeds_left, seed_budget = _seeds_line(service)
    # Arm 1's diagnosis does not flow into the next proposal, so it is also
    # not GIVEN the history: the doc's "Nothing -- fresh proposals" means the
    # iteration is memoryless, which has to be true of the prompt as well as
    # of the mask or arm 1 is just arm 2 with a shorter brief.
    history_block = (
        "# Actions already taken\n\n" + render_history(history)
        if flags["conditioning"] else ""
    )
    brief = DIAGNOSE_BRIEF.format(
        receptor=parent.receptor_name,
        lp=parent.lp_sequence,
        cand=cand_id,
        cp=parent.cp_sequence,
        cyc=parent.cyclization,
        anchors=", ".join(map(str, parent.anchors)),
        k_budget=parent.k_budget,
        k_remaining=max(0, parent.k_budget - service.k_spent),
        seeds_left=seeds_left,
        seed_budget=seed_budget,
        scorecard=render_scorecard(card, parent),
        state=render_residue_table(parent),
        history_block=history_block,
        depth=DEPTH_DIAGNOSE,
        action_formatting=DIAGNOSE_ACTION.format(
            classes=" | ".join(
                d.value for d in DefectClass if d != DefectClass.NO_DEFECT
            ),
            n_pos=parent.n_pos,
        ),
    )
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), brief


def build_specify_select(
    parent: ParentProblem, diagnosis, service, arm: Arm, allowed: list[Operator],
) -> tuple[str, str]:
    """First half of Agent Rosetta's two-call action pattern: pick the action."""
    flags = arm_flags(arm)
    contract_block = (
        CONTRACT_BLOCK.format(contract=render_contract(service.active_contract))
        if flags["contract"] and service.active_contract else NO_CONTRACT_BLOCK
    )
    brief = SPECIFY_SELECT.format(
        defect_class=diagnosis.defect_class.value,
        locus=", ".join(map(str, diagnosis.defect_locus)) or "(none given)",
        operator_docs="\n\n".join(f"  {OPERATOR_DOC[o]}" for o in allowed),
        k_remaining=max(0, parent.k_budget - service.k_spent),
        k_budget=parent.k_budget,
        contract_block=contract_block,
    )
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), brief


def build_specify_fill(
    parent: ParentProblem, card: ScoreCard, diagnosis, operator: Operator | None,
    service, arm: Arm, allowed: list[Operator] | None = None,
    inject_contract: bool = False,
    prior: dict | None = None, reflections: list[str] | None = None,
) -> tuple[str, str]:
    """Second half of the two-call pattern, or the whole of the one-call form.

    `operator=None` is the ONE-CALL scaffold: the model picks the operator and
    fills its parameters in a single reply, so it must see every operator's
    documentation and an open operator field. An earlier version passed a
    placeholder operator here, which showed only REBUILD_SEGMENT's docs and
    pre-filled REBUILD_SEGMENT in the schema -- so the one-call arm was not a
    cheaper scaffold, it was a different and much narrower action space, and
    comparing it with the two-call arm would have measured that instead of the
    scaffold.
    """
    flags = arm_flags(arm)
    seeds_left, _ = _seeds_line(service)
    use_contract = flags["contract"]
    contract_block = (
        CONTRACT_BLOCK.format(contract=render_contract(service.active_contract))
        if use_contract and service.active_contract else NO_CONTRACT_BLOCK
    )
    ops = allowed or list(OPERATOR_DOC)
    if operator is not None:
        lead = f"You chose {operator.value}"
        operator_doc = "  " + OPERATOR_DOC[operator]
        operator_field = operator.value
    else:
        lead = "Choose ONE operator and specify the design problem"
        operator_doc = "Available operators:\n\n" + "\n\n".join(
            f"  {OPERATOR_DOC[o]}" for o in ops
        )
        operator_field = "<ONE OF: " + " | ".join(o.value for o in ops) + ">"
    brief = SPECIFY_FILL.format(
        lead=lead,
        defect_class=diagnosis.defect_class.value,
        locus=", ".join(map(str, diagnosis.defect_locus)) or "(none given)",
        operator_doc=operator_doc,
        cp=parent.cp_sequence,
        k_remaining=max(0, parent.k_budget - service.k_spent),
        k_budget=parent.k_budget,
        anchors=", ".join(map(str, parent.anchors)),
        n_pos=parent.n_pos,
        scorecard="# Current evaluation\n\n" + render_scorecard(card, parent),
        state="# Per-residue diagnostics\n\n" + render_residue_table(parent),
        # The design doc calls arm 4's history a PROPOSAL prior -- "a soft
        # down-weight on edit classes that keep failing". It therefore belongs
        # at the step that selects an edit class, which is this one. It was
        # originally passed only to ALLOCATE, which chooses
        # EDIT/ESCALATE/RELAX/STOP and cannot act on an edit class at all, and
        # the measured consequence was arm 4 reproducing arm 3's outcome
        # distribution exactly (4 repairs, 0 contract-broken, 28 no-change in
        # both). That was a wiring bug being read as a null result.
        history_block=(
            ("\n" + render_prior(prior) + "\n" if arm_flags(arm)["history_prior"]
             and prior else "")
            + ("\n" + render_reflections(reflections) + "\n"
               if arm_flags(arm)["reflections"] and reflections else "")
        ),
        contract_block=contract_block,
        contract_reminder=(
            CONTRACT_REMINDER.format(protected=", ".join(map(str, sorted(
                set(service.active_contract.fixed_identities)
                | set(service.active_contract.required_hbonds)))))
            if inject_contract and use_contract and service.active_contract
            and (service.active_contract.fixed_identities
                 or service.active_contract.required_hbonds)
            else ""
        ),
        depth=DEPTH_SPECIFY_ACTIVE,
        action_formatting=SPECIFY_ACTION.format(
            operator=operator_field,
            max_seeds=service.max_eval_seeds,
            n_pos=parent.n_pos,
            seeds_left=seeds_left,
            contract_field=CONTRACT_FIELD if use_contract else "",
        ),
    )
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), brief


def build_allocate(
    parent: ParentProblem, card: ScoreCard, best_cand: str, service, arm: Arm,
    history: list[dict], iteration: int, max_iters: int, last_seeds: int,
    prior: dict | None = None, reflections: list[str] | None = None,
) -> tuple[str, str]:
    flags = arm_flags(arm)
    seeds_left, seed_budget = _seeds_line(service)
    contract_block = (
        CONTRACT_BLOCK.format(contract=render_contract(service.active_contract))
        if flags["contract"] and service.active_contract else NO_CONTRACT_BLOCK
    )
    can_relax = bool(flags["contract"] and service.active_contract)
    prior_block = (
        render_prior(prior or {}) + "\n\n" if flags["history_prior"] and prior else ""
    )
    reflect_block = (
        render_reflections(reflections or []) + "\n\n"
        if flags["reflections"] and reflections else ""
    )
    brief = ALLOCATE_BRIEF.format(
        iteration=iteration,
        max_iters=max_iters,
        best_cand=best_cand,
        scorecard=render_scorecard(card, parent),
        history_block="# Actions taken\n\n" + render_history(history),
        k_remaining=max(0, parent.k_budget - service.k_spent),
        k_budget=parent.k_budget,
        seeds_left=seeds_left,
        seed_budget=seed_budget,
        contract_block=contract_block,
        prior_block=prior_block,
        reflect_block=reflect_block,
        depth=DEPTH_ALLOCATE.format(
            relax_item=(DEPTH_ALLOCATE_RELAX_ITEM if can_relax else ""),
            last_seeds=last_seeds,
            k_remaining=max(0, parent.k_budget - service.k_spent),
            seeds_left=seeds_left,
        ),
        action_formatting=ALLOCATE_ACTION.format(
            max_seeds=service.max_eval_seeds,
            actions=("EDIT | ESCALATE | RELAX | STOP" if can_relax
                     else "EDIT | ESCALATE | STOP"),
            relax_line=(RELAX_LINE if can_relax else ""),
        ),
    )
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), brief


def build_relax(parent: ParentProblem, card: ScoreCard, service) -> tuple[str, str]:
    brief = RELAX_BRIEF.format(
        contract=render_contract(service.active_contract),
        k_remaining=max(0, parent.k_budget - service.k_spent),
        scorecard="# Current evaluation\n\n" + render_scorecard(card, parent),
    )
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), brief


def render_contract(c) -> str:
    if c is None:
        return "  (none declared)"
    parts = []
    if c.fixed_identities:
        parts.append(f"  fixed identities at positions: "
                     f"{', '.join(map(str, c.fixed_identities))}")
    if c.required_hbonds:
        parts.append(f"  polar contacts that must survive, at positions: "
                     f"{', '.join(map(str, c.required_hbonds))}")
    if c.bsa_floor:
        parts.append(f"  BSA floor: {c.bsa_floor:.0f} A^2")
    if c.min_hbonds:
        parts.append(f"  at least {c.min_hbonds} parent polar contact(s) retained")
    return "\n".join(parts) if parts else "  (empty)"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _tag_re(tag: str) -> re.Pattern:
    return re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)


def extract_tag(text: str, tag: str) -> str | None:
    hits = _tag_re(tag).findall(text or "")
    return hits[-1].strip() if hits else None


def extract_json(text: str, tag: str) -> tuple[dict | None, str]:
    """Pull one JSON object out of a tagged block.

    Returns (obj, status). Status is a reason on failure and is recorded as an
    outcome -- never silently defaulted, because a defaulted design spec would
    be scored as though the model had emitted it.
    """
    raw = extract_tag(text, tag)
    if raw is None:
        # Fall back to the first balanced JSON object anywhere in the reply.
        # Local 8B models very often emit the right JSON with the wrong tag,
        # and failing those as unparseable would charge the model for a
        # formatting slip rather than a decision.
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return None, "no_tag"
        raw = m.group(0)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # One repair pass: trailing commas and single quotes are the two
        # failures that account for nearly all of it.
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        fixed = re.sub(r"(?<![\w\"])'([^']*)'(?=\s*[:,}\]])", r'"\1"', fixed)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    return obj, "ok"


_OP_RE = re.compile(r"<operator>\s*([A-Za-z_]+)\s*</operator>", re.I)


def parse_operator(text: str, allowed: list[Operator]) -> tuple[Operator | None, str]:
    hits = _OP_RE.findall(text or "")
    names = {o.value.upper(): o for o in allowed}
    if hits:
        key = hits[-1].strip().upper()
        if key in names:
            return names[key], "ok"
        return None, "unknown_operator"
    # Loose: a bare operator name on its own line.
    loose = [
        names[tok.strip().strip(".,:;*`").upper()]
        for tok in (text or "").splitlines()
        if tok.strip().strip(".,:;*`").upper() in names
    ]
    if len(set(loose)) == 1:
        return loose[0], "ok_loose"
    return None, "no_tag"


RETRY_TEMPLATE = """Your answer could not be used: {reason}.

Reply with nothing but the block, correctly formed:

{schema}"""

RETRY_REASON = {
    "no_tag": "the required tagged block was not present",
    "bad_json": "the block was not valid JSON",
    "not_an_object": "the block was not a JSON object",
    "unknown_operator": "the operator named is not one of the listed operators",
    "unknown_defect": "the defect_class is not one of the allowed values",
    "out_of_range": "a position was outside the peptide",
}


def build_tool_select(
    parent: ParentProblem, card: ScoreCard, service, menu: str,
    history: list[dict] | None = None,
) -> tuple[str, str]:
    """The brief for the tool action space. One call, one tool name out.

    Deliberately NOT given the diagnosis: in the tool arm the failing
    objectives are read straight off the scorecard, which is in the brief, and
    adding a separate diagnose call would both cost a call and reintroduce the
    step Stage 2 measured as near-lookup. What is under test here is the
    selection, so the selection is the only thing asked for.
    """
    return SYSTEM.format(reasoning_formatting=REASONING_FORMATTING), TOOL_SELECT.format(
        cp=parent.cp_sequence,
        k_remaining=max(0, parent.k_budget - service.k_spent),
        k_budget=parent.k_budget,
        anchors=", ".join(map(str, parent.anchors)),
        n_pos=parent.n_pos,
        scorecard="# Current evaluation\n\n" + render_scorecard(card, parent),
        state="# Per-residue diagnostics\n\n" + render_residue_table(parent),
        menu=menu,
        history_block=(
            "\n# What this campaign has already tried\n\n"
            + render_history(history or []) + "\n"
        ),
        depth=DEPTH_TOOL,
    )


_TOOL_RE = re.compile(r"<tool>\s*([A-Za-z0-9_]+)\s*</tool>", re.I)


def parse_tool(text: str, allowed: tuple[str, ...]) -> tuple[str | None, str]:
    """Same shape as parse_operator, including the loose bare-name fallback."""
    names = {a.upper(): a for a in allowed}
    hits = _TOOL_RE.findall(text or "")
    if hits:
        key = hits[-1].strip().upper()
        if key in names:
            return names[key], "ok"
        return None, "unknown_tool"
    loose = [
        names[tok.strip().strip(".,:;*`-").upper()]
        for tok in (text or "").splitlines()
        if tok.strip().strip(".,:;*`-").upper() in names
    ]
    if len(set(loose)) == 1:
        return loose[0], "ok_loose"
    return None, "no_tag"
