"""The TOOL action space: an operator bound to its own position rule.

Why this module exists
----------------------
Stage 2 measured five models failing at one step: SPECIFY's position choice.
The agent edits where the failing metric is MEASURED rather than where the
evaluator has slack -- on a binding-limited parent it opens the buried
high-contact hotspots, which are already at burf~1 and have no headroom to
gain, so the move is pure interface loss. Diagnosis is near-lookup and operator
choice barely discriminates, so neither explains the loss; the mask does.

PDAgent (Ouyang et al., ICML 2026) does not solve that problem, it deletes it.
Its LLM never names a position: it names one of seven TOOLS, and each tool
carries a hard-coded predicate for where it may act, written in terms of where
an edit can help -- `enhance_solubility` acts on a hydrophobic patch found by a
sliding window, `introduce_disulfide` on residue pairs 10-50 apart that are
already Cys/Ser/Thr, `stabilize_helix` on Pro/Gly at VARIABLE positions only.
The lever-versus-symptom distinction lives in code, not in the model's
reasoning. Their one ablation that isolates the LLM (Table 7) replaces it with
a deterministic policy over the same toolkit: -0.92 CSR where objectives do not
conflict, -12.92 where they do.

A tool here is the same object: (operator, position predicate, edit cost). The
registry below is the WHOLE action space, and it is shared -- `tool_router`
(deterministic) and `ToolSelectController` (LLM) select from this one list, so
the only difference between those two arms is the selector. That is what makes
the comparison a measurement of the selector rather than of two action spaces,
which is the mistake the one-call scaffold made in Stage 2.

Disclosure, because it affects how the resulting number may be read
-------------------------------------------------------------------
`headroom` below reproduces the surrogate's own gain predicate: the 3-6.5 A
band and the (1 - burf) weighting are the v2 response model's, so a tool built
on it is a rule that KNOWS the response model. It sits between `rim_first`
(which knows only that contacts carry binding) and the oracle (which reads the
answer key). Its inputs are nonetheless exactly the two columns every LLM arm
is shown -- `burf` and `mind`, in the residue table, with their meanings
spelled out -- so a model is not missing the information this predicate uses.
Existing baselines already encode comparable knowledge: `greedy_interface` and
`rim_first_closure` are built on `interface_mass`, which is the surrogate's
ddG decomposition. The band constant is the new thing and must be reported as
such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .enums import Operator
from .envbuild.enumerate_actions import edit_cost, ring_span
from .qualification import THRESHOLDS
from .types import DesignSpec, ParentProblem, ScoreCard

# ---------------------------------------------------------------------------
# Position predicates. Measured columns in, a ranking out.
# ---------------------------------------------------------------------------

GAIN_DIST_LO = 3.0
GAIN_DIST_HI = 6.5


def headroom(r) -> float:
    """How much a redesign at this position could ADD to the interface.

    Zero where the residue is already packed against the partner (burf ~ 1,
    nothing to gain) and zero where it cannot reach it at all; largest for a
    partially exposed residue in the 3-6.5 A band. Both inputs are measured
    from deposited coordinates and both are printed in the residue table the
    LLM arms read: `burf` and `mind`.

    This is the LEVER location. It is not where a failing binding metric is
    measured -- that is the buried hotspot, where this function returns ~0.
    """
    if not (GAIN_DIST_LO < r.min_dist < GAIN_DIST_HI):
        return 0.0
    mid = GAIN_DIST_LO + (GAIN_DIST_HI - GAIN_DIST_LO) / 2.0
    prox = 1.0 - abs(r.min_dist - mid) / ((GAIN_DIST_HI - GAIN_DIST_LO) / 2.0)
    return max(0.0, (1.0 - r.buried_frac) * prox)


def interface_carried(r) -> float:
    """How much binding this position already carries. The SYMPTOM location.

    Positive-is-more, unlike the surrogate's signed ddG contribution, so that
    both predicates in this module read "bigger number = more of the thing".
    """
    return 1.0 * r.contacts_5a + 2.0 * r.polar_3p5a + r.dsasa / 100.0


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCtx:
    parent: ParentProblem          # the VIEW (current sequence + current card)
    card: ScoreCard
    k_remaining: int
    seeds: int
    iteration: int
    tried: tuple[str, ...] = ()    # tool names whose candidates are exhausted
    # How many times THIS tool has already been selected in this campaign. A
    # tool returns its nth-ranked candidate, so re-selecting it is a step down
    # its own ranking rather than a repeat of the same intervention.
    #
    # Why this exists: with one candidate per tool, a campaign could reach at
    # most ~6 of the ~150 enumerated interventions for its parent, and a tool
    # whose single guess was wrong lost the parent outright. Ranked candidates
    # turn the menu into a proposal generator the agent can walk down, which is
    # what makes the iteration budget worth anything -- and it does not move
    # the division of labour, because the ranking is still the tool's and the
    # agent still never names a position.
    attempt: int = 0


@dataclass(frozen=True)
class Tool:
    name: str
    operator: Operator | None
    cost: str
    doc: str
    build: Callable[[ToolCtx], DesignSpec | None]


def _res(ctx: ToolCtx) -> dict:
    return {r.pos: r for r in ctx.parent.residues}


def _repack_headroom(ctx: ToolCtx) -> DesignSpec | None:
    """RESTRAIN_AND_REPACK at the positions with the most headroom.

    Costs NO edit budget (RESTRAIN_AND_REPACK is not in BUDGET_CONSUMING) and
    gives up no interface, so where headroom exists this is free upside. No
    Stage 2 baseline uses this operator at all.

    Ranked: attempt 0 repacks the top 3 headroom positions, attempt 1 the next
    3, and so on.
    """
    res = _res(ctx)
    ranked = [p for p in sorted(res, key=lambda p: (-headroom(res[p]), p))
              if headroom(res[p]) > 0.05]
    window = ranked[3 * ctx.attempt: 3 * ctx.attempt + 3]
    if not window:
        return None
    return DesignSpec(
        operator=Operator.RESTRAIN_AND_REPACK,
        open_positions=tuple(sorted(window)),
        n_eval_seeds=ctx.seeds,
        rationale=f"tool:repack_headroom#{ctx.attempt}",
    )


def _substitute_headroom(ctx: ToolCtx) -> DesignSpec | None:
    """SUBSTITUTE at the single position with the most headroom. Costs 1."""
    if ctx.k_remaining < 1:
        return None
    res = _res(ctx)
    ranked = [p for p in sorted(res, key=lambda p: (-headroom(res[p]), p))
              if headroom(res[p]) > 0.05]
    if ctx.attempt >= len(ranked):
        return None
    # Attempt 0 opens the single best position; later attempts walk down the
    # ranking, and every second attempt opens a PAIR, because SUBSTITUTE over
    # two positions is 2 of the 20 repairs in the enumerated table and a
    # one-position tool can never express it.
    if ctx.attempt % 2 == 1 and ctx.k_remaining >= 2 and len(ranked) >= 2:
        i = ctx.attempt // 2
        pick = tuple(sorted(ranked[i:i + 2]))
        if len(pick) < 2:
            return None
    else:
        pick = (ranked[ctx.attempt // 2 if ctx.attempt else 0],)
    return DesignSpec(
        operator=Operator.SUBSTITUTE,
        open_positions=pick,
        n_eval_seeds=ctx.seeds,
        rationale=f"tool:substitute_headroom#{ctx.attempt}",
    )


def _substitute_cheap_rim(ctx: ToolCtx) -> DesignSpec | None:
    """SUBSTITUTE where the LEAST interface is carried. This is the Stage 0
    winner `greedy_interface`, kept in the menu as the incumbent idea."""
    if ctx.k_remaining < 1:
        return None
    res = _res(ctx)
    ranked = sorted(res, key=lambda p: (interface_carried(res[p]), p))
    if ctx.attempt >= len(ranked):
        return None
    return DesignSpec(
        operator=Operator.SUBSTITUTE,
        open_positions=(ranked[ctx.attempt],),
        n_eval_seeds=ctx.seeds,
        rationale=f"tool:substitute_cheap_rim#{ctx.attempt}",
    )


def _narrow_near_closure(ctx: ToolCtx) -> DesignSpec | None:
    """NARROW_PIVOTS over the cheapest span next to the cyclisation bond.
    This is `rim_first_closure`, the arm to beat, as one option in the menu."""
    if ctx.k_remaining < 1:
        return None
    res = _res(ctx)
    n = ctx.parent.n_pos
    near = sorted([p for p in res if res[p].ring_dist <= 2] or list(res),
                  key=lambda p: (interface_carried(res[p]), p))
    if ctx.attempt >= len(near):
        return None
    centre = near[ctx.attempt]
    span = ring_span(((centre - 1) % n) + 1, 2, n)
    seg = (min(span), max(span)) if max(span) - min(span) < n - 1 \
        else (span[-1], span[0])
    return DesignSpec(
        operator=Operator.NARROW_PIVOTS,
        rebuild_segment=seg,
        n_eval_seeds=ctx.seeds,
        rationale="tool:narrow_near_closure",
    )


def _rebuild_at_strain(ctx: ToolCtx) -> DesignSpec | None:
    """REBUILD_SEGMENT centred on the worst measured strain. `fixed_pipeline`."""
    res = _res(ctx)
    n = ctx.parent.n_pos
    ranked_strain = sorted(res, key=lambda p: (-res[p].strain_rel, p))
    locus = list(ctx.card.strain_locus) + [
        p for p in ranked_strain if p not in (ctx.card.strain_locus or ())]
    if ctx.attempt >= len(locus):
        return None
    centre = locus[ctx.attempt]
    span = ring_span(((centre - 2) % n) + 1, 3, n)
    if edit_cost(Operator.REBUILD_SEGMENT, span) > ctx.k_remaining:
        span = ring_span(centre, 2, n)
    if edit_cost(Operator.REBUILD_SEGMENT, span) > ctx.k_remaining:
        return None
    return DesignSpec(
        operator=Operator.REBUILD_SEGMENT,
        rebuild_segment=(min(span), max(span)) if max(span) - min(span) < n - 1
        else (span[-1], span[0]),
        n_eval_seeds=ctx.seeds,
        rationale="tool:rebuild_at_strain",
    )


def _rebuild_cheapest_2span(ctx: ToolCtx) -> DesignSpec | None:
    """REBUILD_SEGMENT over the 2-residue span carrying the least interface.

    This is `rim_first_closure`'s position rule -- spend where the interface is
    cheapest -- with a different OPERATOR and a different SPAN, and that is the
    whole of the difference between them. The oracle table says why it matters:
    of the 20 parents with a contract-holding repair, the single most common
    shape is REBUILD_SEGMENT over a 2-residue span, 7 of 20, and a 2-span costs
    ceil(2/2) = 1 substitution -- the same price as the bar's NARROW_PIVOTS and
    half the price of the 3-span `rebuild_at_strain` uses.

    Reading the enumerated optimum to choose an operator is fitting to the
    answer key, and this tool must be reported as such: it is not something the
    controller discovered. What it establishes is an upper bound on how much of
    this benchmark's headroom a one-line rule captures -- which is the right
    thing to compare an agent against.
    """
    res = _res(ctx)
    n = ctx.parent.n_pos
    if ctx.k_remaining < 1:
        return None
    spans = sorted(
        (ring_span(a, 2, n) for a in range(1, n + 1)),
        key=lambda sp: (sum(interface_carried(res[q]) for q in sp), sp[0]),
    )
    if ctx.attempt >= len(spans):
        return None
    best = spans[ctx.attempt]
    seg = (min(best), max(best)) if max(best) - min(best) < n - 1 \
        else (best[-1], best[0])
    return DesignSpec(
        operator=Operator.REBUILD_SEGMENT,
        rebuild_segment=seg,
        n_eval_seeds=ctx.seeds,
        rationale="tool:rebuild_cheapest_2span",
    )


def _stop(ctx: ToolCtx) -> DesignSpec | None:
    return None


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="repack_headroom",
        operator=Operator.RESTRAIN_AND_REPACK,
        cost="0 substitutions",
        doc=(
            "Restrain the existing contacts and repack side chains at the up to "
            "3 positions with the most room to GAIN interface -- partially "
            "exposed (low burf) and 3-6.5 A from the target (mind). Spends no "
            "substitution budget. Gains nothing at a position that is already "
            "buried against the target, and nothing at a position too far to "
            "reach it. Does not relieve backbone strain."
        ),
        build=_repack_headroom,
    ),
    Tool(
        name="substitute_headroom",
        operator=Operator.SUBSTITUTE,
        cost="1 substitution",
        doc=(
            "Open the single position with the most room to gain interface "
            "(lowest burf in the 3-6.5 A band) for side-chain redesign. Costs "
            "one substitution. Gains interface where there is headroom; at a "
            "buried hotspot it would only give interface up."
        ),
        build=_substitute_headroom,
    ),
    Tool(
        name="substitute_cheap_rim",
        operator=Operator.SUBSTITUTE,
        cost="1 substitution",
        doc=(
            "Open the position carrying the LEAST interface (fewest contacts, "
            "fewest polar pairs, smallest buried area) for side-chain "
            "redesign. Costs one substitution. The cheapest place to spend an "
            "edit if the interface must not be disturbed."
        ),
        build=_substitute_cheap_rim,
    ),
    Tool(
        name="narrow_near_closure",
        operator=Operator.NARROW_PIVOTS,
        cost="1 substitution",
        doc=(
            "Tighten backbone pivots over a 2-residue span beside the "
            "cyclisation bond, centred on whichever of those positions carries "
            "the least interface. Costs one substitution. Relieves closure "
            "strain near the ring bond; gives up some interface in the span."
        ),
        build=_narrow_near_closure,
    ),
    Tool(
        name="rebuild_cheapest_2span",
        operator=Operator.REBUILD_SEGMENT,
        cost="1 substitution",
        doc=(
            "Rebuild the backbone over the 2-residue span that carries the "
            "least interface anywhere in the ring (fewest contacts, fewest "
            "polar pairs, smallest buried area across the pair). Costs one "
            "substitution, because a 2-residue rebuild needs identity freedom "
            "at one position. Gives the backbone somewhere to move at the "
            "cheapest possible price in interface."
        ),
        build=_rebuild_cheapest_2span,
    ),
    Tool(
        name="rebuild_at_strain",
        operator=Operator.REBUILD_SEGMENT,
        cost="2 substitutions",
        doc=(
            "Rebuild the backbone over a 3-residue span centred on the worst "
            "measured per-residue strain. Costs two substitutions. The largest "
            "strain relief available, and the largest interface loss if the "
            "span sits on the interface."
        ),
        build=_rebuild_at_strain,
    ),
    Tool(
        name="stop",
        operator=None,
        cost="0",
        doc=(
            "Commit nothing further and end the campaign. The right answer when "
            "no tool can help: every remaining tool would cost interface or "
            "budget without fixing a failing objective."
        ),
        build=_stop,
    ),
)

BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}
DEFAULT_TOOL = "narrow_near_closure"   # the declared fallback == the arm to beat

# ---------------------------------------------------------------------------
# Menu versions. The action space is a VERSIONED, DECLARED object.
# ---------------------------------------------------------------------------
#
# This exists because of a mistake made here and caught by the trace. The
# registry was edited between two arms of one submission, so `gpt-6-luna` ran
# against six tools while `gpt-6.1-terra` and `gpt-6-sol` ran against seven.
# Comparing those three rows would have measured the action space and called it
# a model comparison -- the identical error the one-call scaffold made in
# Stage 2, where a placeholder operator narrowed the space fivefold and the
# arm was labelled "a cheaper scaffold".
#
# So the menu is now named, selected explicitly, and recorded in every row's
# scaffold tag. The two versions also answer a question that matters for how
# the result may be read:
#
#   v1  the six tools written from the defect structure alone, BEFORE anything
#       was read out of the enumerated oracle table. Nothing here is fitted to
#       the answer key.
#   v2  v1 plus `rebuild_cheapest_2span`, whose operator and span WERE chosen
#       by reading the oracle table (a 2-residue REBUILD_SEGMENT is the most
#       common repair shape in it, 7 of 20, at cost 1).
#
# A headline that holds on v1 is a result about the action space. A headline
# that needs v2 is a result about a rule fitted to the optimum. Reporting both
# is the only way to tell those apart, and they must not be pooled.
MENUS: dict[str, tuple[str, ...]] = {
    "v1": (
        "repack_headroom",
        "substitute_headroom",
        "substitute_cheap_rim",
        "narrow_near_closure",
        "rebuild_at_strain",
        "stop",
    ),
    "v2": (
        "repack_headroom",
        "substitute_headroom",
        "substitute_cheap_rim",
        "narrow_near_closure",
        "rebuild_cheapest_2span",
        "rebuild_at_strain",
        "stop",
    ),
}
ACTIVE_MENU = "v2"

# Registry revision, carried in every row's scaffold tag alongside the menu
# version. Bump it whenever a tool's POSITION RULE changes, even if the menu
# keeps the same names: two rows whose tools behave differently must not be
# able to share a filename, because run_campaign.py is resumable and would
# append one behaviour's rows to the other's file.
#
#   r1  one candidate per tool, retired after a single selection
#   r2  ranked candidates -- a tool returns its nth-best on its nth selection
#       and is retired only when the ranking runs out
TOOLS_REV = "r2"


def set_menu(version: str) -> None:
    """Select the action space. Called once per run, before any prompt."""
    if version not in MENUS:
        raise ValueError(f"unknown tool menu {version!r}; one of {sorted(MENUS)}")
    global ACTIVE_MENU
    ACTIVE_MENU = version


def active_tools() -> tuple[Tool, ...]:
    return tuple(BY_NAME[n] for n in MENUS[ACTIVE_MENU])


def active_names() -> tuple[str, ...]:
    return MENUS[ACTIVE_MENU]


TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in TOOLS)


def render_menu(ctx: ToolCtx | None = None) -> str:
    """The ACTIVE menu as the selector sees it. One block per tool, name first."""
    return "\n\n".join(
        f"  {t.name}  [costs {t.cost}]\n    {t.doc}" for t in active_tools())


def build(name: str, ctx: ToolCtx) -> DesignSpec | None:
    """Run one tool's position rule. Unknown name is a programming error."""
    return BY_NAME[name].build(ctx)


# ---------------------------------------------------------------------------
# The deterministic selector: PDAgent's Table 7 ablation
# ---------------------------------------------------------------------------


def route(ctx: ToolCtx) -> str:
    """Pick a tool from which objectives are FAILING. No model involved.

    This is the control for the LLM tool-selection arm, and it is the whole
    point of having one: PDAgent reports its LLM beating exactly this kind of
    policy by 0.92 CSR points where objectives do not conflict and 12.92 where
    they do, and that claim cannot be checked here without running the same
    toolkit under both selectors.

    The routing table is deliberately the obvious one -- the mapping a careful
    reader would write from the scorecard -- with one piece of knowledge in it
    that no Stage 2 baseline had: a failing BINDING objective is addressed at a
    position with headroom, not at the position carrying the most interface.
    """
    t = THRESHOLDS
    card = ctx.card
    res = _res(ctx)
    bsa_floor = t["bsa_frac_of_native_min"] * max(ctx.parent.native_bsa, 1.0)
    binding_fails = (card.ddg_bind > t["ddg_bind_max"]) or (card.bsa < bsa_floor)
    strain_fails = card.max_strain > t["max_strain_max"]
    any_headroom = any(headroom(r) > 0.05 for r in res.values())

    if binding_fails and any_headroom and "repack_headroom" not in ctx.tried:
        # Free: no budget, no interface given up. Always worth trying first.
        return "repack_headroom"
    if strain_fails:
        # Cheapest first: a 2-span rebuild costs 1 and is the most productive
        # shape in the enumerated table; the 3-span costs 2 for more relief.
        # Guarded on the active menu -- a router that returned a tool the LLM
        # arm was never offered would not be a control for it.
        if ("rebuild_cheapest_2span" in MENUS[ACTIVE_MENU]
                and ctx.k_remaining >= 1
                and "rebuild_cheapest_2span" not in ctx.tried):
            return "rebuild_cheapest_2span"
        if ctx.k_remaining >= 2 and "rebuild_at_strain" not in ctx.tried:
            return "rebuild_at_strain"
        if ctx.k_remaining >= 1:
            return "narrow_near_closure"
        return "stop"
    if binding_fails and ctx.k_remaining >= 1:
        if any_headroom and "substitute_headroom" not in ctx.tried:
            return "substitute_headroom"
        return "substitute_cheap_rim"
    return "stop"


def route_v2(ctx: ToolCtx) -> str:
    """A deterministic policy that does not give up while a tool remains.

    `route` was the first thing a careful reader would write from the
    scorecard, and measuring it exposed a flaw that makes it a BAD control:
    its precedence chain runs out and it returns `stop` on 19 of 32 campaigns,
    at a mean 3.2 selections out of an available 10. A control that quits early
    makes any selector look good by comparison, and the whole point of having a
    deterministic arm is to find out how much of the LLM arm's result is the
    action space rather than the selection.

    So this version keeps the defect-aware head -- free move first, then the
    cheapest repair for whichever objective is failing -- and then falls
    through a fixed cheapest-first order over whatever still has candidates. It
    returns `stop` only when every tool is exhausted. Both routers are kept and
    reported; this is the one the LLM arm has to beat.
    """
    t = THRESHOLDS
    card = ctx.card
    res = _res(ctx)
    bsa_floor = t["bsa_frac_of_native_min"] * max(ctx.parent.native_bsa, 1.0)
    binding_fails = (card.ddg_bind > t["ddg_bind_max"]) or (card.bsa < bsa_floor)
    strain_fails = card.max_strain > t["max_strain_max"]
    done = set(ctx.tried)
    menu = set(MENUS[ACTIVE_MENU])

    def live(name: str, cost: int) -> bool:
        return name in menu and name not in done and ctx.k_remaining >= cost

    head: list[tuple[str, int]] = []
    if binding_fails:
        head += [("repack_headroom", 0), ("substitute_headroom", 1)]
    if strain_fails:
        head += [("rebuild_cheapest_2span", 1), ("narrow_near_closure", 1),
                 ("rebuild_at_strain", 2)]
    # Cheapest-first fall-through: a free move before a 1-edit move before a
    # 2-edit move, whatever is failing.
    tail = [("repack_headroom", 0), ("rebuild_cheapest_2span", 1),
            ("narrow_near_closure", 1), ("substitute_headroom", 1),
            ("substitute_cheap_rim", 1), ("rebuild_at_strain", 2)]
    for name, cost in head + tail:
        if live(name, cost):
            return name
    return "stop"
