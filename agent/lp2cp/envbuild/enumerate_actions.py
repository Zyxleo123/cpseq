"""Exhaustive enumeration of the action space, per parent. The oracle table.

Design doc, Stage 1 step 2: "Enumerate the action space exhaustively per
parent -- every locus x operator x budget-split on a coarse grid, 60-150
interventions each, run to saturation with recorded variance." Step 3: label
every cell under the frozen qualification function AND that parent's
preservation contract.

Why this is the right shape, in the doc's words: "the table answers every query
a controller would have made, so a whole campaign replays offline at any budget
T. End-to-end success becomes a deterministic function of measured
per-decision regret, and hundreds of prompt variants can run against the
simulator in an afternoon."

Two things this file settles that Stage 0 told us to settle.

JOINT INTERVENTIONS ARE IN THE GRID. The doc flags this as a thing to settle
first -- "if the oracle covers only single interventions, you cannot detect
whether the LLM finds coordinated moves (open position 6 AND rebuild 10-12)".
Stage 0 then measured why it matters: at k=2 a perfect greedy controller is
optimal only 45.8% of the time, and mean |epistasis| is 1.29 kcal/mol. So a
singles-only grid would hide the one place an LLM is most likely to beat a
rule. `_joint_cells` below is that part of the grid.

SPANS WRAP AROUND THE RING. A cyclic peptide has no first or last residue; a
span from position 15 to position 2 of a 16-mer is contiguous. Enumerating
only non-wrapping spans would exclude every span that straddles the closure
bond -- which is exactly where the doc's BACKBONE_STRAIN_AT_CLOSURE defect
lives, so the oracle would not contain the repair for its own headline defect.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..enums import BUDGET_CONSUMING, DefectClass, Operator, RepairOutcome
from ..qualification import assert_frozen, qualify
from ..types import ParentProblem, PreservationContract, sha
from . import surrogate
from .build_parents import load_parents

# Coarse grid. Declared constants, not search parameters.
SPAN_LENGTHS = (2, 3, 4)
MAX_CELLS_PER_PARENT = 150
MAX_PAIRS = 40
MAX_JOINT = 36


def edit_cost(operator: Operator, positions: tuple[int, ...]) -> int:
    """Substitutions an intervention spends. The budget model, in one place.

    A backbone rebuild is not free of sequence: closing a span needs some
    identity freedom inside it, modelled as ceil(L/2). A substitution costs one
    per position. Evidence escalation and contract relaxation cost nothing --
    that is the whole point of having them as operators, since the doc's
    iteration 3 is the agent spending evaluation depth INSTEAD of an edit.

    This function is the single definition of cost: the enumerator, the
    execution service's filter stage and the baselines all call it, so a
    campaign cannot disagree with the oracle about what a move cost.
    """
    if operator not in BUDGET_CONSUMING:
        return 0
    L = len(positions)
    if operator == Operator.SUBSTITUTE:
        return L
    if operator == Operator.NARROW_PIVOTS:
        return 1 if L else 0
    if operator == Operator.ANCHOR_SHIFT:
        return 2
    # REBUILD_SEGMENT
    return (L + 1) // 2


def ring_span(lo: int, length: int, n: int) -> tuple[int, ...]:
    """A contiguous span of `length` positions starting at `lo`, around the ring."""
    return tuple(sorted(((lo - 1 + i) % n) + 1 for i in range(length)))


def reference_contract(parent: ParentProblem) -> PreservationContract:
    """The contract a competent agent would declare. The 2->3 label's reference.

    Straight from the doc's worked example: "preserved features = W4 and R7
    identities, both parent H-bonds, and BSA floor 527 (0.85 x parent)". So:
    every position carrying a parent H-bond, the two positions carrying the
    most interface mass, and 0.85 of the parent's own buried area.

    This is NOT what the agent is told -- arms 1 and 2 never see a contract,
    and arm 3+ declares its own. It is the fixed yardstick the oracle table is
    labelled against, so that CONTRACT_HOLDING_REPAIR means the same thing for
    every arm. An arm that declared a weaker contract and then "repaired" under
    it would otherwise score as if it had preserved something.
    """
    by_mass = sorted(
        parent.residues, key=lambda r: surrogate.interface_mass(r)
    )  # most negative = most interface mass, first
    hot = tuple(sorted(r.pos for r in by_mass[:2]))
    return PreservationContract(
        fixed_identities=hot,
        required_hbonds=tuple(parent.scorecard0.hbonds_retained),
        bsa_floor=round(0.85 * parent.scorecard0.bsa, 1),
        min_hbonds=0,
    )


def _single_cells(parent: ParentProblem) -> list[dict]:
    n = parent.n_pos
    cells: list[dict] = []
    for p in range(1, n + 1):
        cells.append({"groups": ((Operator.SUBSTITUTE, (p,)),), "family": "sub1"})
    for L in SPAN_LENGTHS:
        for lo in range(1, n + 1):
            span = ring_span(lo, L, n)
            cells.append({"groups": ((Operator.REBUILD_SEGMENT, span),),
                          "family": f"rebuild{L}"})
    for lo in range(1, n + 1):
        cells.append({"groups": ((Operator.NARROW_PIVOTS, ring_span(lo, 2, n)),),
                      "family": "narrow"})
    for a in parent.anchors:
        for d in (-2, -1, 1, 2):
            tgt = ((a - 1 + d) % n) + 1
            cells.append({"groups": ((Operator.ANCHOR_SHIFT, tuple(sorted({a, tgt}))),),
                          "family": "anchor"})
    hb = tuple(parent.scorecard0.hbonds_retained)
    if hb:
        cells.append({"groups": ((Operator.RESTRAIN_AND_REPACK, hb),),
                      "family": "restrain_hbonds"})
    near = tuple(r.pos for r in parent.residues if r.ring_dist <= 1)
    if near:
        cells.append({"groups": ((Operator.RESTRAIN_AND_REPACK, near),),
                      "family": "restrain_closure"})
    # The do-nothing cell. Without it the table cannot say whether ANY action
    # beats inaction on this parent, and "regret" would be measured against a
    # best-of-actions that might all be worse than leaving the parent alone.
    cells.append({"groups": (), "family": "noop"})
    return cells


def _pair_cells(parent: ParentProblem) -> list[dict]:
    """Substitution pairs. Budget-split k=2 on the coarse grid."""
    n = parent.n_pos
    pairs = list(itertools.combinations(range(1, n + 1), 2))
    # A deterministic priority rather than a random sample: pairs that involve
    # the closure neighbourhood come first, because those are the ones where a
    # closure/interface trade-off can actually arise. Ties by hash so the order
    # does not track position number.
    ring = {r.pos: r.ring_dist for r in parent.residues}
    pairs.sort(key=lambda pq: (min(ring.get(pq[0], 9), ring.get(pq[1], 9)),
                               sha((parent.parent_id, "pair", pq))))
    return [
        {"groups": ((Operator.SUBSTITUTE, pq),), "family": "sub2"}
        for pq in pairs[:MAX_PAIRS]
    ]


def _joint_cells(parent: ParentProblem) -> list[dict]:
    """Coordinated moves: rebuild a span AND substitute somewhere else.

    The doc's example of the move a rule-based controller would miss: "open
    position 6 AND rebuild 10-12". Stage 0's k=2 result is the reason this is
    not optional -- greedy was optimal only 45.8% of the time, so the cells
    where a pair beats the sum of its parts are where the headline could move.
    """
    n = parent.n_pos
    out: list[dict] = []
    ring = {r.pos: r.ring_dist for r in parent.residues}
    mass = {r.pos: surrogate.interface_mass(r) for r in parent.residues}
    # Spans worth rebuilding: the ones containing the most measured strain.
    spans = [ring_span(lo, 3, n) for lo in range(1, n + 1)]
    strain = {r.pos: r.strain_rel for r in parent.residues}
    spans.sort(key=lambda s: -sum(strain.get(p, 0.0) for p in s))
    # Positions worth substituting: cheapest interface mass first, which is
    # where a competent controller would spend a spare edit.
    posns = sorted(range(1, n + 1), key=lambda p: -mass.get(p, 0.0))
    for span in spans[:6]:
        for p in posns:
            if p in span:
                continue
            out.append(
                {
                    "groups": (
                        (Operator.REBUILD_SEGMENT, span),
                        (Operator.SUBSTITUTE, (p,)),
                    ),
                    "family": "joint_rebuild_sub",
                }
            )
            if len(out) >= MAX_JOINT:
                return out
    return out


def enumerate_parent(parent: ParentProblem, max_cells: int = MAX_CELLS_PER_PARENT
                     ) -> list[dict]:
    """Every cell for one parent, labelled. Deterministic and order-independent."""
    contract = reference_contract(parent)
    cells = _single_cells(parent)
    n_single = len(cells)
    if parent.k_remaining >= 2:
        cells += _pair_cells(parent)
        cells += _joint_cells(parent)
    # Singles and spans are never dropped; only the k>=2 grid is trimmed, and
    # from the tail of a declared priority order.
    if len(cells) > max_cells:
        cells = cells[:n_single] + cells[n_single:][: max_cells - n_single]

    rows: list[dict] = []
    for cell in cells:
        groups = cell["groups"]
        touched = tuple(sorted({p for _, ps in groups for p in ps}))
        cost = sum(edit_cost(op, ps) for op, ps in groups)
        iid = "noop" if not groups else "|".join(
            f"{op.value}:{','.join(map(str, ps))}" for op, ps in groups
        )

        if not groups:
            latent = parent.scorecard0
            held = surrogate.evaluate(
                parent, (), "noop", parent.k_spent,
                n_seeds=surrogate.CFG.heldout_seeds, heldout=True,
            )
        else:
            latent = surrogate.latent_card(parent, groups, iid, parent.k_spent + cost)
            held = surrogate.evaluate(
                parent, groups, iid, parent.k_spent + cost,
                n_seeds=surrogate.CFG.heldout_seeds, heldout=True,
            )

        # Budget feasibility. An over-budget cell stays in the table labelled
        # INFEASIBLE rather than being filtered out: the agent can propose it,
        # and the table has to be able to say that it was proposable and wrong.
        over_budget = (parent.k_spent + cost) > parent.k_budget
        verdict = qualify(latent, parent.native_bsa, parent.k_budget)
        held_verdict = qualify(held, parent.native_bsa, parent.k_budget)
        ok_contract, violations = contract.holds(
            latent, parent.cp_sequence, parent.cp_sequence
        )

        if over_budget:
            outcome = RepairOutcome.INFEASIBLE
        elif verdict.qualifies and ok_contract:
            outcome = RepairOutcome.CONTRACT_HOLDING_REPAIR
        elif verdict.qualifies and not ok_contract:
            outcome = RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN
        elif (
            verdict.defect_class != parent.true_defect_class
            and parent.true_defect_class != DefectClass.NO_DEFECT
            and not ok_contract
        ):
            # The original defect went away but a different objective now
            # fails AND the contract broke. Still "resolved, contract broken":
            # the doc's C-041 row, which is the 2->3 ablation in one cell.
            outcome = RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN
        else:
            outcome = RepairOutcome.NO_CHANGE

        rows.append(
            {
                "parent_id": parent.parent_id,
                "cluster_id": parent.cluster_id,
                "split": parent.split,
                "intervention_id": iid,
                "family": cell["family"],
                "groups": [[op.value, list(ps)] for op, ps in groups],
                "touched": list(touched),
                "edit_cost": cost,
                "k_after": parent.k_spent + cost,
                "over_budget": over_budget,
                "latent": latent.to_row(),
                "verdict": verdict.to_row(),
                "heldout": held.to_row(),
                "heldout_verdict": held_verdict.to_row(),
                "contract_ok": ok_contract,
                "contract_violations": list(violations),
                "outcome": outcome.value,
                # The scalar the controller is scored against. Defined below.
                "utility": utility(latent, parent, ok_contract, over_budget),
            }
        )
    return rows


def utility(card, parent: ParentProblem, contract_ok: bool, over_budget: bool) -> float:
    """The scalar an ideal controller maximises. One definition, used everywhere.

    A campaign's goal is a CONTRACT-HOLDING qualifying design, so the utility
    is lexicographic in that and only then continuous:

        +100   qualifies and the reference contract holds
        + 40   qualifies, contract broken   (arm 2 counts this; arm 3 does not)
        continuous progress term otherwise, so a controller that moved three
        objectives most of the way scores above one that moved nothing --
        without which almost every cell on a hard parent would tie at zero and
        regret would be uninformative.
        -100   over budget

    Progress is the mean shortfall across the scored objectives, negated: 0.0
    means every objective is met. It reads the SAME frozen margins the
    qualification function uses, so utility and qualification cannot drift
    apart.
    """
    from ..qualification import _margins

    if over_budget:
        return -100.0
    m = _margins(card, parent.native_bsa, parent.k_budget)
    scored = ("iptm", "consensus", "bsa", "ddg_bind", "max_strain",
              "free_conformer_frac")
    shortfall = sum(max(m[k], 0.0) for k in scored) / len(scored)
    qualifies = all(m[k] <= 1e-9 for k in scored)
    if qualifies:
        return 100.0 if contract_ok else 40.0
    return -10.0 * shortfall


def build(parents_path: Path, out_dir: Path, qual_path: Path,
          max_cells: int = MAX_CELLS_PER_PARENT) -> dict:
    assert_frozen(qual_path)
    parents = load_parents(parents_path, with_labels=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "oracle_table.jsonl"

    fam = Counter()
    outc = Counter()
    n = 0
    per_parent: dict[str, dict] = {}
    with dest.open("w") as fh:
        for p in parents:
            rows = enumerate_parent(p, max_cells)
            best = max(rows, key=lambda r: r["utility"])
            per_parent[p.parent_id] = {
                "n_cells": len(rows),
                "best_intervention": best["intervention_id"],
                "best_utility": round(best["utility"], 4),
                "best_outcome": best["outcome"],
                "n_contract_holding": sum(
                    1 for r in rows
                    if r["outcome"] == RepairOutcome.CONTRACT_HOLDING_REPAIR.value),
                "n_resolved_broken": sum(
                    1 for r in rows
                    if r["outcome"] == RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN.value),
                "noop_utility": round(
                    next(r["utility"] for r in rows if r["family"] == "noop"), 4),
                "true_defect_class": p.true_defect_class.value,
                "reference_contract": asdict(reference_contract(p)),
            }
            for r in rows:
                fam[r["family"]] += 1
                outc[r["outcome"]] += 1
                fh.write(json.dumps(r) + "\n")
                n += 1
            print(f"[enum] {p.parent_id:30s} n={len(rows):4d} "
                  f"best={best['intervention_id'][:42]:42s} "
                  f"u={best['utility']:7.2f} {best['outcome']}")

    summary = {
        "n_parents": len(parents),
        "n_cells": n,
        "cells_per_parent_mean": round(n / max(len(parents), 1), 1),
        "family_census": dict(fam),
        "outcome_census": dict(outc),
        "per_parent": per_parent,
        "surrogate": surrogate.provenance(),
        "oracle_table_sha": sha(sorted(per_parent), 32),
        "max_cells_per_parent": max_cells,
    }
    (out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[enum] wrote {n} cells over {len(parents)} parents -> {dest}")
    print(f"[enum] outcomes: {dict(outc)}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", required=True, help="parents_full.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qualification", required=True)
    ap.add_argument("--max-cells", type=int, default=MAX_CELLS_PER_PARENT)
    args = ap.parse_args()
    build(Path(args.parents), Path(args.out), Path(args.qualification), args.max_cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
