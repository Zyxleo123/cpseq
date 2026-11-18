"""Episodic archive, scoped retrieval, and reflection. Arms 4 and 5.

The design doc separates three mechanisms and insists they are three, not two:
iterative refinement (the core, present in every arm from 2 up), episodic
memory (arm 4), and reflection (arm 5, "an ablation whose benefit must be
measured, not assumed").

Two properties of this implementation come straight from the doc and are the
reason arm 4 is not simply "a longer prompt":

STORING INFLUENCES NOTHING; RETRIEVAL CONDITIONS A DECISION. "The archive
stores everything for reproducibility; storing alone influences nothing.
Retrieval is what conditions a decision, and it is scoped -- current
diagnostics plus relevant prior successes, failures and contradictory evidence,
never the whole archive in every prompt." So `Archive.write` is called in every
arm (the trace is complete regardless), and only `retrieve` is gated on the
arm.

HISTORY IS A PROPOSAL PRIOR, NOT A BAN. "a soft down-weight on edit classes
that keep failing, scoped to where it was observed and gated on a count fixed
in advance. Not a ban: a ban drawn from n=1 noise permanently deletes a region
of design space and you never notice." So `prior()` returns weights with their
supporting counts attached, the prompt says low counts mean little, and the
weight is bounded below so no edit class can reach zero.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .enums import DefectClass, Operator, RepairOutcome

# The count gate, fixed in advance. Below this an edit class's prior stays at
# 1.0 -- it is reported with its count so the model can see the evidence is
# thin, but it is not down-weighted at all.
MIN_N_FOR_PRIOR = 2
# The floor on the weight. A class that has failed every time is discouraged,
# never deleted.
MIN_WEIGHT = 0.25


@dataclass(frozen=True)
class Episode:
    """One intervention and what it did. The archive's unit."""

    parent_id: str
    cluster_id: str
    iteration: int
    defect_class: str
    operator: str
    locus: tuple[int, ...]
    ring_dist_min: int
    edit_cost: int
    outcome: str
    d_strain: float
    d_bsa: float
    d_ddg: float
    contract_broken: bool

    @property
    def edit_class(self) -> str:
        """The granularity the prior is kept at.

        Operator crossed with where it acted relative to the closure bond --
        not the exact positions, which never repeat across parents, and not the
        operator alone, which would pool a rebuild at the closure with a
        rebuild on the far side of the ring. "Scoped to where it was observed"
        in the doc's phrase.
        """
        band = "at_closure" if self.ring_dist_min <= 1 else (
            "near_closure" if self.ring_dist_min <= 3 else "far_from_closure"
        )
        return f"{self.operator}@{band}"


class Archive:
    """Everything that happened, plus the scoped reads that condition a decision."""

    def __init__(self, path: str | Path | None = None):
        self.episodes: list[Episode] = []
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    d["locus"] = tuple(d.get("locus") or ())
                    self.episodes.append(Episode(**d))

    def write(self, ep: Episode) -> None:
        """Called in EVERY arm. Storing alone conditions nothing."""
        self.episodes.append(ep)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(ep) | {"locus": list(ep.locus)}) + "\n")

    # -- arm 4 ------------------------------------------------------------
    def prior(
        self, defect_class: DefectClass, exclude_cluster: str | None = None
    ) -> dict[str, dict]:
        """Soft weights per edit class, scoped to this defect class.

        `exclude_cluster` drops episodes from the SAME target. Without it, arm 4
        would be retrieving outcomes measured on the very receptor it is now
        designing against -- which is leakage across the by-target split, and
        would make arm 4's advantage over arm 3 an artifact of having seen the
        answer rather than of having a history.
        """
        buckets: dict[str, list[Episode]] = defaultdict(list)
        for ep in self.episodes:
            if ep.defect_class != defect_class.value:
                continue
            if exclude_cluster and ep.cluster_id == exclude_cluster:
                continue
            buckets[ep.edit_class].append(ep)

        out: dict[str, dict] = {}
        for key, eps in buckets.items():
            n = len(eps)
            repairs = sum(
                1 for e in eps
                if e.outcome == RepairOutcome.CONTRACT_HOLDING_REPAIR.value
            )
            broken = sum(
                1 for e in eps
                if e.outcome == RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN.value
            )
            if n < MIN_N_FOR_PRIOR:
                weight = 1.0
            else:
                rate = repairs / n
                weight = max(MIN_WEIGHT, min(1.0, 0.35 + 0.9 * rate))
            out[key] = {
                "n": n, "repairs": repairs, "broken": broken,
                "weight": round(weight, 3),
            }
        return out

    # -- arm 5 ------------------------------------------------------------
    def reflections(
        self, defect_class: DefectClass, exclude_cluster: str | None = None,
        max_n: int = 4,
    ) -> list[str]:
        """Distilled lessons, each tagged with the attempts supporting it.

        Deliberately generated from the archive deterministically rather than
        by asking the model to summarise its own history. The doc's arm 5 asks
        whether DISTILLING experience beats RETRIEVING the evidence itself; if
        the distillation were itself an LLM call, arm 5 would differ from arm 4
        by an extra call as well as by a mechanism, and the extra compute would
        confound the comparison the doc is set up to make.
        """
        pri = self.prior(defect_class, exclude_cluster)
        out: list[str] = []
        for key, v in sorted(pri.items(), key=lambda kv: (kv[1]["weight"], -kv[1]["n"])):
            if v["n"] < MIN_N_FOR_PRIOR:
                continue
            op, _, band = key.partition("@")
            where = band.replace("_", " ")
            if v["repairs"] == 0 and v["broken"] > 0:
                out.append(
                    f"On {defect_class.value}, {op} {where} has resolved the "
                    f"defect but broken the preservation contract in "
                    f"{v['broken']} of {v['n']} attempts, and produced no "
                    f"contract-holding repair. Hypothesis: it buys closure by "
                    f"spending interface. (n={v['n']})"
                )
            elif v["repairs"] == 0:
                out.append(
                    f"On {defect_class.value}, {op} {where} has not repaired "
                    f"in {v['n']} attempts. (n={v['n']})"
                )
            elif v["repairs"] == v["n"]:
                out.append(
                    f"On {defect_class.value}, {op} {where} has held the "
                    f"contract in all {v['n']} attempts. (n={v['n']})"
                )
            if len(out) >= max_n:
                break
        return out


def episode_from(
    parent, iteration: int, diagnosis, groups, outcome: RepairOutcome,
    card_before, card_after, edit_cost: int, contract_broken: bool,
) -> Episode:
    ring = {r.pos: r.ring_dist for r in parent.residues}
    locus = tuple(sorted({p for _, ps in groups for p in ps})) if groups else ()
    op = groups[0][0].value if groups else Operator.STOP.value
    return Episode(
        parent_id=parent.parent_id,
        cluster_id=parent.cluster_id,
        iteration=iteration,
        defect_class=(
            diagnosis.defect_class.value if diagnosis else DefectClass.NO_DEFECT.value
        ),
        operator=op,
        locus=locus,
        ring_dist_min=min((ring.get(p, 9) for p in locus), default=9),
        edit_cost=edit_cost,
        outcome=outcome.value,
        d_strain=round(card_after.max_strain - card_before.max_strain, 3),
        d_bsa=round(card_after.bsa - card_before.bsa, 1),
        d_ddg=round(card_after.ddg_bind - card_before.ddg_bind, 2),
        contract_broken=contract_broken,
    )
