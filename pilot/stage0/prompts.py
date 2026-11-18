"""Prompt scaffold and action schema for the Stage 0 allocator.

Shape borrowed from Agent Rosetta (the only harness published at template
granularity): a system prompt with a reasoning-format slot, a design brief
templating the environment state, explicit action-formatting instructions, and
an error-correction re-prompt on a malformed action. Tagged output
(`<choice>...</choice>`) follows ABLE's convention so the field can be
extracted without parsing prose.

Two things here are load-bearing for interpretability, not style.

Forced depth, in EVERY arm. BioDesignBench's finding is that the
evidence-allocation deficit is behavioural rather than capability-limited: a
plain instruction to evaluate thoroughly recovers most of the apparent gap. If
only some arms were forced-depth prompted, any win would be confounded with
that instruction. So the deliberation instruction is part of the shared
scaffold and the arms differ only in which EVIDENCE they receive.

The LLM never names a residue. It emits a position choice -- a design problem.
The identity is the solver's, which is what makes a win attributable to
allocation quality rather than to luck on chemistry.
"""

from __future__ import annotations

import re

REASONING_FORMATTING = (
    "Think step by step inside <reasoning>...</reasoning>. Keep it under 200 words."
)

SYSTEM = """You are an expert structural biologist allocating a scarce mutation \
budget on a protein-protein interface.

You do NOT choose amino-acid identities. A calibrated downstream solver does \
that. Your only job is to decide WHERE the edit budget is spent.

{reasoning_formatting}

Then give your final answer in the exact form:
<choice>CAND_ID</choice>
using one of the candidate ids listed in the brief, and nothing else inside \
the tag."""

TASK_BRIEF = """# Design problem

Complex: {complex_id} ({pdb_id})
  side 1: chains {chains1} -- {protein1}
  side 2: chains {chains2} -- {protein2}

You must mutate EXACTLY {k} interface position(s), chosen from the candidate \
list below. {solver_sentence}

Objective: the mutation is mandatory, so choose the allocation that costs the \
LEAST binding affinity. Equivalently: of the options below, which position(s) \
can this interface most afford to lose?

{state}

# How to decide

{depth_instruction}

{action_formatting_instructions}"""

# Applied to every arm. See the module docstring on why this is shared and not
# an experimental condition.
DEPTH_INSTRUCTION = """Before answering, work through the evidence deliberately:
1. Identify which candidates carry the most interface burial and the most \
cross-interface contacts -- those are the expensive places to spend an edit.
2. Identify which candidates look peripheral or substitution-tolerant.
3. Consider explicitly whether the cheapest-looking candidate has a \
compensating liability (for example a buried polar contact that a smaller \
residue would leave unsatisfied, or a position whose burial is low only \
because the side chain points away from the partner).
4. Commit to the single best allocation. Do not hedge across candidates."""

ACTION_FORMATTING = """Respond with your reasoning, then exactly one line:
<choice>CAND_ID</choice>

CAND_ID must be copied verbatim from the Candidate column. Choose exactly one."""

SOLVER_SENTENCE = {
    "alanine": (
        "Each chosen position will be mutated to ALANINE (side chain truncated)."
    ),
    "best_measured": (
        "A calibrated side-chain solver will then choose the best amino-acid "
        "identity for the position(s) you open, so assume the identity is "
        "chosen well."
    ),
}

# ---------------------------------------------------------------------------
# Evidence rendering. Which columns exist IS the experimental condition.
# ---------------------------------------------------------------------------

STRUCTURE_COLUMNS = [
    ("class", "SKEMPI interface class (COR=core, SUP=support, RIM=rim)"),
    ("dSASA", "A^2 of side-chain surface buried on binding"),
    ("burfrac", "fraction of this residue's surface buried by the partner"),
    ("cont5A", "heavy-atom contacts to the partner within 5 A"),
    ("pol3.5A", "polar (N/O) pairs to the partner within 3.5 A"),
    ("mindist", "closest approach to the partner, A"),
]
CONSERVATION_COLUMNS = [
    ("esm_H", "ESM-2 masked entropy, 0=one residue tolerated, 3.0=all tolerated"),
    ("wt_p", "ESM-2 probability of the residue currently there"),
    ("wt_rk", "its rank among the 20 amino acids"),
]

ARMS = {
    # name:            (structure, conservation, per_position)
    "full": (True, True, True),
    "structure_only": (True, False, True),
    "conservation_only": (False, True, True),
    "aggregate_only": (True, True, False),
    "no_evidence": (False, False, True),
}


def _fmt(val, spec: str = "", dash: str = "-") -> str:
    if val is None:
        return dash
    try:
        return format(val, spec) if spec else str(val)
    except (TypeError, ValueError):
        return str(val)


def render_state(
    decision: dict,
    features: dict,
    conservation: dict,
    arm: str,
    order: list[str],
) -> str:
    """The `{state}` block: the evidence table, in the given candidate order."""
    use_struct, use_cons, per_position = ARMS[arm]
    by_id = {c["cand_id"]: c for c in decision["candidates"]}
    rows = []

    for cand_id in order:
        cand = by_id[cand_id]
        site_labels = []
        struct_vals, cons_vals = [], []
        for (chain, pos), wt, loc in zip(
            cand["pdb_sites"], cand["wt"], cand["locations"]
        ):
            site_labels.append(f"{wt}{chain}{pos}")
            f = features.get(f"{chain}:{pos}", {})
            c = conservation.get(f"{chain}:{pos}", {})
            struct_vals.append(
                (
                    loc or "?",
                    f.get("dsasa"),
                    f.get("buried_frac_on_binding"),
                    f.get("n_contacts_5A"),
                    f.get("n_polar_pairs_3p5A"),
                    f.get("min_dist_to_partner"),
                )
            )
            cons_vals.append(
                (c.get("entropy"), c.get("wt_prob"), c.get("wt_rank"))
            )

        cells = [cand_id, "+".join(site_labels)]
        if use_struct:
            # For k>1 the per-position numbers are joined, not summed: summing
            # dSASA would invent a joint burial the structure never reported.
            cells += [
                "/".join(_fmt(v[0]) for v in struct_vals),
                "/".join(_fmt(v[1], ".0f") for v in struct_vals),
                "/".join(_fmt(v[2], ".2f") for v in struct_vals),
                "/".join(_fmt(v[3]) for v in struct_vals),
                "/".join(_fmt(v[4]) for v in struct_vals),
                "/".join(_fmt(v[5], ".1f") for v in struct_vals),
            ]
        if use_cons:
            cells += [
                "/".join(_fmt(v[0], ".2f") for v in cons_vals),
                "/".join(_fmt(v[1], ".3f") for v in cons_vals),
                "/".join(_fmt(v[2]) for v in cons_vals),
            ]
        rows.append(cells)

    header = ["Candidate", "Positions"]
    legend = []
    if use_struct:
        header += [h for h, _ in STRUCTURE_COLUMNS]
        legend += [f"  {h}: {d}" for h, d in STRUCTURE_COLUMNS]
    if use_cons:
        header += [h for h, _ in CONSERVATION_COLUMNS]
        legend += [f"  {h}: {d}" for h, d in CONSERVATION_COLUMNS]

    if not per_position:
        # The "aggregate only" ablation: the agent is told what the interface
        # looks like overall and which candidates exist, but not how the
        # evidence differs BETWEEN candidates. This is the arm that tests
        # whether per-position diagnostics are doing any work at all.
        vals = [
            features.get(f"{chain}:{pos}", {})
            for cand in decision["candidates"]
            for chain, pos in cand["pdb_sites"]
        ]
        vals = [v for v in vals if v]
        def _mean(key):
            got = [v[key] for v in vals if v.get(key) is not None]
            return sum(got) / len(got) if got else None
        lines = [
            "# Interface summary (aggregate only -- no per-candidate breakdown)",
            f"  candidate positions: {len(vals)}",
            f"  mean dSASA over candidates: {_fmt(_mean('dsasa'), '.0f')} A^2",
            f"  mean contacts within 5 A: {_fmt(_mean('n_contacts_5A'), '.1f')}",
            f"  mean buried fraction: {_fmt(_mean('buried_frac_on_binding'), '.2f')}",
            "",
            "# Candidates",
        ]
        for cand_id in order:
            cand = by_id[cand_id]
            labels = "+".join(
                f"{wt}{ch}{pos}"
                for (ch, pos), wt in zip(cand["pdb_sites"], cand["wt"])
            )
            lines.append(f"  {cand_id}  {labels}")
        return "\n".join(lines)

    if not use_struct and not use_cons:
        lines = ["# Candidates (no diagnostics provided)"]
        for cells in rows:
            lines.append(f"  {cells[0]}  {cells[1]}")
        return "\n".join(lines)

    widths = [
        max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    out = ["# Evidence", "", "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    out.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        out.append("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(header))))
    out += ["", "Column meanings:"] + legend
    return "\n".join(out)


def build_prompt(
    decision: dict,
    features: dict,
    conservation: dict,
    arm: str,
    order: list[str],
) -> tuple[str, str]:
    state = render_state(decision, features, conservation, arm, order)
    brief = TASK_BRIEF.format(
        complex_id=decision["complex_id"],
        pdb_id=decision["pdb_id"],
        chains1=decision["chains1"],
        chains2=decision["chains2"],
        protein1=decision["protein1"],
        protein2=decision["protein2"],
        k=decision["k"],
        solver_sentence=SOLVER_SENTENCE.get(
            decision.get("solver_model", "best_measured"), ""
        ),
        state=state,
        depth_instruction=DEPTH_INSTRUCTION,
        action_formatting_instructions=ACTION_FORMATTING,
    )
    system = SYSTEM.format(reasoning_formatting=REASONING_FORMATTING)
    return system, brief


_CHOICE_RE = re.compile(r"<choice>\s*([A-Za-z0-9_\-]+)\s*</choice>", re.I)


def parse_choice(text: str, valid_ids: set[str]) -> tuple[str | None, str]:
    """Extract the chosen candidate id, or a reason it could not be parsed.

    Returns (cand_id, status). Status is one of ok / no_tag / unknown_id /
    multiple_tags. An unparseable answer is NEVER silently turned into a
    default or a null that later arithmetic treats as a real choice -- it is
    an outcome in its own right and scoring gates on it.
    """
    hits = _CHOICE_RE.findall(text or "")
    if not hits:
        # Last-resort: a bare candidate id alone on a line.
        loose = [
            tok
            for line in (text or "").splitlines()
            for tok in [line.strip().strip(".,:;*`")]
            if tok in valid_ids
        ]
        if len(set(loose)) == 1:
            return loose[0], "ok_loose"
        return None, "no_tag"
    uniq = {h.strip() for h in hits}
    if len(uniq) > 1:
        return None, "multiple_tags"
    cand = hits[-1].strip()
    if cand not in valid_ids:
        return None, "unknown_id"
    return cand, "ok"


RETRY_TEMPLATE = """Your answer could not be used: {reason}.

Reply with nothing but the final action line, choosing exactly one candidate id \
from the list you were given:
<choice>CAND_ID</choice>"""

RETRY_REASON = {
    "no_tag": "no <choice>...</choice> tag was present",
    "unknown_id": "the id in the tag is not one of the listed candidates",
    "multiple_tags": "more than one different candidate id was tagged",
}
