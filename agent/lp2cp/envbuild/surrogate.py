"""The declared response model. THIS IS THE ONE PART THAT IS NOT MEASURED.

Read this before reading any number this project reports.

What is real: the parents are experimental cyclic peptide-receptor complexes
from CPSea_PDB. Their per-residue burial, cross-interface contacts, polar
pairs, backbone dihedrals, ring topology and buried area are measured from
deposited coordinates by `diagnostics.py`. So WHERE each parent's
closure-interface conflict sits, HOW MUCH interface is at stake, and HOW HARD
a given parent is all come from real structure.

What is modelled: how those objectives RESPOND to a hypothetical edit. No
Rosetta, no Boltz-2, no GenKIC ran here. `apply()` is a closed-form function of
the measured diagnostics plus a seeded deterministic noise stream.

Why that is the right instrument for the question being asked. The claim under
test is about a CONTROLLER -- can an LLM read structured diagnostics and decide
where to spend a scarce edit budget when closure and interface compete for the
same positions. Measuring a controller needs an environment whose optimum is
known exactly, which no real backend gives you: with Boltz-2 in the loop you
get a number per campaign and no idea what the best attainable was, so "regret"
is not computable and a win cannot be separated from an easy parent. This
environment is enumerable, so every controller is scored against the exact
oracle of the same action space. The cost is that a qualifying design here is
not a biophysical claim, and nothing in this repo says otherwise.

What this therefore does and does not license:
  - LICENSED: "controller A allocates budget better than controller B under a
    closure/interface conflict", "the execution service catches N% of scope
    leaks", "relaxation rate rises as k falls", "escalating evidence instead of
    editing changes the search-time-to-held-out gap by X".
  - NOT LICENSED: any statement that a sequence produced here binds anything,
    or any ddG / ipTM / BSA number quoted as a prediction.

Every ScoreCard carries `provenance="surrogate_v1:<hash>"` so a row cannot be
mistaken for a measurement downstream. Swapping in a real backend is a
`backends/` subclass; nothing in the agent, the service or the scorer changes.

Calibration anchors, so the constants are not arbitrary. Two are TUNED and one
is EMERGENT, and the difference is stated because a tuned quantity cannot then
be offered as evidence that the environment is realistic:

  - TUNED. Mean |epistasis| for joint interventions is set to 1.29, the value
    Stage 0 MEASURED on SKEMPI 2.0 k=2 decisions (see ../../HANDOFF.md).
    Verified on the built benchmark: 1.324 REU over 1,240 pairs.
  - TUNED. ipTM, BSA, ddG and strain scales are set so that an unmodified
    parent sits just below the design doc's worked-example thresholds -- a near
    miss, which is what iteration 0 of the doc produces (zero qualify, 14 near
    misses). The per-objective constants carry their measured distributions.
  - EMERGENT, not tuned. How often a greedy singles controller is optimal at
    k=2: 37.5% here (12 of 32 parents), against the 45.8% Stage 0 measured on
    SKEMPI. Nothing was fitted to that number -- it falls out of the measured
    interface geometry plus the epistasis term. It landing in the same regime
    is weak corroboration that the action space has a comparable amount of
    coordination in it; it is NOT a calibration success, and the gap means
    coordinated moves matter somewhat MORE here than in SKEMPI.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ..enums import Operator
from ..types import ParentProblem, ResidueDiag, ScoreCard

SURROGATE_VERSION = "surrogate-v2"


@dataclass(frozen=True)
class SurrogateConfig:
    """Frozen constants. Part of the environment hash; never tuned per parent."""

    # --- ddG_bind, REU-like. Built from measured interface terms. ------------
    # Calibrated over 150 CPSea parents so the iteration-0 population STRADDLES
    # the design doc's -30 REU ceiling (measured median -29.6 at these
    # weights). An earlier set was ~2x stronger and put the median at -58,
    # which made ddG unfailable and so removed INTERFACE_CONTACT_LOSS from the
    # defect enum entirely -- the benchmark cannot contain a defect class no
    # parent can exhibit.
    ddg_per_contact: float = -0.080
    ddg_per_polar: float = -0.75
    ddg_per_100a2_dsasa: float = -1.30
    ddg_offset: float = -4.0
    # Fraction of a position's interface term that a SUBSTITUTE gives up. The
    # solver is assumed to pick a good identity, so this is well below 1.
    sub_interface_give_up: float = 0.42
    # ... and what a backbone REBUILD gives up at a position inside the span.
    rebuild_interface_give_up: float = 0.72
    narrow_interface_give_up: float = 0.26
    anchor_interface_give_up: float = 0.80

    # --- interface GAIN (v2) ------------------------------------------------
    # A substitution at a position with HEADROOM can ADD burial: the solver
    # picks a larger side chain that fills a gap, which is what ProteinMPNN
    # does at a partially exposed interface position. Without this term no
    # operator could ever increase BSA, and v1 measured the consequence: the
    # BURIAL_DEFICIT class had 0 of 7 repairable parents and 22 of 32 parents
    # had no contract-holding repair anywhere in a ~140-cell action space.
    #
    # Only SUBSTITUTE and RESTRAIN_AND_REPACK gain. A backbone rebuild moves
    # the segment wholesale and is modelled as giving up contact, not finding
    # it -- which is what keeps the closure/interface conflict intact.
    sub_interface_gain: float = 0.55
    repack_interface_gain: float = 0.22
    gain_dist_lo: float = 3.0
    gain_dist_hi: float = 6.5
    gain_dsasa_per_unit: float = 46.0
    gain_ddg_per_unit: float = -2.1

    # --- BSA ----------------------------------------------------------------
    # Fraction of native BSA an unguided cyclisation already costs, per unit of
    # interface mass sitting next to the closure bond. This is what makes
    # iteration 0 a near miss rather than a pass.
    #
    # Calibrated: the measured closure/interface overlap over 150 CPSea parents
    # runs 0.10-0.46 with a median of 0.23, so a penalty of 0.75 puts the
    # median parent at 0.83 of native BSA -- just above the 0.80 floor -- and
    # fails the upper third. At the original 0.30 no parent could ever fail
    # BURIAL_DEFICIT, which is the defect the LP->CP conflict is most directly
    # about.
    cyc_bsa_penalty: float = 0.75

    # --- strain -------------------------------------------------------------
    # How much of the measured strain inside a rebuilt span is relieved.
    rebuild_strain_relief: float = 0.82
    narrow_strain_relief: float = 0.55
    sub_strain_relief: float = 0.12
    # Strain the rebuild ITSELF introduces, per residue moved. A rebuild is not
    # free: widening the span relieves more and costs more, which is what makes
    # "which span" a real decision rather than "the biggest span wins".
    rebuild_strain_cost_per_res: float = 0.21
    restrain_strain_cost: float = 0.15

    # --- free-peptide conformer population ----------------------------------
    # Rigidification from a rebuild near the closure bond, and the sequence
    # terms that set the parent's baseline flexibility.
    conf_base: float = 0.085
    conf_per_gly: float = -0.011
    conf_per_pro: float = 0.016
    conf_per_ring_res: float = -0.0032
    conf_rebuild_gain: float = 0.055
    conf_narrow_gain: float = 0.022
    conf_sub_gain: float = 0.010

    # --- ipTM ---------------------------------------------------------------
    # Logistic in normalised (ddG, BSA, strain). Scales chosen so a healthy
    # complex lands near 0.85 and a strained near-miss near 0.78.
    iptm_bias: float = 1.05
    iptm_w_ddg: float = 0.055
    iptm_w_bsa: float = 1.35
    iptm_w_strain: float = -0.085

    # --- evaluation noise ---------------------------------------------------
    # The latent truth is fixed per candidate; an evaluation at n seeds returns
    # truth + noise/sqrt(n). This is what makes EVIDENCE_ESCALATION a real
    # decision: at 5 seeds a candidate can look qualifying and fail at 10.
    noise_iptm: float = 0.055
    noise_bsa_frac: float = 0.075
    noise_ddg: float = 3.1
    noise_strain: float = 0.55
    noise_conf: float = 0.021
    max_seeds: int = 20

    # --- epistasis ----------------------------------------------------------
    # Mean |epistasis| for a joint intervention, REU-like. Set to Stage 0's
    # MEASURED 1.29 kcal/mol on SKEMPI k=2; 1 REU is treated as 1 kcal/mol for
    # this purpose, which is the usual rough equivalence and is declared here
    # rather than assumed silently.
    epistasis_scale: float = 1.29

    # --- held-out protocol --------------------------------------------------
    # The doc's held-out protocol: fresh folding seeds, a second predictor,
    # extended free-peptide sampling. Modelled as an independent noise stream
    # plus the second predictor's systematic offset, which is what makes a
    # search-time-to-held-out GAP exist at all.
    heldout_seeds: int = 10
    heldout_predictor_bias_iptm: float = -0.022
    heldout_predictor_bias_ddg: float = 1.6
    heldout_conf_bias: float = -0.009

    # --- backend fidelity ---------------------------------------------------
    # PROTEAN's measured finding is that backend SCOPE LEAKAGE and refolding
    # failure are the two dominant bottlenecks, not constraint compilation. A
    # backend that never leaks would make the execution service's verification
    # step untestable, so the simulator leaks at a declared rate and the
    # service is measured on how much of it it catches.
    scope_leak_rate: float = 0.12
    budget_overrun_rate: float = 0.08
    refold_failure_rate: float = 0.07

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]


CFG = SurrogateConfig()


def provenance(cfg: SurrogateConfig = CFG) -> str:
    return f"{SURROGATE_VERSION}:{cfg.digest()}"


# ---------------------------------------------------------------------------
# Deterministic noise
# ---------------------------------------------------------------------------

def _unit(*parts) -> float:
    """A deterministic draw in [0,1) from the hash of its arguments.

    Not an RNG object: a seeded `random.Random` would make a draw depend on how
    many draws preceded it, so adding a metric would silently change every
    number already recorded. Hashing the full key makes each draw independent
    of evaluation order, which is what lets the enumerated table be rebuilt in
    any order and come out identical.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _normal(*parts) -> float:
    """Standard normal from two independent uniforms (Box-Muller)."""
    u1 = max(_unit("n1", *parts), 1e-12)
    u2 = _unit("n2", *parts)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Interface and closure terms, from MEASURED diagnostics
# ---------------------------------------------------------------------------

def interface_mass(r: ResidueDiag, cfg: SurrogateConfig = CFG) -> float:
    """How much binding this position is worth, in REU-like units.

    All three inputs are measured: heavy-atom contacts within 5 A, polar pairs
    within 3.5 A, and side-chain area buried on binding. Returns a NEGATIVE
    number -- the contribution to ddG_bind -- so "more negative" means "this
    position is carrying more of the interface".
    """
    return (
        cfg.ddg_per_contact * r.contacts_5a
        + cfg.ddg_per_polar * r.polar_3p5a
        + cfg.ddg_per_100a2_dsasa * (r.dsasa / 100.0)
    )


def interface_headroom(r: ResidueDiag, cfg: SurrogateConfig = CFG) -> float:
    """How much contact a redesign at this position could ADD. Measured inputs.

    Zero where the residue is already packed against the partner (nothing to
    gain) and zero where it is too far to reach it; largest for a partially
    exposed residue in the 3-6.5 A band, where a bigger side chain has
    somewhere to go. Both inputs are measured from deposited coordinates.
    """
    if r.min_dist <= cfg.gain_dist_lo or r.min_dist >= cfg.gain_dist_hi:
        return 0.0
    span = cfg.gain_dist_hi - cfg.gain_dist_lo
    mid = cfg.gain_dist_lo + span / 2.0
    prox = 1.0 - abs(r.min_dist - mid) / (span / 2.0)
    return max(0.0, (1.0 - r.buried_frac) * prox)


def closure_weight(r: ResidueDiag, n_pos: int) -> float:
    """How much this position's strain belongs to the ring closure. Measured.

    1.0 at an anchor, decaying with distance AROUND THE RING. The decay length
    is two residues, which is what the design doc's worked example shows
    (strain at 10-12 in a 12-mer, i.e. the closure bond and its two
    neighbours).
    """
    return math.exp(-r.ring_dist / 2.0)


def parent_scorecard(
    residues: tuple[ResidueDiag, ...],
    native_bsa: float,
    sequence: str,
    anchors: tuple[int, ...],
    parent_id: str,
    n_seeds: int = 5,
    cfg: SurrogateConfig = CFG,
) -> ScoreCard:
    """Iteration 0: what an unguided cyclisation of this parent evaluates to.

    The design doc's iteration 0 produces zero qualifying designs and 14 near
    misses, so this deliberately lands just below the thresholds for most
    parents: the cyclisation keeps most of the measured interface but pays a
    burial penalty proportional to how much of that interface sits next to the
    closure bond -- which is the conflict, read straight off the structure.
    """
    n = len(residues)
    total_mass = sum(interface_mass(r, cfg) for r in residues)
    # The share of interface mass sitting within the closure neighbourhood.
    # Entirely measured: it is high when the binding hotspots happen to be next
    # to where the ring has to close, which is parent-specific and real.
    near = sum(interface_mass(r, cfg) * closure_weight(r, n) for r in residues)
    overlap = (near / total_mass) if abs(total_mass) > 1e-9 else 0.0

    bsa = native_bsa * (1.0 - cfg.cyc_bsa_penalty * overlap)
    ddg = cfg.ddg_offset + total_mass * (1.0 - 0.5 * cfg.cyc_bsa_penalty * overlap)
    strains = [r.strain_rel for r in residues]
    max_strain = max(strains) if strains else 0.0
    locus = tuple(r.pos for r in residues if r.strain_rel >= max_strain - 1e-6)
    conf = (
        cfg.conf_base
        + cfg.conf_per_gly * sequence.count("G")
        + cfg.conf_per_pro * sequence.count("P")
        + cfg.conf_per_ring_res * n
    )
    hbonds = tuple(r.pos for r in residues if r.polar_3p5a > 0)
    latent = {
        "iptm": _iptm(ddg, bsa, native_bsa, max_strain, cfg),
        "bsa": bsa,
        "ddg": ddg,
        "strain": max_strain,
        "conf": max(conf, 0.001),
    }
    return _observe(latent, locus, hbonds, 0, parent_id, "parent", n_seeds, cfg)


def _iptm(ddg: float, bsa: float, native_bsa: float, strain: float,
          cfg: SurrogateConfig) -> float:
    z = (
        cfg.iptm_bias
        + cfg.iptm_w_ddg * (-ddg)
        + cfg.iptm_w_bsa * (bsa / max(native_bsa, 1.0))
        + cfg.iptm_w_strain * strain
    )
    return 1.0 / (1.0 + math.exp(-(z - 2.0)))


# ---------------------------------------------------------------------------
# Observation: latent truth -> what an evaluation at n seeds reports
# ---------------------------------------------------------------------------

def _observe(
    latent: dict,
    locus: tuple[int, ...],
    hbonds: tuple[int, ...],
    edit_distance: int,
    parent_id: str,
    cand_key: str,
    n_seeds: int,
    cfg: SurrogateConfig = CFG,
    heldout: bool = False,
) -> ScoreCard:
    """Report a latent candidate at a given evaluation depth.

    Noise falls as 1/sqrt(n_seeds), so escalating evidence genuinely buys
    certainty and genuinely costs the budget it was spent from. The noise
    stream is keyed by (parent, candidate, metric, depth, protocol) and nothing
    else -- in particular not by evaluation ORDER -- so re-running a campaign
    reproduces every number.

    `heldout=True` switches to an independent stream and adds the second
    predictor's systematic offset. That is the design doc's held-out protocol,
    and the gap between the two is the number the doc says to report.
    """
    stream = "heldout" if heldout else "search"
    n = max(1, min(int(n_seeds), cfg.max_seeds))
    s = 1.0 / math.sqrt(n)

    def draw(metric: str, sd: float) -> float:
        return sd * s * _normal(stream, parent_id, cand_key, metric, n)

    iptm = latent["iptm"] + draw("iptm", cfg.noise_iptm)
    bsa = latent["bsa"] * (1.0 + draw("bsa", cfg.noise_bsa_frac))
    ddg = latent["ddg"] + draw("ddg", cfg.noise_ddg)
    strain = latent["strain"] + draw("strain", cfg.noise_strain)
    conf = latent["conf"] + draw("conf", cfg.noise_conf)
    if heldout:
        iptm += cfg.heldout_predictor_bias_iptm
        ddg += cfg.heldout_predictor_bias_ddg
        conf += cfg.heldout_conf_bias

    # Seed mode-consensus: how many of n seeds land in the modal pose. Driven
    # by the latent ipTM, so a marginal candidate also has a marginal
    # consensus -- which is why the doc pairs the two in one clause.
    p_mode = min(max(latent["iptm"] * 1.05, 0.05), 0.985)
    agree = sum(
        1 for i in range(n)
        if _unit(stream, parent_id, cand_key, "seed", n, i) < p_mode
    )
    return ScoreCard(
        iptm=round(min(max(iptm, 0.0), 1.0), 4),
        seed_consensus=int(agree),
        n_seeds=n,
        bsa=round(max(bsa, 0.0), 1),
        ddg_bind=round(ddg, 2),
        max_strain=round(max(strain, 0.0), 3),
        strain_locus=tuple(locus),
        free_conformer_frac=round(min(max(conf, 0.0), 1.0), 4),
        edit_distance=int(edit_distance),
        hbonds_retained=tuple(sorted(hbonds)),
        provenance=provenance(cfg),
    )


# ---------------------------------------------------------------------------
# Applying an intervention
# ---------------------------------------------------------------------------

GIVE_UP = {
    Operator.SUBSTITUTE: "sub_interface_give_up",
    Operator.REBUILD_SEGMENT: "rebuild_interface_give_up",
    Operator.NARROW_PIVOTS: "narrow_interface_give_up",
    Operator.ANCHOR_SHIFT: "anchor_interface_give_up",
    Operator.RESTRAIN_AND_REPACK: None,
}
STRAIN_RELIEF = {
    Operator.SUBSTITUTE: "sub_strain_relief",
    Operator.REBUILD_SEGMENT: "rebuild_strain_relief",
    Operator.NARROW_PIVOTS: "narrow_strain_relief",
    Operator.ANCHOR_SHIFT: "rebuild_strain_relief",
    Operator.RESTRAIN_AND_REPACK: None,
}
CONF_GAIN = {
    Operator.SUBSTITUTE: "conf_sub_gain",
    Operator.REBUILD_SEGMENT: "conf_rebuild_gain",
    Operator.NARROW_PIVOTS: "conf_narrow_gain",
    Operator.ANCHOR_SHIFT: "conf_rebuild_gain",
    Operator.RESTRAIN_AND_REPACK: None,
}


Group = tuple[Operator, tuple[int, ...]]


def latent_of_groups(
    parent: ParentProblem,
    groups: tuple[Group, ...],
    intervention_id: str,
    cfg: SurrogateConfig = CFG,
) -> tuple[dict, tuple[int, ...], tuple[int, ...]]:
    """The latent (noise-free) outcome of one intervention on one parent.

    An intervention is a set of (operator, positions) GROUPS, not a single
    operator, because the design doc's open question requires coordinated
    moves: "open position 6 AND rebuild 10-12". A single-operator intervention
    is just a one-group intervention. Per-group operators matter because the
    two halves of a coordinated move pay different costs -- a substitution
    gives up ~0.42 of a position's interface mass, a backbone rebuild ~0.72 --
    and collapsing them to one rate would make coordinated moves look uniformly
    cheaper or dearer than they are.

    Returns (latent metrics, strain locus, retained H-bond positions).

    The conflict lives in two places in this function and nowhere else: strain
    relief is proportional to the MEASURED strain inside the touched span, and
    interface cost is proportional to the MEASURED interface mass inside the
    same span. When a parent's hotspots happen to sit next to its closure bond,
    those two pull against each other and no fixed rule resolves it; when they
    do not, the decision is easy. Which parent is which is read off the
    structure, not chosen.
    """
    res = {r.pos: r for r in parent.residues}
    n = parent.n_pos
    card0 = parent.scorecard0

    # Resolve each position to the operator acting on it. A position named by
    # two groups takes the one that gives up MORE -- a backbone rebuild through
    # a position is not made gentler by also substituting there.
    per_pos: dict[int, Operator] = {}
    for op, positions in groups:
        for p in positions:
            if p not in res:
                continue
            prev = per_pos.get(p)
            if prev is None:
                per_pos[p] = op
            else:
                def gu(o: Operator) -> float:
                    name = GIVE_UP.get(o)
                    return getattr(cfg, name) if name else 0.0
                per_pos[p] = op if gu(op) > gu(prev) else prev
    touched = tuple(sorted(per_pos))

    def give_up_of(p: int) -> float:
        name = GIVE_UP.get(per_pos[p])
        return getattr(cfg, name) if name else 0.0

    # --- interface cost -----------------------------------------------------
    lost_mass = sum(interface_mass(res[p], cfg) * give_up_of(p) for p in touched)
    _GAIN = {Operator.SUBSTITUTE: "sub_interface_gain",
             Operator.RESTRAIN_AND_REPACK: "repack_interface_gain"}
    gained = 0.0
    for p in touched:
        name = _GAIN.get(per_pos[p])
        if name:
            gained += interface_headroom(res[p], cfg) * getattr(cfg, name)
    # lost_mass is negative*positive = negative; giving it up makes ddG LESS
    # negative, so it is SUBTRACTED. The gain term pushes it back down.
    ddg = card0.ddg_bind - lost_mass + cfg.gain_ddg_per_unit * gained
    bsa = (card0.bsa - sum(res[p].dsasa * give_up_of(p) for p in touched)
           + cfg.gain_dsasa_per_unit * gained)

    # --- epistasis for joint interventions ---------------------------------
    # Only for 2+ touched positions, seeded per unordered pair so that the same
    # pair always interacts the same way regardless of which intervention
    # contains it. Mean |value| is the config's epistasis_scale, which is
    # Stage 0's measured 1.29 kcal/mol.
    if len(touched) >= 2:
        eps = 0.0
        for i, a in enumerate(touched):
            for b in touched[i + 1:]:
                # E[|N(0,1)|] = sqrt(2/pi); dividing by it makes the mean
                # absolute epistasis equal to epistasis_scale exactly.
                eps += (
                    cfg.epistasis_scale
                    / math.sqrt(2.0 / math.pi)
                    * _normal("epi", parent.parent_id, a, b)
                )
        ddg += eps

    # --- strain -------------------------------------------------------------
    new_strain: dict[int, float] = {}
    for p, r in res.items():
        relief_name = STRAIN_RELIEF.get(per_pos[p]) if p in per_pos else None
        relief = getattr(cfg, relief_name) if relief_name else 0.0
        new_strain[p] = r.strain_rel * (1.0 - relief)

    moved = [p for p in touched if per_pos[p] in (
        Operator.REBUILD_SEGMENT, Operator.ANCHOR_SHIFT, Operator.NARROW_PIVOTS)]
    if moved:
        # A rebuild introduces strain of its own at the span's edges: the span
        # has to hand off to residues that did not move. Scaling with span
        # length is what stops "rebuild everything" from being free.
        cost = cfg.rebuild_strain_cost_per_res * len(moved)
        for p in moved:
            new_strain[p] += cost * (0.5 if len(moved) > 1 else 1.0)
    for p in touched:
        if per_pos[p] == Operator.RESTRAIN_AND_REPACK:
            new_strain[p] += cfg.restrain_strain_cost

    max_strain = max(new_strain.values()) if new_strain else 0.0
    locus = tuple(p for p, s in new_strain.items() if s >= max_strain - 1e-6)

    # --- free-peptide conformer population ---------------------------------
    conf = card0.free_conformer_frac
    for p in touched:
        name = CONF_GAIN.get(per_pos[p])
        gain = getattr(cfg, name) if name else (
            cfg.conf_narrow_gain if per_pos[p] == Operator.RESTRAIN_AND_REPACK else 0.0
        )
        # Rigidification only counts near the closure bond: restraining a
        # position on the far side of the ring does not pre-organise the turn
        # that was floppy. closure_weight is measured ring geometry.
        conf += gain * closure_weight(res[p], n)

    # --- H-bonds ------------------------------------------------------------
    # A polar contact at a touched position survives with a probability that
    # falls with how much the operator gives up there. This is the mechanism
    # behind the doc's iteration 2: "the turn rebuild rotated the backbone
    # enough to pull R7 off E191", recovered by narrowing pivots and adding
    # restraints -- NARROW_PIVOTS gives up less, so it keeps more.
    kept: list[int] = []
    for p in card0.hbonds_retained:
        if p not in per_pos:
            kept.append(p)
            continue
        survive = (
            0.95 if per_pos[p] == Operator.RESTRAIN_AND_REPACK
            else 1.0 - give_up_of(p)
        )
        if _unit("hbond", parent.parent_id, intervention_id, p) < survive:
            kept.append(p)
    for p in touched:
        # Restrain-and-repack can RECOVER a contact the previous step lost,
        # which is the only operator that can, and is why it is worth a slot.
        if per_pos[p] != Operator.RESTRAIN_AND_REPACK:
            continue
        r = res[p]
        if r.polar_3p5a > 0 and p not in kept:
            if _unit("recover", parent.parent_id, intervention_id, p) < 0.45:
                kept.append(p)

    latent = {
        "iptm": _iptm(ddg, bsa, parent.native_bsa, max_strain, cfg),
        "bsa": bsa,
        "ddg": ddg,
        "strain": max_strain,
        "conf": max(conf, 0.001),
    }
    return latent, locus, tuple(sorted(set(kept)))


def latent_of(
    parent: ParentProblem,
    operator: Operator,
    touched: tuple[int, ...],
    intervention_id: str,
    cfg: SurrogateConfig = CFG,
) -> tuple[dict, tuple[int, ...], tuple[int, ...]]:
    """Single-operator convenience wrapper over `latent_of_groups`."""
    return latent_of_groups(parent, ((operator, tuple(touched)),), intervention_id, cfg)


def evaluate(
    parent: ParentProblem,
    groups: tuple[Group, ...],
    intervention_id: str,
    edit_distance: int,
    n_seeds: int = 5,
    heldout: bool = False,
    cfg: SurrogateConfig = CFG,
) -> ScoreCard:
    """Full path: latent outcome, then an observation at the given depth."""
    latent, locus, hbonds = latent_of_groups(parent, groups, intervention_id, cfg)
    return _observe(
        latent, locus, hbonds, edit_distance, parent.parent_id,
        intervention_id, n_seeds, cfg, heldout=heldout,
    )


def latent_card(
    parent: ParentProblem,
    groups: tuple[Group, ...],
    intervention_id: str,
    edit_distance: int,
    cfg: SurrogateConfig = CFG,
) -> ScoreCard:
    """The noise-free truth. Used ONLY by the oracle and the scorer.

    No agent-visible path reaches this. An arm that could read it would be
    scoring itself against the answer key, which is the self-grading problem
    the design doc says to avoid.
    """
    latent, locus, hbonds = latent_of_groups(parent, groups, intervention_id, cfg)
    return ScoreCard(
        iptm=round(latent["iptm"], 4),
        seed_consensus=int(round(min(max(latent["iptm"] * 1.05, 0.0), 1.0) * cfg.heldout_seeds)),
        n_seeds=cfg.heldout_seeds,
        bsa=round(latent["bsa"], 1),
        ddg_bind=round(latent["ddg"], 2),
        max_strain=round(latent["strain"], 3),
        strain_locus=tuple(locus),
        free_conformer_frac=round(latent["conf"], 4),
        edit_distance=int(edit_distance),
        hbonds_retained=hbonds,
        provenance=provenance(cfg) + ":latent",
    )
