"""Does the EXACT pll optimum introduce chemistry no one asked it about?

WHY THIS EXISTS. `TargetPSSM.best_subs` returns the exact argmax of a mean of
independent per-position terms, so on the single objective it cannot be beaten
by any search -- that is measured (6/6 free optimum, docs/pll_inloop_results).
The open question is not whether it optimizes pll. It is whether the sequence it
returns is a peptide anyone could make or use. pll_bg knows about one axis. A
free thiol, an aspartimide-forming D-G, a trypsin site the native did not have:
the table has no column for any of them and will take them for a few millinats.

So this module does NOT propose a better optimizer. It measures the damage the
exact optimizer does on axes outside its objective, on the targets and the
cached PSSMs we already have, for CPU seconds. If that damage is small, the
multi-objective angle is weak and we learn it for one job. If it is large, this
file is also the grading instrument: `violations introduced` is the number the
constraint-specification arms get compared on.

TWO AXES PER CHECK, AND THEY ANSWER DIFFERENT QUESTIONS.

  `klass` -- how hard is this to OPTIMIZE once you know you want it?
      SEPARABLE  per-position; folds into the L x 20 table, still one sort.
      LOCAL      adjacent/window motif; exact by DP over positions.
      GLOBAL     linear in residue counts (charge, GRAVY, MW); exact by ILP.
      JOINT      not a function of per-position or count features at all;
                 needs an evaluation of the whole sequence. No table exists.
      Everything except JOINT is solvable EXACTLY and FREELY. So `klass` is the
      bad news for an LLM search arm, not the good news: annotating the table
      with these columns and running a DP/ILP beats a search by construction.

  `needs_ctx` -- does the RIGHT RULE vary by target?
      False  one universal rule is correct everywhere (D-G forms aspartimide in
             every peptide).
      True   the rule depends on target-specific facts a universal column set
             does not contain: a peptide that is already disulfide-cyclized
             needs exactly two Cys AT GIVEN POSITIONS, while a linear one needs
             at most one anywhere. Hardcode either and you are exactly optimal
             against the wrong feasible set on the other.

  `needs_ctx` is the whole case for a specification arm. A violation with
  needs_ctx=False is arm-1 territory: write the column once, by hand, and the
  solver prevents it forever. A violation with needs_ctx=True is the only thing
  a per-target specifier can buy that a careful hand-written universal spec
  cannot. Report the two separately or the angle will look stronger than it is.

NATIVE-RELATIVE, DELIBERATELY. Several natives already carry liabilities --
they are real peptides, not clean-room sequences. Counting a D-G the native
brought with it as a failure of the optimizer is wrong and would inflate every
arm equally. `audit` reports `introduced` (present in the mutant, absent in the
native) as the headline and `absolute` beside it.

WHAT THIS BATTERY IS NOT. It is a battery of RULES, every one of them a
literature heuristic with a `source` string, not a measurement. The JOINT
checks are declared and deliberately NOT implemented: aggregation propensity,
membrane permeability, immunogenicity and off-target binding are the axes that
cannot be tabulated, and listing them as `tool_required` is the honest
statement that this file's coverage stops before the interesting part. Do not
report a clean battery as a clean peptide.

Run:
  python agent/lnr/pepliab.py --selftest
  python agent/lnr/pepliab.py --audit --pssm-dir /zfsauton/scratch/yixiz/lp2cp_stage2/lnr/pssm \
      --pdb-dir /zfsauton/scratch/yixiz/LNR/pdbs --k-mut 2 --out liab_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Separability classes. The ordering is "how much structure an exact solver can
# exploit", worst (easiest for the solver) to best (impossible for it).
# ---------------------------------------------------------------------------

SEPARABLE = "separable"   # per-position  -> in the table, one sort
LOCAL = "local"           # motif/window  -> exact by DP
GLOBAL = "global"         # linear counts -> exact by ILP
JOINT = "joint"           # whole-sequence-> no table; evaluation required

SOLVABLE_EXACTLY = (SEPARABLE, LOCAL, GLOBAL)

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Kyte & Doolittle 1982 hydropathy. Used only for GRAVY, which is a mean of
# per-residue values -- hence GLOBAL/linear, hence exactly constrainable.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}

# Charge at ~pH 7.4, counting only the dominant ionizable side chains. His is
# deliberately given 0.1 rather than 0 or 1: at pH 7.4 it is mostly neutral.
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# Residues whose runs drive aggregation / poor aqueous solubility. This is a
# coarse set; the check it feeds is a heuristic flag, not a predictor.
HYDROPHOBIC = set("AILMFWVY")


@dataclass(frozen=True)
class Violation:
    check: str
    klass: str
    needs_ctx: bool
    positions: tuple          # 0-based positions implicated
    detail: str

    def key(self):
        """Identity for native-vs-mutant set difference. Includes positions, so
        a D-G that MOVED counts as introduced at its new site and resolved at
        the old one -- which is the truthful accounting for a point mutation."""
        return (self.check, self.positions)

    def to_json(self):
        return {"check": self.check, "klass": self.klass,
                "needs_ctx": self.needs_ctx,
                "positions": list(self.positions), "detail": self.detail}


@dataclass
class Ctx:
    """Target-specific facts a universal column set does not contain.

    Every field here is a reason some check's correct form differs by target.
    An empty Ctx is the "universal spec" case and is what arm 1 gets."""
    target_id: str = ""
    ss_pairs: tuple = ()        # (i, j) 0-based peptide positions bonded to EACH
                                # OTHER in the deposited structure (SSBOND)
    ss_anchor: tuple = ()       # 0-based peptide positions disulfide-bonded to
                                # the RECEPTOR -- a covalent tether, not an
                                # internal staple. See `receptor_disulfide`.
    cyclic: bool = False        # head-to-tail cyclized
    charge_window: tuple = (-2.0, 4.0)   # inclusive (lo, hi)
    gravy_max: float = 1.0
    notes: str = ""
    # Provenance of the SSBOND mapping, so a zero can never pass as a finding.
    # See `ctx_from_pdb`: `ss_seen` counts records naming the peptide chain and
    # `ss_mapped` the ones whose residue numbers actually resolved to sequence
    # indices carrying a Cys. ss_seen > ss_mapped means the mapping FAILED.
    ss_seen: int = 0
    ss_mapped: int = 0

    @property
    def ss_positions(self):
        return {p for pair in self.ss_pairs for p in pair}


# ---------------------------------------------------------------------------
# The checks.
#
# Signature: f(seq, ctx) -> list[Violation]. Each is registered with its class
# and its ctx-dependence, and carries a `source` so a threshold cannot enter
# this file without somewhere to point. A check with no source is a check
# someone invented.
# ---------------------------------------------------------------------------

CHECKS = []


def check(name, klass, needs_ctx, source):
    def deco(fn):
        fn.check_name = name
        fn.klass = klass
        fn.needs_ctx = needs_ctx
        fn.source = source
        CHECKS.append(fn)
        return fn
    return deco


def _v(fn, positions, detail):
    return Violation(fn.check_name, fn.klass, fn.needs_ctx,
                     tuple(positions), detail)


@check("unpaired_cys", GLOBAL, True,
       "free thiol: disulfide scrambling / covalent dimerization")
def unpaired_cys(seq, ctx):
    """THE needs_ctx EXEMPLAR. What counts as a Cys violation is a different
    predicate depending on the target's own topology:

      disulfide-cyclized native  -> exactly the two bonded positions must stay
                                    Cys, and any THIRD Cys is a scrambling risk
      linear / head-to-tail      -> any Cys at all is an unpaired thiol

    A universal "<= 1 Cys" column is wrong in the first case (it forbids the
    bond the peptide needs) and too permissive in the second. There is no single
    number that is right for both, which is exactly the claim this module exists
    to test."""
    cys = [i for i, a in enumerate(seq) if a == "C"]
    out = []
    # Positions whose Cys is ACCOUNTED FOR by a bond in the deposited structure
    # -- an internal staple or a tether to the receptor. A Cys here is not a
    # free thiol, and counting it as one would double-report the same residue
    # that `receptor_disulfide` already owns.
    bonded = ctx.ss_positions | set(ctx.ss_anchor)
    if bonded:
        # Only the internal staple is this check's business; losing a receptor
        # tether is `receptor_disulfide`'s, and it is a different failure.
        lost = sorted(p for p in ctx.ss_positions if seq[p] != "C")
        if lost:
            out.append(_v(unpaired_cys, lost,
                          f"mutated away a bonded Cys at {lost}; the "
                          f"disulfide in the deposited structure cannot form"))
        extra = sorted(set(cys) - bonded)
        if extra:
            out.append(_v(unpaired_cys, extra,
                          f"{len(extra)} Cys beyond the native disulfide(s) "
                          f"-> scrambling with the existing bond"))
    elif cys:
        out.append(_v(unpaired_cys, cys,
                      f"{len(cys)} free Cys in a peptide with no native "
                      f"disulfide -> unpaired thiol"))
    return out


@check("receptor_disulfide", GLOBAL, True,
       "peptide Cys disulfide-bonded to the RECEPTOR: a covalent tether, and "
       "mutating it away removes the bond that holds the complex together")
def receptor_disulfide(seq, ctx):
    """THE strongest needs_ctx case in this battery, and the one I got wrong
    first time round.

    An earlier version of this file ignored interchain SSBOND records on the
    grounds that "a bond to the receptor is not something a peptide
    substitution can break". That is false, and backwards: replacing the
    PEPTIDE's Cys is precisely how you break it. On 2hwl the peptide's residue 1
    is covalently bonded to receptor chain B, and the pll table -- which has no
    column for covalent chemistry -- will trade that bond away for a few
    millinats at that position.

    Nothing universal can catch this. It is not a threshold that varies by
    target, it is a FACT about one deposited structure: which peptide position,
    if any, is covalently tethered. A hand-written column set that did not go
    and read every PDB cannot contain it, which is the entire argument for a
    per-target specification step."""
    lost = sorted(p for p in ctx.ss_anchor if p < len(seq) and seq[p] != "C")
    if not lost:
        return []
    return [_v(receptor_disulfide, lost,
               f"mutated away the receptor-tethering Cys at {lost}; the "
               f"interchain disulfide in the deposited structure is destroyed")]


@check("asp_gly", LOCAL, False,
       "Asp-Gly: aspartimide/succinimide formation under Fmoc SPPS and storage")
def asp_gly(seq, ctx):
    return [_v(asp_gly, (i, i + 1), "D-G aspartimide hotspot")
            for i in range(len(seq) - 1) if seq[i] == "D" and seq[i + 1] == "G"]


@check("asp_pro", LOCAL, False,
       "Asp-Pro: acid-labile amide bond, cleaves during TFA deprotection")
def asp_pro(seq, ctx):
    return [_v(asp_pro, (i, i + 1), "D-P acid-labile bond")
            for i in range(len(seq) - 1) if seq[i] == "D" and seq[i + 1] == "P"]


@check("deamidation", LOCAL, False,
       "Asn-Gly / Asn-Ser: deamidation hotspot")
def deamidation(seq, ctx):
    out = []
    for i in range(len(seq) - 1):
        if seq[i] == "N" and seq[i + 1] in "GS":
            out.append(_v(deamidation, (i, i + 1),
                          f"N-{seq[i + 1]} deamidation hotspot"))
    return out


@check("nglyc_sequon", LOCAL, False,
       "N-X-S/T (X != P): N-linked glycosylation sequon")
def nglyc_sequon(seq, ctx):
    out = []
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            out.append(_v(nglyc_sequon, (i, i + 1, i + 2),
                          f"sequon N-{seq[i+1]}-{seq[i+2]}"))
    return out


@check("dkp_risk", LOCAL, True,
       "X-Pro at the N-terminus: diketopiperazine formation during synthesis")
def dkp_risk(seq, ctx):
    """needs_ctx because a head-to-tail cyclized peptide has no N-terminus, so
    the check does not apply at all -- not a different threshold, a different
    question. A universal column would flag every cyclic peptide falsely."""
    if ctx.cyclic or len(seq) < 2:
        return []
    if seq[1] == "P":
        return [_v(dkp_risk, (0, 1), f"{seq[0]}-P at the N-terminus -> DKP")]
    return []


@check("oxidation_prone", SEPARABLE, False,
       "Met / Trp: oxidation liability (per-residue, so it folds into the table)")
def oxidation_prone(seq, ctx):
    return [_v(oxidation_prone, (i,), f"{a} oxidation-prone")
            for i, a in enumerate(seq) if a in "MW"]


@check("trypsin_site", LOCAL, False,
       "K/R not followed by Pro: tryptic cleavage, serum half-life")
def trypsin_site(seq, ctx):
    out = []
    for i, a in enumerate(seq):
        if a not in "KR":
            continue
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if nxt != "P":
            out.append(_v(trypsin_site, (i,), f"{a}{i} tryptic site"))
    return out


@check("hydrophobic_run", LOCAL, False,
       ">=3 consecutive strongly hydrophobic residues: aggregation / solubility")
def hydrophobic_run(seq, ctx):
    out = []
    for m in re.finditer(r"[AILMFWVY]{3,}", seq):
        out.append(_v(hydrophobic_run, tuple(range(m.start(), m.end())),
                      f"hydrophobic run '{m.group()}'"))
    return out


@check("charge_run", LOCAL, False,
       ">=4 consecutive like-signed charges: synthesis and nonspecific binding")
def charge_run(seq, ctx):
    out = []
    for pat, label in ((r"[KR]{4,}", "cationic"), (r"[DE]{4,}", "anionic")):
        for m in re.finditer(pat, seq):
            out.append(_v(charge_run, tuple(range(m.start(), m.end())),
                          f"{label} run '{m.group()}'"))
    return out


@check("net_charge", GLOBAL, True,
       "net charge at pH 7.4 outside the target-class window")
def net_charge(seq, ctx):
    """needs_ctx because the window is a property of where the peptide has to
    work -- a plasma-exposed binder and a cell-penetrant macrocycle do not share
    one. Linear in residue counts, so once the window IS known an ILP enforces
    it exactly; the hard part is the number, not the optimization."""
    q = sum(CHARGE.get(a, 0.0) for a in seq)
    lo, hi = ctx.charge_window
    if q < lo or q > hi:
        return [_v(net_charge, tuple(range(len(seq))),
                   f"net charge {q:+.1f} outside [{lo:+.1f}, {hi:+.1f}]")]
    return []


@check("gravy", GLOBAL, True,
       "GRAVY (mean Kyte-Doolittle) above the solubility cutoff")
def gravy(seq, ctx):
    g = sum(KD[a] for a in seq) / max(1, len(seq))
    if g > ctx.gravy_max:
        return [_v(gravy, tuple(range(len(seq))),
                   f"GRAVY {g:+.2f} > {ctx.gravy_max:+.2f}")]
    return []


# ---------------------------------------------------------------------------
# The axes this battery CANNOT cover. Declared, not implemented.
#
# These are JOINT: not a function of per-position values or residue counts, so
# there is no column to add to the table and no ILP to write. They are the only
# place a search arm is structurally necessary -- and every one of them costs a
# real evaluation, which is why none is in the battery above. A run that reports
# zero violations has said nothing about any of these.
# ---------------------------------------------------------------------------

TOOL_REQUIRED = (
    ("aggregation_propensity", "window-and-context predictor; not additive"),
    ("membrane_permeability", "whole-molecule conformational property"),
    ("immunogenicity", "MHC-II presentation over the full sequence"),
    ("off_target_panel", "binding against other receptors; needs folding"),
    ("synthesis_yield", "sequence-dependent coupling efficiency, empirical"),
)


def battery(seq, ctx=None):
    ctx = ctx or Ctx()
    out = []
    for fn in CHECKS:
        out.extend(fn(seq, ctx))
    return out


def introduced(native, mutant, ctx=None):
    """Violations the mutation ADDED. Set difference on (check, positions)."""
    ctx = ctx or Ctx()
    before = {v.key(): v for v in battery(native, ctx)}
    after = {v.key(): v for v in battery(mutant, ctx)}
    return [v for k, v in after.items() if k not in before]


def tally(vs):
    """Counts split the two ways that matter: by solver class, and by whether a
    universal spec could have contained the rule at all."""
    t = {"n": len(vs),
         "by_klass": {k: 0 for k in (SEPARABLE, LOCAL, GLOBAL, JOINT)},
         "by_check": {},
         "n_needs_ctx": 0, "n_universal": 0}
    for v in vs:
        t["by_klass"][v.klass] += 1
        t["by_check"][v.check] = t["by_check"].get(v.check, 0) + 1
        if v.needs_ctx:
            t["n_needs_ctx"] += 1
        else:
            t["n_universal"] += 1
    return t


# ---------------------------------------------------------------------------
# Target context from the deposited structure.
# ---------------------------------------------------------------------------

def chain_resnums(pdb_path, chain):
    """The peptide chain's residue ids IN SEQUENCE ORDER, as `(resSeq, iCode)`.

    THE INSERTION CODE IS PART OF THE KEY, not a detail. On 2hwl chain A the
    first residues are numbered `1D, 1E, ...`, so resSeq alone is not unique
    within the chain: keying on the number alone resolved `CYS A 1` to the Gly
    at `1D` and the Cys guard then (correctly) refused the whole record. A
    residue id in the PDB is the pair, and both this function and the SSBOND
    reader have to treat it that way.

    This exists because of a silent failure that produced a confident wrong
    answer. `TargetPSSM.positions` is not a list of PDB residue numbers -- it is
    `[0, 1, ..., n-1]`, plain sequence indices. Using it as a resnum->index map
    made SSBOND 4-12 on 4w50 resolve position 4 and drop position 12, so the
    pair was discarded and the audit reported "disulfide-bearing targets: 0"
    across the whole set. Zero liabilities looked like a result.

    So the mapping comes from the deposited coordinates instead: one entry per
    residue of `chain`, in file order, which is the order the peptide sequence
    is read in. Insertion codes and non-contiguous numbering therefore cost
    nothing, and the result is checkable against the sequence -- which
    `ctx_from_pdb` does, by requiring the mapped residue to be a Cys."""
    seen, order = set(), []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 27 or line[21] != chain:
            continue
        key = line[22:27]                    # resSeq + insertion code
        if key in seen:
            continue
        seen.add(key)
        try:
            order.append((int(line[22:26]), line[26].strip(),
                          line[17:20].strip()))
        except ValueError:
            order.append(None)
    return order


def ctx_from_pdb(pdb_path, pep_chain, pep_seq, target_id=""):
    """SSBOND records involving the peptide chain, mapped to sequence indices.

    TWO KINDS, and the distinction is the point:
      intra-peptide (`ss_pairs`)   an internal staple. Both partners are peptide
                                   positions; the peptide needs both to stay Cys
                                   and a third Cys is a scrambling risk.
      peptide<->receptor (`ss_anchor`)  a covalent tether to the target. Only the
                                   peptide side is ours to mutate, and mutating
                                   it destroys the bond. An earlier version of
                                   this function discarded these on reasoning
                                   that was simply wrong; see
                                   `receptor_disulfide`.

    Every mapped index is VERIFIED to carry a Cys in `pep_seq`, and records that
    fail to map are counted into `ss_seen`/`ss_mapped` rather than dropped in
    silence. An audit must be able to say "I looked and found none", which is a
    different statement from "my mapping broke".

    Falls back to an empty Ctx when the file is missing -- the honest default,
    and also exactly the "universal spec" arm."""
    p = Path(pdb_path)
    if not p.exists():
        return Ctx(target_id=target_id)
    order = chain_resnums(p, pep_chain)
    exact, by_num = {}, {}
    for i, r in enumerate(order):
        if r is None:
            continue
        num, icode, name = r
        exact.setdefault((num, icode), i)
        by_num.setdefault(num, []).append((i, name))

    def idx_of(num, icode):
        """Resolve a SSBOND partner to a 0-based sequence index.

        Exact (resSeq, iCode) first. When the SSBOND record leaves the code
        blank but the chain uses codes -- 2hwl writes `CYS A 1` while chain A
        runs `1D, 1E, ...` -- fall back to the residues with that number and
        disambiguate BY RESIDUE NAME, which the record itself asserts is CYS.
        Still ambiguous, or the chosen index is not a Cys in the sequence we
        are scoring: refuse and let `seen > mapped` report it. Guessing here is
        how a wrong position becomes a confident finding."""
        i = exact.get((num, icode))
        if i is None:
            cand = [j for j, name in by_num.get(num, []) if name == "CYS"]
            if len(cand) == 1:
                i = cand[0]
        if i is None or i >= len(pep_seq):
            return None
        # the verification that makes a silent mis-mapping impossible
        return i if pep_seq[i] == "C" else None

    pairs, anchors, seen, mapped = [], [], 0, 0
    for line in p.read_text(errors="ignore").splitlines():
        if not line.startswith("SSBOND"):
            continue
        try:                                 # fixed-width per the PDB spec:
            c1 = line[15]                    # chainID1 col 16
            r1 = (int(line[17:21]), line[21].strip())   # resSeq1 + iCode1
            c2 = line[29]                    # chainID2 col 30
            r2 = (int(line[31:35]), line[35].strip())   # resSeq2 + iCode2
            if len(line) < 36:
                raise IndexError("truncated SSBOND record")
        except (ValueError, IndexError):
            continue
        if pep_chain not in (c1, c2):
            continue
        seen += 1
        if c1 == pep_chain and c2 == pep_chain:
            i, j = idx_of(*r1), idx_of(*r2)
            if i is not None and j is not None:
                pairs.append((min(i, j), max(i, j)))
                mapped += 1
        else:
            r = r1 if c1 == pep_chain else r2
            i = idx_of(*r)
            if i is not None:
                anchors.append(i)
                mapped += 1
    return Ctx(target_id=target_id,
               ss_pairs=tuple(sorted(set(pairs))),
               ss_anchor=tuple(sorted(set(anchors))),
               ss_seen=seen, ss_mapped=mapped)


# ---------------------------------------------------------------------------
# The audit. Reads only CACHED PSSMs -- no folding, no GPU, CPU seconds.
# ---------------------------------------------------------------------------

PSSM_RE = re.compile(r"^pssm_(?P<tid>[^_]+)_(?P<chain>[^_]+)_(?P<model>.+)\.json$")


def audit(pssm_dir, pdb_dir, out_path, k_mut=2, scope="all", metric=None,
          use_ctx=True, verbose=True):
    """For every cached PSSM: native, the exact k_mut optimum, and the exact
    unconstrained optimum -- then the battery on each.

    THREE SEQUENCES, because they bound the question differently.
      native        what the liabilities already are. The baseline.
      best_subs(k)  what the GRADED optimum does at the loop's budget. This is
                    the number that matters for the current experiment.
      best_sequence the free optimum: every position at its argmax. This is the
                    limit of what the single objective wants, and where the
                    damage should be largest.
    """
    from lnr import bindscore as bs  # noqa: E402  (heavy import, CLI-only)

    interface_only = (scope == "iface")
    rows, n_bad = [], 0
    files = sorted(Path(pssm_dir).glob("pssm_*.json"))
    for f in files:
        m = PSSM_RE.match(f.name)
        if not m:
            continue
        tid, chain = m.group("tid"), m.group("chain")
        try:
            p = bs.TargetPSSM.load(f)
        except Exception as e:                       # noqa: BLE001
            n_bad += 1
            if verbose:
                print(f"SKIP {f.name}: {type(e).__name__}: {e}")
            continue
        native = p.pep_seq
        ctx = (ctx_from_pdb(Path(pdb_dir) / f"{tid}.pdb", chain, native, tid)
               if use_ctx else Ctx(target_id=tid))
        subs, _ = p.best_subs(native, k_mut, metric=metric,
                              interface_only=interface_only)
        kseq = list(native)
        for pos, aa in subs:
            kseq[pos] = aa
        kseq = "".join(kseq)
        free = p.best_sequence(metric=metric)

        row = {"target_id": tid, "pep_chain": chain, "model": p.model_name,
               "k_mut": k_mut, "scope": scope, "length": p.n,
               "use_ctx": bool(use_ctx),
               "ss_pairs": [list(x) for x in ctx.ss_pairs],
               "ss_anchor": list(ctx.ss_anchor),
               "ss_seen": ctx.ss_seen, "ss_mapped": ctx.ss_mapped,
               "native_seq": native, "kopt_seq": kseq, "free_seq": free,
               "kopt_subs": [[int(a), b] for a, b in subs],
               "native_score": p.score(native, metric=metric,
                                       interface_only=interface_only),
               "kopt_score": p.score(kseq, metric=metric,
                                     interface_only=interface_only),
               "free_score": p.score(free, metric=metric,
                                     interface_only=interface_only)}
        for label, seq in (("kopt", kseq), ("free", free)):
            ins = introduced(native, seq, ctx)
            row[f"{label}_introduced"] = tally(ins)
            row[f"{label}_introduced_detail"] = [v.to_json() for v in ins]
        row["native_absolute"] = tally(battery(native, ctx))
        row["kopt_absolute"] = tally(battery(kseq, ctx))
        row["free_absolute"] = tally(battery(free, ctx))
        rows.append(row)
        if verbose:
            print(f"{tid}/{chain}  L={p.n:2d}  "
                  f"ss={len(ctx.ss_pairs)}i/{len(ctx.ss_anchor)}r"
                  f"{'' if ctx.ss_seen == ctx.ss_mapped else ' MAPFAIL'}  "
                  f"kopt +{row['kopt_introduced']['n']} "
                  f"(ctx {row['kopt_introduced']['n_needs_ctx']})  "
                  f"free +{row['free_introduced']['n']} "
                  f"(ctx {row['free_introduced']['n_needs_ctx']})")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summarize(rows, n_bad=n_bad, n_files=len(files))
    print(f"\nwrote {len(rows)} rows -> {out}")
    return rows


def summarize(rows, n_bad=0, n_files=0):
    if not rows:
        print("GATE: no rows -- nothing to summarize. "
              "Check --pssm-dir; this is not a result.")
        return
    n = len(rows)
    print(f"\n{'=' * 72}\nLIABILITIES INTRODUCED BY THE EXACT pll OPTIMUM"
          f"\n{'=' * 72}")
    print(f"targets: {n}  (skipped {n_bad} of {n_files} pssm files)")
    # PROVENANCE FIRST. A "0" on the lines below means one of two very different
    # things -- the set has no disulfides, or the mapping failed -- and the first
    # version of this file reported the second as the first across 87 targets.
    seen = sum(r.get("ss_seen", 0) for r in rows)
    mapped = sum(r.get("ss_mapped", 0) for r in rows)
    mapfail = [r["target_id"] for r in rows
               if r.get("ss_seen", 0) != r.get("ss_mapped", 0)]
    print(f"SSBOND records naming the peptide chain: {seen} seen, "
          f"{mapped} mapped to a Cys position")
    if mapfail:
        print(f"  MAPPING FAILED on {len(mapfail)} target(s): "
              f"{', '.join(mapfail[:12])}")
        print(f"  Do NOT read the disulfide lines below as a finding.")
    print(f"  intra-peptide staples : "
          f"{sum(1 for r in rows if r['ss_pairs'])} target(s)")
    print(f"  receptor tethers      : "
          f"{sum(1 for r in rows if r.get('ss_anchor'))} target(s)")
    for label in ("kopt", "free"):
        T = [r[f"{label}_introduced"] for r in rows]
        tot = sum(t["n"] for t in T)
        hit = sum(1 for t in T if t["n"])
        ctxn = sum(t["n_needs_ctx"] for t in T)
        ctxhit = sum(1 for t in T if t["n_needs_ctx"])
        print(f"\n-- {label} --")
        print(f"  targets with >=1 introduced violation : {hit}/{n} "
              f"({100.0 * hit / n:.0f}%)")
        print(f"  total introduced violations           : {tot} "
              f"({tot / n:.2f} per target)")
        print(f"  of which needs_ctx (per-target rule)  : {ctxn} "
              f"on {ctxhit}/{n} targets")
        print(f"  of which a universal column fixes     : "
              f"{tot - ctxn}")
        byk = {}
        byc = {}
        for t in T:
            for k, v in t["by_klass"].items():
                byk[k] = byk.get(k, 0) + v
            for c, v in t["by_check"].items():
                byc[c] = byc.get(c, 0) + v
        print(f"  by solver class: " + "  ".join(
            f"{k}={byk.get(k, 0)}" for k in (SEPARABLE, LOCAL, GLOBAL, JOINT)))
        for c, v in sorted(byc.items(), key=lambda kv: -kv[1]):
            print(f"      {c:<20s} {v}")
    print(f"\nREAD THIS BEFORE QUOTING THE NUMBERS ABOVE:")
    print(f"  * Everything counted here is in a class an exact solver handles"
          f" ({'/'.join(SOLVABLE_EXACTLY)}).")
    print(f"    A high count motivates ANNOTATING THE TABLE, not a search arm.")
    print(f"  * The case for a per-target specification arm rests on the"
          f" needs_ctx line only.")
    print(f"  * {len(TOOL_REQUIRED)} JOINT axes are declared and NOT measured: "
          f"{', '.join(c for c, _ in TOOL_REQUIRED)}.")
    print(f"    Zero violations here does not mean a usable peptide.")


# ---------------------------------------------------------------------------
# Selftest. Runs without the heavy imports, so it is the fast gate.
# ---------------------------------------------------------------------------

def selftest():
    ok = True

    def eq(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {label}: {got!r}"
              + ("" if good else f" != {want!r}"))

    print("registry")
    eq("n_checks", len(CHECKS), 13)
    eq("every check has a source",
       all(getattr(f, "source", "") for f in CHECKS), True)
    eq("every klass is known",
       all(f.klass in (SEPARABLE, LOCAL, GLOBAL, JOINT) for f in CHECKS), True)
    eq("no JOINT check is implemented",
       [f.check_name for f in CHECKS if f.klass == JOINT], [])

    print("motifs")
    eq("asp_gly", [v.positions for v in asp_gly("ADGK", Ctx())], [(1, 2)])
    eq("asp_pro", [v.positions for v in asp_pro("ADPK", Ctx())], [(1, 2)])
    eq("sequon NAT", len(nglyc_sequon("GNATG", Ctx())), 1)
    eq("sequon NPT is not one", len(nglyc_sequon("GNPTG", Ctx())), 0)
    eq("hydrophobic run", [v.positions for v in
                           hydrophobic_run("GAILG", Ctx())], [(1, 2, 3)])
    eq("trypsin KP exempt", len(trypsin_site("GKPG", Ctx())), 0)
    eq("trypsin K counted", len(trypsin_site("GKGG", Ctx())), 1)

    print("needs_ctx: the same sequence, two verdicts")
    seq = "CGGCGGW"
    linear = unpaired_cys(seq, Ctx())
    bonded = unpaired_cys(seq, Ctx(ss_pairs=((0, 3),)))
    eq("linear: both Cys are free thiols",
       [v.positions for v in linear], [(0, 3)])
    eq("bonded: the native pair is fine", bonded, [])
    eq("bonded + a third Cys is flagged",
       [v.positions for v in unpaired_cys("CGGCGCW", Ctx(ss_pairs=((0, 3),)))],
       [(5,)])
    eq("bonded Cys mutated away is flagged",
       [v.check for v in unpaired_cys("CGGAGGW", Ctx(ss_pairs=((0, 3),)))],
       ["unpaired_cys"])
    eq("dkp applies to linear", len(dkp_risk("APGG", Ctx())), 1)
    eq("dkp does not apply to cyclic",
       len(dkp_risk("APGG", Ctx(cyclic=True))), 0)

    print("receptor tether: the check an earlier version wrongly excluded")
    eq("tethering Cys kept is fine",
       receptor_disulfide("CGGW", Ctx(ss_anchor=(0,))), [])
    eq("tethering Cys mutated away is flagged",
       [v.positions for v in receptor_disulfide("AGGW", Ctx(ss_anchor=(0,)))],
       [(0,)])
    eq("a receptor tether is invisible without ctx",
       receptor_disulfide("AGGW", Ctx()), [])
    # A KEPT tethered Cys is the real double-count case: it is a Cys, so the
    # free-thiol check sees it, but it is bonded, so it is not a free thiol.
    eq("a kept tethered Cys is not a free thiol",
       [v.check for v in battery("CGGW", Ctx(ss_anchor=(0,)))
        if v.check in ("receptor_disulfide", "unpaired_cys")], [])
    eq("the same Cys with NO ctx is a free thiol",
       [v.check for v in battery("CGGW", Ctx())
        if v.check == "unpaired_cys"], ["unpaired_cys"])
    eq("a LOST tether is reported once, by its own check",
       [v.check for v in battery("AGGW", Ctx(ss_anchor=(0,)))
        if v.check in ("receptor_disulfide", "unpaired_cys")],
       ["receptor_disulfide"])
    eq("an extra Cys beside a tether still scrambles",
       [v.positions for v in battery("CGCW", Ctx(ss_anchor=(0,)))
        if v.check == "unpaired_cys"], [(2,)])

    print("global, linear in counts")
    eq("net charge in window", len(net_charge("GKGDG", Ctx())), 0)
    eq("net charge too positive",
       len(net_charge("KKKKKG", Ctx(charge_window=(-2.0, 4.0)))), 1)
    eq("gravy flags an all-Ile peptide",
       len(gravy("IIIIII", Ctx(gravy_max=1.0))), 1)

    print("native-relative accounting")
    eq("native D-G is not blamed on the mutant",
       [v.check for v in introduced("ADGK", "ADGW", Ctx())],
       ["oxidation_prone"])
    eq("a moved motif is introduced at its new site",
       [v.positions for v in introduced("ADGKK", "AKDGK", Ctx())
        if v.check == "asp_gly"], [(2, 3)])
    eq("identity introduces nothing", introduced("ADGK", "ADGK", Ctx()), [])

    print("SSBOND mapping (regression: PSSM `positions` are NOT resnums)")
    import tempfile
    seq = "APYCVYRGSWSC"            # 4w50's peptide; Cys at indices 3 and 11
    # Numbered from 4, so resnum != index and resnum != index+1 either. The
    # original bug used the PSSM's [0..n-1] as a resnum map, which half-matched
    # and silently dropped the pair.
    lines = [f"SSBOND   1 CYS F    7    CYS F   15"
             f"                          1555   1555  2.03",
             f"SSBOND   2 CYS F    7    CYS A  200"
             f"                          1555   1555  2.03"]
    for i, a in enumerate(seq):
        lines.append(f"ATOM  {i * 4 + 1:5d}  CA  XXX F{4 + i:4d}    "
                     f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    with tempfile.TemporaryDirectory() as d:
        pdb = Path(d) / "t.pdb"
        pdb.write_text("\n".join(lines) + "\n")
        eq("residue ids read in file order", chain_resnums(pdb, "F")[:2],
           [(4, "", "XXX"), (5, "", "XXX")])
        c = ctx_from_pdb(pdb, "F", seq, "t")
        eq("intra pair maps to the Cys indices", c.ss_pairs, ((3, 11),))
        eq("receptor tether maps too", c.ss_anchor, (3,))
        eq("every seen record was mapped",
           (c.ss_seen, c.ss_mapped), (2, 2))
        # the guard: a bond naming a non-Cys index must NOT map
        bad = ctx_from_pdb(pdb, "F", "A" * len(seq), "t")
        eq("non-Cys target refuses to map", bad.ss_pairs, ())
        eq("and says so via seen != mapped",
           bad.ss_seen > bad.ss_mapped, True)

    # 2hwl's shape: the chain uses insertion codes (1D, 1E, ...) while the
    # SSBOND record writes a BARE `CYS A 1`. Keying on resSeq alone resolved
    # that to the Gly at 1D; the fix disambiguates by residue name, which the
    # record itself asserts.
    iseq = "GCA"
    ilines = ["SSBOND   1 CYS A    1    CYS B  122"
              "                          1555   1555  2.04"]
    for i, (num, ic, nm) in enumerate([(1, "D", "GLY"), (1, "E", "CYS"),
                                       (2, " ", "ALA")]):
        ilines.append(f"ATOM  {i + 1:5d}  CA  {nm} A{num:4d}{ic}   "
                      f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    with tempfile.TemporaryDirectory() as d:
        pdb = Path(d) / "i.pdb"
        pdb.write_text("\n".join(ilines) + "\n")
        eq("insertion codes are kept in the id",
           chain_resnums(pdb, "A")[:2], [(1, "D", "GLY"), (1, "E", "CYS")])
        c = ctx_from_pdb(pdb, "A", iseq, "i")
        eq("bare SSBOND resolves to the CYS, not the GLY", c.ss_anchor, (1,))
        eq("and it is counted as mapped", (c.ss_seen, c.ss_mapped), (1, 1))

    print("tally")
    t = tally(battery("CGGCGGWDG", Ctx()))
    eq("tally splits by ctx",
       t["n"] == t["n_needs_ctx"] + t["n_universal"], True)
    eq("no JOINT ever counted", t["by_klass"][JOINT], 0)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--pssm-dir",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/pssm")
    ap.add_argument("--pdb-dir", default="/zfsauton/scratch/yixiz/LNR/pdbs")
    ap.add_argument("--out", default="pepliab_audit.jsonl")
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--scope", default="all", choices=("all", "iface"))
    ap.add_argument("--metric", default="")
    ap.add_argument("--no-ctx", action="store_true",
                    help="run with an EMPTY Ctx on every target -- the "
                         "'universal spec' arm. The difference against the "
                         "default run IS the measurement.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.audit:
        audit(a.pssm_dir, a.pdb_dir, a.out, k_mut=a.k_mut, scope=a.scope,
              metric=(a.metric or None), use_ctx=not a.no_ctx)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
