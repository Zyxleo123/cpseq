"""The sequence-mutation lever for LNR: mutate, then close, scored by REAL binding.

This adds a second lever to the real ring-closure environment (lnr/env.py). The
proven LNR action is a held window [a,b] -- which residues keep their bound pose
while the rest are released to close the ring (lnr/run.py). This module lets a
controller ALSO edit the peptide SEQUENCE before that window is closed, and
scores the resulting cyclic design by the binding metric the project already
reports: energy-weighted interface retention x native interface dG.

The one correctness point, stated once (it is the project's recurring bug class)
-------------------------------------------------------------------------------
report.py scores a held window's binding from `rosetta_interface.jsonl`, which is
the WILD-TYPE peptide's Rosetta alanine scan. The moment the sequence is mutated
that table describes a DIFFERENT complex. Scored against it, a beneficial
mutation is invisible (native_dG cannot move) and a mutation at a held hotspot
falsely inherits the WT residue's energy. So this module NEVER reuses the WT
scan for a mutant: `mutant_scan()` recomputes native_dG AND the per-residue
alanine scan on the MUTANT bound pose (clean -> apply substitutions + repack ->
FastRelax -> InterfaceAnalyzer -> alanine scan), the same pipeline as
rosetta_interface.py with a mutation step inserted. The empty-substitution scan
is byte-identical to the WT scan, so `nomut` reproduces the existing number.

Why the closure loop is untouched (orthogonality)
-------------------------------------------------
score_window(), analytic_feasible() and both clash gates read backbone N/CA/C
only. A side-chain substitution changes none of them, so the held-window closure
campaign is IDENTICAL whether or not the sequence was mutated. v1 therefore
commits a window with the existing, already-studied closure controller
(default `placement_search`, the strong deterministic bar) ONCE per target, then
varies only the mutation arm and re-scores binding on that committed window. The
paired question is clean: holding the closure policy fixed, does a mutation arm
increase the binding energy the closed ring retains?
  (v2 would couple them: let Gly/Pro widen the torsion-feasible set so a mutation
  can also buy closure. That breaks the orthogonality on purpose; it is not v1.)

Action set, v1 (deliberately not committing to one fork of the survey)
----------------------------------------------------------------------
A mutation is a set of substitutions {(local_index, amino_acid)} under a small
mutation budget k_mut, separate from the release budget. v1 arms:
  nomut          no substitution -- the existing clean-binding number, the bar
                 the lever must beat.
  random_mut     k_mut random substitutions at interface positions, seeded by
                 target id -- the NULL: does naming mutations by chance help?
                 (the lnr `rule:random` analogue, the floor a real arm must clear)
  llm            the controller names the substitutions (position AND identity)
                 -- this is the invariant-BREAKING fork of the survey (the LLM
                 emits residue identities). It is wired to the SAME scorer; run
                 it only after the deterministic arms validate the scorer.
The invariant-PRESERVING fork (controller names the position, a structure-
conditioned packer names the identity) is `packer_design` and is left as a
v2 arm: it needs a PackRotamersMover design step, where v1's arms all hand the
scorer concrete (position, identity) pairs and so share one simple subprocess.
Both forks score through this module unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Pure scoring -- no Rosetta, no solver. Matches report.py byte-for-byte.
# ---------------------------------------------------------------------------

def energy_weighted_retention(per_res_ddg_ala: dict, a: int, b: int) -> float:
    """Fraction of native interface ENERGY the held window [a,b] carries.

    Identical to report.py:binding_retention -- the numerator sums the positive
    per-residue alanine-scan ddG (a residue's favourable interface contribution)
    over held positions, the denominator over all positions. This is the ONLY
    definition of binding retention, imported by the scorer, the arms and the
    report, so they cannot drift. `per_res_ddg_ala` MUST be the MUTANT's scan
    when the sequence was edited (see the module docstring).
    """
    pr = {int(k): (v or 0.0) for k, v in per_res_ddg_ala.items()}
    tot = sum(v for v in pr.values() if v > 0) or 1.0
    kept = sum(pr[k] for k in range(a, b + 1) if k in pr and pr[k] > 0)
    return kept / tot


def recovered_binding_energy(per_res_ddg_ala: dict, native_dG, a: int, b: int):
    """retention x native_dG: the ABSOLUTE interface energy (REU) the closed ring
    retains. native_dG is negative for a favourable interface, so this is
    negative and MORE NEGATIVE IS BETTER -- more of the parent's binding energy
    kept on a ring that closes. Returns None if the mutant was never scored.
    """
    if native_dG is None:
        return None
    return energy_weighted_retention(per_res_ddg_ala, a, b) * float(native_dG)


def subs_key(subs) -> str:
    """Canonical cache/identity key for a substitution set. Order-independent."""
    if not subs:
        return ""
    return ",".join(f"{int(i)}:{a.upper()}" for i, a in sorted(subs))


def apply_subs(seq: str, subs) -> str:
    s = list(seq)
    for i, a in subs:
        if 0 <= int(i) < len(s) and a.upper() in AA20:
            s[int(i)] = a.upper()
    return "".join(s)


# ---------------------------------------------------------------------------
# The mutant Rosetta scan -- subprocess-isolated, resumable, cached.
# Mirrors rosetta_interface.py:_score_one with a mutation step before relax.
# ---------------------------------------------------------------------------

def _scan_one(path, pep_chain, rec_chain, subs, relax_repeats, q):
    """Child process: clean -> apply substitutions + repack -> relax -> native dG
    -> per-residue alanine scan, ALL on the mutant bound pose. `subs` is a list
    of (local_peptide_index, one_letter_aa). Result (or an 'error' dict) goes on
    the queue; a PyRosetta segfault reaches the parent only as a nonzero exit.
    """
    try:
        import os
        import tempfile

        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.toolbox import mutate_residue

        pyrosetta.init("-mute all -ignore_unrecognized_res "
                       "-ignore_zero_occupancy false")
        sf = pyrosetta.get_fa_scorefxn()

        # Clean two-chain PDB on disk -- exactly rosetta_interface.py's filter, so
        # the WT scan (subs == []) reproduces rosetta_interface.jsonl.
        AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
               "TYR", "VAL", "MSE"}
        lines = []
        for ln in open(path):
            if ln.startswith("ATOM") and ln[21] in (pep_chain, rec_chain) \
                    and ln[17:20].strip() in AA3 and ln[16] in (" ", "A"):
                lines.append(ln)
            elif ln.startswith("TER"):
                lines.append(ln)
        fd, tmp = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, "w") as fo:
            fo.writelines(lines)
            fo.write("END\n")
        clean = pyrosetta.pose_from_pdb(tmp)
        os.unlink(tmp)
        cpdbi = clean.pdb_info()
        pep_pose = [i for i in range(1, clean.total_residue() + 1)
                    if cpdbi.chain(i) == pep_chain]
        chains = f"{rec_chain}_{pep_chain}"

        # Apply the substitutions in the bound pose, repacking the local shell
        # (pack_radius 8 A), BEFORE relax -- the mutant side chain(s) get a pose
        # and a local environment before the interface energy is measured.
        applied = []
        for local, aa in subs:
            local = int(local)
            aa = aa.upper()
            if not (0 <= local < len(pep_pose)) or aa not in AA20:
                continue
            ri = pep_pose[local]
            if clean.residue(ri).name1() == aa:
                continue  # no-op substitution
            mutate_residue(clean, ri, aa, pack_radius=8.0, pack_scorefxn=sf)
            applied.append([local, aa])

        fr = FastRelax(relax_repeats)
        fr.set_scorefxn(sf)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(clean)

        def idg(pose):
            ia = InterfaceAnalyzerMover(chains)
            ia.set_pack_separated(True)
            ia.set_compute_packstat(False)
            ia.apply(pose)
            return ia.get_interface_dG()

        dg0 = idg(clean)
        per_res = {}
        for local, ri in enumerate(pep_pose):
            if clean.residue(ri).name1() == "A":
                per_res[local] = 0.0
                continue
            m = clean.clone()
            mutate_residue(m, ri, "A", pack_radius=8.0, pack_scorefxn=sf)
            per_res[local] = round(idg(m) - dg0, 3)
        q.put({"native_dG": round(dg0, 3),
               "per_residue_ddG_ala": per_res,
               "applied_subs": applied})
    except Exception as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}"[:200]})


class ScanCache:
    """Resumable store of mutant scans, keyed by (target_id, subs_key).

    One JSONL row per scan, appended as it completes, so a time limit costs a
    resubmission not a restart (the LNR convention). The empty-subs row for a
    target IS its WT scan, so `--seed-wt rosetta_interface.jsonl` pre-loads those
    to make `nomut` free.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mem: dict = {}
        # get/put/_append are called concurrently when seqmut_iter scans a
        # generation's offspring across threads (each thread blocks on its own
        # PyRosetta subprocess); this guards the dict and the file append so
        # lines never interleave. Single-threaded callers pay a trivial cost.
        self._lock = __import__("threading").RLock()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.mem[(r["target_id"], r.get("subs_key", ""))] = r
        self._fh = None

    def seed_wt(self, rosetta_interface_jsonl: Path):
        if not Path(rosetta_interface_jsonl).exists():
            return 0
        n = 0
        for line in Path(rosetta_interface_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "per_residue_ddG_ala" not in r:
                continue
            key = (r["target_id"], "")
            if key in self.mem:
                continue
            row = {"target_id": r["target_id"], "subs_key": "",
                   "native_dG": r.get("native_dG"),
                   "per_residue_ddG_ala": r["per_residue_ddG_ala"],
                   "applied_subs": [], "seeded_from_wt": True}
            self.mem[key] = row
            self._append(row)
            n += 1
        return n

    def seed_scans(self, path):
        """Preload ALL scans (any subs_key) from another scan cache, read-only
        reuse: a breadth job writes its OWN cache but inherits the Rosetta work
        already done elsewhere, so it only recomputes genuinely new (target,
        subs). Reading a file another job is still appending to is safe -- a
        consistent prefix is loaded, later appends are simply missed."""
        p = Path(path)
        if not p.exists():
            return 0
        n = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("target_id"), r.get("subs_key", ""))
            if key[0] is None or key in self.mem:
                continue
            self.mem[key] = r
            self._append(r)
            n += 1
        return n

    def get(self, target_id, subs):
        with self._lock:
            return self.mem.get((target_id, subs_key(subs)))

    def put(self, row):
        with self._lock:
            self.mem[(row["target_id"], row.get("subs_key", ""))] = row
            self._append(row)

    def _append(self, row):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a")
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def mutant_scan(target, subs, cache: ScanCache, relax_repeats: int = 1,
                timeout: float = 900.0) -> dict:
    """native_dG + per-residue alanine scan of `target` with `subs` applied.
    Cached; subprocess-isolated; a crash/timeout returns an error row (not a
    throw) so a batch continues. `target` is an env.LNRTarget."""
    hit = cache.get(target.target_id, subs)
    if hit is not None:
        return hit
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_scan_one,
                    args=(target.pdb_path, target.pep_chain, target.rec_chain,
                          [(int(i), a) for i, a in subs], relax_repeats, q))
    t0 = time.time()
    p.start()
    p.join(timeout)
    row = {"target_id": target.target_id, "subs_key": subs_key(subs),
           "subs": [[int(i), a.upper()] for i, a in subs],
           "mutant_seq": apply_subs(target.seq, subs)}
    if p.is_alive():
        p.terminate(); p.join()
        row["error"] = "timeout"
    elif p.exitcode != 0:
        row["error"] = f"crashed_signal_{-p.exitcode}"
    else:
        try:
            res = q.get_nowait()
        except Exception:
            res = {"error": "no_result"}
        row.update(res)
    row["seconds"] = round(time.time() - t0, 1)
    cache.put(row)
    return row


# ---------------------------------------------------------------------------
# Committing a clean closing window (reuses the existing closure controller)
# ---------------------------------------------------------------------------

def committed_clean_window(t, history, max_check: int = 8,
                           n_restarts: int = 6, max_seconds: float = 8.0):
    """The window the closure controller commits to: among the campaign's
    CLOSING windows, the highest CA-contact-retention one that is also CLASH-FREE
    (the project's clean-closure headline). Walks the closers in retention order
    and returns the first clash-free one; falls back to the best geometric closer
    (clean=False) if none is clash-free, or None if nothing closed.

    The controller commits by the cheap signal it has in a campaign (contact
    retention) -- binding energy needs Rosetta and is the held-out evaluation, so
    it does NOT steer the commit here. That keeps the binding number honest.
    """
    from lnr import env as E

    closers = sorted((h for h in history if h.get("closes")),
                     key=lambda h: -h.get("retention", 0.0))
    if not closers:
        return None
    checked = 0
    for h in closers:
        if checked >= max_check:
            break
        checked += 1
        rep = E.clash_report(t, h["a"], h["b"], n_restarts=n_restarts,
                             max_seconds=max_seconds)
        if rep.get("clash_free"):
            return {"a": h["a"], "b": h["b"], "retention": h["retention"],
                    "clean": True}
    top = closers[0]
    return {"a": top["a"], "b": top["b"], "retention": top["retention"],
            "clean": False}


# ---------------------------------------------------------------------------
# Mutation arms: propose the substitution set for one (target, committed window)
# ---------------------------------------------------------------------------

def _norm_ala(wt_ala) -> dict:
    """WT alanine-scan per-residue binding, coerced to {int_index: float}."""
    if not wt_ala:
        return {}
    return {int(k): (v or 0.0) for k, v in wt_ala.items()}


def propose_subs(arm: str, t, window: dict, k_mut: int, seq: str,
                 wt_ala=None) -> list:
    """Return a list of (local_index, amino_acid). The window is the committed
    clean window, so an arm MAY choose to spend its edits inside it (to
    strengthen retained hotspots) or outside (to lift overall binding).

    Deterministic bars that bracket the LLM:
      random_mut   random AA at random interface positions -- the null floor.
      conservative random AA at the LOWEST-binding interface positions (smallest
                   alanine-scan ddG) -- "don't touch the hotspots"; should lose
                   less than random, a smarter floor.
      anti         the HOTSPOT positions (largest ddG) -> Ala -- a deliberate
                   anti-baseline that MUST be catastrophic, confirming the scorer
                   charges for interface damage (the Stage-0 max_dsasa role).
    conservative/anti need the WT alanine scan; with none they degrade to random.
    """
    if arm == "nomut" or k_mut <= 0:
        return []
    L = len(seq)
    iface = [i for i in range(L) if i < len(t.per_res_contacts)
             and t.per_res_contacts[i] > 0] or list(range(L))

    if arm in ("conservative", "anti"):
        pr = _norm_ala(wt_ala)
        if not pr:
            return propose_subs("random_mut", t, window, k_mut, seq)
        scored = [(i, pr.get(i, 0.0)) for i in iface]
        if arm == "anti":
            # hotspots first (largest favourable ddG), knock them to alanine
            order = sorted(scored, key=lambda x: -x[1])
            out = [(i, "A") for i, _ in order[:k_mut]
                   if (seq[i] if i < len(seq) else "A") != "A"]
            return out
        # conservative: least-binding interface positions first, random non-WT AA
        order = sorted(scored, key=lambda x: x[1])
        h = hashlib.sha256(f"{t.target_id}|cons|{k_mut}".encode()).digest()
        out = []
        for bi, (i, _) in enumerate(order[:k_mut]):
            wt = seq[i] if i < len(seq) else "A"
            ch = [a for a in AA20 if a != wt]
            out.append((i, ch[h[bi % len(h)] % len(ch)]))
        return out

    if arm == "random_mut":
        # seeded by target id so the null floor is reproducible and paired.
        h = hashlib.sha256(f"{t.target_id}|seqmut|{k_mut}".encode()).digest()
        picks = []
        used = set()
        bi = 0
        order = sorted(iface,
                       key=lambda i: hashlib.sha256(
                           f"{t.target_id}|pos|{i}".encode()).digest())
        for i in order:
            if len(picks) >= k_mut:
                break
            if i in used:
                continue
            wt = seq[i] if i < len(seq) else "A"
            choices = [a for a in AA20 if a != wt]
            aa = choices[h[bi % len(h)] % len(choices)]
            bi += 1
            picks.append((i, aa))
            used.add(i)
        return picks
    raise ValueError(f"unknown deterministic mutation arm {arm!r} "
                     f"(the 'llm' arm is proposed by the controller, not here)")


DET_MUT_ARMS = ("nomut", "random_mut", "conservative", "anti")
# fork B: the controller NAMES the substitutions. The iter-* arms are driven by
# seqmut_iter.py (an evolutionary loop), NOT this single-shot module, but they are
# listed here so _arm_label tags their finalist rows with the model -- so a
# two-model run never collides and boltz_validate's mut-held can address them.
LLM_MUT_ARMS = ("llm", "llm-ala", "iter-llm", "iter-llm-ala")
# iterative-search arm names, owned by seqmut_iter.py; defined here so this
# single-shot driver can reject them and _arm_label can label them consistently.
S_ITER_ARMS = ("iter-llm", "iter-llm-ala", "iter-ea")


# ---------------------------------------------------------------------------
# The LLM mutation arm (fork B of the survey -- the controller names identity)
# ---------------------------------------------------------------------------
#
# This is the invariant-BREAKING fork, run deliberately as a scouting arm: the
# controller emits (position, amino-acid). It goes through the SAME scorer as
# every other arm (score_target's subs_override), so if the invariant-preserving
# fork A (controller names the position, a packer names the identity) is wired
# later it is measured on an identical instrument.
#
# The scorer is POSE-PRESERVING (mutation threaded onto the native bound pose,
# local repack, relax, alanine scan), so it credits only substitutions that keep
# the native binding mode -- it cannot see a fold/binding-mode change. The system
# prompt states this explicitly, so the model is asked for exactly what the
# instrument can measure, and is not graded on moves it was never shown how to
# have scored. The real limitation (no re-fold/re-dock) is carried in the module
# docstring and belongs to a v2 scorer, not to the prompt.

MUT_SYSTEM = (
    "You are a protein engineer helping design a head-to-tail cyclic-peptide "
    "analog of a linear peptide bound to a receptor in a known, experimentally "
    "deposited pose. You may propose a few amino-acid substitutions in the "
    "peptide to improve or preserve its interface with the receptor. Each "
    "candidate is scored by threading the substitution onto the native bound "
    "pose, repacking the local side chains, relaxing, and measuring the Rosetta "
    "interface energy -- so a useful substitution strengthens binding WITHOUT "
    "requiring a different binding mode or fold. This is a computational methods "
    "benchmark on public deposited structures; nothing is synthesized.\n\n"
    "Reason briefly, then end your reply with the required tag and nothing after it."
)

# Real-score-in-loop variant: the in-loop fitness FOLDS the mutant from its
# sequence (Boltz x4 -> Rosetta dG) instead of threading onto the fixed native
# pose. The proxy framing above is then actively WRONG -- it tells the model its
# edits cannot change the fold, when the folded scorer rewards exactly that -- so
# the folded LLM arm uses this instead.
# Pose-conditioned-likelihood variant (`--evaluator pll`). Neither prompt above
# fits: the fitness is not a Rosetta energy at all but an inverse-folding model's
# log-odds that each residue belongs at its position in the deposited complex. A
# model told it is being scored by a relaxed interface energy will reason about
# packing and strain, which this fitness cannot see.
MUT_SYSTEM_PLL = (
    "You are a protein engineer helping design a head-to-tail cyclic-peptide "
    "analog of a linear peptide bound to a receptor in a known, experimentally "
    "deposited pose. You may propose a few amino-acid substitutions in the "
    "peptide. Each candidate is scored by an inverse-folding model: holding the "
    "deposited complex fixed, it reads the receptor's backbone and side chains "
    "and the peptide's BACKBONE ONLY, and reports how strongly each peptide "
    "position prefers the residue you put there, as a log-odds against that "
    "residue's background frequency. A useful substitution puts a residue where "
    "the pocket's own geometry and chemistry call for it -- shape and "
    "electrostatic complementarity to the facing receptor atoms -- rather than "
    "one that is merely large or merely hydrophobic. The score cannot see "
    "strain, packing clashes or any change of fold. This is a computational "
    "methods benchmark on public deposited structures; nothing is synthesized."
    "\n\n"
    "Reason briefly, then end your reply with the required tag and nothing after it."
)

MUT_SYSTEM_FOLDED = (
    "You are a protein engineer helping design a head-to-tail cyclic-peptide "
    "analog of a linear peptide bound to a receptor in a known, experimentally "
    "deposited pose. You may propose a few amino-acid substitutions in the "
    "peptide to improve or preserve its interface with the receptor. Each "
    "candidate is scored by FOLDING the cyclized peptide from its sequence with a "
    "structure predictor (its backbone held at the retained window), then "
    "measuring the predicted ring's Rosetta interface energy against the receptor "
    "-- so your edits are judged on the complex they actually fold into, and a "
    "substitution that changes how the peptide folds or binds is fair game as long "
    "as it scores better. This is a computational methods benchmark on public "
    "deposited structures; nothing is synthesized.\n\n"
    "Reason briefly, then end your reply with the required tag and nothing after it."
)

_MUT = re.compile(r"<mut>\s*(.*?)\s*</mut>", re.I | re.S)
_SUB_TOK = re.compile(r"^(\d+)\s*[:=]?\s*([A-Za-z])$")


def parse_subs(text: str, L: int, k_mut: int):
    """Parse `<mut>pos:AA,pos:AA</mut>` (or `<mut>none</mut>`). Returns
    (subs, status). subs is a list of (pos, AA); [] with status 'ok_none' is an
    explicit choice NOT to mutate. None means unparseable/no valid substitution
    (the caller retries, then records it malformed -- never silently scored as
    nomut). Out-of-range positions, non-standard residues and duplicate
    positions are dropped (rejection, as Wang et al.), and the list is clamped
    to the k_mut budget."""
    m = _MUT.findall(text or "")
    if not m:
        return None, "no_tag"
    body = m[-1].strip()
    if body.lower() in ("none", "", "-", "no", "nomut"):
        return [], "ok_none"
    subs, seen = [], set()
    for tok in re.split(r"[,\s;]+", body):
        tok = tok.strip()
        if not tok:
            continue
        mm = _SUB_TOK.match(tok)
        if not mm:
            continue
        pos, aa = int(mm.group(1)), mm.group(2).upper()
        if not (0 <= pos < L) or aa not in AA20 or pos in seen:
            continue
        seen.add(pos)
        subs.append((pos, aa))
        if len(subs) >= k_mut:
            break
    if not subs:
        return None, "no_valid_subs"
    return subs, "ok"


def receptor_context(t, max_per_pos: int = 6):
    """Per-position receptor contact partners, or None if unavailable.

    Never raises: a brief that loses one column is a worse brief, but a brief
    that cannot be built at all is a dead generation.
    """
    try:
        from lnr import env as _env
        return _env.receptor_contacts_by_position(t, max_per_pos=max_per_pos)
    except Exception:    # noqa: BLE001  a diagnostic must never fail a run
        return None


def _residue_table(t, wt_ala=None, rec_ctx=None) -> str:
    """idx / aa / contacts, and (for the -ala evidence arm) the per-residue WT
    interface energy from the alanine scan -- the binding diagnostics the LLM
    allocates its edits against.

    `rec_ctx` adds WHAT EACH POSITION FACES (nearest receptor residues first).
    Without it the brief asks the model to match "the shape and chemistry of the
    receptor atoms across from it" while telling it only how MANY atoms those
    are -- a contact count of 15 does not distinguish an aromatic pocket from a
    salt bridge, so the structural reasoning the arm is supposed to contribute
    has nothing to run on. The partners are read off the deposited complex, not
    out of the scorer, so this does not make the fitness self-referential.
    """
    head, dash = "  idx  aa  contacts", "  ---  --  --------"
    if wt_ala:
        head += "  bindE(ddG_ala)"; dash += "  --------------"
    if rec_ctx:
        head += "  faces (receptor, nearest first)"
        dash += "  -------------------------------"
    rows = [head, dash]
    for i in range(t.length):
        aa = t.seq[i] if i < len(t.seq) else "X"
        c = t.per_res_contacts[i] if i < len(t.per_res_contacts) else 0
        r = f"  {i:3d}  {aa}   {c:3d}"
        if wt_ala:
            e = wt_ala.get(i, wt_ala.get(str(i)))
            es = f"{e:+.2f}" if isinstance(e, (int, float)) else "   .  "
            r += f"       {es:>8s}"
        if rec_ctx:
            partners = rec_ctx[i] if i < len(rec_ctx) else []
            r += "  " + (" ".join(partners) if partners else "(no contacts)")
        rows.append(r)
    return "\n".join(rows)


def mut_brief(t, window: dict, k_mut: int, wt_ala=None) -> str:
    a, b = window["a"], window["b"]
    held = f"[{a},{b}]"
    note = (" The bindE column is each residue's current interface energy "
            "(higher = it carries more binding)." if wt_ala else "")
    return (
        f"Peptide length {t.length} (indices 0..{t.length-1}), "
        f"{t.n_contacts} interface contacts.\n\n"
        f"Per-residue table:{note}\n{_residue_table(t, wt_ala)}\n\n"
        f"On cyclization the residues in window {held} are HELD at their bound "
        f"pose and keep the interface; residues outside {held} are released so "
        f"the ring can close. A substitution inside {held} changes the interface "
        f"that is retained; one outside can change overall binding but that "
        f"position moves.\n\n"
        f"Propose up to {k_mut} substitution(s) that improve or preserve the "
        f"receptor interface IN THE NATIVE BINDING MODE (the scorer repacks "
        f"locally; it cannot reward a changed fold). You may propose fewer, or "
        f"none.\n\n"
        f"End with exactly one line:\n"
        f"<mut>pos:AA,pos:AA</mut>   (0-indexed position, one-letter amino "
        f"acid), or <mut>none</mut> to make no substitution."
    )


def llm_propose_subs(llm, t, window: dict, k_mut: int, max_retries: int = 1,
                     wt_ala=None, save_raw: bool = False):
    """One LLM proposal of a substitution set. Returns (subs, meta). subs is
    always a list (empty on decline/malformed); meta.llm_status records which, so
    the report can separate a real 'chose not to mutate' (ok_none) from a
    malformed or safety-declined call -- the run.py accounting, reused."""
    from lp2cp.enums import StepStatus

    brief = mut_brief(t, window, k_mut, wt_ala=wt_ala)
    msgs = [{"role": "user", "content": brief}]
    meta = {"model_id": getattr(llm, "model_id", "llm"), "llm_cost_usd": 0.0,
            "llm_status": "malformed", "llm_calls": 0, "llm_raw": ""}
    subs = []
    for r in range(max_retries + 1):
        reply = llm.complete(MUT_SYSTEM, msgs)
        meta["llm_calls"] += 1
        meta["llm_cost_usd"] = round(meta["llm_cost_usd"] + reply.cost_usd, 6)
        if reply.status == StepStatus.DECLINED:
            meta["llm_status"] = "declined"
            break
        if reply.status in (StepStatus.ERROR, StepStatus.BUDGET_CAP):
            meta["llm_status"] = reply.status.value.lower()
            break
        if save_raw:
            meta["llm_raw"] = (reply.text or "")[:800]
        val, st = parse_subs(reply.text, t.length, k_mut)
        if val is not None:
            subs = val
            meta["llm_status"] = st   # 'ok' or 'ok_none'
            break
        if r < max_retries:
            msgs += [{"role": "assistant", "content": reply.text},
                     {"role": "user", "content":
                      "Could not parse. End with exactly "
                      "<mut>pos:AA,pos:AA</mut> or <mut>none</mut> and nothing after."}]
    meta["n_proposed"] = len(subs)
    return subs, meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def score_target(t, mut_arm: str, window: dict, cache: ScanCache,
                 k_mut: int, relax_repeats: int, timeout: float,
                 subs_override=None, wt_ala=None) -> dict:
    """One (target, mutation-arm) row: propose substitutions, re-scan the mutant,
    score binding on the committed window. `subs_override` lets the LLM arm hand
    in its own (position, identity) list through the identical scorer; `wt_ala`
    is the WT alanine scan the conservative/anti rules allocate against."""
    subs = (subs_override if subs_override is not None
            else propose_subs(mut_arm, t, window, k_mut, t.seq, wt_ala=wt_ala))
    scan = mutant_scan(t, subs, cache, relax_repeats=relax_repeats,
                       timeout=timeout)
    a, b = window["a"], window["b"]
    row = {
        "target_id": t.target_id, "mut_arm": mut_arm, "length": t.length,
        "k_mut": k_mut, "subs": [[int(i), a_.upper()] for i, a_ in subs],
        "subs_key": subs_key(subs),
        "committed_window": [a, b], "window_clean": window.get("clean"),
        "contact_retention": round(window.get("retention", 0.0), 4),
        "native_dG": scan.get("native_dG"),
        "scan_error": scan.get("error"),
    }
    if scan.get("per_residue_ddG_ala") is not None and scan.get("native_dG") is not None:
        ewr = energy_weighted_retention(scan["per_residue_ddG_ala"], a, b)
        row["energy_retention"] = round(ewr, 4)
        row["recovered_binding_energy"] = round(ewr * float(scan["native_dG"]), 3)
    else:
        row["energy_retention"] = None
        row["recovered_binding_energy"] = None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--out", default="", help="rows jsonl (resumable)")
    ap.add_argument("--scan-cache", default="",
                    help="mutant-scan cache jsonl (default <out dir>/seqmut_scans.jsonl)")
    ap.add_argument("--seed-wt", default="",
                    help="rosetta_interface.jsonl to pre-load WT scans (makes nomut free)")
    ap.add_argument("--mut-arms", default="nomut,random_mut",
                    help=f"deterministic {DET_MUT_ARMS} and/or LLM {LLM_MUT_ARMS} "
                    "('llm-ala' shows the WT alanine-scan binding column as evidence)")
    ap.add_argument("--closure-arm", default="placement_search",
                    help="the (deterministic) closure controller that commits the window")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--per-target-timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=0)
    # LLM backend (only used when a mut arm is in LLM_MUT_ARMS)
    ap.add_argument("--backend", default="", choices=["", "openai", "api", "scripted"])
    ap.add_argument("--model", default="")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-spend-usd", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--seed-scans", default="",
                    help="comma-separated prior scan caches to preload read-only "
                    "(reuse Rosetta work across breadth jobs)")
    ap.add_argument("--report-dir", default="",
                    help="aggregate every seqmut*_rows.jsonl in this dir and print "
                    "the combined cross-arm table, then exit (no run)")
    args = ap.parse_args()

    if args.report_dir:
        report_dir(Path(args.report_dir))
        return 0
    if not (args.manifest and args.pdb_dir and args.out):
        raise SystemExit("[seqmut] --manifest, --pdb-dir and --out are required "
                         "unless --report-dir is given")
    mut_arms = [a.strip() for a in args.mut_arms.split(",") if a.strip()]
    for a in mut_arms:   # fail fast, before loading the manifest
        if a.startswith("iter-"):
            raise SystemExit(f"[seqmut] {a!r} is an iterative-search arm; run it "
                             "with seqmut_iter.py, not this single-shot driver.")

    from lnr import env as E
    from lnr import run as R

    targets = E.load_manifest(args.manifest, args.pdb_dir)
    targets.sort(key=lambda t: t.target_id)
    if args.limit:
        targets = targets[: args.limit]
    allowed = (set(DET_MUT_ARMS) | set(LLM_MUT_ARMS)) - set(S_ITER_ARMS)
    for a in mut_arms:
        if a not in allowed:
            raise SystemExit(f"[seqmut] unknown mut arm {a!r}; allowed: {sorted(allowed)}")
    llm_arms = [a for a in mut_arms if a in LLM_MUT_ARMS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = ScanCache(Path(args.scan_cache) if args.scan_cache
                      else out.parent / "seqmut_scans.jsonl")
    if args.seed_wt:
        n = cache.seed_wt(Path(args.seed_wt))
        print(f"[seqmut] seeded {n} WT scans from {args.seed_wt}", flush=True)
    for sf in [s.strip() for s in args.seed_scans.split(",") if s.strip()]:
        n = cache.seed_scans(Path(sf))
        print(f"[seqmut] seeded {n} prior scans from {sf}", flush=True)

    llm = None
    if llm_arms:
        if not args.backend:
            raise SystemExit(f"[seqmut] LLM arms {llm_arms} need --backend "
                             "(openai|api|scripted) and --model")
        llm = R.make_llm(args.backend, args.model, args.effort,
                         args.max_spend_usd, args.max_tokens)
        print(f"[seqmut] llm backend={args.backend} model={args.model} "
              f"cap=${args.max_spend_usd}", flush=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["target_id"], r["mut_arm"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[seqmut] {len(targets)} targets, mut_arms={mut_arms}, "
          f"closure={args.closure_arm}, k_mut={args.k_mut}, "
          f"{len(done)} rows done", flush=True)

    win_cache: dict = {}
    with out.open("a") as fh:
        for t in targets:
            if all((t.target_id, m) in done for m in mut_arms):
                continue
            # Commit the window ONCE per target: closure is substitution-invariant
            # (backbone only), so every mutation arm shares it.
            if t.target_id not in win_cache:
                cl_is_llm = args.closure_arm.endswith(("mask", "tool"))
                if cl_is_llm and llm is None:
                    raise SystemExit(f"[seqmut] closure-arm {args.closure_arm} is an "
                                     "LLM arm but no --backend/--model was given")
                camp = R.run_campaign(args.closure_arm, t, args.budget,
                                      llm=(llm if cl_is_llm else None),
                                      max_seconds=args.max_seconds)
                win = committed_clean_window(t, camp["history"],
                                             max_seconds=args.max_seconds)
                win_cache[t.target_id] = win
            win = win_cache[t.target_id]
            if win is None:
                for m in mut_arms:
                    if (t.target_id, m) in done:
                        continue
                    row = {"target_id": t.target_id, "mut_arm": m,
                           "length": t.length, "committed_window": None,
                           "note": "no_closing_window", "recovered_binding_energy": None}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[seqmut] {t.target_id:8s} no closing window -- skipped",
                      flush=True)
                continue
            # WT alanine scan (for the -ala evidence arm), from the seeded cache.
            wt = cache.get(t.target_id, [])
            wt_ala = (wt or {}).get("per_residue_ddG_ala")
            for m in mut_arms:
                if (t.target_id, m) in done:
                    continue
                meta = None
                if m in LLM_MUT_ARMS:
                    subs, meta = llm_propose_subs(
                        llm, t, win, args.k_mut, max_retries=args.max_retries,
                        wt_ala=(wt_ala if m == "llm-ala" else None),
                        save_raw=args.save_raw)
                    row = score_target(t, m, win, cache, args.k_mut,
                                       args.relax_repeats, args.per_target_timeout,
                                       subs_override=subs)
                    row["model_id"] = meta["model_id"]
                    row["llm_status"] = meta["llm_status"]
                    row["llm_cost_usd"] = meta["llm_cost_usd"]
                    row["n_proposed"] = meta["n_proposed"]
                    if args.save_raw:
                        row["llm_raw"] = meta["llm_raw"]
                else:
                    row = score_target(t, m, win, cache, args.k_mut,
                                       args.relax_repeats, args.per_target_timeout,
                                       wt_ala=wt_ala)
                row["closure_arm"] = args.closure_arm
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rbe = row.get("recovered_binding_energy")
                tag = (f" {meta['llm_status']} ${meta['llm_cost_usd']:.4f}"
                       if meta else "")
                print(f"[seqmut] {t.target_id:8s} {m:12s} "
                      f"win=[{win['a']},{win['b']}] clean={win['clean']} "
                      f"subs={row.get('subs')} "
                      f"eRet={row.get('energy_retention')} "
                      f"natdG={row.get('native_dG')} "
                      f"recovered={rbe}{tag}", flush=True)
    cache.close()

    # paired summary: does a mutation arm beat nomut on recovered binding energy?
    _summarize(out, mut_arms)
    print("[seqmut] done", flush=True)
    return 0


def _arm_label(r) -> str:
    """Display/grouping key. llm arms carry the model, and EVERY arm carries its
    mutation budget k, so neither a two-model run nor a k-sweep collapses two
    distinct configs into one cell (the Stage-2 campaign_id-collision bug class).
    The bare mutation-arm name is always the first ':'-field, so the nomut
    baseline is found by that field regardless of the model/k suffix."""
    m = r["mut_arm"]
    parts = [m]
    if m in LLM_MUT_ARMS and r.get("model_id"):
        parts.append(r["model_id"])
    if r.get("k_mut") is not None:
        parts.append(f"k{r['k_mut']}")
    # the closure controller is part of the config: a coupled (LLM-committed
    # window) run must not merge with a placement_search run on the same label.
    cl = r.get("closure_arm")
    if cl and cl != "placement_search":
        parts.append(f"cl:{cl}")
    return ":".join(parts)


def _summarize(out: Path, mut_arms):
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    _summarize_rows(rows)


def report_dir(d: Path):
    """Aggregate every seqmut*_rows.jsonl in a directory and print one combined
    table -- how the breadth jobs (each its own rows file) read out together.
    Dedups on (arm_label, target), keeping the last row seen."""
    rows = []
    for f in sorted(Path(d).glob("seqmut*_rows.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    seen = {}
    for r in rows:
        seen[(_arm_label(r), r.get("target_id"))] = r
    print(f"[seqmut] aggregated {len(rows)} rows -> {len(seen)} (arm,target) cells "
          f"from {d}")
    _summarize_rows(list(seen.values()))


def _summarize_rows(rows):
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(_arm_label(r), {})[r["target_id"]] = r
    print("\n" + "=" * 100)
    print("SEQMUT -- native dG (linear bound pose) and recovered binding energy "
          "(retention x native_dG), REU; more negative = stronger binding")
    print("=" * 100)
    print(f"{'arm':35s} {'n':>3s} {'meanRecov':>10s} {'meanNatDG':>10s} "
          f"{'meanERet':>9s} {'nMut':>5s} {'status':>18s}")
    for label in sorted(by_arm):
        vals = list(by_arm[label].values())
        v = [x for x in vals if x.get("recovered_binding_energy") is not None]
        # status breakdown for llm arms (ok / ok_none / malformed / declined)
        st = {}
        for x in vals:
            if x.get("llm_status"):
                st[x["llm_status"]] = st.get(x["llm_status"], 0) + 1
        sts = ",".join(f"{k}:{v_}" for k, v_ in sorted(st.items())) if st else "-"
        if not v:
            print(f"{label:35s} {len(vals):3d}  (no scored rows) {sts}")
            continue
        mr = sum(x["recovered_binding_energy"] for x in v) / len(v)
        mn = sum(x["native_dG"] for x in v) / len(v)
        me = sum(x["energy_retention"] for x in v) / len(v)
        nmut = sum(1 for x in v if x.get("subs"))
        print(f"{label:35s} {len(v):3d} {mr:10.2f} {mn:10.2f} "
              f"{me:9.3f} {nmut:5d} {sts:>18s}")
    # Pair each arm against the nomut baseline with the SAME closure controller:
    # a coupled (LLM-committed window) arm must not be compared to the
    # placement_search nomut -- the windows differ, so the delta would conflate
    # the mutation effect with the closure effect. Within one job nomut and the
    # mutation arms share the per-target committed window, so same-closure
    # pairing is matched.
    def _closure_of(cells):
        return (next(iter(cells.values())).get("closure_arm")
                or "placement_search")
    nomut_by_cl = {_closure_of(cells): cells
                   for lab, cells in by_arm.items()
                   if lab.split(":")[0] == "nomut"}
    if nomut_by_cl:
        print("  paired vs nomut with MATCHING closure (more negative Δ = more "
              "binding kept; random is the NULL floor):")
        for label in sorted(by_arm):
            if label.split(":")[0] == "nomut":
                continue
            cells = by_arm[label]
            base = nomut_by_cl.get(_closure_of(cells))
            if base is None:
                continue
            shared = sorted(set(base) & set(cells))
            paired = [(cells[p], base[p]) for p in shared
                      if cells[p].get("recovered_binding_energy") is not None
                      and base[p].get("recovered_binding_energy") is not None]
            d = [arm["recovered_binding_energy"] - wt["recovered_binding_energy"]
                 for arm, wt in paired]
            if not d:
                continue
            d_native = [arm["native_dG"] - wt["native_dG"]
                        for arm, wt in paired]
            better = sum(1 for x in d if x < -1e-6)
            worse = sum(1 for x in d if x > 1e-6)
            print(f"    {label:28s} n={len(d):3d} mean_d={sum(d)/len(d):+.2f} REU "
                  f"native_d={sum(d_native)/len(d_native):+.2f} REU "
                  f"(better/worse = {better}/{worse})")


if __name__ == "__main__":
    raise SystemExit(main())
