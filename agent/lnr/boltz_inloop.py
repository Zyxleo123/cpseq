"""Real-score IN-LOOP evaluator: fold a (sequence, held window) with Boltz x4 and
score the ring with Rosetta, returning a denoised dG as the search fitness.

Why this exists
---------------
The pose-preserving Rosetta retention proxy was shown to DISAGREE with the real
folded score (1bjr: proxy ranked the edits as big wins; the real Boltz x4 -> Rosetta
score ranked nomut best and every edit worse). A search that selects on a proxy
that does not rank like truth optimises noise. So the loop now selects on the real
number itself. This module is the in-loop fitness; `seqmut_iter.py` calls it in
place of `seqmut.mutant_scan` when `--evaluator boltz`.

The firewall is preserved by INDEPENDENT fold draws, not by a different metric
----------------------------------------------------------------------------
The loop folds with one Boltz `seed`; the believed-number report (boltz_validate
mut-held) re-folds the finalist with a DIFFERENT seed -- an independent x4 draw.
Selection noise from the loop's draw therefore cannot inflate the reported draw
(the cross-fit idea, at the fold-draw level). Both are Boltz; only the random draw
differs -- plus the deferred perpendicular folder for cross-lineage confirmation.

This reuses boltz_validate's fold-one machinery verbatim (build_held_template_pdb
-> CIF -> YAML -> predict -> score_replicas median), so the in-loop number and the
report number are computed by identical code, differing only in the fold seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import boltz_validate as BV  # noqa: E402  the fold-one machinery (reused)
from lnr import seqmut as S           # noqa: E402  apply_subs / subs_key


def variant_id(target_id, a, b, seq) -> str:
    key = hashlib.sha256(f"{target_id}|{a}|{b}|{seq}".encode()).hexdigest()[:12]
    return f"{target_id}_{a}_{b}_{key}"


def fold_score_one(t, subs, window, cfg, work: Path, replicas: int = 4,
                   draw: int = 0) -> dict:
    """Fold the MUTANT peptide held at `window` with Boltz (`replicas` samples from
    one trunk pass), score every replica's ring with Rosetta, return the denoised
    row (median dG_refined over sequence/ring-valid folds, + spread). Mirrors
    boltz_validate.run_mut_held's per-variant block exactly.

    `draw` indexes an INDEPENDENT re-measurement of the same variant: it shifts the
    Boltz seed and gives the prediction its own working directory, so draw 1 cannot
    read back draw 0's structures. draw 0 keeps the original seed, stem and paths,
    so every fold already in the cache stays valid and is never recomputed."""
    from proteinfoundation.cpseq.adapters.boltz2 import build_input_yaml, predict
    from proteinfoundation.cpseq.targets import write_target_template_cif

    a, b = window["a"], window["b"]
    seq = S.apply_subs(t.seq, subs)
    vid = variant_id(t.target_id, a, b, seq)
    stem = f"{vid}_d{int(draw)}" if draw else vid
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    # variant_id stays CANONICAL (the identity boltz_validate/report rows use);
    # cache_key carries the draw so repeats are stored side by side.
    rec = {"target_id": t.target_id, "variant_id": vid, "cache_key": stem,
           "draw": int(draw), "window": [a, b],
           "subs": [[int(i), aa.upper()] for i, aa in subs], "peptide_sequence": seq,
           "subs_key": S.subs_key(subs)}
    try:
        tmpl = work / f"tmpl_{stem}.pdb"
        cif = work / f"tmpl_{stem}.cif"
        yml = work / f"held_{stem}.yaml"
        rec["n_held"] = BV.build_held_template_pdb(
            t, a, b, str(tmpl), peptide_sequence=seq, backbone_only=True)
        write_target_template_cif(tmpl, cif, BV.BOLTZ_PY)
        recseq, pocket = BV.receptor_seq_and_pocket(t)
        yml.write_text(build_input_yaml(
            peptide_sequence=seq, receptor_sequences={t.rec_chain: recseq},
            chemistry="mainchain", pocket_contacts=pocket,
            receptor_msa_path=None, target_template_path=None,
        ) + BV._held_templates_block(str(cif)))
        odir = work / f"heldout_{stem}"
        pred = predict(yaml_path=yml, out_dir=odir, config=cfg,
                       python_exec=BV.BOLTZ_PY, boltz_cache=BV.BOLTZ_CACHE,
                       devices=1, accelerator="gpu", timeout_s=3600)
        rec["predict_status"] = pred.get("status")
        rec["predict_runtime_s"] = pred.get("runtime_s")
        if pred.get("status") != "ok":
            rec["error"] = (pred.get("stderr_tail") or "prediction failed")[-500:]
            return rec
        pdbs = BV.find_all_pdbs(odir)
        if not pdbs:
            rec["error"] = "prediction produced no PDB"
            return rec
        rec["boltz_pdb"] = str(pdbs[0])
        agg, reps = BV.score_replicas(
            pdbs, binder_chain="B", receptor_chains=["A"], chemistry="mainchain",
            requested_sequence=seq, refined_dir=work, stem=stem,
            rmsd_fn=lambda p: BV.held_rmsd(p, t, a, b))
        rec["n_folds"] = agg["n_folds"]
        rec["n_valid_folds"] = agg["n_valid"]
        rec["dG_refined"] = agg["dG_refined"]["median"]
        rec["dG_refined_std"] = agg["dG_refined"]["std"]
        rec["dG_raw"] = agg["dG_raw"]["median"]
        rec["held_rmsd"] = agg["held_rmsd"]["median"]
        # per-fold dG list (valid folds only): lets a consumer recompute the SEM
        # at any n_folds and check whether the diffusion samples are independent
        # draws. Cheap to carry; the in-loop search ignores it.
        rec["per_fold_dG"] = [r.get("dG_refined") for r in reps if r.get("valid")]
        rec["valid"] = agg["n_valid"] > 0
        rec["score_status"] = "ok" if agg["n_valid"] > 0 else "no_valid_fold"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return rec


class BoltzScorer:
    """In-loop fitness = denoised Boltz x4 -> Rosetta median dG. Resumable, cached
    by variant_id (a re-proposed sequence/window is free), so the loop's budget is
    spent on genuinely new folds. Drop-in for seqmut_iter.Population via `.score`.

    fitness convention matches the proxy: dG_refined, MORE NEGATIVE = better; None
    when no replica produced a valid ring (the candidate is unscored, never 0)."""

    is_gpu = True       # folds saturate the one GPU -> Population evaluates serially
    folds = True        # re-folds the mutant -> LLM brief uses the folded objective
    stochastic = True   # a fitness is a DRAW: `draw=k` re-measures independently

    def __init__(self, cache_path, work, replicas: int = 4, seed: int = 1234,
                 use_potentials: bool = True):
        from proteinfoundation.cpseq.adapters.boltz2 import Boltz2Config
        self.cfg = Boltz2Config()
        self.cfg.use_potentials = use_potentials
        self.base_seed = int(seed)
        self.cfg.seed = self.base_seed                  # the loop's fold draw
        BV.configure_replicas(self.cfg, replicas)       # diffusion_samples=replicas
        self.replicas = replicas
        self.work = Path(work)
        self.path = Path(cache_path)
        self.mem: dict = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        # cache_key is draw-aware; pre-draw rows key on variant_id
                        self.mem[r.get("cache_key") or r["variant_id"]] = r
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._fh = None

    def sem(self, row) -> float:
        """Standard error of this row's median dG over its own folds -- the noise
        estimate available before a genome has been measured twice."""
        if not row:
            return 0.0
        sd, n = row.get("dG_refined_std"), row.get("n_valid_folds")
        if not isinstance(sd, (int, float)) or not isinstance(n, int) or n < 1:
            return 0.0
        return float(sd) / (float(n) ** 0.5)

    def score(self, t, subs, window, draw: int = 0):
        """Returns (fitness_or_None, row). Cached by (variant, draw). `draw > 0`
        is an INDEPENDENT re-measurement: a different Boltz seed and its own output
        directory, so it is a genuine second sample of the same genome's fitness
        rather than a cache hit."""
        a, b = window["a"], window["b"]
        seq = S.apply_subs(t.seq, subs)
        vid = variant_id(t.target_id, a, b, seq)
        key = f"{vid}_d{int(draw)}" if draw else vid
        row = self.mem.get(key)
        if row is None:
            cfg = self.cfg
            if draw:
                cfg = copy.copy(self.cfg)
                cfg.seed = self.base_seed + 7919 * int(draw)   # a distinct draw
            row = fold_score_one(t, subs, window, cfg, self.work, self.replicas,
                                 draw=int(draw))
            self.mem[key] = row
            self._append(row)
        return row.get("dG_refined"), row

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


class ProxyScorer:
    """The original pose-preserving Rosetta retention proxy, wrapped in the same
    `.score` interface so Population is evaluator-agnostic. Kept so `--evaluator
    proxy` reproduces the earlier behaviour exactly (and the unit tests run with no
    GPU)."""

    is_gpu = False
    folds = False       # fixed-pose retention proxy -> brief uses the proxy framing
    stochastic = False  # deterministic: a redraw is the same number, so no repeats

    def __init__(self, cache, relax_repeats: int = 1, timeout: float = 900.0):
        self.cache = cache
        self.relax_repeats = relax_repeats
        self.timeout = timeout

    def sem(self, row) -> float:
        return 0.0

    def score(self, t, subs, window, draw: int = 0):
        scan = S.mutant_scan(t, list(subs), self.cache,
                             relax_repeats=self.relax_repeats, timeout=self.timeout)
        rec = None
        if scan.get("per_residue_ddG_ala") is not None and scan.get("native_dG") is not None:
            a, b = window["a"], window["b"]
            rec = round(S.energy_weighted_retention(scan["per_residue_ddG_ala"], a, b)
                        * float(scan["native_dG"]), 3)
        return rec, scan

    def close(self):
        pass
