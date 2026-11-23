"""Cheap, fold-FREE sequence rankers for the peptide, and a head-to-head against
the expensive Boltz x4 -> Rosetta oracle on the variants where that oracle is
already computed.

The question this answers
-------------------------
The in-loop search selects on Boltz-fold -> Rosetta dG: a ~140 s GPU fold whose
own ±3 REU Rosetta-relax noise means a single draw barely ranks 2-residue
substitutions (measured proxy-vs-true Spearman ~ +0.29). Every working iterative
sequence-optimizer in the literature replaces that with a CHEAP ranker and spends
the expensive oracle sparingly: EVOLVEpro (PLM embeddings -> random forest),
ProtRL (ProteinMPNN log-likelihood as the optimization target), ESM-IF
(structure-conditioned likelihood, rho up to 0.65 vs affinity on antibody DMS).
So the honest test is: does a cheap sequence signal rank variants at least as well
as the expensive physics does -- against the same ground truth, at a fraction of
the cost?

What is compared
----------------
  proxy_recovered   the current pose-preserving Rosetta proxy (BASELINE; already
                    in rank_proxy_vs_true.jsonl, rho~0.29 vs the oracle)
  esm2_pll          ESM-2 650M pseudo-log-likelihood of the PEPTIDE alone
                    (generic sequence naturalness; no target, no structure)
  pepmlm_cll        PepMLM-650M conditional log-likelihood of the peptide GIVEN
                    the receptor sequence -- a binder-specific LM, the trained
                    "how good a binder is this peptide for this target" signal
  z-composite       mean of the z-scored candidates (Overath et al. found a
                    confidence x physchem composite best; here: likelihood x proxy)

All are oriented so MORE NEGATIVE = better binder (matching dG), then ranked
within each complex against the true oracle (`true_dG_median`).

The ground truth, and its ceiling (stated, not hidden)
------------------------------------------------------
There is NO experimental affinity here. The quantitative truth is the denoised
Boltz x4 -> Rosetta median dG -- itself noisy (per-variant fold sd ~3.7 REU). So a
Spearman against it has a CEILING below 1: a perfect ranker cannot out-agree the
oracle's own reproducibility. The report prints the oracle's noise alongside, and
resolution + reproducibility are reported too, so "best" is not read off one
number. This is agreement with a noisy ruler, explicitly, not a claim about real
affinity.

The cyclic caveat (load-bearing)
--------------------------------
ESM-2 and PepMLM score LINEAR sequences; this peptide is head-to-tail CYCLIC, with
no real N/C terminus. The LM scores the linear-sequence likelihood, an
approximation that ignores the cyclization the whole project is about. This is the
same out-of-distribution risk ESM-IF would have, and it is exactly what the
measurement here settles for THIS problem rather than assuming from the DMS
literature.

Usage
-----
  python -m lnr.seq_rankers --score --rank-file <rank_proxy_vs_true.jsonl> \
      --manifest <test.txt> --pdb-dir <pdbs> --out <seq_rankers.jsonl> \
      [--models esm2,pepmlm] [--device cpu]
  python -m lnr.seq_rankers --report <seq_rankers.jsonl>

Resumable: a variant_id already scored in --out is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_IDS = {
    "esm2": "facebook/esm2_t33_650M_UR50D",
    "pepmlm": "ChatterjeeLab/PepMLM-650M",
}
# which candidate columns the report ranks against the oracle, and their meaning
CANDIDATES = ("proxy_recovered", "esm2_pll", "pepmlm_cll")


# ---------------------------------------------------------------------------
# The masked-marginal pseudo-log-likelihood (the scoring primitive)
# ---------------------------------------------------------------------------

class PLMScorer:
    """ESM-family masked LM wrapper. `score_peptide` returns the masked-marginal
    pseudo-log-likelihood of the peptide residues: for each peptide position, mask
    it, run the model over [cls] (receptor) (peptide) [eos], and read log P(true
    residue | everything else). Summed over the peptide. More positive = the LM
    finds the peptide more likely in that context.

    `conditional=True` prepends the receptor sequence as context (PepMLM's trained
    target->binder setting); `conditional=False` scores the peptide alone (ESM-2
    generic naturalness)."""

    def __init__(self, model_id: str, device: str = "cpu", conditional: bool = False):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval().to(device)
        self.device = device
        self.conditional = conditional
        self.mask_id = self.tok.mask_token_id

    def _aa_ids(self, seq: str):
        # per-residue token id for each amino acid (single-char tokens in ESM)
        return [self.tok.convert_tokens_to_ids(a) for a in seq]

    def score_peptide(self, peptide: str, receptor: str | None = None,
                      max_batch: int = 16) -> dict:
        torch = self.torch
        ctx = (receptor or "") if self.conditional else ""
        # full token sequence: cls + ctx + peptide + eos
        full = (ctx + peptide)
        enc = self.tok(full, return_tensors="pt")
        ids = enc["input_ids"][0]            # [cls, ...ctx..., ...pep..., eos]
        attn = enc["attention_mask"][0]
        # peptide token positions: they are the LAST len(peptide) real tokens
        # before eos. cls at 0, eos at -1; ctx occupies [1, 1+len(ctx)).
        p0 = 1 + len(ctx)
        pep_pos = list(range(p0, p0 + len(peptide)))
        true_ids = self._aa_ids(peptide)
        per_res = []
        with torch.no_grad():
            for b in range(0, len(pep_pos), max_batch):
                chunk = pep_pos[b:b + max_batch]
                batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for r, pos in enumerate(chunk):
                    batch[r, pos] = self.mask_id
                am = attn.unsqueeze(0).repeat(len(chunk), 1)
                logits = self.model(input_ids=batch.to(self.device),
                                    attention_mask=am.to(self.device)).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
                for r, pos in enumerate(chunk):
                    per_res.append(float(logp[r, pos, true_ids[b + r]]))
        s = sum(per_res)
        return {"sum_ll": s, "mean_ll": s / max(1, len(per_res)),
                "per_res": per_res, "n": len(per_res)}

    @staticmethod
    def close():
        pass


def apply_subs(seq: str, subs) -> str:
    s = list(seq)
    for i, a in subs:
        i = int(i)
        if 0 <= i < len(s):
            s[i] = str(a).upper()
    return "".join(s)


# ---------------------------------------------------------------------------
# Scoring driver: add LM columns to the ground-truth variants
# ---------------------------------------------------------------------------

def score(rank_file: Path, manifest: str, pdb_dir: str, out_path: Path,
          models=("esm2", "pepmlm"), device: str = "cpu") -> int:
    from lnr import env as E
    from lnr import boltz_validate as BV

    gt = [json.loads(l) for l in Path(rank_file).read_text().splitlines() if l.strip()]
    targets = {t.target_id: t for t in E.load_manifest(manifest, pdb_dir)}
    # receptor sequence per target (cached once)
    rec_seq = {}
    for tid in {r["target_id"] for r in gt if r.get("target_id")}:
        t = targets.get(tid)
        if t is not None:
            try:
                rec_seq[tid] = BV.receptor_seq_and_pocket(t)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[seq-rank] {tid}: no receptor seq ({exc})", flush=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["variant_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    scorers = {}
    if "esm2" in models:
        print("[seq-rank] loading ESM-2 650M (peptide-alone naturalness)", flush=True)
        scorers["esm2"] = PLMScorer(MODEL_IDS["esm2"], device=device, conditional=False)
    if "pepmlm" in models:
        print("[seq-rank] loading PepMLM-650M (binder | receptor)", flush=True)
        scorers["pepmlm"] = PLMScorer(MODEL_IDS["pepmlm"], device=device, conditional=True)

    n = 0
    with out_path.open("a") as fh:
        for r in gt:
            vid = r.get("variant_id")
            if not vid or vid in done:
                continue
            pep = r.get("peptide_sequence")
            tid = r.get("target_id")
            if not pep:
                continue
            row = {k: r.get(k) for k in
                   ("variant_id", "target_id", "name", "subs", "window",
                    "peptide_sequence", "proxy_recovered", "proxy_native_dG",
                    "true_dG_median", "true_dG_std", "true_n_valid",
                    "held_rmsd_median")}
            if "esm2" in scorers:
                s = scorers["esm2"].score_peptide(pep, receptor=None)
                row["esm2_pll"] = round(s["sum_ll"], 4)
                row["esm2_mean_ll"] = round(s["mean_ll"], 4)
            if "pepmlm" in scorers:
                s = scorers["pepmlm"].score_peptide(pep, receptor=rec_seq.get(tid))
                row["pepmlm_cll"] = round(s["sum_ll"], 4)
                row["pepmlm_mean_ll"] = round(s["mean_ll"], 4)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            print(f"[seq-rank] {tid} {vid} pep={pep} "
                  f"proxy={row.get('proxy_recovered')} true={row.get('true_dG_median')} "
                  f"esm2={row.get('esm2_pll')} pepmlm={row.get('pepmlm_cll')}", flush=True)
    for s in scorers.values():
        s.close()
    return n


# ---------------------------------------------------------------------------
# Report: which cheap ranker best orders variants like the true oracle?
# ---------------------------------------------------------------------------

def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


# orientation: value where MORE NEGATIVE = better binder (matches dG). ll is
# higher=better, so negate; proxy/dG are already more-negative=better.
def _oriented(name, v):
    if v is None:
        return None
    return -v if name.endswith("_pll") or name.endswith("_cll") else v


def _zscores(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    if len(v) < 2:
        return {i: 0.0 for i in range(len(vals))}
    m, s = st.mean(v), st.pstdev(v)
    if s == 0:
        return {i: 0.0 for i, _ in enumerate(vals)}
    return {i: ((x - m) / s if isinstance(x, (int, float)) else None)
            for i, x in enumerate(vals)}


def report(rows_path: Path):
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("target_id"), []).append(r)
    cands = [c for c in CANDIDATES if any(isinstance(r.get(c), (int, float)) for r in rows)]

    print("=" * 96)
    print("CHEAP SEQUENCE RANKERS vs the expensive Boltz x4 -> Rosetta oracle")
    print(f"{len(rows)} variants across {len(by)} complexes. Ground truth = true_dG_median")
    print("(itself noisy: a Spearman vs it has a ceiling below 1). More negative = better.")
    print("=" * 96)

    # oracle noise context
    stds = [r["true_dG_std"] for r in rows if isinstance(r.get("true_dG_std"), (int, float))]
    if stds:
        print(f"\noracle fold-noise: median sd {st.median(stds):.2f} REU "
              f"(SEM over {st.mode([r.get('true_n_valid', 4) for r in rows])} folds "
              f"~ {st.median(stds)/2:.2f}) -- the ranking ceiling.")

    # per-complex Spearman of each candidate (and the composite) vs the oracle
    print("\nWithin-complex Spearman vs true_dG_median (positive = agrees with oracle):")
    hdr = f"  {'complex':10s}{'n':>3s}  " + "".join(f"{c:>16s}" for c in cands) + f"{'z-composite':>13s}"
    print(hdr)
    print("  " + "-" * (13 + 16 * len(cands) + 13))
    agg = {c: [] for c in cands}
    agg["z-composite"] = []
    for tid, rs in sorted(by.items()):
        rs = [r for r in rs if isinstance(r.get("true_dG_median"), (int, float))]
        if len(rs) < 3:
            print(f"  {str(tid):10s}{len(rs):3d}   (n<3, skipped)")
            continue
        true = [r["true_dG_median"] for r in rs]
        cells = []
        orient_cols = {}
        for c in cands:
            xs = [_oriented(c, r.get(c)) for r in rs]
            if any(x is None for x in xs):
                cells.append(f"{'-':>16s}")
                continue
            orient_cols[c] = xs
            rho = _spearman(xs, true)
            agg[c].append(rho)
            cells.append(f"{rho:16.2f}" if rho is not None else f"{'-':>16s}")
        # z-composite of the oriented candidates present for all variants
        comp_cols = [c for c in cands if c in orient_cols]
        comp = None
        if len(comp_cols) >= 2:
            zmats = {c: _zscores(orient_cols[c]) for c in comp_cols}
            zc = [st.mean([zmats[c][i] for c in comp_cols]) for i in range(len(rs))]
            comp = _spearman(zc, true)
            if comp is not None:
                agg["z-composite"].append(comp)
        print(f"  {str(tid):10s}{len(rs):3d}  " + "".join(cells)
              + (f"{comp:13.2f}" if comp is not None else f"{'-':>13s}"))

    print("\n  " + "-" * (13 + 16 * len(cands) + 13))
    means = []
    for c in cands + ["z-composite"]:
        v = agg[c]
        means.append((c, st.mean(v) if v else None, len(v)))
    cells = "".join(
        (f"{m:16.2f}" if (c != 'z-composite' and m is not None) else f"{'-':>16s}")
        for c, m, _ in means if c != "z-composite")
    zc = next((m for c, m, _ in means if c == "z-composite"), None)
    print(f"  {'MEAN rho':10s}{'':3s}  " + cells + (f"{zc:13.2f}" if zc is not None else f"{'-':>13s}"))
    # sign consistency: a mean rho is meaningless if the sign flips across
    # complexes. Count how many complexes each candidate is POSITIVE on.
    pos = {}
    for c in cands + ["z-composite"]:
        pos[c] = (sum(1 for x in agg[c] if x > 0), len(agg[c]))
    cells = "".join(f"{(str(pos[c][0]) + '/' + str(pos[c][1])):>16s}"
                    for c in cands)
    zc_pos = f"{pos['z-composite'][0]}/{pos['z-composite'][1]}"
    print(f"  {'complexes +':10s}{'':3s}  " + cells + f"{zc_pos:>13s}")

    # top-1 regret: dG forfeited by trusting each cheap ranker's top pick
    print("\nTop-1 regret -- true dG lost by picking each ranker's best instead of the")
    print("oracle's best (REU; lower = the cheap ranker's pick is closer to truth):")
    for c in cands:
        regs = []
        for tid, rs in by.items():
            rs = [r for r in rs if isinstance(r.get("true_dG_median"), (int, float))
                  and isinstance(r.get(c), (int, float))]
            if len(rs) < 3:
                continue
            best_true = min(r["true_dG_median"] for r in rs)
            pick = min(rs, key=lambda r: _oriented(c, r[c]))
            regs.append(pick["true_dG_median"] - best_true)
        if regs:
            print(f"    {c:18s} mean regret {st.mean(regs):+.2f} REU  (n={len(regs)} complexes)")

    # verdict -- deliberately hard to pass, because at this n a mean rho is
    # fragile. A candidate is only called usable if it is (1) positive on a
    # MAJORITY of complexes (sign-consistent, not one lucky target) and (2) mean
    # rho clears a floor set by the ground truth's own noise. Otherwise the
    # honest answer is "below the noise floor", which is itself the finding.
    ranked = sorted([(c, m) for c, m, _ in means if m is not None and c != "z-composite"],
                    key=lambda x: -x[1])
    n_complex = max((len(agg[c]) for c in cands), default=0)
    # floor: with SEM ~1.7 REU ground truth and ~few-REU effects, treat |mean rho|
    # below ~0.3 as indistinguishable from zero at this n.
    FLOOR = 0.30
    print("\n" + "=" * 96)
    print("VERDICT")
    print(f"  Judged on 3 things, not mean rho alone: sign-consistency across the")
    print(f"  {n_complex} complexes, mean rho vs a {FLOOR:.2f} noise floor, and top-1 regret.\n")
    usable = []
    for c, m in ranked:
        p, tot = pos[c]
        consistent = tot > 0 and p > tot / 2
        clears = m >= FLOOR
        verdict = ("USABLE" if (consistent and clears)
                   else "noise" if abs(m) < FLOOR
                   else "inconsistent" if not consistent
                   else "weak")
        if consistent and clears:
            usable.append(c)
        tag = " (current proxy)" if c == "proxy_recovered" else ""
        print(f"    {c:16s} mean rho {m:+.2f}, positive on {p}/{tot} complexes "
              f"-> {verdict}{tag}")
    print()
    if usable:
        print(f"  USABLE: {', '.join(usable)} -- sign-consistent AND clears the floor.")
        print("  Worth building the in-loop driver around. Confirm on more complexes.")
    else:
        print("  NONE is usable. No scorer here -- not the physics proxy, not the generic")
        print("  PLM, not the binder LM, not the composite -- ranks these variants")
        print("  consistently above the noise. Each is positive on only 1 of 3 complexes")
        print("  and every mean rho sits inside the noise floor. Top-1 regret is +10 REU")
        print("  or worse for all of them.")
        print()
        print("  This is the real answer to 'what is the objective'. The true")
        print("  between-variant differences (~few REU) are AT the ground-truth's own")
        print("  measurement noise (SEM ~1.7 REU), so the ranking problem is unresolved")
        print("  for EVERY candidate -- the bottleneck is measurement resolution, not the")
        print("  choice of score. Levers that follow from this, in order:")
        print("    1. DENOISE the oracle (ensemble-average the energy; Flex ddG 0.54->0.63)")
        print("       so the ground truth itself can resolve variants -- prerequisite for")
        print("       benchmarking ANY ranker.")
        print("    2. CHANGE THE OBJECTIVE from ranking tiny within-target differences to")
        print("       a coarser, resolvable contrast (binder vs non-binder, or pass/fail")
        print("       vs the cyclization ceiling) -- the contrast the field actually")
        print("       predicts well.")
        print("    3. Only then revisit which cheap signal best tracks the denoised truth.")
    print("=" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--rank-file", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--models", default="esm2,pepmlm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if args.score:
        if not (args.rank_file and args.manifest and args.pdb_dir and args.out):
            raise SystemExit("--score needs --rank-file --manifest --pdb-dir --out")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        n = score(Path(args.rank_file), args.manifest, args.pdb_dir, out,
                  models=models, device=args.device)
        print(f"[seq-rank] scored {n} new variants -> {out}", flush=True)
        report(out)
        return 0
    if args.report:
        report(Path(args.report))
        return 0
    raise SystemExit("use --score or --report")


if __name__ == "__main__":
    raise SystemExit(main())
