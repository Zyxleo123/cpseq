"""Boltz-in-the-loop validation: fold cyclic peptides and score binding directly.

Modes:
  target    fold the native sequence without a held window and correlate its
            Rosetta score with geometric retention across targets;
  held      force-template the native held window for LLM and rule closure arms;
  mut-held  use each seqmut arm's actual sequence with its committed held window,
            compare against nomut with the same window, and score the predicted
            ring directly with Rosetta. The two members of a pair share receptor
            coordinates and held backbone input, while Boltz builds each peptide
            sequence's side chains and released loop.

The target-level reference folds the NATIVE cyclic peptide sequence with
Boltz-2 (cyclic head-to-tail, pocket-constrained to the native interface,
msa empty), relaxes + scores Rosetta InterfaceAnalyzer dG_separated, and
correlates across targets vs:
  * native_dG            -- the native LINEAR binding (how much survives cyclising)
  * LNR retention ceiling -- the best clash-free retention LNR says is achievable.

If targets LNR rates high-retention are the ones Boltz folds into strong binders,
LNR's geometric read has merit as a cheap proxy; if not, it is a geometric toy.

CAVEATS (pilot): no receptor MSA (msa:empty). The target mode has no receptor
template; the held modes template the receptor and a peptide window. These
are first-pass structure and binding comparisons, not experimental affinities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

PC = "/zfsauton2/home/yixiz/Proteina-Complexa"
for _p in (f"{PC}/src", PC, "/zfsauton2/home/yixiz/lp2cp-agent/agent"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import env as E  # noqa: E402

BOLTZ_PY = "/zfsauton/scratch/yixiz/envs/boltz2/.venv/bin/python"
BOLTZ_CACHE = "/zfsauton/scratch/yixiz/boltz_cache"


def receptor_seq_and_pocket(t, max_pocket: int = 24):
    """Receptor one-letter sequence in rec_ca order, and the 1-based interface
    residue indices (pocket). rec_ca is built from CA lines (altloc blank/A) in
    file order, so reading resnames the same way aligns index k <-> rec_ca[k]."""
    seq = []
    for line in open(t.pdb_path):
        if not line.startswith(("ATOM", "HETATM")) or line[21] != t.rec_chain:
            continue
        if line[16] not in (" ", "A") or line[12:16].strip() != "CA":
            continue
        seq.append(E._THREE_TO_ONE.get(line[17:20].strip(), "X"))
    seq = "".join(seq)
    rec_idx = sorted({r for (_, r) in t.contacts})
    # keep the interface residues closest to the peptide, capped
    pocket = [(t.rec_chain, k + 1) for k in rec_idx if k < len(seq)][:max_pocket]
    return seq, pocket


def find_pdb(out_dir: Path):
    pdbs = sorted(out_dir.rglob("*_model_0.pdb")) or sorted(out_dir.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


# ---------------------------------------------------------------------------
# Replica folding: denoise the real dG by scoring every diffusion sample
# ---------------------------------------------------------------------------
#
# Boltz draws `diffusion_samples` structures from ONE trunk pass and writes them
# as *_model_0.pdb .. *_model_{N-1}.pdb. The single-fold path scored model_0 and
# discarded the rest, so a reported dG was one draw off a stochastic denoiser --
# the fold noise was unmeasured and uncontrolled. Scoring ALL samples turns that
# into a denoised median plus the spread that quantifies the noise, for the price
# of extra (CPU) Rosetta relaxes; the extra samples themselves are nearly free
# because the expensive trunk pass is shared across them.

def find_all_pdbs(out_dir: Path) -> list[Path]:
    """Every diffusion-sample model PDB, ordered by model index (model_0 first)."""
    pdbs = list(out_dir.rglob("*_model_*.pdb"))
    if not pdbs:
        pdbs = list(out_dir.rglob("*.pdb"))

    def idx(p: Path) -> int:
        m = re.search(r"_model_(\d+)\.pdb$", p.name)
        return int(m.group(1)) if m else 1_000_000
    return sorted(pdbs, key=idx)


def _agg(vals) -> dict:
    """median/mean/std/min/max over the non-None values; n is how many there were.

    std is the POPULATION std over the folds (the observed spread, not an estimate
    of a larger population's), and is 0.0 for a single fold by definition."""
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 3),
        "mean": round(statistics.fmean(vals), 3),
        "std": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


def score_replicas(pdbs, *, binder_chain, receptor_chains, chemistry,
                   requested_sequence, refined_dir, stem, rmsd_fn=None):
    """Rosetta-score every fold replica and aggregate over the VALID ones.

    A replica counts toward the denoised dG only if it scored, is the requested
    sequence, and declared a closed ring -- identical to the single-fold gate, now
    applied per replica so one sample that failed to close does not poison the
    mean. Returns (aggregate, per_replica_rows). The aggregate's `dG_refined` is
    the headline denoised binding energy; its `std` is the single-fold noise."""
    from proteinfoundation.cpseq.scorer import score_structure
    reps = []
    for k, pdb in enumerate(pdbs):
        refined = Path(refined_dir) / f"refined_{stem}_r{k}.pdb"
        rep = {"model": k, "pdb": str(pdb)}
        try:
            sc = score_structure(
                str(pdb), binder_chain=binder_chain,
                receptor_chains=list(receptor_chains), chemistry=chemistry,
                requested_sequence=requested_sequence, refined_out_path=refined)
            rep["score_status"] = sc.get("status")
            rep["reason"] = sc.get("reason")
            rep["sequence_matches"] = sc.get("sequence_matches")
            rep["ring_declared"] = (sc.get("ring_declared_complex") or {}).get("declared")
            rep["dG_raw"] = (sc.get("raw") or {}).get("dG_separated")
            rep["dG_refined"] = (sc.get("refined") or {}).get("dG_separated")
            rep["valid"] = (rep["score_status"] == "ok"
                            and rep["sequence_matches"] is True
                            and rep["ring_declared"] is True
                            and rep["dG_refined"] is not None)
            if rmsd_fn is not None:
                rep["held_rmsd"] = rmsd_fn(pdb)
                if refined.exists():
                    rep["refined_pdb"] = str(refined)
                    rep["held_rmsd_refined"] = rmsd_fn(refined)
        except Exception as exc:  # noqa: BLE001
            rep["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rep["valid"] = False
        reps.append(rep)
    valid = [r for r in reps if r.get("valid")]
    agg = {
        "n_folds": len(reps),
        "n_valid": len(valid),
        "dG_refined": _agg([r.get("dG_refined") for r in valid]),
        "dG_raw": _agg([r.get("dG_raw") for r in valid]),
        "held_rmsd": _agg([r.get("held_rmsd") for r in valid]),
        "held_rmsd_refined": _agg([r.get("held_rmsd_refined") for r in valid]),
    }
    return agg, reps


def configure_replicas(cfg, replicas: int) -> str | None:
    """Make the fold draw `replicas` structures from one trunk pass, so a single
    noisy dG becomes a denoised median+/-std.

    Above boltz's default sample count the denoiser mis-counts its chunks
    (`boltz_chunkfix`): raising diffusion_samples without the patch collapses the
    batch into one oversized chunk -- an OOM with inference potentials on, which
    the force-held modes use. For replicas>4 we therefore put the patched shadow on
    PYTHONPATH and cap the per-batch size so the chunking is correct. Returns the
    shadow root if one was engaged, else None. Idempotent; the shadow (~47 MB) is
    built once and reused."""
    cfg.diffusion_samples = int(replicas)
    if replicas <= 4:
        return None
    from proteinfoundation.cpseq.adapters.boltz_chunkfix import ensure_patched_boltz
    info = ensure_patched_boltz(
        python_exec=BOLTZ_PY, cache_root="/zfsauton/scratch/yixiz/cpseq/_chunkfix")
    cfg.chunk_fix = True
    cfg.chunk_fix_root = info["root"]
    # Cap the per-batch size at the proven value. With potentials on the denoiser
    # multiplicity is 3*diffusion_samples, and ceil(mult/mps) chunks each hold
    # ceil(mult/chunks) samples; mps=4 keeps the batch at the size the single-fold
    # (ds=4) runs used on the v100, so raising the replica count does not silently
    # raise GPU memory into an OOM. More, smaller chunks cost a little time, not VRAM.
    cfg.max_parallel_samples = min(int(replicas), 4)
    return info["root"]


def _relabel(line: str, chain: str, resseq: int) -> str:
    """Rewrite a PDB ATOM line's chain id and residue number (blank the altloc/icode)."""
    return f"{line[:16]} {line[17:21]}{chain}{resseq:4d} {line[27:]}"


_AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", (
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER "
    "THR VAL TRP TYR").split()))


def build_held_template_pdb(t, a, b, out_pdb: str,
                            peptide_sequence: str | None = None,
                            backbone_only: bool = False) -> int:
    """Template for a FORCE-held fold: receptor (chain A, native coords) + the HELD
    peptide window [a,b] (chain B, native coords) at entity positions a+1..b+1.
    Boltz is then told (force:true) to keep these near native while it folds the
    released residues and closes the ring. Returns #held residues written."""
    from lnr.rescore_rings import read_full_residues
    if peptide_sequence is not None and len(peptide_sequence) != t.length:
        raise ValueError("template peptide sequence length does not match target")
    rec = read_full_residues(t.pdb_path, t.rec_chain)
    pep = read_full_residues(t.pdb_path, t.pep_chain)
    lines = []
    for i, (_nm, atoms) in enumerate(rec, start=1):
        for _atom, line in atoms.items():
            lines.append(_relabel(line, "A", i))
    lines.append("TER\n")
    nheld = 0
    for k in range(a, b + 1):
        if k >= len(pep):
            break
        nheld += 1
        _nm, atoms = pep[k]
        # number the template peptide residues CONTIGUOUSLY from 1 (not a+1..b+1):
        # Boltz builds a per-chain sequence of just these residues and indexes it by
        # residue number, so a chain starting at 6 overruns a 4-long sequence. Boltz
        # then sequence-aligns this sub-chain to the full peptide entity to place it.
        for atom, line in atoms.items():
            if backbone_only and atom not in ("N", "CA", "C", "O"):
                continue
            if peptide_sequence is not None:
                line = f"{line[:17]}{_AA3[peptide_sequence[k]]}{line[20:]}"
            lines.append(_relabel(line, "B", nheld))
    if nheld != b - a + 1:
        raise ValueError(f"held window [{a},{b}] has only {nheld} template residues")
    lines.append("TER\nEND\n")
    Path(out_pdb).write_text("".join(lines))
    return nheld


def _held_templates_block(tmpl_cif: str, threshold: float = 2.0) -> str:
    # CIF (not PDB): boltz's PDB template parser crashes on hand-built partial chains
    # (IndexError in parse_polymer); a gemmi-built CIF whose entity sequence matches its
    # coordinates is what survives the alignment (see cpseq targets.write_target_template_cif).
    return ("templates:\n"
            f"  - cif: {tmpl_cif}\n"
            "    chain_id: [A, B]\n"
            "    force: true\n"
            f"    threshold: {threshold}\n")


def _chain_ca(pdb, chain):
    out = {}
    for line in open(pdb):
        if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA":
            out[int(line[22:26])] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return out


def held_rmsd(pdb, t, a, b):
    """Held peptide CA RMSD after aligning the predicted receptor to native."""
    import numpy as np

    from lnr.rescore_rings import kabsch

    pr_rec = _chain_ca(pdb, "A")
    nat_rec = {k + 1: t.rec_ca[k] for k in range(len(t.rec_ca))}
    common = sorted(set(pr_rec) & set(nat_rec))
    if len(common) < 3:
        return None
    P = np.array([pr_rec[i] for i in common])
    Q = np.array([nat_rec[i] for i in common])
    R, cP, cQ = kabsch(P, Q)
    pr_pep = _chain_ca(pdb, "B")
    ds = []
    for k in range(a, b + 1):
        if k + 1 in pr_pep:
            x = R @ (np.array(pr_pep[k + 1]) - cP) + cQ
            ds.append(float(((x - t.CA[k]) ** 2).sum()))
    return round(float(np.sqrt(sum(ds) / len(ds))), 3) if len(ds) == b - a + 1 else None


def run_held(args, targets, d, work, cfg, bind):
    """Window-level: fold the LLM's committed window vs the rule's, each FORCE-held
    at its native bound pose, and compare the Boltz binding dG. Tests whether the
    LLM's allocation lets a real folder close a better-binding ring than the rule's."""
    import numpy as np

    from lnr.rescore_rings import kabsch
    from proteinfoundation.cpseq.adapters.boltz2 import build_input_yaml, predict
    from proteinfoundation.cpseq.scorer import score_structure
    from proteinfoundation.cpseq.targets import write_target_template_cif

    # force-template steering uses a potential, which is gated behind --use_potentials
    cfg.use_potentials = True

    def load(p):
        p = Path(p)
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    clash = {(c["target_id"], c["a"], c["b"]): c.get("clash_free", False)
             for c in load(d / "clashes.jsonl")}
    llm_rows, ps_rows = {}, {}
    for f in sorted(d.glob("llm_*.jsonl")):
        for r in load(f):
            if r["arm"] == "llm-mask" and r.get("model_id") == "gpt-6-luna":
                llm_rows[r["target_id"]] = r
    for f in sorted(d.glob("rules*.jsonl")):
        for r in load(f):
            if r["arm"] == "placement_search":
                ps_rows[r["target_id"]] = r

    def best_clean(r):
        best = (-1.0, None)
        for h in r.get("history", []):
            if (h.get("closes") and clash.get((r["target_id"], h["a"], h["b"]), False)
                    and h["retention"] > best[0]):
                best = (h["retention"], (h["a"], h["b"]))
        return best[1]

    def held_rmsd(pdb, t, a, b):
        try:
            pr_rec = _chain_ca(pdb, "A")
            nat_rec = {k + 1: t.rec_ca[k] for k in range(len(t.rec_ca))}
            common = sorted(set(pr_rec) & set(nat_rec))
            if len(common) < 3:
                return None
            P = np.array([pr_rec[i] for i in common]); Q = np.array([nat_rec[i] for i in common])
            R, cP, cQ = kabsch(P, Q)
            pr_pep = _chain_ca(pdb, "B")
            ds = []
            for k in range(a, b + 1):
                if k + 1 in pr_pep:
                    x = R @ (np.array(pr_pep[k + 1]) - cP) + cQ
                    ds.append(float(((x - t.CA[k]) ** 2).sum()))
            return round(float(np.sqrt(sum(ds) / len(ds))), 3) if ds else None
        except Exception:
            return None

    out = d / "boltz_held.jsonl"
    rows = []
    with out.open("w") as fh:
        for tid in [x.strip() for x in args.targets.split(",") if x.strip()]:
            if tid not in targets:
                continue
            t = targets[tid]
            recseq, pocket = receptor_seq_and_pocket(t)
            arms = {
                "llm": best_clean(llm_rows.get(tid, {})),
                "rule": (tuple(ps_rows[tid]["best_window"])
                         if tid in ps_rows and ps_rows[tid].get("best_window") else None),
            }
            for armname, w in arms.items():
                rec = {"target_id": tid, "arm": armname,
                       "window": list(w) if w else None, "native_dG": bind.get(tid)}
                if not w:
                    rec["error"] = "no window"; rows.append(rec)
                    fh.write(json.dumps(rec) + "\n"); continue
                a, b = w
                tmpl = str(work / f"tmpl_{tid}_{armname}.pdb")
                rec["n_held"] = build_held_template_pdb(t, a, b, tmpl)
                cif = work / f"tmpl_{tid}_{armname}.cif"
                try:
                    write_target_template_cif(Path(tmpl), cif, BOLTZ_PY)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"cif: {type(exc).__name__}: {exc}"[:150]
                    rows.append(rec); fh.write(json.dumps(rec) + "\n"); print(rec, flush=True); continue
                yaml_text = build_input_yaml(
                    peptide_sequence=t.seq, receptor_sequences={t.rec_chain: recseq},
                    chemistry="mainchain", pocket_contacts=pocket,
                    receptor_msa_path=None, target_template_path=None,
                ) + _held_templates_block(str(cif))
                yml = work / f"held_{tid}_{armname}.yaml"
                yml.write_text(yaml_text)
                odir = work / f"heldout_{tid}_{armname}"
                try:
                    st = predict(yaml_path=yml, out_dir=odir, config=cfg, python_exec=BOLTZ_PY,
                                 boltz_cache=BOLTZ_CACHE, devices=1, accelerator="gpu", timeout_s=3600)
                    rec["predict_status"] = st.get("status") if isinstance(st, dict) else str(st)
                    if isinstance(st, dict) and st.get("status") != "ok":
                        rec["boltz_stderr"] = (st.get("stderr_tail") or "")[-500:]
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"predict: {type(exc).__name__}: {exc}"[:150]
                    rows.append(rec); fh.write(json.dumps(rec) + "\n"); print(rec, flush=True); continue
                pdbs = find_all_pdbs(odir)
                if not pdbs:
                    rec["error"] = "no pdb"
                    rows.append(rec); fh.write(json.dumps(rec) + "\n"); print(rec, flush=True); continue
                rec["boltz_pdb"] = str(pdbs[0])
                agg, reps = score_replicas(
                    pdbs, binder_chain="B", receptor_chains=["A"],
                    chemistry="mainchain", requested_sequence=t.seq,
                    refined_dir=work, stem=f"held_{tid}_{armname}",
                    rmsd_fn=lambda p: held_rmsd(p, t, a, b))
                rec["replicas"] = reps
                rec["n_folds"] = agg["n_folds"]
                rec["n_valid_folds"] = agg["n_valid"]
                rec["held_rmsd"] = agg["held_rmsd"]["median"]
                rec["score_status"] = "ok" if agg["n_valid"] > 0 else "unscored"
                # denoised binding energy: median over valid folds, with the spread
                rec["boltz_dG_refined"] = agg["dG_refined"]["median"]
                rec["boltz_dG_std"] = agg["dG_refined"]["std"]
                rec["boltz_dG_min"] = agg["dG_refined"]["min"]
                rows.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[held] {tid:6s} {armname:4s} win={rec['window']} "
                      f"held_rmsd={rec.get('held_rmsd')} native={rec.get('native_dG')} "
                      f"boltz_dG={rec.get('boltz_dG_refined')}±{rec.get('boltz_dG_std')} "
                      f"(n_valid={rec.get('n_valid_folds')}/{rec.get('n_folds')}) "
                      f"{rec.get('error','')}", flush=True)

    print("\n==== LLM window vs RULE window, force-held in Boltz ====")
    print(f"{'target':7s} {'native':>7s} | {'LLM win':>8s} {'dG':>7s} {'rmsd':>5s} | "
          f"{'RULE win':>8s} {'dG':>7s} {'rmsd':>5s}")
    by = {}
    for r in rows:
        by.setdefault(r["target_id"], {})[r["arm"]] = r
    for tid, aa in by.items():
        lm = aa.get("llm", {}); ru = aa.get("rule", {})
        print(f"{tid:7s} {str(lm.get('native_dG','')):>7s} | "
              f"{str(lm.get('window','')):>8s} {str(lm.get('boltz_dG_refined', lm.get('error','')))[:7]:>7s} "
              f"{str(lm.get('held_rmsd','')):>5s} | "
              f"{str(ru.get('window','')):>8s} {str(ru.get('boltz_dG_refined', ru.get('error','')))[:7]:>7s} "
              f"{str(ru.get('held_rmsd','')):>5s}")
    print(f"\nwrote {out}")
    return 0


def run_mut_held(args, targets, d, work, cfg):
    """Fold each sequence arm with its paired nomut held window, then score the ring.

    The window and receptor/template geometry are fixed within a pair. Peptide
    side chains are omitted from BOTH templates, so a mutated held residue does
    not inherit wild-type side-chain atoms and the nomut input has the same
    amount of structural conditioning. The sequence is supplied through Boltz's
    cyclic-peptide YAML; Rosetta scores the predicted ring directly.
    """
    from lnr.seqmut import apply_subs, _arm_label
    from proteinfoundation.cpseq.adapters.boltz2 import build_input_yaml, predict
    from proteinfoundation.cpseq.scorer import score_structure
    from proteinfoundation.cpseq.targets import write_target_template_cif

    cfg.use_potentials = True
    source = Path(args.seqmut_dir) if args.seqmut_dir else d
    wanted = [s.strip() for s in args.mut_arms.split(",") if s.strip()]
    if not wanted:
        raise SystemExit("--mut-arms needs at least one seqmut arm label")
    indexed = {}
    for f in sorted(source.glob("seqmut*_rows.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    indexed[(r["target_id"], _arm_label(r))] = r
                except (json.JSONDecodeError, KeyError):
                    continue

    requested = ([t.strip() for t in args.targets.split(",") if t.strip()]
                 if args.targets != "all" else sorted(targets))
    specs = []
    for tid in requested:
        if tid not in targets:
            print(f"[mut-held] {tid}: missing target", flush=True)
            continue
        t = targets[tid]
        added_baselines = set()
        for label in wanted:
            arm = indexed.get((tid, label))
            if arm is None or arm.get("committed_window") is None:
                print(f"[mut-held] {tid} {label}: missing scored seqmut row", flush=True)
                continue
            cl = arm.get("closure_arm") or "placement_search"
            k = arm.get("k_mut")
            base_label = f"nomut:k{k}" + (f":cl:{cl}" if cl != "placement_search" else "")
            base = indexed.get((tid, base_label))
            if base is None or base.get("committed_window") != arm["committed_window"]:
                print(f"[mut-held] {tid} {label}: missing matched nomut window", flush=True)
                continue
            if base_label not in added_baselines:
                specs.append((t, base_label, base))
                added_baselines.add(base_label)
            specs.append((t, label, arm))

    work = work / "mut_held"
    work.mkdir(parents=True, exist_ok=True)
    out = d / args.mut_output
    old = []
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    old.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    completed = {(r.get("target_id"), r.get("arm_label"), r.get("variant_id")): r
                 for r in old if r.get("score_status") == "ok"}
    variants = {(r.get("target_id"), r.get("variant_id")): r
                for r in old if r.get("score_status") == "ok"}

    n_new = 0
    with out.open("a") as fh:
        for t, label, row in specs:
            a, b = row["committed_window"]
            subs = [(int(i), aa) for i, aa in row.get("subs", [])]
            seq = apply_subs(t.seq, subs)
            key = hashlib.sha256(f"{t.target_id}|{a}|{b}|{seq}".encode()).hexdigest()[:12]
            variant_id = f"{t.target_id}_{a}_{b}_{key}"
            if not args.dry and (t.target_id, label, variant_id) in completed:
                continue
            rec = {
                "target_id": t.target_id, "arm_label": label,
                "variant_id": variant_id, "window": [a, b],
                "window_clean": row.get("window_clean"),
                "subs": row.get("subs", []), "peptide_sequence": seq,
                "native_linear_dG": row.get("native_dG"),
                "recovered_proxy_dG": row.get("recovered_binding_energy"),
            }
            prior = variants.get((t.target_id, variant_id))
            if prior is not None and not args.dry:
                for field in ("predict_status", "boltz_pdb", "held_rmsd",
                              "refined_pdb", "held_rmsd_refined",
                              "score_status", "score_reason", "sequence_matches",
                              "ring_declared", "dG_raw", "dG_refined",
                              "dG_refined_std", "dG_refined_min", "n_folds",
                              "n_valid_folds", "replicas",
                              "valid", "predict_runtime_s"):
                    rec[field] = prior.get(field)
                rec["reused_from"] = prior["arm_label"]
            else:
                stem = variant_id
                tmpl = work / f"tmpl_{stem}.pdb"
                cif = work / f"tmpl_{stem}.cif"
                yml = work / f"held_{stem}.yaml"
                try:
                    rec["n_held"] = build_held_template_pdb(
                        t, a, b, str(tmpl), peptide_sequence=seq, backbone_only=True)
                    write_target_template_cif(tmpl, cif, BOLTZ_PY)
                    recseq, pocket = receptor_seq_and_pocket(t)
                    yml.write_text(build_input_yaml(
                        peptide_sequence=seq,
                        receptor_sequences={t.rec_chain: recseq},
                        chemistry="mainchain", pocket_contacts=pocket,
                        receptor_msa_path=None, target_template_path=None,
                    ) + _held_templates_block(str(cif)))
                    rec["input_yaml"] = str(yml)
                    if args.dry:
                        print(f"[mut-held] prepared {t.target_id} {label}: {yml}", flush=True)
                        continue
                    odir = work / f"heldout_{stem}"
                    pred = predict(yaml_path=yml, out_dir=odir, config=cfg,
                                   python_exec=BOLTZ_PY, boltz_cache=BOLTZ_CACHE,
                                   devices=1, accelerator="gpu", timeout_s=3600)
                    rec["predict_status"] = pred.get("status")
                    rec["predict_runtime_s"] = pred.get("runtime_s")
                    if rec["predict_status"] != "ok":
                        rec["error"] = (pred.get("stderr_tail") or "prediction failed")[-500:]
                    else:
                        pdbs = find_all_pdbs(odir)
                        if not pdbs:
                            rec["error"] = "prediction produced no PDB"
                        else:
                            rec["boltz_pdb"] = str(pdbs[0])
                            agg, reps = score_replicas(
                                pdbs, binder_chain="B", receptor_chains=["A"],
                                chemistry="mainchain", requested_sequence=seq,
                                refined_dir=work, stem=stem,
                                rmsd_fn=lambda p: held_rmsd(p, t, a, b))
                            rec["replicas"] = reps
                            rec["n_folds"] = agg["n_folds"]
                            rec["n_valid_folds"] = agg["n_valid"]
                            # Denoised dG: MEDIAN over the folds that scored a valid
                            # ring; dG_refined_std / _min report the single-fold
                            # spread this replication removes. The single scalar
                            # fields keep the median so the downstream report and the
                            # resume-dedup read a denoised number, not one lucky draw.
                            rec["dG_refined"] = agg["dG_refined"]["median"]
                            rec["dG_refined_std"] = agg["dG_refined"]["std"]
                            rec["dG_refined_min"] = agg["dG_refined"]["min"]
                            rec["dG_raw"] = agg["dG_raw"]["median"]
                            # Do NOT gate on held_rmsd. Cyclisation necessarily
                            # moves residues off the LINEAR bound pose -- that
                            # displacement is the very dG loss this project
                            # measures -- so demanding the held window stay <=2 A
                            # of the linear native contradicts the premise and
                            # rejected every fold (all ~4-6 A). held_rmsd is kept
                            # as a cyclisation-deviation DIAGNOSTIC, not a gate.
                            rec["held_rmsd"] = agg["held_rmsd"]["median"]
                            rec["held_rmsd_refined"] = agg["held_rmsd_refined"]["median"]
                            v0 = next((r for r in reps if r.get("valid")), reps[0])
                            if v0.get("refined_pdb"):
                                rec["refined_pdb"] = v0["refined_pdb"]
                            # Row-level validity: at least one fold scored a valid
                            # ring of the requested sequence (same gate, per replica).
                            rec["score_status"] = ("ok" if agg["n_valid"] > 0
                                                   else v0.get("score_status"))
                            rec["score_reason"] = None if agg["n_valid"] > 0 else v0.get("reason")
                            rec["sequence_matches"] = any(r.get("sequence_matches") for r in reps)
                            rec["ring_declared"] = any(r.get("ring_declared") for r in reps)
                            rec["valid"] = agg["n_valid"] > 0
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_new += 1
            if rec.get("score_status") == "ok":
                variants[(t.target_id, variant_id)] = rec
            print(f"[mut-held] {t.target_id:6s} {label:30s} win=[{a},{b}] "
                  f"subs={rec['subs']} rmsd={rec.get('held_rmsd')}/"
                  f"{rec.get('held_rmsd_refined')} "
                  f"raw={rec.get('dG_raw')} "
                  f"refined={rec.get('dG_refined')}±{rec.get('dG_refined_std')} "
                  f"(n_valid={rec.get('n_valid_folds')}/{rec.get('n_folds')}) "
                  f"valid={rec.get('valid')} {rec.get('error', '')}", flush=True)

    if args.dry:
        print(f"[mut-held] prepared {len(specs)} arm rows in {work}")
        return 0
    print(f"[mut-held] wrote {n_new} new rows to {out}")
    report_mut_held(out)
    return 0


def _is_valid(r) -> bool:
    """Recomputed validity, independent of any stale stored `valid` (which may
    have been written under the old held_rmsd<=2 gate). A fold counts if it
    scored, is the requested sequence, closed a ring, and has a refined dG --
    held_rmsd is a diagnostic, NOT a gate (see run_mut_held)."""
    return (r.get("score_status") == "ok"
            and r.get("sequence_matches") is True
            and r.get("ring_declared") is True
            and r.get("dG_refined") is not None)


def report_mut_held(out: Path, rmsd_max: float = 8.0):
    """Paired direct Rosetta scores over sequence/ring-valid folds. held_rmsd is
    a cyclisation-deviation diagnostic, not a validity gate, but a pair is only
    averaged when BOTH folds stayed within `rmsd_max` of the native interface --
    a grossly mislocalised fold is dropped (and counted), never silently pooled."""
    if not out.exists():
        print(f"[mut-held] no results at {out}")
        return
    rows = {}
    for line in out.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                rows[(r["target_id"], r["arm_label"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    print("\n==== force-held mutant-sequence Boltz -> direct Rosetta (REU) ====")
    print("dG columns are the MEDIAN over valid fold replicas; +/-sigma is the "
          "single-fold spread, n = valid/total folds.")
    print(f"{'target':7s} {'arm':30s} {'window':10s} {'held Å':>7s} "
          f"{'refined dG (median+/-sd)':>24s} {'n':>6s} {'valid':>5s}")
    for (_, _), r in sorted(rows.items()):
        sd = r.get("dG_refined_std")
        dg = r.get("dG_refined")
        dgs = f"{dg:+.2f}+/-{sd:.2f}" if dg is not None and sd is not None else str(dg)
        nf = f"{r.get('n_valid_folds','?')}/{r.get('n_folds','?')}"
        print(f"{r['target_id']:7s} {r['arm_label']:30s} "
              f"{str(r.get('window')):10s} {str(r.get('held_rmsd')):>7s} "
              f"{dgs:>24s} {nf:>6s} {str(r.get('valid')):>5s}")
    for label in sorted({r["arm_label"] for r in rows.values()
                         if not r["arm_label"].startswith("nomut:")}):
        paired, excluded = [], 0
        for (tid, arm_label), arm in rows.items():
            if arm_label != label or not _is_valid(arm):
                continue
            parts = label.split(":")
            cl = parts[parts.index("cl") + 1] if "cl" in parts else None
            k = next((x for x in parts if x.startswith("k") and x[1:].isdigit()), None)
            base_label = f"nomut:{k}" + (f":cl:{cl}" if cl else "")
            base = rows.get((tid, base_label))
            if not (base and _is_valid(base) and base.get("window") == arm.get("window")):
                continue
            # pair-quality guard: a pair is only comparable if BOTH folds stayed
            # near the native interface. ~4-6 A is the expected cyclisation
            # deviation; a fold that flew to >rmsd_max (e.g. 1d4t's 12 A mutant)
            # is mislocalised, not a worse binder, and conflates the two -- drop
            # it from the mean but count it, never silently average it in.
            ra, rb = arm.get("held_rmsd"), base.get("held_rmsd")
            if ra is None or rb is None or ra > rmsd_max or rb > rmsd_max:
                excluded += 1
                continue
            sa, sb = arm.get("dG_refined_std"), base.get("dG_refined_std")
            # fold-noise sigma on the PAIRED delta: independent folds, so the
            # variances add. None (single-fold / legacy rows) -> unknown, dropped
            # from the pooled estimate rather than counted as zero noise.
            dsig = ((sa ** 2 + sb ** 2) ** 0.5
                    if sa is not None and sb is not None else None)
            paired.append((arm["dG_refined"] - base["dG_refined"], tid, ra, rb, dsig))
        if paired:
            better = sum(dg < 0 for dg, *_ in paired)
            rmsds = [r for (_, _, ra, rb, *_) in paired for r in (ra, rb)]
            mean_delta = sum(dg for dg, *_ in paired) / len(paired)
            dsigs = [d for *_, d in paired if d is not None]
            noise = f"; mean fold-noise σ(Δ)≈{sum(dsigs)/len(dsigs):.2f} REU" if dsigs else ""
            print(f"  {label}: n={len(paired)} mean ΔdG={mean_delta:+.2f} REU; "
                  f"better(mut binds stronger)={better}/{len(paired)}; "
                  f"mean held_rmsd≈{sum(rmsds)/len(rmsds):.1f} Å{noise}"
                  + (f"; {excluded} pair(s) dropped (fold >{rmsd_max} Å)" if excluded else ""))
            print("    " + ", ".join(
                f"{tid}:{dg:+.2f}" + (f"(σΔ{d:.1f})" if d is not None else "")
                for dg, tid, _ra, _rb, d in paired))
        else:
            print(f"  {label}: no comparable folds yet"
                  + (f" ({excluded} dropped for fold >{rmsd_max} Å)" if excluded else ""))
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--dry", action="store_true", help="build YAML only, no GPU fold")
    ap.add_argument("--mode", choices=["target", "held", "mut-held"], default="held",
                    help="target = fold native cyclic seq, correlate retention vs dG; "
                    "held = fold LLM vs rule committed window force-held, compare dG; "
                    "mut-held = fold seqmut sequences with their paired held window")
    ap.add_argument("--seqmut-dir", default="",
                    help="directory containing seqmut*_rows.jsonl (default out-dir)")
    ap.add_argument("--mut-arms", default="llm:gpt-6-luna:k2,llm-ala:gpt-6.1-terra:k2",
                    help="comma-separated exact seqmut arm labels for mut-held mode")
    ap.add_argument("--mut-output", default="boltz_mut_held.jsonl",
                    help="output filename for mut-held mode (within out-dir)")
    ap.add_argument("--work", default="")
    ap.add_argument("--replicas", type=int, default=5,
                    help="diffusion samples to fold and score per variant; the real "
                    "dG is the MEDIAN over the valid ones and +/-sigma is the fold "
                    "noise. >4 engages the boltz_chunkfix shadow (see "
                    "configure_replicas). 1 reproduces the old single-fold behaviour.")
    args = ap.parse_args()

    from proteinfoundation.cpseq.adapters.boltz2 import (Boltz2Config,
                                                         build_input_yaml, predict)

    d = Path(args.out_dir)
    work = Path(args.work) if args.work else d / "boltz_val"
    work.mkdir(parents=True, exist_ok=True)
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}

    def load(p):
        p = Path(p)
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    frontier = {r["target_id"]: r for r in load(d / "frontier.jsonl")}
    bind = {b["target_id"]: b.get("native_dG")
            for b in load(d / "rosetta_interface.jsonl") if "native_dG" in b}

    cfg = Boltz2Config()
    shadow = configure_replicas(cfg, args.replicas)
    print(f"[boltz] folding {cfg.diffusion_samples} replicas/variant"
          + (f"; chunkfix shadow on PYTHONPATH: {shadow}" if shadow else "")
          + f"; max_parallel_samples={cfg.max_parallel_samples}", flush=True)
    if args.mode == "mut-held":
        return run_mut_held(args, targets, d, work, cfg)
    if args.mode == "held" and not args.dry:
        return run_held(args, targets, d, work, cfg, bind)

    rows = []
    out = d / "boltz_validate.jsonl"
    fh = out.open("w")
    for tid in [x.strip() for x in args.targets.split(",") if x.strip()]:
        rec = {"target_id": tid}
        if tid not in targets:
            rec["error"] = "not in manifest"; rows.append(rec)
            fh.write(json.dumps(rec) + "\n"); continue
        t = targets[tid]
        recseq, pocket = receptor_seq_and_pocket(t)
        rec["pep_seq"] = t.seq
        rec["n_pocket"] = len(pocket)
        rec["native_dG"] = bind.get(tid)
        fr = frontier.get(tid, {}).get("ceiling")
        rec["retention_ceiling"] = fr["retention"] if fr else None
        yaml_text = build_input_yaml(
            peptide_sequence=t.seq,
            receptor_sequences={t.rec_chain: recseq},
            chemistry="mainchain",
            pocket_contacts=pocket,
            receptor_msa_path=None,
            target_template_path=None,
        )
        yml = work / f"{tid}.yaml"
        yml.write_text(yaml_text)
        rec["yaml"] = str(yml)
        if args.dry:
            print(f"[dry] {tid}: pep_len={len(t.seq)} rec_len={len(recseq)} "
                  f"pocket={len(pocket)} native_dG={rec['native_dG']} "
                  f"ret_ceil={rec['retention_ceiling']}")
            rows.append(rec); fh.write(json.dumps(rec) + "\n"); continue

        odir = work / f"out_{tid}"
        try:
            status = predict(yaml_path=yml, out_dir=odir, config=cfg,
                             python_exec=BOLTZ_PY, boltz_cache=BOLTZ_CACHE,
                             devices=1, accelerator="gpu", timeout_s=3600)
            rec["predict_status"] = status.get("status") if isinstance(status, dict) else str(status)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"predict: {type(exc).__name__}: {exc}"[:200]
            rows.append(rec); fh.write(json.dumps(rec) + "\n"); print(rec); continue
        pdbs = find_all_pdbs(odir)
        if not pdbs:
            rec["error"] = "no pdb produced"
            rows.append(rec); fh.write(json.dumps(rec) + "\n"); print(rec); continue
        rec["boltz_pdb"] = str(pdbs[0])
        try:
            agg, reps = score_replicas(
                pdbs, binder_chain="B", receptor_chains=["A"],
                chemistry="mainchain", requested_sequence=t.seq,
                refined_dir=work, stem=f"out_{tid}")
            rec["replicas"] = reps
            rec["n_folds"] = agg["n_folds"]
            rec["n_valid_folds"] = agg["n_valid"]
            rec["score_status"] = "ok" if agg["n_valid"] > 0 else "unscored"
            # denoised binding energy of the freely-cyclised native sequence
            rec["boltz_dG_refined"] = agg["dG_refined"]["median"]
            rec["boltz_dG_std"] = agg["dG_refined"]["std"]
            rec["boltz_dG_min"] = agg["dG_refined"]["min"]
            rec["boltz_dG_raw"] = agg["dG_raw"]["median"]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"score: {type(exc).__name__}: {exc}"[:200]
        rows.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"[boltz] {tid}: native={rec.get('native_dG')} "
              f"ret_ceil={rec.get('retention_ceiling')} "
              f"boltz_dG={rec.get('boltz_dG_refined')}±{rec.get('boltz_dG_std')} "
              f"(n_valid={rec.get('n_valid_folds')}/{rec.get('n_folds')}) "
              f"status={rec.get('score_status')} {rec.get('error','')}", flush=True)
    fh.close()

    # correlation summary (if we have folded numbers)
    pts = [(r["retention_ceiling"], r.get("boltz_dG_refined"), r.get("native_dG"), r["target_id"])
           for r in rows if r.get("boltz_dG_refined") is not None
           and r.get("retention_ceiling") is not None]
    if len(pts) >= 3:
        def spearman(x, y):
            n = len(x)
            def rk(v):
                o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n
                for p, i in enumerate(o): r[i] = p
                return r
            rx, ry = rk(x), rk(y); mx = sum(rx) / n; my = sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            dx = sum((a - mx) ** 2 for a in rx) ** .5; dy = sum((b - my) ** 2 for b in ry) ** .5
            return num / (dx * dy) if dx and dy else None
        ret = [p[0] for p in pts]; bz = [p[1] for p in pts]; nat = [p[2] for p in pts if p[2] is not None]
        print(f"\n==== VALIDATION (n={len(pts)}) ====")
        print(f"Spearman(retention_ceiling, boltz_dG) = {spearman(ret, bz):+.2f}  "
              f"(negative = high retention -> stronger/more-negative dG, the HOPED sign)")
        print(f"{'target':8s} {'ret_ceil':>8s} {'boltz_dG':>9s} {'native_dG':>9s}")
        for r0, b0, n0, tid in sorted(pts):
            print(f"{tid:8s} {r0:8.3f} {b0:9.1f} {str(n0):>9s}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
