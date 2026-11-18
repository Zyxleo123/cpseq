"""Does the pose-preserving PROXY rank mutations within a complex like the real
4-fold Boltz->Rosetta score does?

The proxy (lnr.seqmut: recovered = energy_weighted_retention x native_dG on the
native bound pose, local repack) is cheap and drives the search. The question
that decides whether it is a usable SELECTION oracle is not "does it match the
absolute dG" but "does it RANK candidates within one complex the way the true
score does" -- if yes, you can search on it; if no, it misleads selection (which
would explain the ~50/50 real win rate).

Per complex this holds the window fixed (the nomut committed window, which is
sequence-independent) and varies only the SEQUENCE over a spread of candidates
(WT + seeded random single mutations + the LLM's pick). Each candidate is scored
BOTH ways:
  proxy : seqmut.mutant_scan -> recovered_binding_energy (CPU, cached)
  true  : Boltz fold with diffusion_samples=4 (native 4 folds) -> score_replicas
          median dG_refined over the valid samples (denoised, per the other
          session's replica machinery).
Then Spearman(proxy, true) WITHIN each complex. A separate --report pass prints
the per-complex rho and the pooled picture.

Self-contained on purpose: it reuses importable helpers from boltz_validate
(templating, fold discovery, replica scoring) but does NOT edit that file, which
another session is live in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PC = "/zfsauton2/home/yixiz/Proteina-Complexa"
for _p in (f"{PC}/src", PC, "/zfsauton2/home/yixiz/lp2cp-agent/agent",
           "/zfsauton2/home/yixiz/lp2cp-agent/pilot/stage0"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import env as E  # noqa: E402
from lnr import seqmut as S  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _candidates(t, window, k_mut, n_random, extra_subs):
    """WT + n_random seeded single mutations at interface positions + any extra
    (e.g. the LLM's pick). Deduped by resulting sequence. Returns [(name, subs)]."""
    L = t.length
    iface = [i for i in range(L) if i < len(t.per_res_contacts)
             and t.per_res_contacts[i] > 0] or list(range(L))
    out = [("wt", [])]
    seen = {t.seq}
    h = hashlib.sha256(f"{t.target_id}|rank".encode()).digest()
    bi = 0
    order = sorted(iface, key=lambda i: hashlib.sha256(
        f"{t.target_id}|rk|{i}".encode()).digest())
    for i in order:
        if len([c for c in out if c[0].startswith("rand")]) >= n_random:
            break
        wt = t.seq[i]
        choices = [a for a in AA20 if a != wt]
        aa = choices[h[bi % len(h)] % len(choices)]
        bi += 1
        subs = [(i, aa)]
        seq = S.apply_subs(t.seq, subs)
        if seq in seen:
            continue
        seen.add(seq)
        out.append((f"rand_{i}{aa}", subs))
    for nm, subs in extra_subs:
        seq = S.apply_subs(t.seq, subs)
        if seq not in seen:
            seen.add(seq)
            out.append((nm, [(int(i), a) for i, a in subs]))
    return out


def _templates_block(cif: str) -> str:
    return (f"templates:\n  - cif: {cif}\n    chain_id: [A, B]\n"
            "    force: true\n    threshold: 2.0\n")


def run(args):
    from lnr.boltz_validate import (BOLTZ_CACHE, BOLTZ_PY, find_all_pdbs,
                                    build_held_template_pdb, held_rmsd,
                                    receptor_seq_and_pocket, score_replicas)
    from proteinfoundation.cpseq.adapters.boltz2 import (Boltz2Config,
                                                         build_input_yaml, predict)
    from proteinfoundation.cpseq.targets import write_target_template_cif

    d = Path(args.out_dir)
    work = Path(args.work) if args.work else d / "rankval"
    work.mkdir(parents=True, exist_ok=True)
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}

    # committed window + the LLM pick per target, from the seqmut rows
    win, llm_pick = {}, {}
    for f in sorted(d.glob("seqmut*_rows.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mut_arm") == "nomut" and r.get("committed_window"):
                win[r["target_id"]] = r["committed_window"]
            if (r.get("mut_arm") == "llm" and r.get("model_id") == "gpt-6-luna"
                    and r.get("k_mut") == args.k_mut and r.get("subs")):
                llm_pick[r["target_id"]] = r["subs"]

    cache = S.ScanCache(d / "seqmut_scans.jsonl")   # reuse the proxy cache
    cfg = Boltz2Config()
    cfg.diffusion_samples = 4        # native 4 folds per candidate
    cfg.use_potentials = True

    out = d / args.out_file
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["target_id"], r["variant_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    wanted = [x.strip() for x in args.targets.split(",") if x.strip()]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with out.open("a") as fh:
        for tid in wanted:
            if tid not in targets or tid not in win:
                print(f"[rank] {tid}: no target/window", flush=True)
                continue
            t = targets[tid]
            a, b = win[tid]
            extra = [("llm_luna", llm_pick[tid])] if tid in llm_pick else []
            cands = _candidates(t, (a, b), args.k_mut, args.n_random, extra)
            recseq, pocket = receptor_seq_and_pocket(t)
            for name, subs in cands:
                seq = S.apply_subs(t.seq, subs)
                vid = f"{tid}_{a}_{b}_{hashlib.sha256(seq.encode()).hexdigest()[:10]}"
                if (tid, vid) in done:
                    continue
                rec = {"target_id": tid, "variant_id": vid, "name": name,
                       "window": [a, b], "subs": [[int(i), aa] for i, aa in subs],
                       "peptide_sequence": seq}
                # ---- PROXY (pose-preserving scan) ----
                scan = S.mutant_scan(t, subs, cache, relax_repeats=args.relax_repeats,
                                     timeout=args.proxy_timeout)
                if scan.get("per_residue_ddG_ala") is not None and scan.get("native_dG") is not None:
                    rec["proxy_recovered"] = round(
                        S.energy_weighted_retention(scan["per_residue_ddG_ala"], a, b)
                        * float(scan["native_dG"]), 3)
                    rec["proxy_native_dG"] = scan["native_dG"]
                else:
                    rec["proxy_recovered"] = None
                    rec["proxy_error"] = scan.get("error")
                # ---- TRUE (Boltz 4 folds -> replica median) ----
                try:
                    tmpl = work / f"t_{vid}.pdb"
                    build_held_template_pdb(t, a, b, str(tmpl),
                                            peptide_sequence=seq, backbone_only=True)
                    cif = work / f"t_{vid}.cif"
                    write_target_template_cif(tmpl, cif, BOLTZ_PY)
                    yml = work / f"t_{vid}.yaml"
                    yml.write_text(build_input_yaml(
                        peptide_sequence=seq,
                        receptor_sequences={t.rec_chain: recseq},
                        chemistry="mainchain", pocket_contacts=pocket,
                        receptor_msa_path=None, target_template_path=None,
                    ) + _templates_block(str(cif)))
                    odir = work / f"o_{vid}"
                    t0 = time.time()
                    predict(yaml_path=yml, out_dir=odir, config=cfg,
                            python_exec=BOLTZ_PY, boltz_cache=BOLTZ_CACHE,
                            devices=1, accelerator="gpu", timeout_s=3600)
                    pdbs = find_all_pdbs(odir)
                    rec["n_folds"] = len(pdbs)
                    if pdbs:
                        agg, reps = score_replicas(
                            pdbs, binder_chain="B", receptor_chains=["A"],
                            chemistry="mainchain", requested_sequence=seq,
                            refined_dir=work, stem=f"rank_{vid}")
                        rec["true_dG_median"] = agg["dG_refined"]["median"]
                        rec["true_dG_std"] = agg["dG_refined"]["std"]
                        rec["true_n_valid"] = agg["dG_refined"]["n"]
                        rmsds = [held_rmsd(str(p), t, a, b) for p in pdbs]
                        rmsds = [x for x in rmsds if x is not None]
                        rec["held_rmsd_median"] = round(sorted(rmsds)[len(rmsds)//2], 3) if rmsds else None
                    rec["fold_seconds"] = round(time.time() - t0, 1)
                except Exception as exc:  # noqa: BLE001
                    rec["true_error"] = f"{type(exc).__name__}: {exc}"[:200]
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[rank] {tid} {name:12s} proxy={rec.get('proxy_recovered')} "
                      f"true={rec.get('true_dG_median')}±{rec.get('true_dG_std')} "
                      f"n={rec.get('true_n_valid')} rmsd={rec.get('held_rmsd_median')} "
                      f"{rec.get('true_error','')}", flush=True)
    cache.close()
    report(out)


def report(out: Path, rmsd_max: float = 8.0):
    rows = [json.loads(l) for l in Path(out).read_text().splitlines() if l.strip()]
    bytid = {}
    for r in rows:
        bytid.setdefault(r["target_id"], []).append(r)

    def spearman(x, y):
        n = len(x)
        if n < 4:
            return None
        def rk(v):
            o = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
            while i < n:
                j = i
                while j+1 < n and v[o[j+1]] == v[o[i]]:
                    j += 1
                for t in range(i, j+1):
                    r[o[t]] = (i+j)/2+1
                i = j+1
            return r
        rx, ry = rk(x), rk(y); mx = sum(rx)/n; my = sum(ry)/n
        num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
        dx = sum((a-mx)**2 for a in rx)**.5; dy = sum((b-my)**2 for b in ry)**.5
        return num/(dx*dy) if dx and dy else None

    def pearson(x, y):
        # Secondary: linear agreement of magnitudes. Reported alongside Spearman,
        # NOT as the headline -- the proxy is a SELECTION oracle (ordering is what
        # the loop uses), the proxy/true scales differ (different pipelines, and
        # retention saturates at 1 so the map can be monotonic-but-curved), and at
        # n~5-7 a single drifted-fold outlier dominates Pearson. Spearman is the
        # robust, decision-aligned metric; Pearson flags whether a LINEAR surrogate
        # could be fit.
        n = len(x)
        if n < 4:
            return None
        mx = sum(x)/n; my = sum(y)/n
        num = sum((a-mx)*(b-my) for a, b in zip(x, y))
        dx = sum((a-mx)**2 for a in x)**.5; dy = sum((b-my)**2 for b in y)**.5
        return num/(dx*dy) if dx and dy else None

    print("\n" + "="*78)
    print("PROXY vs TRUE (4-fold Boltz->Rosetta) -- within-complex RANKING")
    print("proxy = recovered (retention x native_dG, pose-preserving); "
          "true = median dG over 4 folds")
    print("="*78)
    allr = []; allp = []; regrets = []; top1_hits = 0; top1_n = 0
    for tid in sorted(bytid):
        pairs = [(r["proxy_recovered"], r["true_dG_median"], r.get("name"),
                  r.get("held_rmsd_median"))
                 for r in bytid[tid]
                 if r.get("proxy_recovered") is not None
                 and r.get("true_dG_median") is not None
                 and (r.get("held_rmsd_median") is None or r["held_rmsd_median"] <= rmsd_max)]
        if len(pairs) < 4:
            print(f"  {tid}: n={len(pairs)} (<4, no rho)")
            continue
        px = [p[0] for p in pairs]; tr = [p[1] for p in pairs]
        rho = spearman(px, tr); r_lin = pearson(px, tr)
        if rho is not None:
            allr.append(rho)
        if r_lin is not None:
            allp.append(r_lin)
        rtxt = f"{rho:+.2f}" if rho is not None else "n/a"
        ptxt = f"{r_lin:+.2f}" if r_lin is not None else "n/a"
        print(f"  {tid}: n={len(pairs)}  Spearman={rtxt}  Pearson={ptxt}")
        # selection view: argmin dG (= best binder) under each scorer.
        best_proxy = min(pairs, key=lambda p: p[0])
        best_true = min(pairs, key=lambda p: p[1])
        top1_n += 1
        hit = best_proxy[2] == best_true[2]
        if hit:
            top1_hits += 1
        # regret: true-dG lost by trusting the proxy's pick (0 = proxy picked the
        # true-best; positive REU = how much worse the proxy's choice really binds).
        regret = best_proxy[1] - best_true[1]
        regrets.append(regret)
        print(f"      proxy picks '{best_proxy[2]}' as best; true picks "
              f"'{best_true[2]}'" + ("  (AGREE)" if hit else "  (DISAGREE)")
              + f"; regret={regret:+.1f} REU")
    if allr:
        print(f"\n  mean within-complex Spearman = {sum(allr)/len(allr):+.2f}  "
              f"(headline: selection oracle = ordering), over {len(allr)} complexes")
    if allp:
        print(f"  mean within-complex Pearson  = {sum(allp)/len(allp):+.2f}  "
              f"(secondary: linear surrogate feasibility; outlier-sensitive at small n)")
    if top1_n:
        print(f"  top-1 agreement = {top1_hits}/{top1_n} complexes "
              f"(proxy-best == true-best)")
    if regrets:
        print(f"  mean regret of trusting proxy-best = {sum(regrets)/len(regrets):+.1f} REU "
              f"(0 = proxy always picks the true-best mutation)")
    print("  (Spearman>0 => proxy orders mutations like the real score; near 0 or "
          "high regret => proxy misleads selection, loop needs the 4-fold score.)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default="")
    ap.add_argument("--out-file", default="rank_proxy_vs_true.jsonl")
    ap.add_argument("--n-random", type=int, default=7)
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--relax-repeats", type=int, default=1)
    ap.add_argument("--proxy-timeout", type=float, default=900.0)
    ap.add_argument("--work", default="")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir) / args.out_file
    if args.report_only:
        report(out)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
