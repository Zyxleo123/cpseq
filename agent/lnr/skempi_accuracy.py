"""THE ACCURACY GATE: does our Rosetta interface-dG scorer rank REAL mutations by
their MEASURED binding change?

Everything in this project so far measured the oracle's PRECISION (agreement with
itself). This measures its ACCURACY (agreement with reality) -- the one thing that
decides whether any of the search/denoising work is worth continuing. It uses
SKEMPI v2: ~7000 protein-protein interface mutations with experimentally measured
ddG (from Kd), and the 90 complex crystal structures the stage0 pilot already has
on disk.

What it does
------------
For each selected single interface mutation: load the crystal complex, relax +
score the interface dG (Rosetta InterfaceAnalyzer dG_separated) for WT and for the
mutant (R independent relaxes each, averaged to beat the ~1.6 REU relax noise),
take ddG_pred = dG_mut - dG_wt, and correlate ddG_pred against the measured ddG
across all records (Spearman + Pearson).

Sign convention: measured ddG = RT ln(Kd_mut/Kd_wt), so POSITIVE = the mutation
WEAKENS binding; dG_separated is more-negative = stronger, so ddG_pred positive =
weaker too. A working scorer gives a POSITIVE correlation.

Why this isolates the scorer (and what it does NOT test)
--------------------------------------------------------
This scores the CRYSTAL structure directly -- no Boltz fold -- so it tests the
Rosetta scoring component alone, free of fold noise, in the best case (a real
bound structure). If the scorer cannot rank mutations HERE, the Boltz-fold version
(which adds ~4 REU fold noise and an OOD cyclic backbone) has no chance, and the
objective must change. If it CAN (Flex ddG, the ensemble gold standard, reaches
Pearson ~0.6 here; a single-relax scorer will be lower), the scorer is usable and
the project's problem is fold noise + cyclization, not the energy function.

Caveat: SKEMPI is protein-PROTEIN interface mutations. This validates the
interface-dG scorer on PPIs; our actual target is a cyclic PEPTIDE, still OOD. A
good number here is necessary, not sufficient; a bad number is decisive.

Usage
-----
  python -m lnr.skempi_accuracy --score --csv <skempi_v2.csv> --pdb-dir <pdb/> \
      --out <skempi_ddg.jsonl> --max-records 60 --replicas 3 --workers 16
  python -m lnr.skempi_accuracy --report <skempi_ddg.jsonl>
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

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
AA1 = set(AA3to1.values())


# ---------------------------------------------------------------------------
# Selection (pure, testable)
# ---------------------------------------------------------------------------

def _split_pos(pos: str):
    """'38' -> (38, '') ; '38A' -> (38, 'A') (insertion code)."""
    pos = str(pos).strip()
    i = 0
    while i < len(pos) and (pos[i].isdigit() or (i == 0 and pos[i] == "-")):
        i += 1
    if i == 0:
        return None, None
    return int(pos[:i]), pos[i:].strip()


def pdb_path_for(pdb_dir, pdb_id):
    for name in (pdb_id, pdb_id.lower(), pdb_id.upper()):
        p = Path(pdb_dir) / f"{name}.pdb"
        if p.exists():
            return p
    return None


def _count_ca(pdb_path, keep_chains) -> int:
    """CA atoms in the kept chains -- the relax workload / memory proxy."""
    keep = set(keep_chains)
    n = 0
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA " and line[21] in keep:
            n += 1
    return n


def select_records(records, pdb_dir, max_records=0, max_ca=0):
    """Single interface mutations with a resolvable PDB residue and standard AAs.
    `max_ca` drops complexes larger than that many CA residues (the 1000+ residue
    antibody complexes OOM a shared-memory relax and are far from our ~300-residue
    peptide-receptor scale anyway). Returns (selected, reasons); a shrinking
    denominator must be visible, not silent."""
    reasons = {"not_single": 0, "not_interface": 0, "no_pdb": 0,
               "nonstandard_aa": 0, "bad_position": 0, "too_big": 0, "kept": 0}
    out = []
    for r in records:
        if r.k != 1:
            reasons["not_single"] += 1
            continue
        if not r.at_interface:
            reasons["not_interface"] += 1
            continue
        pp = pdb_path_for(pdb_dir, r.pdb_id)
        if pp is None:
            reasons["no_pdb"] += 1
            continue
        m = r.mutations[0]
        if m.wt not in AA1 or m.mut not in AA1:
            reasons["nonstandard_aa"] += 1
            continue
        ch, pos = m.pdb_site
        num, icode = _split_pos(pos)
        if num is None:
            reasons["bad_position"] += 1
            continue
        if max_ca:
            try:
                if _count_ca(pp, r.chains1 + r.chains2) > max_ca:
                    reasons["too_big"] += 1
                    continue
            except OSError:
                reasons["no_pdb"] += 1
                continue
        reasons["kept"] += 1
        out.append({
            "complex_id": r.complex_id, "pdb_id": r.pdb_id,
            "pdb_path": str(pp), "chains1": r.chains1, "chains2": r.chains2,
            "interface": f"{r.chains1}_{r.chains2}",
            "keep_chains": r.chains1 + r.chains2,
            "mut_chain": ch, "mut_resnum": num, "mut_icode": icode,
            "wt": m.wt, "mut": m.mut, "mut_str": str(m),
            "measured_ddg": r.ddg, "location": ",".join(r.locations),
        })
    out.sort(key=lambda d: (d["pdb_id"], d["mut_resnum"], d["mut"]))
    if max_records and len(out) > max_records:
        # spread across complexes, don't take all of one: round-robin by pdb
        by = {}
        for d in out:
            by.setdefault(d["pdb_id"], []).append(d)
        picked, i = [], 0
        order = sorted(by)
        while len(picked) < max_records and any(by[p] for p in order):
            p = order[i % len(order)]
            if by[p]:
                picked.append(by[p].pop(0))
            i += 1
        out = picked
    return out, reasons


# ---------------------------------------------------------------------------
# Correlation (pure, testable)
# ---------------------------------------------------------------------------

def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rr = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rr[order[k]] = avg
            i = j + 1
        return rr
    rx, ry = rank(xs), rank(ys)
    return _pearson(rx, ry)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else None


def aggregate_predictions(rows):
    """rows: per-(complex,label,replica) dG records. label 'WT' is per-pdb; a
    mutant label is the mut_str. Returns per-record ddG_pred = mean(mut) -
    mean(WT of that pdb), paired with measured_ddg."""
    wt, mut = {}, {}
    meta = {}
    for r in rows:
        if not r.get("valid") or not isinstance(r.get("dG"), (int, float)):
            continue
        key = r["complex_id"]
        if r["label"] == "WT":
            wt.setdefault(key, []).append(r["dG"])
        else:
            mut.setdefault((key, r["label"]), []).append(r["dG"])
            meta[(key, r["label"])] = r.get("measured_ddg")
    preds = []
    for (key, lab), dgs in mut.items():
        if key not in wt or not wt[key]:
            continue
        ddg_pred = st.mean(dgs) - st.mean(wt[key])
        preds.append({"complex_id": key, "mut": lab,
                      "ddg_pred": round(ddg_pred, 3),
                      "measured_ddg": meta.get((key, lab)),
                      "n_mut": len(dgs), "n_wt": len(wt[key])})
    return preds


# ---------------------------------------------------------------------------
# Scoring worker (PyRosetta; not unit-tested)
# ---------------------------------------------------------------------------

def _fail(task, msg):
    return {"complex_id": task["complex_id"], "label": task["label"],
            "replica": task["replica"], "measured_ddg": task.get("measured_ddg"),
            "valid": False, "error": msg[:200]}


def score_one(task):
    """One independent relax+score of WT or a mutant of one complex. Fresh-process
    PyRosetta init => independent relax draw. Returns {label, replica, dG, valid}."""
    rec = {"complex_id": task["complex_id"], "label": task["label"],
           "replica": task["replica"], "measured_ddg": task.get("measured_ddg")}
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.toolbox import mutate_residue
        if not getattr(score_one, "_init", False):
            pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                           "-load_PDB_components false -relax:default_repeats 2 "
                           "-no_fconfig -use_input_sc", silent=True)
            score_one._init = True
        pose = pyrosetta.pose_from_pdb(task["pdb_path"])
        # keep only the two interacting chain groups
        keep = set(task["keep_chains"])
        info = pose.pdb_info()
        # delete residues not in the kept chains (from the end, to keep indices valid)
        for i in range(pose.total_residue(), 0, -1):
            if info.chain(i) not in keep:
                pose.delete_residue_slow(i)
        info = pose.pdb_info()
        if task["label"] != "WT":
            idx = info.pdb2pose(task["mut_chain"],
                                int(task["mut_resnum"]), task.get("mut_icode") or " ")
            if idx == 0:
                rec["error"] = "residue not found in pose"
                rec["valid"] = False
                return rec
            mutate_residue(pose, idx, task["mut"], pack_radius=8.0)
        sf = pyrosetta.get_fa_scorefxn()
        fr = FastRelax(sf, 1)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)
        ia = InterfaceAnalyzerMover(task["interface"])
        ia.set_pack_separated(True)
        ia.apply(pose)
        rec["dG"] = float(ia.get_interface_dG())
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["valid"] = False
    return rec


def _worker(task, q):
    try:
        q.put(score_one(task))
    except Exception as exc:  # noqa: BLE001
        q.put(_fail(task, f"{type(exc).__name__}: {exc}"))


def _run_isolated(tasks, workers, timeout):
    """Run each task in its OWN short-lived process. A task that OOMs, segfaults or
    hangs costs exactly one failed row -- never the whole run (the ProcessPoolExecutor
    failure mode that killed the first attempt: one dead worker -> BrokenProcessPool
    -> everything lost). Each process also frees all PyRosetta memory on exit, so
    memory does not accumulate across the big complexes."""
    import multiprocessing as mp
    import time
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    running = {}   # proc -> (task, queue, start)
    while pending or running:
        while pending and len(running) < workers:
            t = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(t, q), daemon=True)
            p.start()
            running[p] = (t, q, time.time())
        time.sleep(0.5)
        for p in list(running):
            t, q, start = running[p]
            if not p.is_alive():
                p.join()
                try:
                    rec = q.get(timeout=10)
                except Exception:  # noqa: BLE001  died before putting a result
                    rec = _fail(t, f"worker died (exitcode {p.exitcode}; likely OOM/segfault)")
                running.pop(p, None)
                yield rec
            elif time.time() - start > timeout:
                p.terminate(); p.join()
                running.pop(p, None)
                yield _fail(t, f"timeout after {timeout}s")


def score(csv_path, pdb_dir, out_path, max_records, replicas, workers,
          max_ca=0, timeout=1800, _scorer=None):
    from skempi import load_records

    records, report = load_records(csv_path)
    selected, reasons = select_records(records, pdb_dir, max_records=max_records,
                                       max_ca=max_ca)
    print(f"[skempi-acc] {report['csv_rows']} csv rows -> {len(records)} parsed; "
          f"selection {reasons}", flush=True)
    print(f"[skempi-acc] {len(selected)} mutations across "
          f"{len({s['complex_id'] for s in selected})} complexes", flush=True)

    done = set()
    if Path(out_path).exists():
        for line in Path(out_path).read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["complex_id"], r["label"], r["replica"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    # tasks: WT per complex (dedup) + each mutant, R replicas each
    seen_wt, tasks = set(), []
    for s in selected:
        if s["complex_id"] not in seen_wt:
            seen_wt.add(s["complex_id"])
            for rep in range(replicas):
                t = dict(s, label="WT", replica=rep, measured_ddg=None)
                if (t["complex_id"], "WT", rep) not in done:
                    tasks.append(t)
        for rep in range(replicas):
            t = dict(s, label=s["mut_str"], replica=rep)
            if (t["complex_id"], s["mut_str"], rep) not in done:
                tasks.append(t)

    print(f"[skempi-acc] {len(tasks)} relax+score tasks ({len(done)} done)", flush=True)
    n = 0
    with Path(out_path).open("a") as fh:
        if _scorer is not None:                     # tests: serial, in-process
            for t in tasks:
                fh.write(json.dumps(_scorer(t)) + "\n"); fh.flush(); n += 1
        else:                                       # real: crash-isolated subprocesses
            total = len(tasks)
            for rec in _run_isolated(tasks, workers, timeout):
                fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                if n % 10 == 0 or not rec.get("valid"):
                    tag = "" if rec.get("valid") else f"  FAIL {rec.get('error','')}"
                    print(f"[skempi-acc] {n}/{total}{tag}", flush=True)
    return n


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(out_path):
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    preds = aggregate_predictions(rows)
    preds = [p for p in preds if isinstance(p["measured_ddg"], (int, float))
             and isinstance(p["ddg_pred"], (int, float))]
    print("=" * 90)
    print("ACCURACY GATE -- Rosetta interface ddG vs MEASURED ddG (SKEMPI v2)")
    print("positive ddG = mutation WEAKENS binding (both axes). A working scorer => positive rho.")
    print("=" * 90)
    n_fail = sum(1 for r in rows if not r.get("valid"))
    if n_fail:
        print(f"\n({n_fail} relax/score tasks failed; excluded)")
    if len(preds) < 3:
        print(f"\nonly {len(preds)} scored mutations -- need >=3. Still running?")
        return
    xs = [p["ddg_pred"] for p in preds]
    ys = [p["measured_ddg"] for p in preds]
    rho = _spearman(xs, ys)
    r = _pearson(xs, ys)
    # sign accuracy: does predicted sign match measured sign (destabilizing or not)?
    sign_ok = sum(1 for a, b in zip(xs, ys) if (a > 0) == (b > 0)) / len(xs)
    print(f"\nn = {len(preds)} mutations, {len({p['complex_id'] for p in preds})} complexes")
    print(f"  Spearman rho = {rho:+.3f}" if rho is not None else "  Spearman: n/a")
    print(f"  Pearson  r   = {r:+.3f}" if r is not None else "  Pearson: n/a")
    print(f"  sign agreement = {100*sign_ok:.0f}%  (predicted stabilizing/destabilizing matches)")
    print(f"  measured ddG range [{min(ys):+.2f}, {max(ys):+.2f}] kcal/mol; "
          f"pred range [{min(xs):+.2f}, {max(xs):+.2f}] REU")
    # worst disagreements
    pairs = sorted(preds, key=lambda p: -abs(p["ddg_pred"] - p["measured_ddg"]))
    print("\n  largest disagreements (pred REU vs measured kcal/mol):")
    for p in pairs[:5]:
        print(f"    {p['complex_id']:14s} {p['mut']:8s} pred {p['ddg_pred']:+7.2f} "
              f"measured {p['measured_ddg']:+6.2f}")
    print("\n" + "=" * 90)
    print("VERDICT")
    print("  reference: Flex ddG (35-model ensemble) reaches Pearson ~0.6 here;")
    print("  our single-relax scorer should land below that if it works at all.")
    print()
    best = max([x for x in (rho, r) if x is not None], default=0.0)
    if best >= 0.4:
        print(f"  PASS: the scorer ranks real mutations (best |corr| {best:.2f}). The")
        print("  energy function is usable; the project's problem is fold noise +")
        print("  cyclization OOD, not the score. Worth denoising the fold (more Boltz")
        print("  samples) and continuing -- after checking it survives the cyclic case.")
    elif best >= 0.2:
        print(f"  WEAK: corr {best:.2f} -- real but marginal signal, consistent with a")
        print("  single-relax (no ensemble) scorer. Ranking is possible but hard; it")
        print("  will only get worse through the noisy Boltz fold. Favor the coarser")
        print("  objective (binder/non-binder, pass/fail vs ceiling).")
    else:
        print(f"  FAIL: corr {best:.2f} ~ 0. The scorer does not track real binding even")
        print("  on crystal structures with no fold noise. Ranking affinity is not the")
        print("  objective to pursue -- change it, or change the scorer entirely.")
    print("=" * 90)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--csv", default="")
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-records", type=int, default=60)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-ca", type=int, default=600,
                    help="skip complexes larger than this many CA residues (0=all)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-task wall limit (s); a task over it is one failed row")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    if args.score:
        if not (args.csv and args.pdb_dir and args.out):
            raise SystemExit("--score needs --csv --pdb-dir --out")
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        n = score(args.csv, args.pdb_dir, out, args.max_records, args.replicas,
                  args.workers, max_ca=args.max_ca, timeout=args.timeout)
        print(f"[skempi-acc] wrote {n} task rows -> {out}", flush=True)
        report(out)
        return 0
    if args.report:
        report(Path(args.report))
        return 0
    raise SystemExit("use --score or --report")


if __name__ == "__main__":
    raise SystemExit(main())
