"""Re-grade every LNR campaign under the FINER backbone-triplet clash gate.

Companion to clashcheck.py. That one grades closure at CA resolution (3.0 A);
this one grades the same closing windows under a backbone-triplet gate (N/CA/C
on both sides, receptor read at backbone resolution, 2.5 A cutoff calibrated so
every deposited native stays clash-free). It writes BOTH verdicts per window
(`clash_free` CA + `clash_free_bb` backbone) from one re-solve, so the report
can switch gate with --clash-field and the clean-closure result can be shown to
survive a 3x finer physical check, not just the coarse CA one.

Writes a SEPARATE file (clashes_bb.jsonl) -- it never touches clashes.jsonl, so
it cannot collide with a running CA-gate re-grade.

Resumable: one row per (target, a, b).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from lnr import env as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="lnr outdir with *.jsonl")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--n-restarts", type=int, default=8)
    ap.add_argument("--self-cut", type=float, default=2.5)
    ap.add_argument("--rec-cut", type=float, default=2.5)
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out) if args.out else d / "clashes_bb.jsonl"

    targets = {t.target_id: t for t in
               E.load_manifest(args.manifest, args.pdb_dir)}

    # Native floor sanity: the deposited (linear, real) peptides must be
    # clash-free under this gate, or the cutoff is mis-set. Checks the smallest
    # non-adjacent backbone-atom distance in each deposited peptide.
    import numpy as np
    bad = []
    for tid, t in targets.items():
        L = t.length
        pep = np.stack([t.N, t.CA, t.C], axis=1).reshape(-1, 3)
        res_of = np.repeat(np.arange(L), 3)
        dd = np.linalg.norm(pep[:, None, :] - pep[None, :, :], axis=-1)
        di = np.abs(res_of[:, None] - res_of[None, :])
        far = di >= 2                      # linear peptide: no ring wrap
        mind = dd[far].min() if far.any() else np.inf
        if mind < args.self_cut:
            bad.append((tid, float(mind)))
    if bad:
        print(f"[clash-bb] WARNING: {len(bad)} deposited natives fail the "
              f"self cutoff {args.self_cut} A: {bad[:5]} ... cutoff may be too "
              f"high", flush=True)
    else:
        print(f"[clash-bb] native-floor OK: all {len(targets)} deposited "
              f"peptides clash-free at self_cut={args.self_cut} A", flush=True)

    want: set[tuple[str, int, int]] = set()
    for f in sorted(d.glob("rules*.jsonl")) + sorted(d.glob("llm_*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r["target_id"]
            if tid not in targets:
                continue
            for h in r.get("history", []):
                if h.get("closes"):
                    want.add((tid, h["a"], h["b"]))

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done.add((o["target_id"], o["a"], o["b"]))
    todo = sorted(want - done)
    print(f"[clash-bb] {len(want)} closing windows, {len(done)} done, "
          f"{len(todo)} to check", flush=True)

    with out.open("a") as fh:
        for i, (tid, a, b) in enumerate(todo, 1):
            t = targets[tid]
            rep = E.clash_report_fine(t, a, b, self_cut_A=args.self_cut,
                                      rec_cut_A=args.rec_cut,
                                      max_seconds=args.max_seconds,
                                      n_restarts=args.n_restarts)
            rep.update({"target_id": tid, "a": a, "b": b})
            fh.write(json.dumps(rep) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[clash-bb] {i}/{len(todo)}", flush=True)
    print("[clash-bb] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
