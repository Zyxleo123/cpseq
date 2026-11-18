"""Post-hoc: re-grade every LNR campaign under CLEAN (clash-free) closure.

The campaigns scored GEOMETRIC closure -- closing torsions exist within
tolerance. A Rosetta-free steric check (`env.clash_report`) shows that is too
permissive: some closing windows drive the backbone through itself or the
receptor. Clean closure is the more legitimate objective.

This does NOT re-run campaigns. It reads every window each campaign actually
evaluated, clash-checks the ones that geometrically closed (keyed and cached on
(target, a, b) so it is computed once across all arms), and writes a clash map.
The report then recomputes each campaign's best CLEAN-closing retention from
windows the controller really tried -- a fair re-grade on the stricter metric.

Resumable: one row per (target, a, b) in clashes.jsonl.
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
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out) if args.out else d / "clashes.jsonl"

    targets = {t.target_id: t for t in
               E.load_manifest(args.manifest, args.pdb_dir)}

    # collect every geometrically-closing window any campaign evaluated
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
    print(f"[clash] {len(want)} closing windows, {len(done)} done, "
          f"{len(todo)} to check", flush=True)

    with out.open("a") as fh:
        for i, (tid, a, b) in enumerate(todo, 1):
            t = targets[tid]
            rep = E.clash_report(t, a, b, max_seconds=args.max_seconds,
                                 n_restarts=args.n_restarts)
            rep.update({"target_id": tid, "a": a, "b": b})
            fh.write(json.dumps(rep) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[clash] {i}/{len(todo)}", flush=True)
    print("[clash] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
