"""The achievable-retention ceiling per LNR target. Context, not a regret oracle.

The user's framing is that controllers are compared relative to each other, not
against a per-cell regret oracle. This computes, for each target, the best REAL
(torsion-confirmed) retention attainable -- the top of the frontier -- so the
arm table can report how much interface was on the table without that number
entering any arm's score.

It is cheap because the analytic bound is a true upper bound: windows are walked
in descending analytic retention and torsion-confirmed until one closes, so at
most a handful of solves per target are spent, and nothing above the first
confirmed window can exist.
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
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-top", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    targets = E.load_manifest(args.manifest, args.pdb_dir)
    targets.sort(key=lambda t: t.target_id)
    if args.limit:
        targets = targets[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["target_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[frontier] {len(targets)} targets, {len(done)} done", flush=True)
    with out.open("a") as fh:
        for t in targets:
            if t.target_id in done:
                continue
            fr = E.frontier(t, confirm_top=args.confirm_top,
                            max_seconds=args.max_seconds)
            fr["seq"] = t.seq
            fh.write(json.dumps(fr) + "\n")
            fh.flush()
            c = fr["ceiling"]
            desc = "none" if c is None else f"{c['retention']:.3f} rel={c['release_count']}"
            print(f"[frontier] {t.target_id:8s} L={t.length:2d} ceiling={desc}",
                  flush=True)
    print("[frontier] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
