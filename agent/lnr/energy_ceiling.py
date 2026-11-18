"""The realistic binding-energy ceiling a cyclization can look up to.

recovered(a,b) = energy_weighted_retention(WT alanine scan, [a,b]) * native_dG
is the interface energy a closed ring KEEPS when window [a,b] is held. The
`placement_search` closure controller commits the max-CONTACT-retention window;
contacts are only a ~0.37-rho proxy for binding energy, so that window is not
the one that keeps the most dG. This enumerates every window, keeps the
analytically-feasible ones, torsion-confirms + clash-checks the top ones by
ENERGY retention, and reports the max recovered over windows that close CLEAN:

    energy ceiling   -- best recovered a (no-mutation) cyclization can reach by
                        choosing the window for ENERGY, not contacts.
    native_dG        -- the retention=1 upper bound; UNREACHABLE (holding every
                        residue cannot close). The gap ceiling->native is the
                        FUNDAMENTAL cyclization loss: hotspot energy that no
                        closing window can hold.

Still the retention model (energy carried by held native residues), not a
re-fold/re-dock -- the Boltz2/Rosetta re-scorer gives the physical version. This
is the cheap, exact ceiling for the question "how much dG could we keep if we
chose the window perfectly," which the nomut/placement baseline should be read
against, and which the mutation lever must beat to be worth anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import env as E        # noqa: E402
from lnr import seqmut as S     # noqa: E402


def wt_scans(path: str) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if "per_residue_ddG_ala" in r:
                pr = {int(k): (v or 0.0) for k, v in r["per_residue_ddG_ala"].items()}
                out[r["target_id"]] = (pr, r.get("native_dG"))
    return out


def energy_ceiling(t, per_ala: dict, native_dG, confirm_top: int = 40,
                   n_restarts: int = 8, max_seconds: float = 12.0):
    """Max recovered over clash-free closing windows, ranked by energy retention.

    Walks analytically-feasible windows in descending ENERGY retention and
    torsion-confirms + backbone-clash-checks until one closes clean -- that
    first clean closer is the energy ceiling (the analytic bound guarantees
    nothing above it in the list truly closes any better). Also returns the
    best CONTACT-retention clean window for contrast, so the gap between the
    energy ceiling and what a contact-ranked rule would pick is visible.
    """
    L = t.length
    ewin = []     # (energy_retention, a, b)
    for a in range(L):
        for b in range(a, L):
            if E.analytic_feasible(t, a, b):
                ewin.append((S.energy_weighted_retention(per_ala, a, b), a, b))
    ewin.sort(key=lambda x: -x[0])
    best = None
    tried = 0
    for er, a, b in ewin:
        if tried >= confirm_top:
            break
        tried += 1
        # MATCH the committed-window gate (lnr.seqmut.committed_clean_window uses
        # E.clash_report = CA 3.0 A). Using the stricter backbone gate here made
        # the ceiling a smaller feasible set than placement, so it could come out
        # BELOW placement (1ssc) and miss 1jrr -- not a valid upper bound. With
        # the same gate, placement's own window is always a candidate, so
        # ceiling >= placement by construction.
        rep = E.clash_report(t, a, b, n_restarts=n_restarts,
                             max_seconds=max_seconds)
        if rep.get("closed") and rep.get("clash_free"):
            best = {"a": a, "b": b, "energy_retention": round(er, 4),
                    "recovered": round(er * native_dG, 2) if native_dG else None,
                    "release": L - (b - a + 1),
                    "contact_retention": round(E.retention_of(t, a, b), 4)}
            break
    return best, tried, len(ewin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--scans", required=True, help="rosetta_interface.jsonl (WT scans)")
    ap.add_argument("--targets", default="", help="comma list; empty = all scanned")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-top", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scans = wt_scans(args.scans)
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}
    want = [x.strip() for x in args.targets.split(",") if x.strip()] or list(scans)
    if args.limit:
        want = want[: args.limit]

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
    print(f"[ceil] {len(want)} targets, {len(done)} done", flush=True)

    with out.open("a") as fh:
        for tid in want:
            if tid in done or tid not in targets or tid not in scans:
                continue
            t = targets[tid]
            per_ala, nat = scans[tid]
            best, tried, nfeas = energy_ceiling(
                t, per_ala, nat, args.confirm_top, max_seconds=args.max_seconds)
            row = {"target_id": tid, "native_dG": nat, "length": t.length,
                   "n_feasible": nfeas, "confirmed": tried, "energy_ceiling": best}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if best and nat:
                print(f"[ceil] {tid:8s} native={nat:8.1f}  ceiling_recovered="
                      f"{best['recovered']:8.1f} ({100*best['recovered']/nat:3.0f}% of "
                      f"native)  win=[{best['a']},{best['b']}] eRet={best['energy_retention']} "
                      f"cRet={best['contact_retention']} rel={best['release']}", flush=True)
            else:
                print(f"[ceil] {tid:8s} native={nat}  NO clash-free closer in top "
                      f"{tried}", flush=True)
    print("[ceil] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
