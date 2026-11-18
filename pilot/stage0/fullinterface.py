"""Build the full-interface secondary benchmark.

The primary arms choose from the positions SKEMPI measured. That is clean to
score but it hands the agent the oracle's support: every option it is shown is
one somebody thought worth assaying. This variant shows the agent the WHOLE
interface on the mutated side and lets it pick anywhere.

The cost is that most of the interface was never measured, so many picks cannot
be scored. That is a property of the benchmark, not a failure of the agent, and
it is reported as its own metric:

  coverage        fraction of picks that landed on a measured position
  regret          computed on the measured subset only, and therefore on a
                  BIASED subset -- never quoted as the headline

What this arm actually establishes: if the agent's measured-coverage is far
above the rate a uniform pick over the interface would give, it is finding the
positions worth assaying. And if its regret on the covered subset matches its
regret in the restricted arm, the restriction was not carrying the result.

k=1 only. The full interface has 20-80 positions, so k=2 would be thousands of
pairs -- a different experiment (candidate-set size) wearing this one's name.

Nothing here reveals which positions were measured. Candidate ids are assigned
in (chain, residue number) order over the whole interface, so measured
positions are scattered through the id space, and the presentation order is a
fresh seeded shuffle on top.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# A residue counts as interface if it touches the partner at all. Deliberately
# permissive: a tighter cut would quietly re-impose a curated candidate list,
# which is the very thing this arm exists to remove.
MIN_CONTACTS = 1
SPLIT_SALT = "lp2cp-stage0-fullif-v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="ala_k1")
    args = ap.parse_args()

    bench = Path(args.bench)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    public = json.loads((bench / "decisions_public.json").read_text())
    oracle = {
        o["decision_id"]: o
        for o in json.loads((bench / "decisions_oracle.json").read_text())
    }
    feats = json.loads((Path(args.features) / "features.json").read_text())

    new_public: list[dict] = []
    new_oracle: list[dict] = []
    report_rows = []

    for d in public:
        if d["slice"] != args.slice or d["k"] != 1:
            continue
        complex_feats = feats.get(d["complex_id"])
        if not complex_feats:
            continue
        ora = oracle[d["decision_id"]]

        # Which side(s) were mutated. Only those are opened up: offering the
        # partner's interface would change the design problem, not widen it.
        measured_chains = {
            chain for cand in d["candidates"] for chain, _ in cand["pdb_sites"]
        }
        measured_sites = {
            tuple(cand["pdb_sites"][0]): cand["cand_id"] for cand in d["candidates"]
        }
        value_by_site = {
            tuple(c["sites"][0]): c["value"]
            for c in ora["candidates"]
        }
        # The oracle file keys on CLEANED numbering while features key on PDB
        # numbering, so the bridge is the public file, which carries both.
        pdb_to_clean = {
            tuple(cand["pdb_sites"][0]): tuple(cand["sites"][0])
            for cand in d["candidates"]
        }

        # The pool is the structural interface UNIONED with every position
        # SKEMPI classified as interface. A 5 A heavy-atom contact is stricter
        # than SKEMPI's COR/SUP/RIM call -- rim residues can sit further out --
        # and on 1F47_A_B and 3BT1_A_U the contact cut alone returned fewer
        # positions than SKEMPI had measured. Dropping those decisions would
        # discard the two most heavily scanned interfaces for disagreeing with
        # a threshold, so SKEMPI's own classification is taken as the second
        # source of truth about what counts as interface.
        pool_keys: dict[tuple[str, str], tuple[str, str, str]] = {}
        for key, f in complex_feats.items():
            if f["chain"] not in measured_chains:
                continue
            if (f.get("n_contacts_5A") or 0) < MIN_CONTACTS:
                continue
            pool_keys[(f["chain"], f["resnum"])] = (f["chain"], f["resnum"], f["aa"])
        for site, cand_id in measured_sites.items():
            if site in pool_keys:
                continue
            f = complex_feats.get(f"{site[0]}:{site[1]}")
            if f is None:
                continue
            pool_keys[site] = (site[0], site[1], f["aa"])
        pool = list(pool_keys.values())
        if len(pool) < len(d["candidates"]):
            report_rows.append(
                {"decision_id": d["decision_id"], "status": "pool_smaller_than_measured",
                 "pool": len(pool), "measured": len(d["candidates"])}
            )
            continue

        pool.sort(key=lambda t: (t[0], int("".join(ch for ch in t[1] if ch.isdigit() or ch == "-") or 0), t[1]))
        cands_pub, cands_ora = [], []
        n_measured = 0
        for idx, (chain, resnum, aa) in enumerate(pool):
            cand_id = f"f{idx:03d}"
            site = (chain, resnum)
            cands_pub.append(
                {
                    "cand_id": cand_id,
                    "sites": [list(pdb_to_clean.get(site, site))],
                    "pdb_sites": [[chain, resnum]],
                    "wt": [aa],
                    # SKEMPI's class is only defined for measured positions, so
                    # it is withheld here for ALL of them. Supplying it only
                    # where it exists would mark the measured ones.
                    "locations": [""],
                }
            )
            clean = pdb_to_clean.get(site)
            value = value_by_site.get(clean) if clean else None
            if value is not None:
                n_measured += 1
            cands_ora.append(
                {
                    "cand_id": cand_id,
                    "sites": [list(clean or site)],
                    "value": value,          # None => unmeasured, unscorable
                    "value_mean": value,
                    "n_records": 1 if value is not None else 0,
                    "substitutions": [],
                    "additive_from_singles": None,
                    "epistasis": None,
                    "was_measured": value is not None,
                }
            )

        vals = [c["value"] for c in cands_ora if c["value"] is not None]
        if not vals:
            continue
        did = f"fullif_{args.slice}__{d['complex_id']}"
        rng = random.Random(f"{SPLIT_SALT}|{did}")
        order = [c["cand_id"] for c in cands_pub]
        rng.shuffle(order)

        new_public.append(
            {
                **{k: v for k, v in d.items()
                   if k not in ("candidates", "presentation_order",
                                "presentation_order_alt", "decision_id", "slice",
                                "n_candidates")},
                "decision_id": did,
                "slice": f"fullif_{args.slice}",
                "n_candidates": len(cands_pub),
                "candidates": cands_pub,
                "presentation_order": order,
                "presentation_order_alt": order[::-1],
            }
        )
        new_oracle.append(
            {
                "decision_id": did,
                "slice": f"fullif_{args.slice}",
                "split": d["split"],
                "pdb_id": d["pdb_id"],
                "k": 1,
                "oracle_value": min(vals),
                "worst_value": max(vals),
                "median_value": sorted(vals)[len(vals) // 2],
                "spread": max(vals) - min(vals),
                "candidates": cands_ora,
                # The rate a blind pick would achieve: the yardstick the
                # agent's coverage has to beat for coverage to mean anything.
                "measured_fraction": n_measured / len(cands_pub),
            }
        )
        report_rows.append(
            {
                "decision_id": did,
                "status": "ok",
                "interface_positions": len(cands_pub),
                "measured": n_measured,
                "measured_fraction": round(n_measured / len(cands_pub), 4),
            }
        )

    (out / "decisions_public.json").write_text(
        json.dumps(new_public, indent=1, sort_keys=True)
    )
    (out / "decisions_oracle.json").write_text(
        json.dumps(new_oracle, indent=1, sort_keys=True)
    )

    ok = [r for r in report_rows if r["status"] == "ok"]
    mf = [r["measured_fraction"] for r in ok]
    meta = {
        "bench_version": "stage0-fullif-v1",
        "derived_from": str(bench.resolve()),
        "source_slice": args.slice,
        "min_contacts": MIN_CONTACTS,
        "n_decisions": len(new_public),
        "median_interface_positions": (
            sorted(r["interface_positions"] for r in ok)[len(ok) // 2] if ok else 0
        ),
        "mean_measured_fraction": round(sum(mf) / len(mf), 4) if mf else 0.0,
        "blind_pick_coverage_note": (
            "mean_measured_fraction IS the coverage a uniform pick over the "
            "interface would achieve; an arm's coverage must beat it to mean "
            "anything"
        ),
        "rows": report_rows,
    }
    pub_sha = hashlib.sha256((out / "decisions_public.json").read_bytes()).hexdigest()
    meta["decisions_public_sha256"] = pub_sha
    (out / "bench_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))

    # Leak check: the public file must not say which positions were measured.
    blob = json.loads((out / "decisions_public.json").read_text())
    def _walk(node):
        hits = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("value", "was_measured", "n_records", "measured_fraction"):
                    hits.append(k)
                hits += _walk(v)
        elif isinstance(node, list):
            for v in node:
                hits += _walk(v)
        return hits
    leaks = _walk(blob)
    if leaks:
        print(f"[fullif] FATAL: public file marks measured positions: {set(leaks)}")
        return 1

    print(f"[fullif] {len(new_public)} decisions from slice {args.slice}")
    print(f"[fullif] median interface positions: {meta['median_interface_positions']}")
    print(f"[fullif] blind-pick coverage (bar to beat): "
          f"{meta['mean_measured_fraction']:.3f}")
    print(f"[fullif] leak check OK; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
