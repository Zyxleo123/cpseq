"""Freeze the Stage 0 decision set.

A "decision" is one allocation problem: a complex, an edit budget k, and a list
of candidate position-sets of size k that SKEMPI actually measured. The agent
picks exactly one candidate. The oracle is the candidate with the lowest ddG,
so regret is measured in kcal/mol of binding energy thrown away.

Two files are written, and the split is the point:

  decisions_public.json  -- what any arm is allowed to see. Carries no ddG.
  decisions_oracle.json  -- the values, read only by score.py.

Keeping the labels in a separate file makes leakage a structural impossibility
instead of a thing we remember not to do.

Slices
------
ala_k1     Alanine scan, k=1. Exactly one measured substitution per candidate,
           so the candidate's value cannot be improved by having been measured
           more often. PRIMARY.
allsub_k1  Every substitution, k=1, candidate value = best (lowest) measured
           ddG at that position -- an optimistic solver that always finds the
           best identity. Carries a draw-count confound (a position measured
           ten times gets ten draws at the minimum), which is why the
           `most_measured` control baseline exists. SECONDARY.
allsub_k2  Real measured double mutants, k=2. This is the slice where budget
           SPLITTING is tested, and where a coordinated move can beat picking
           the two best singles independently.
allsub_k3  Same at k=3. Small; reported as a trend, not a test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from skempi import INTERFACE_CLASSES, Record, load_records

BENCH_VERSION = "stage0-v1"

# A decision needs enough candidates that a random pick is clearly worse than a
# good one; below ~8 the regret distribution is dominated by how few options
# there were rather than by the quality of the choice. Multi-position budgets
# are measured far more sparsely, so they get a lower floor -- at 8 the k=2
# slice holds 15 decisions, which cannot support a test.
MIN_CANDIDATES = {1: 8, 2: 5, 3: 5}

# Below this a slice is reported but not tested: a "trend" over five paired
# decisions is an anecdote, and labelling it as a result is how an
# underpowered comparison gets quoted later as evidence.
MIN_DECISIONS_TESTABLE = 20

# Fraction of complexes reserved for prompt engineering. Declared here, before
# any prompt exists, so that tuning on dev cannot quietly become tuning on test.
DEV_FRACTION = 1.0 / 3.0
SPLIT_SALT = "lp2cp-stage0-split-v1"


def _interface_only(records: list[Record]) -> list[Record]:
    return [r for r in records if r.all_at_interface]


def _group_split(group_key: str) -> str:
    """Deterministic dev/test assignment, keyed on the grouping unit.

    Hash-based rather than random.shuffle so the split is reproducible from the
    salt alone and does not depend on dict ordering or on how many complexes
    happened to survive filtering.
    """
    h = hashlib.sha256(f"{SPLIT_SALT}|{group_key}".encode()).digest()
    frac = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "dev" if frac < DEV_FRACTION else "test"


def _candidate_rows(
    records: list[Record], k: int, alanine_only: bool
) -> dict[tuple, list[Record]]:
    """Measured records bucketed by the position-set they mutate."""
    buckets: dict[tuple, list[Record]] = defaultdict(list)
    for r in records:
        if r.k != k:
            continue
        if alanine_only and not all(m.mut == "A" and m.wt != "A" for m in r.mutations):
            continue
        buckets[r.site_set].append(r)
    return buckets


def build_slice(
    records: list[Record], slice_name: str, k: int, alanine_only: bool
) -> tuple[list[dict], list[dict]]:
    by_complex: dict[str, list[Record]] = defaultdict(list)
    for r in _interface_only(records):
        by_complex[r.complex_id].append(r)

    public: list[dict] = []
    oracle: list[dict] = []

    for complex_id in sorted(by_complex):
        recs = by_complex[complex_id]
        buckets = _candidate_rows(recs, k, alanine_only)
        if len(buckets) < MIN_CANDIDATES.get(k, 8):
            continue

        head = recs[0]
        # Singles table for this complex, used for the additive prediction at
        # k>=2. Built from the all-substitution singles regardless of slice,
        # because that is what a greedy controller would have had available.
        singles: dict[tuple[str, str], float] = {}
        for r in recs:
            if r.k == 1:
                site = r.mutations[0].site
                v = r.ddg
                if site not in singles or v < singles[site]:
                    singles[site] = v

        pub_cands: list[dict] = []
        ora_cands: list[dict] = []
        for idx, site_set in enumerate(sorted(buckets)):
            group = buckets[site_set]
            by_site = {m.site: m for r in group for m in r.mutations}
            sites = list(site_set)
            values = [r.ddg for r in group]

            additive = None
            if k > 1 and all(s in singles for s in sites):
                additive = sum(singles[s] for s in sites)

            pub_cands.append(
                {
                    "cand_id": f"c{idx:03d}",
                    "sites": [list(s) for s in sites],
                    "pdb_sites": [list(by_site[s].pdb_site) for s in sites],
                    "wt": [by_site[s].wt for s in sites],
                    "locations": [
                        next(
                            (
                                loc
                                for r in group
                                for m, loc in zip(r.mutations, r.locations)
                                if m.site == s and loc in INTERFACE_CLASSES
                            ),
                            "",
                        )
                        for s in sites
                    ],
                }
            )
            ora_cands.append(
                {
                    "cand_id": f"c{idx:03d}",
                    "sites": [list(s) for s in sites],
                    # Best measured substitution at this position set: the
                    # "trusted solver" model. For ala_k1 there is exactly one.
                    "value": min(values),
                    "value_mean": sum(values) / len(values),
                    "n_records": len(group),
                    "substitutions": sorted(
                        ",".join(str(m) for m in r.mutations) for r in group
                    ),
                    "additive_from_singles": additive,
                    "epistasis": (
                        None if additive is None else min(values) - additive
                    ),
                }
            )

        vals = [c["value"] for c in ora_cands]
        decision_id = f"{slice_name}__{complex_id}"
        split = _group_split(head.pdb_id)

        public.append(
            {
                "decision_id": decision_id,
                "slice": slice_name,
                "split": split,
                "complex_id": complex_id,
                "pdb_id": head.pdb_id,
                "chains1": head.chains1,
                "chains2": head.chains2,
                "protein1": head.protein1,
                "protein2": head.protein2,
                "hold_out_type": head.hold_out_type,
                "k": k,
                "solver_model": "alanine" if alanine_only else "best_measured",
                "n_candidates": len(pub_cands),
                "candidates": pub_cands,
            }
        )
        oracle.append(
            {
                "decision_id": decision_id,
                "slice": slice_name,
                "split": split,
                "complex_id": complex_id,
                "pdb_id": head.pdb_id,
                "k": k,
                "oracle_value": min(vals),
                "worst_value": max(vals),
                "median_value": sorted(vals)[len(vals) // 2],
                "spread": max(vals) - min(vals),
                "candidates": ora_cands,
            }
        )
    return public, oracle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, report = load_records(args.csv)
    print(f"[build] parsed {report['records']} records "
          f"over {report['complexes']} complexes")
    for key, val in report.items():
        print(f"[build]   {key}: {val}")

    specs = [
        ("ala_k1", 1, True),
        ("allsub_k1", 1, False),
        ("allsub_k2", 2, False),
        ("allsub_k3", 3, False),
    ]
    all_public: list[dict] = []
    all_oracle: list[dict] = []
    slice_report: dict[str, dict] = {}
    for name, k, ala in specs:
        pub, ora = build_slice(records, name, k, ala)
        all_public.extend(pub)
        all_oracle.extend(ora)
        n_dev = sum(1 for d in pub if d["split"] == "dev")
        slice_report[name] = {
            "n_decisions": len(pub),
            "n_dev": n_dev,
            "n_test": len(pub) - n_dev,
            "median_candidates": (
                sorted(d["n_candidates"] for d in pub)[len(pub) // 2] if pub else 0
            ),
            "median_spread": (
                round(sorted(o["spread"] for o in ora)[len(ora) // 2], 3) if ora else 0
            ),
            "testable": len(pub) >= MIN_DECISIONS_TESTABLE,
        }
        print(f"[build] slice {name:11s} n={len(pub):3d} "
              f"(dev {n_dev}, test {len(pub) - n_dev})  "
              f"median_cands={slice_report[name]['median_candidates']}  "
              f"median_spread={slice_report[name]['median_spread']}"
              f"{'' if slice_report[name]['testable'] else '   [NOT TESTABLE]'}")

    # Presentation order: one fixed shuffle per decision, written into the
    # frozen benchmark so every arm sees candidates in the SAME order, and a
    # second independent shuffle so a position-bias control can re-ask the same
    # decision with the list permuted. Without this, "the model picks the first
    # option" is indistinguishable from "the model picks the best option"
    # whenever the candidates arrive in a meaningful order.
    for d in all_public:
        ids = [c["cand_id"] for c in d["candidates"]]
        rng = random.Random(f"{SPLIT_SALT}|order|{d['decision_id']}")
        order = ids[:]
        rng.shuffle(order)
        d["presentation_order"] = order
        rng2 = random.Random(f"{SPLIT_SALT}|order2|{d['decision_id']}")
        order2 = ids[:]
        rng2.shuffle(order2)
        d["presentation_order_alt"] = order2

    meta = {
        "bench_version": BENCH_VERSION,
        "source_csv": str(Path(args.csv).resolve()),
        "source_csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        "min_candidates": MIN_CANDIDATES,
        "min_decisions_testable": MIN_DECISIONS_TESTABLE,
        "dev_fraction": DEV_FRACTION,
        "split_salt": SPLIT_SALT,
        "split_unit": "pdb_id",
        "parse_report": report,
        "slices": slice_report,
        "sign_convention": "ddG = RT ln(Kd_mut/Kd_wt); lower is better; oracle minimises",
    }

    pub_path = out / "decisions_public.json"
    ora_path = out / "decisions_oracle.json"
    pub_path.write_text(json.dumps(all_public, indent=1, sort_keys=True))
    ora_path.write_text(json.dumps(all_oracle, indent=1, sort_keys=True))
    meta["decisions_public_sha256"] = hashlib.sha256(pub_path.read_bytes()).hexdigest()
    meta["decisions_oracle_sha256"] = hashlib.sha256(ora_path.read_bytes()).hexdigest()
    (out / "bench_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))

    # A public file that still contains a ddG would invalidate every arm, so it
    # is checked rather than assumed.
    # A public file that still carries a label would invalidate every arm, so
    # the keys are checked rather than trusted. Matching on JSON keys and not
    # on substrings: "solver_model" legitimately contains neither a value nor
    # the word, but a substring search for "value" flags it.
    banned_keys = (
        "value", "value_mean", "ddg", "oracle_value", "worst_value",
        "median_value", "spread", "epistasis", "n_records", "substitutions",
        "additive_from_singles",
    )
    def _walk(node, path=""):
        hits = []
        if isinstance(node, dict):
            for key, child in node.items():
                if key in banned_keys:
                    hits.append(f"{path}.{key}")
                hits += _walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for i, child in enumerate(node):
                hits += _walk(child, f"{path}[{i}]")
        return hits
    leaks = _walk(json.loads(pub_path.read_text()))
    if leaks:
        print(f"[build] FATAL: public decisions file carries labels: {leaks[:5]}")
        return 1
    print(f"[build] leak check OK; wrote {pub_path} and {ora_path}")
    print(f"[build] public sha256 {meta['decisions_public_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
