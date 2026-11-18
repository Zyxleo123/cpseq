"""Select the failed cyclic parents and declare the held-out split.

Design doc, Stage 1 step 1: "Fix 20-40 failed parents from one unguided
generation run, stratified across the `defect_class` enum with roughly balanced
cells."

And its first "settle before starting" item, settled here:

    Held-out split, declared first. Tuning prompts against the oracle table is
    fitting to it. Split BY TARGET, not by parent, and freeze before touching
    anything. Otherwise Stage 2 underperforms Stage 1 and the cause is
    unrecoverable.

CPSea ships a `cluster_id` per complex -- a sequence/structure cluster over
receptors -- so the split is on that column and no cluster appears on both
sides. Two peptides against the same receptor cannot land in different splits,
which is the leak the doc is warning about: the same receptor's hotspot pattern
would otherwise be visible in dev and scored in test.

Deduplication matters here. CPSea contains near-identical rows (the same
peptide bound to several copies of the same receptor in one asymmetric unit --
1E2T chains A/B/C/D are four copies of one complex). Keeping them would make
the benchmark look four times larger than it is and would make a per-parent
mean a weighted average over whichever complexes happened to crystallise with
more copies. One parent per (cluster_id, sequence).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ..enums import DefectClass
from ..qualification import QUAL_HASH, freeze_to, qualify
from ..types import ParentProblem, ResidueDiag, sha
from . import surrogate
from .diagnostics import measure

CPSEA_DEFAULT = "/zfsauton/scratch/yixiz/CPSea/CPSea_PDB/preprocessed"

# A parent is a NEAR MISS or it is not a parent.
#
# The design doc's iteration 0 is explicit about this: 200 candidates, "Zero
# qualify; 14 near misses", and the one it carries forward (C-017) fails
# exactly one clause -- strain -- while passing ipTM, BSA and ddG. A candidate
# failing four or five objectives at once is not a repair problem, it is a
# failed topology, and the doc discards those.
#
# Measured here, and the reason this constant exists: with no cap, 19 of 32
# selected parents failed 3 or more objectives, and 26 of 32 had NO intervention
# anywhere in a 141-cell action space that produced a contract-holding repair.
# The whole table then sat within 4 utility points of doing nothing, so every
# controller tied and the benchmark measured nothing. Capping at 2 keeps the
# parents that a one- or two-move campaign can actually repair.
MAX_FAILED_OBJECTIVES = 2


def _candidate_rows(cpsea_root: Path, splits: tuple[str, ...], max_scan: int) -> list[dict]:
    """Metadata rows for complexes worth measuring, deduplicated.

    Reads the parquet metadata rather than globbing the PDB directory: the
    metadata carries `cluster_id` (the split key), `peptide_length` and
    `cyclization_type`, and a glob would force a structure parse per file just
    to find out whether the file is usable.
    """
    import pandas as pd

    frames = []
    for sp in splits:
        p = cpsea_root / "metadata" / f"cpsea_{sp}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no CPSea metadata parquet under {cpsea_root}/metadata")
    df = pd.concat(frames, ignore_index=True)

    # 8-16 residues. Below 8 the action space is too small for a budget split
    # to be a decision at all (every position is within one step of the closure
    # bond, so closure and interface cannot be traded against each other);
    # above 16 is outside the dataset's own curation rule.
    df = df[(df.peptide_length >= 8) & (df.peptide_length <= 16)]
    df = df.sort_values("example_id").reset_index(drop=True)
    return df.head(max_scan).to_dict("records")


def build(
    cpsea_root: Path,
    out_dir: Path,
    n_parents: int = 32,
    k_budget: int = 4,
    k_spent0: int = 0,
    splits: tuple[str, ...] = ("test", "val"),
    max_scan: int = 900,
    test_frac: float = 0.5,
    seed: str = "lp2cp-stage2",
) -> dict:
    """Measure, label, stratify, split, freeze. Writes four files.

      parents_public.json   what the agent may see  (no true_defect_class)
      parents_full.json     the same plus labels    (scorer / oracle only)
      qualification.json    the frozen hashed predicate
      env_manifest.json     every hash the campaign checks at startup

    `k_spent0` is how much of the mutation allowance iteration 0 already spent.
    It is 0 here, and that is a consequence of how the linear parent is
    defined rather than a free choice: the LP is the deposited peptide's OWN
    sequence, so the cyclic parent is at edit distance 0 from it and nothing
    has been spent. (The doc's worked example starts at 2 of 4 because its
    parent came out of a generative run that had already substituted twice.)
    Setting it to 2 here would also have put every coordinated move out of
    budget -- a 3-residue rebuild costs 2 and a rebuild-plus-substitution
    costs 3 -- which would have removed from the grid the exact cells the doc
    says must be in it.
    """
    rows = _candidate_rows(cpsea_root, splits, max_scan)
    print(f"[parents] scanning {len(rows)} CPSea rows from {cpsea_root}")

    seen: set[tuple[str, str]] = set()
    measured: list[tuple[dict, dict]] = []
    n_err = 0
    for row in rows:
        path = Path(row["path"])
        if not path.exists():
            continue
        try:
            m = measure(path, binder_chain=str(row.get("binder_chain_id") or "B"))
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 5:
                print(f"[parents] skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        key = (str(row["cluster_id"]), m["sequence"])
        if key in seen:
            continue
        seen.add(key)
        measured.append((row, m))
    print(f"[parents] measured {len(measured)} unique (cluster, sequence) parents "
          f"({n_err} unusable)")

    # --- score iteration 0 and assign the TRUE defect class -----------------
    staged: list[dict] = []
    for row, m in measured:
        pid = str(row["example_id"])
        card0 = surrogate.parent_scorecard(
            residues=m["residues"],
            native_bsa=m["native_bsa"],
            sequence=m["sequence"],
            anchors=m["anchors"],
            parent_id=pid,
            n_seeds=5,
        )
        verdict = qualify(card0, m["native_bsa"], k_budget)
        staged.append(
            {
                "row": row, "m": m, "card0": card0, "verdict": verdict,
                "parent_id": pid,
            }
        )

    # The doc's iteration 0 produces zero qualifying designs and 14 near
    # misses, so a parent that already qualifies is not a parent -- there is
    # nothing to repair and including it would inflate every arm's success rate
    # identically while adding no decision.
    unqualified = [s for s in staged if not s["verdict"].qualifies]
    failed = [
        s for s in unqualified
        if len([f for f in s["verdict"].failed if f != "edit_distance"])
        <= MAX_FAILED_OBJECTIVES
    ]
    print(f"[parents] {len(unqualified)}/{len(staged)} fail qualification at "
          f"iteration 0; {len(failed)} of those are NEAR MISSES "
          f"(<= {MAX_FAILED_OBJECTIVES} failing objectives) and are eligible")

    # --- stratify across the defect enum with roughly balanced cells --------
    by_class: dict[DefectClass, list[dict]] = defaultdict(list)
    for s in failed:
        by_class[s["verdict"].defect_class].append(s)
    for v in by_class.values():
        # Deterministic within a cell: ordered by a hash of the parent id, not
        # by file order, so the pick does not track the PDB alphabet.
        v.sort(key=lambda s: sha((seed, s["parent_id"])))
    print("[parents] defect-class census before balancing: "
          + ", ".join(f"{k.value}={len(v)}" for k, v in sorted(
              by_class.items(), key=lambda kv: -len(kv[1]))))

    chosen: list[dict] = []
    classes = sorted(by_class, key=lambda c: c.value)
    i = 0
    while len(chosen) < n_parents and any(by_class[c] for c in classes):
        c = classes[i % len(classes)]
        if by_class[c]:
            chosen.append(by_class[c].pop(0))
        i += 1

    # --- the held-out split: BY TARGET (cluster), declared here -------------
    clusters = sorted({str(s["row"]["cluster_id"]) for s in chosen})
    clusters.sort(key=lambda c: sha((seed, "split", c)))
    n_test = max(1, int(round(test_frac * len(clusters))))
    test_clusters = set(clusters[:n_test])
    split_of = {c: ("test" if c in test_clusters else "dev") for c in clusters}

    parents: list[ParentProblem] = []
    for s in chosen:
        row, m = s["row"], s["m"]
        cid = str(row["cluster_id"])
        parents.append(
            ParentProblem(
                parent_id=s["parent_id"],
                pdb_id=s["parent_id"].split("_")[0],
                receptor_name=f"{s['parent_id'].split('_')[0]} receptor "
                              f"({m['receptor_n_res']} res)",
                cluster_id=cid,
                split=split_of[cid],
                # The LINEAR parent is the peptide's own sequence: the campaign
                # is "make a cyclic analog of this linear binder", so edit
                # distance is measured against it throughout.
                lp_sequence=m["sequence"],
                cp_sequence=m["sequence"],
                cyclization=str(row.get("cyclization_type") or "head_tail"),
                k_budget=k_budget,
                k_spent=k_spent0,
                anchors=tuple(m["anchors"]),
                residues=tuple(m["residues"]),
                native_bsa=m["native_bsa"],
                true_defect_class=s["verdict"].defect_class,
                scorecard0=s["card0"],
            )
        )
    parents.sort(key=lambda p: p.parent_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    pub = [p.public() for p in parents]
    full = [
        asdict(p) | {"true_defect_class": p.true_defect_class.value,
                     "verdict0": s["verdict"].to_row()}
        for p, s in zip(parents, sorted(chosen, key=lambda s: s["parent_id"]))
    ]
    (out_dir / "parents_public.json").write_text(json.dumps(pub, indent=1, default=str))
    (out_dir / "parents_full.json").write_text(json.dumps(full, indent=1, default=str))
    qual = freeze_to(out_dir / "qualification.json")

    census = defaultdict(int)
    for p in parents:
        census[p.true_defect_class.value] += 1
    manifest = {
        "env_version": "stage2-env-v1",
        "surrogate": surrogate.provenance(),
        "surrogate_config": asdict(surrogate.CFG),
        "qualification_hash": qual["qualification_hash"],
        "cpsea_root": str(cpsea_root),
        "cpsea_splits_scanned": list(splits),
        "n_parents": len(parents),
        "k_budget": k_budget,
        "k_spent_at_iteration0": k_spent0,
        "defect_census": dict(census),
        "clusters": {c: split_of[c] for c in clusters},
        "n_dev_parents": sum(1 for p in parents if p.split == "dev"),
        "n_test_parents": sum(1 for p in parents if p.split == "test"),
        "split_seed": seed,
        "parents_public_sha": sha(pub, 32),
        "WARNING": (
            "Structures, per-residue diagnostics and native BSA are MEASURED "
            "from experimental coordinates. The response of every objective to "
            "an edit is a DECLARED SURROGATE (see envbuild/surrogate.py). No "
            "number here is a biophysical prediction."
        ),
    }
    manifest["env_hash"] = sha(
        {k: v for k, v in manifest.items() if k not in ("WARNING",)}, 16
    )
    (out_dir / "env_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"[parents] wrote {len(parents)} parents to {out_dir}")
    print(f"[parents] defect census: {dict(census)}")
    print(f"[parents] split by TARGET: {manifest['n_dev_parents']} dev / "
          f"{manifest['n_test_parents']} test over {len(clusters)} clusters")
    print(f"[parents] env_hash={manifest['env_hash']} qual_hash={QUAL_HASH}")
    return manifest


def load_parents(path: str | Path, with_labels: bool = False) -> list[ParentProblem]:
    """Rehydrate ParentProblem objects from disk.

    `with_labels=False` (the default, and what the campaign runner uses) leaves
    `true_defect_class` as NO_DEFECT -- not the real label. Any code that wants
    the label has to ask for it explicitly, which keeps a prompt from reaching
    it by accident.
    """
    from ..types import ScoreCard

    raw = json.loads(Path(path).read_text())
    out: list[ParentProblem] = []
    for d in raw:
        card = d["scorecard0"]
        out.append(
            ParentProblem(
                parent_id=d["parent_id"],
                pdb_id=d["pdb_id"],
                receptor_name=d["receptor_name"],
                cluster_id=d["cluster_id"],
                split=d["split"],
                lp_sequence=d["lp_sequence"],
                cp_sequence=d["cp_sequence"],
                cyclization=d["cyclization"],
                k_budget=int(d["k_budget"]),
                k_spent=int(d["k_spent"]),
                anchors=tuple(d["anchors"]),
                residues=tuple(ResidueDiag(**r) for r in d["residues"]),
                native_bsa=float(d["native_bsa"]),
                true_defect_class=(
                    DefectClass(d["true_defect_class"])
                    if with_labels and d.get("true_defect_class")
                    else DefectClass.NO_DEFECT
                ),
                scorecard0=ScoreCard(
                    **{**card,
                       "strain_locus": tuple(card["strain_locus"]),
                       "hbonds_retained": tuple(card["hbonds_retained"])}
                ),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpsea", default=CPSEA_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-parents", type=int, default=32)
    ap.add_argument("--k-budget", type=int, default=4)
    ap.add_argument("--k-spent0", type=int, default=0)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--max-scan", type=int, default=900)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()
    build(
        cpsea_root=Path(args.cpsea),
        out_dir=Path(args.out),
        n_parents=args.n_parents,
        k_budget=args.k_budget,
        k_spent0=args.k_spent0,
        splits=tuple(s for s in args.splits.split(",") if s),
        max_scan=args.max_scan,
        test_frac=args.test_frac,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
