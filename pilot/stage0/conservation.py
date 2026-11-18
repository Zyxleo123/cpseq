"""ESM-2 sequence-tolerance diagnostics for the candidate positions.

The doc's kill criterion names a "hotspot/conservation heuristic" as the bar to
beat, so a conservation signal has to exist as a baseline and as evidence the
agent can read. No MSA pipeline is involved: ESM-2's masked distribution stands
in for one, which is the standard substitute and is what the rest of this
project already has cached.

Masked marginals, not wild-type marginals, and only at candidate positions.
Masking is the more faithful estimate of "what else could sit here" -- the
wild-type-marginal shortcut leaves the real residue visible in the input, so
the model mostly reports its own copy of it. Masking every residue in every
chain would be tens of thousands of forward passes; masking only the ~16
candidate positions per decision is ~1500 passes in total.

Reported per position:
  entropy      Shannon entropy over the 20 amino acids. HIGH = tolerant, which
               for a damage-minimisation task is where an edit should go.
  wt_prob      model's probability for the residue actually there.
  wt_rank      its rank among the 20. 1 = the model agrees strongly.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser, PDBParser  # noqa: E402

try:
    from Bio.PDB.Polypeptide import protein_letters_3to1 as _T31
except ImportError:  # pragma: no cover
    from Bio.Data.IUPACData import protein_letters_3to1_extended as _T31

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022  # ESM-2 positional limit, minus BOS/EOS


def chain_sequences(model) -> dict[str, list[tuple[str, str]]]:
    """chain id -> ordered [(res_key, one_letter)] over standard residues."""
    out: dict[str, list[tuple[str, str]]] = {}
    for chain in model:
        seq: list[tuple[str, str]] = []
        for res in chain:
            het, num, icode = res.get_id()
            if het.strip():
                continue
            one = _T31.get(res.get_resname().strip(), "X")
            if one not in AA20:
                continue
            seq.append((f"{chain.id}:{num}{icode.strip()}", one))
        if seq:
            out[chain.id] = seq
    return out


def _window(seq: list[tuple[str, str]], wanted: set[str]) -> list[tuple[str, str]]:
    """Crop to <= MAX_LEN tokens, keeping every wanted position if possible."""
    if len(seq) <= MAX_LEN:
        return seq
    idx = [i for i, (key, _) in enumerate(seq) if key in wanted]
    if not idx:
        return seq[:MAX_LEN]
    lo, hi = min(idx), max(idx)
    if hi - lo + 1 > MAX_LEN:  # cannot cover all; centre on the span midpoint
        mid = (lo + hi) // 2
        lo, hi = mid - MAX_LEN // 2, mid + MAX_LEN // 2
    pad = (MAX_LEN - (hi - lo + 1)) // 2
    lo = max(0, lo - pad)
    return seq[lo : lo + MAX_LEN]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pdb-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "conservation.json"
    store: dict[str, dict] = json.loads(dest.read_text()) if dest.exists() else {}

    decisions = json.loads(
        (Path(args.bench) / "decisions_public.json").read_text()
    )
    # Which positions matter, per complex, in PDB numbering.
    wanted: dict[str, dict] = {}
    for d in decisions:
        slot = wanted.setdefault(
            d["complex_id"],
            {"pdb_id": d["pdb_id"], "chains": set(d["chains1"] + d["chains2"]),
             "sites": set()},
        )
        for cand in d["candidates"]:
            for chain, pos in cand["pdb_sites"]:
                slot["sites"].add(f"{chain}:{pos}")
    if args.limit:
        wanted = {k: wanted[k] for k in sorted(wanted)[: args.limit]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cons] device={device} model={MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    net = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device).eval()
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA20], device=device)

    todo = [c for c in sorted(wanted) if c not in store]
    print(f"[cons] {len(todo)} complexes to do ({len(store)} already cached)")
    failures: dict[str, str] = {}

    for n, complex_id in enumerate(todo, 1):
        spec = wanted[complex_id]
        pdb_id = spec["pdb_id"].lower()
        path = next(
            (
                p
                for p in (
                    Path(args.pdb_cache) / f"{pdb_id}.pdb",
                    Path(args.pdb_cache) / f"{pdb_id}.cif",
                )
                if p.exists()
            ),
            None,
        )
        if path is None:
            failures[complex_id] = "no_structure"
            print(f"[cons] {n:3d}/{len(todo)} {complex_id:14s} NO STRUCTURE")
            continue
        try:
            parser = (
                MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
            )
            model = parser.get_structure(pdb_id, str(path))[0]
            seqs = chain_sequences(model)
        except Exception as exc:  # noqa: BLE001
            failures[complex_id] = f"{type(exc).__name__}: {exc}"
            print(f"[cons] {n:3d}/{len(todo)} {complex_id:14s} PARSE FAILED {exc}")
            continue

        result: dict[str, dict] = {}
        for chain_id, seq in seqs.items():
            if chain_id not in spec["chains"]:
                continue
            seq = _window(seq, spec["sites"])
            keys = [k for k, _ in seq]
            letters = [a for _, a in seq]
            targets = [i for i, k in enumerate(keys) if k in spec["sites"]]
            if not targets:
                continue
            text = "".join(letters)
            base = tok(text, return_tensors="pt")
            ids = base["input_ids"].to(device)
            attn = base["attention_mask"].to(device)
            # token index = sequence index + 1 (BOS)
            for start in range(0, len(targets), args.batch):
                block = targets[start : start + args.batch]
                batch_ids = ids.repeat(len(block), 1).clone()
                for row, pos in enumerate(block):
                    batch_ids[row, pos + 1] = tok.mask_token_id
                with torch.no_grad():
                    logits = net(
                        input_ids=batch_ids, attention_mask=attn.repeat(len(block), 1)
                    ).logits
                for row, pos in enumerate(block):
                    vec = logits[row, pos + 1, aa_ids].float()
                    prob = torch.softmax(vec, dim=-1)
                    p = prob.cpu().numpy()
                    wt = letters[pos]
                    wt_i = AA20.index(wt) if wt in AA20 else None
                    ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
                    entry = {
                        "entropy": round(ent, 4),
                        "entropy_frac_of_max": round(ent / math.log(20), 4),
                        "wt_prob": round(float(p[wt_i]), 5) if wt_i is not None else None,
                        "wt_rank": (
                            int((p > p[wt_i]).sum()) + 1 if wt_i is not None else None
                        ),
                        "top_alt": AA20[int(np.argmax(p))],
                    }
                    result[keys[pos]] = entry
        store[complex_id] = result
        print(f"[cons] {n:3d}/{len(todo)} {complex_id:14s} {len(result):3d} positions")
        if n % 10 == 0:
            dest.write_text(json.dumps(store, sort_keys=True))
    dest.write_text(json.dumps(store, sort_keys=True))

    covered = sum(
        1
        for d in decisions
        if all(
            f"{c}:{p}" in store.get(d["complex_id"], {})
            for cand in d["candidates"]
            for c, p in cand["pdb_sites"]
        )
    )
    report = {
        "model": MODEL_ID,
        "method": "masked marginals at candidate positions",
        "n_complexes": len(store),
        "failures": failures,
        "decisions_fully_covered": covered,
        "decisions_total": len(decisions),
    }
    (out / "conservation_report.json").write_text(json.dumps(report, indent=1))
    print(f"[cons] decisions fully covered {covered}/{len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
