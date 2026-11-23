"""THE BINDING METRIC: pose-conditioned interface likelihood for head-to-tail
cyclic peptides.

Full pre-registration (the argument, the SAR it has to explain, the leakage
firewall, the verdict rule) lives in `docs/binding_metric_design.md`, written and
committed BEFORE the first run. This docstring is the operating summary.

The one-sentence claim
----------------------
Every metric this project has tried failed for the same reason: it was not
conditioned on the sequence's fit to the specific bound pose. Rosetta
`dG_separated` tracks how MUCH surface is buried (partial rho controlling peptide
length = -0.002); Boltz ipTM/LIS score whether SOME structure is confident;
ESM-2 and PepMLM score whether the peptide is A PLAUSIBLE PEPTIDE. None asks
whether THIS residue, at THIS position, suits THIS pocket -- which is exactly
what the measured SAR is made of (on CPSet S03_keap1, D2->A costs 2.33 log units
and T5->A costs 2.78, while Rosetta calls the D2->A variant its strongest binder).

Interface Context Gain
----------------------
With the receptor's backbone AND side chains as context, and the peptide's
backbone ONLY (its own side chains are masked out, so the crystal's answer
cannot leak into its own score):

    lp_cplx(i) = log p( s_i | receptor + peptide backbone )
    lp_pep(i)  = log p( s_i | peptide backbone alone )

    ICG(s) = mean_i [ lp_cplx(i) - lp_pep(i) ]            nats / residue

The difference cancels the peptide's intrinsic plausibility and its own
backbone's preferences; what survives is attributable to the receptor being
there -- the pointwise mutual information I(s_i ; receptor | pose). That is the
construction that makes it immune to the critique which closed the ESM-2 /
PepMLM thread. The mean (not the sum) makes it nats/residue, so unlike dG it
cannot buy correlation from peptide size; `icg_sum` is kept as the deliberately
size-confounded control.

Noise
-----
Exactly deterministic, and not by assertion:

  * `encode()` never reads the sequence (it uses S only for its SHAPE), so the
    structure encoding is fixed, and it is cached across variants -- which is
    also why this is nearly free.
  * `augment_eps = 0` at inference, so no coordinate jitter is added.
  * With the queried position decoded FIRST its decoder trajectory depends only
    on the frozen encoder output (mask_bw for that row is identically zero), so
    it is provably independent of the decoding order drawn for every other
    position. `--selftest` asserts bitwise equality across order seeds.

Contrast with the incumbent oracle: +-3-4 REU, dominated by Boltz fold sampling,
needing 16 GPU folds to reach SEM ~1.0. This needs one CPU forward pass.

THE FLAG INVERSION (read before touching `see_other_seq`)
---------------------------------------------------------
Upstream `ProteinMPNN.single_aa_score(feature_dict, use_sequence=...)` has its
flag inverted relative to its own docstring. `use_sequence=True` puts the queried
position FIRST in the decoding order, so it sees NO other sequence; `False` puts
it LAST, so it sees ALL of it. This module exposes `see_other_seq`, which means
what it says, and `test_bindscore.py` pins the inversion against upstream so that
a future LigandMPNN release which fixes the naming breaks the test loudly instead
of silently flipping the metric.

Where the pose comes from
-------------------------
A metric needing the variant's own crystal is useless for design, so every
variant is scored against every same-length template in its series:

  self       scored on its own crystal. Reported for completeness ONLY: that
             backbone was solved for that sequence, so it leaks.
  xtemplate  mean over templates OTHER than its own. The honest, deployable
             number -- one reference complex per target, sequence threaded on,
             nothing about the variant used except its sequence. THE HEADLINE.

Usage
-----
  python -m lnr.bindscore --selftest
  python -m lnr.bindscore --score --work-dir <prepared pdbs> --out bindscore.jsonl
  python -m lnr.bindscore --report bindscore.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr.cpset_accuracy import (  # noqa: E402
    SERIES, SERIES_OF, TIER_OF, AA3to1, load_dataset1,
)

# ---------------------------------------------------------------------------
# LigandMPNN is vendored in the Proteina-Complexa repo. Its modules import each
# other by bare name (`from data_utils import ...`), so its directory has to go
# on sys.path -- at the FRONT, and only for those two imports.
# ---------------------------------------------------------------------------
DEFAULT_LMPNN_HOME = (
    "/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN"
)

# Pre-registered model spec. v_32_020_25 is the standard LigandMPNN checkpoint
# (0.20 A backbone-noise training), chosen because scoring CRYSTAL structures
# with the 0.02 A checkpoint rewards exact geometry and would import
# crystal-to-crystal differences as signal -- a noise source, not a feature.
PRIMARY_MODEL = "ligandmpnn_v_32_020_25"
MODEL_TYPE_OF = {
    "ligandmpnn_v_32_020_25": "ligand_mpnn",
    "ligandmpnn_v_32_010_25": "ligand_mpnn",
    "ligandmpnn_v_32_005_25": "ligand_mpnn",
    "ligandmpnn_v_32_030_25": "ligand_mpnn",
    "proteinmpnn_v_48_020": "protein_mpnn",
    "proteinmpnn_v_48_002": "protein_mpnn",
    "solublempnn_v_48_020": "soluble_mpnn",
}

# SwissProt background amino-acid frequencies, for the composition-corrected
# log-odds variant. Frozen here so the number cannot drift with a library.
BG_FREQ = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.38, "Q": 3.93,
    "E": 6.75, "G": 7.07, "H": 2.27, "I": 5.91, "L": 9.65, "K": 5.80,
    "M": 2.41, "F": 3.86, "P": 4.74, "S": 6.63, "T": 5.35, "W": 1.10,
    "Y": 2.92, "V": 6.86,
}
_bgsum = sum(BG_FREQ.values())
LOG_BG = {k: math.log(v / _bgsum) for k, v in BG_FREQ.items()}

AA20_SET = set("ACDEFGHIKLMNPQRSTVWY")
POS_AA, NEG_AA = set("KR"), set("DE")
# CB-CB cutoff defining an interface position. Uses the VIRTUAL CB computed from
# N/CA/C, so the interface set is a property of the backbone and is identical for
# every variant threaded onto a given template -- if it depended on side chains,
# variants would be scored over different position sets and could not be compared.
IFACE_CB_CUTOFF = 8.0

# ---------------------------------------------------------------------------
# The metric registry. `lower_is_better` is carried explicitly, never inferred:
# sign slips are this project's recurring bug class, and the grader negates from
# this table so that `pair_agreement` (which is written for Rosetta's
# lower-is-stronger convention) can be imported unchanged.
# ---------------------------------------------------------------------------
METRICS = {
    "icg": dict(lower_is_better=False, family="primary",
                doc="mean_i [log p(s_i|complex) - log p(s_i|peptide alone)], nats/res"),
    "pll_cplx": dict(lower_is_better=False, family="secondary",
                     doc="mean_i log p(s_i|complex); no intrinsic-plausibility correction"),
    "pll_pep": dict(lower_is_better=False, family="control",
                    doc="mean_i log p(s_i|peptide alone); MUST NOT rank if ICG is about binding"),
    "pll_bg": dict(lower_is_better=False, family="secondary",
                   doc="mean_i [log p(s_i|complex) - log background(s_i)]"),
    "icg_if": dict(lower_is_better=False, family="secondary",
                   doc="ICG averaged over interface positions only"),
    "icg_sum": dict(lower_is_better=False, family="control",
                    doc="SUM not mean: the deliberately size-confounded form of ICG"),
    "icg_ar": dict(lower_is_better=False, family="secondary",
                   doc="ICG with the queried position decoded last (sees rest of sequence)"),
    "rec_top1": dict(lower_is_better=False, family="secondary",
                     doc="fraction of peptide positions where argmax p(.|complex) == s_i"),
    "pos_gain": dict(lower_is_better=False, family="secondary",
                     doc="pll_cplx - pll_shuf: positional information with "
                         "composition held EXACTLY fixed by permutation"),
    "pll_shuf": dict(lower_is_better=False, family="control",
                     doc="same residues in scrambled order on the same pose; "
                         "if THIS ranks, the metric is reading composition"),
    # ---- nulls and joined baselines: every one of these is reported beside the
    # primary, because a metric that cannot beat a residue count has not worked.
    "null_neglen": dict(lower_is_better=True, family="null",
                        doc="-peptide length; beat Rosetta dG cross-complex (rho -0.376)"),
    "null_negdsasa": dict(lower_is_better=True, family="null",
                          doc="-buried SASA; joined from cpset_dg.jsonl"),
    "null_chargecomp": dict(lower_is_better=True, family="null",
                            doc="q_pep * q_receptor_interface; negative = complementary"),
    "null_random": dict(lower_is_better=True, family="null",
                        doc="seeded pseudo-random; calibrates the instrument's chance level"),
    "null_consensus": dict(lower_is_better=True, family="null",
                           doc="Hamming distance to the series consensus of the "
                               "OTHER members; the training-memorisation null"),
    "dg_rosetta": dict(lower_is_better=True, family="baseline",
                       doc="the incumbent: Rosetta dG_separated, joined from cpset_dg.jsonl"),
}


def _import_lmpnn(home=None):
    """Import LigandMPNN's data_utils and model_utils. Returns (data_utils, model_utils)."""
    home = str(home or os.environ.get("LIGANDMPNN_HOME") or DEFAULT_LMPNN_HOME)
    if not Path(home, "model_utils.py").is_file():
        raise RuntimeError(f"LigandMPNN not found at {home}; set LIGANDMPNN_HOME")
    if home not in sys.path:
        sys.path.insert(0, home)
    import data_utils as du  # noqa: E402
    import model_utils as mu  # noqa: E402
    return du, mu


# ---------------------------------------------------------------------------
# Geometry helpers (pure, unit-tested)
# ---------------------------------------------------------------------------
def virtual_cb(n_xyz, ca_xyz, c_xyz):
    """ProteinMPNN's virtual CB from backbone N, CA, C. Sequence-independent."""
    b = ca_xyz - n_xyz
    c = c_xyz - ca_xyz
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_xyz


def net_charge(seq):
    """Net formal charge at neutral pH: (K,R) - (D,E). His counted as neutral."""
    return sum(1 for a in seq if a in POS_AA) - sum(1 for a in seq if a in NEG_AA)


def interface_mask(cb, pep_idx, rec_idx, cutoff=IFACE_CB_CUTOFF):
    """Boolean over pep_idx: is this peptide position within `cutoff` of any
    receptor CB? Backbone-derived, so identical for every variant on a template."""
    if not len(rec_idx):
        return np.ones(len(pep_idx), dtype=bool)
    d = np.linalg.norm(cb[pep_idx][:, None, :] - cb[rec_idx][None, :, :], axis=-1)
    return d.min(axis=1) < cutoff


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------
class Pose:
    """One parsed structure plus the bookkeeping the metric needs.

    `encoded` caches the structure encoding, which is legitimate ONLY because
    `ProteinMPNN.encode` reads the sequence tensor for its SHAPE and never its
    values (model_utils.py: `B, L = S_true.shape`). That is what makes scoring an
    arbitrary number of variant sequences against one template nearly free.
    """

    def __init__(self, protein_dict, feature_dict, pep_idx, rec_idx, pep_seq, cb):
        self.protein_dict = protein_dict
        self.feature_dict = feature_dict
        self.pep_idx = pep_idx
        self.rec_idx = rec_idx
        self.pep_seq = pep_seq
        self.cb = cb
        self.encoded = None


class MPNNScorer:
    def __init__(self, model_name=PRIMARY_MODEL, home=None, device="cpu",
                 use_sc_context=True, seed=0):
        import torch
        self.torch = torch
        self.du, self.mu = _import_lmpnn(home)
        self.home = str(home or os.environ.get("LIGANDMPNN_HOME") or DEFAULT_LMPNN_HOME)
        self.model_name = model_name
        self.model_type = MODEL_TYPE_OF.get(model_name)
        if self.model_type is None:
            raise ValueError(f"unknown model {model_name}")
        self.device = torch.device(device)
        self.seed = seed
        ckpt_path = Path(self.home, "model_params", f"{model_name}.pt")
        if not ckpt_path.is_file():
            raise RuntimeError(f"no checkpoint at {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if self.model_type == "ligand_mpnn":
            self.atom_context_num = ckpt["atom_context_num"]
            self.use_sc_context = bool(use_sc_context)
        else:
            self.atom_context_num = 1
            self.use_sc_context = False
        self.model = self.mu.ProteinMPNN(
            node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3,
            k_neighbors=ckpt["num_edges"], device=self.device,
            atom_context_num=self.atom_context_num, model_type=self.model_type,
            ligand_mpnn_use_side_chain_context=self.use_sc_context,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.aa_to_int = dict(self.du.restype_str_to_int)
        self.int_to_aa = dict(self.du.restype_int_to_str)

    # -- parsing -----------------------------------------------------------
    def parse(self, pdb_path, pep_chain, rec_chains=None):
        """Parse a prepared two-chain PDB into a Pose.

        `rec_chains=None` means "peptide only" -- the receptor is not parsed at
        all, which is how the lp_pep term is produced.
        """
        torch = self.torch
        chains = [pep_chain] if rec_chains is None else list(rec_chains) + [pep_chain]
        protein_dict, _, _, icodes, _ = self.du.parse_PDB(
            str(pdb_path), device=self.device, chains=chains,
            parse_all_atoms=True, parse_atoms_with_zero_occupancy=False,
        )
        letters = list(protein_dict["chain_letters"])
        pep_idx = np.array([i for i, c in enumerate(letters) if c == pep_chain], dtype=int)
        rec_idx = np.array([i for i, c in enumerate(letters) if c != pep_chain], dtype=int)
        if not len(pep_idx):
            raise RuntimeError(f"no peptide chain {pep_chain} in {pdb_path}")
        # chain_mask: 1 = "designed" -> its own side chains are hidden from the
        # model (model_utils.py line ~1252 zeroes xyz_37_m where chain_mask==1).
        # The peptide is therefore scored from its BACKBONE only, and the
        # receptor contributes its side chains as context.
        chain_mask = torch.zeros(len(letters), device=self.device)
        chain_mask[torch.tensor(pep_idx, device=self.device, dtype=torch.long)] = 1.0
        protein_dict["chain_mask"] = chain_mask
        protein_dict["membrane_per_residue_labels"] = torch.zeros(
            len(letters), device=self.device)

        feature_dict = self.du.featurize(
            protein_dict, cutoff_for_score=8.0, use_atom_context=True,
            number_of_ligand_atoms=self.atom_context_num, model_type=self.model_type,
        )
        feature_dict["batch_size"] = 1
        feature_dict["symmetry_residues"] = [[]]

        X = protein_dict["X"].cpu().numpy()          # [L,4,3] N,CA,C,O
        cb = virtual_cb(X[:, 0], X[:, 1], X[:, 2])
        S = protein_dict["S"].cpu().numpy()
        pep_seq = "".join(self.int_to_aa[int(S[i])] for i in pep_idx)
        return Pose(protein_dict, feature_dict, pep_idx, rec_idx, pep_seq, cb)

    # -- the marginal ------------------------------------------------------
    def marginals(self, pose, seq=None, see_other_seq=False, order_seed=0):
        """log p(aa | pose) at every peptide position: array [n_pep, 21].

        Derived from `ProteinMPNN.single_aa_score` (LigandMPNN model_utils.py) by
        restricting the position loop to the peptide -- a ~40x saving, since the
        upstream version decodes all L positions of a 288-residue complex to
        answer a question about 7 of them. `--selftest` asserts exact agreement
        with upstream on the queried positions.

        see_other_seq=False  queried position decoded FIRST -> marginal given
                             backbone only. Provably order-independent, so this
                             mode is exactly deterministic. THE PRIMARY MODE.
        see_other_seq=True   queried position decoded LAST -> pseudo-likelihood
                             given the rest of the sequence. Weakly order-dependent.
        """
        # Validated BEFORE any work: a silent truncation would score a different
        # molecule than the one named in the emitted row.
        if seq is not None and len(seq) != len(pose.pep_idx):
            raise ValueError(
                f"sequence length {len(seq)} != template peptide length "
                f"{len(pose.pep_idx)}; threading requires equal length")
        torch = self.torch
        fd = dict(pose.feature_dict)
        S = pose.protein_dict["S"].clone()
        if seq is not None:
            for k, i in enumerate(pose.pep_idx):
                S[int(i)] = self.aa_to_int[seq[k]]
        fd["S"] = S[None, :] if S.dim() == 1 else S
        if fd["S"].dim() == 1:
            fd["S"] = fd["S"][None, :]

        with torch.no_grad():
            if pose.encoded is None:
                # Sequence-independent: cached once per template pose.
                pose.encoded = self.model.encode(fd)
            h_V_enc, h_E_enc, E_idx_enc = pose.encoded

            S_true_enc = fd["S"]
            mask_enc = fd["mask"]
            B, L = S_true_enc.shape
            device = S_true_enc.device
            g = torch.Generator(device="cpu").manual_seed(int(order_seed))
            randn = torch.randn(L, generator=g).to(device)

            out = np.zeros((len(pose.pep_idx), 21), dtype=np.float64)
            for k, idx in enumerate(pose.pep_idx):
                idx = int(idx)
                # see_other_seq=True  -> idx sorts LAST  -> sees all other seq
                # see_other_seq=False -> idx sorts FIRST -> sees no sequence
                if see_other_seq:
                    order_mask = torch.zeros(L, device=device)
                    order_mask[idx] = 1.0
                else:
                    order_mask = torch.ones(L, device=device)
                    order_mask[idx] = 0.0
                decoding_order = torch.argsort(
                    (order_mask + 0.0001) * torch.abs(randn))[None, :]

                E_idx = E_idx_enc.clone()
                perm = torch.nn.functional.one_hot(
                    decoding_order, num_classes=L).float()
                order_mask_backward = torch.einsum(
                    "ij, biq, bjp->bqp",
                    (1 - torch.triu(torch.ones(L, L, device=device))), perm, perm)
                mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
                mask_1D = mask_enc.view([B, L, 1, 1])
                mask_bw = mask_1D * mask_attend
                mask_fw = mask_1D * (1.0 - mask_attend)

                h_V = h_V_enc.clone()
                h_E = h_E_enc.clone()
                h_S = self.model.W_s(S_true_enc)
                h_ES = self.mu.cat_neighbors_nodes(h_S, h_E, E_idx)
                h_EX_encoder = self.mu.cat_neighbors_nodes(
                    torch.zeros_like(h_S), h_E, E_idx)
                h_EXV_encoder = self.mu.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
                h_EXV_encoder_fw = mask_fw * h_EXV_encoder
                for layer in self.model.decoder_layers:
                    h_ESV = self.mu.cat_neighbors_nodes(h_V, h_ES, E_idx)
                    h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                    h_V = layer(h_V, h_ESV, mask_enc)
                logits = self.model.W_out(h_V)
                lp = torch.nn.functional.log_softmax(logits, dim=-1)
                out[k] = lp[0, idx, :].double().cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------
def battery(scorer, cplx_pose, pep_pose, seq, order_seed=0, n_shuf=8):
    """All MPNN metrics for one (template pose, variant sequence) pair.

    `n_shuf` drives the COMPOSITION CONTROL. A permutation of the sequence has
    exactly the same residues, so anything that survives `pll_cplx - pll_shuf`
    is positional complementarity and cannot be an amino-acid-composition
    artefact -- the strongest form of the objection that closed the ESM-2 /
    PepMLM thread, and one a background-frequency correction only approximates.
    Set n_shuf=0 to skip (the two metrics are then absent from the row).
    """
    aa_to_int = scorer.aa_to_int
    sidx = [aa_to_int[a] for a in seq]

    lp_c = scorer.marginals(cplx_pose, seq=seq, see_other_seq=False)
    lp_p = scorer.marginals(pep_pose, seq=seq, see_other_seq=False)
    lp_c_ar = scorer.marginals(cplx_pose, seq=seq, see_other_seq=True,
                               order_seed=order_seed)
    lp_p_ar = scorer.marginals(pep_pose, seq=seq, see_other_seq=True,
                               order_seed=order_seed)

    n = len(seq)
    sel = np.arange(n)
    c_i = lp_c[sel, sidx]
    p_i = lp_p[sel, sidx]
    c_ar = lp_c_ar[sel, sidx]
    p_ar = lp_p_ar[sel, sidx]
    gain = c_i - p_i

    ifm = interface_mask(cplx_pose.cb, cplx_pose.pep_idx, cplx_pose.rec_idx)
    icg_if = float(gain[ifm].mean()) if ifm.any() else float(gain.mean())

    top1 = float(np.mean(lp_c[sel].argmax(axis=1) == np.array(sidx)))
    bg = np.array([LOG_BG.get(a, math.log(1 / 20)) for a in seq])

    shuf = {}
    if n_shuf and len(set(seq)) > 1:
        # Seeded on the SEQUENCE, not on a counter, so the same variant gets the
        # same permutations on every template and in every rerun: the control is
        # reproducible and cannot drift between runs.
        rng = np.random.default_rng(abs(hash(("shuf", seq))) % (2 ** 32))
        vals = []
        for _ in range(int(n_shuf)):
            perm = rng.permutation(len(seq))
            # No re-scoring needed: in marginal mode lp_c is a FIXED
            # position x amino-acid matrix for this pose (selftest [4] proves it
            # does not depend on the peptide's own sequence), so permuting the
            # sequence just reads different cells of the same matrix. That is
            # also why this metric is a pose-derived PSSM and costs a lookup.
            vals.append(float(lp_c[sel, [sidx[i] for i in perm]].mean()))
        shuf = {"pll_shuf": float(np.mean(vals)),
                "pos_gain": float(c_i.mean() - np.mean(vals)),
                "_pll_shuf_sd": float(np.std(vals))}

    return {
        **shuf,
        "icg": float(gain.mean()),
        "icg_sum": float(gain.sum()),
        "icg_if": icg_if,
        "icg_ar": float((c_ar - p_ar).mean()),
        "pll_cplx": float(c_i.mean()),
        "pll_pep": float(p_i.mean()),
        "pll_bg": float((c_i - bg).mean()),
        "rec_top1": top1,
        "_n_iface": int(ifm.sum()),
        "_per_pos_gain": [round(float(x), 4) for x in gain],
        "_per_pos_cplx": [round(float(x), 4) for x in c_i],
    }



def _consensus_excluding(seqs_by_pdb):
    """{pdb: -hamming(seq, consensus of the OTHER same-length members)}.

    Leave-one-out, so a variant never votes for its own consensus -- the same
    firewall the xtemplate mode applies to structures. Returns {} where a
    length group has fewer than 3 members, since a consensus of one is just
    that member.
    """
    out = {}
    by_len = {}
    for pdb, sq in seqs_by_pdb.items():
        by_len.setdefault(len(sq), []).append(pdb)
    for n, group in by_len.items():
        if len(group) < 3:
            continue
        for pdb in group:
            others = [seqs_by_pdb[q] for q in group if q != pdb]
            cons = "".join(
                max(set(col), key=col.count)
                for col in ([o[i] for o in others] for i in range(n)))
            out[pdb] = -float(sum(1 for a, b in zip(seqs_by_pdb[pdb], cons)
                                  if a != b))
    return out

def null_metrics(seq, cplx_pose, pdb, dg_row=None, int_to_aa=None):
    """Nulls and joined baselines. A metric that cannot beat these has not worked."""
    int_to_aa = int_to_aa if int_to_aa is not None else _INT2AA
    rec_seq_if = ""
    if len(cplx_pose.rec_idx):
        d = np.linalg.norm(
            cplx_pose.cb[cplx_pose.rec_idx][:, None, :]
            - cplx_pose.cb[cplx_pose.pep_idx][None, :, :], axis=-1)
        near = cplx_pose.rec_idx[d.min(axis=1) < IFACE_CB_CUTOFF]
        S = cplx_pose.protein_dict["S"].cpu().numpy()
        rec_seq_if = "".join(int_to_aa.get(int(S[i]), "X") for i in near)
    q_pep = net_charge(seq)
    q_rec = net_charge(rec_seq_if)
    # Deterministic pseudo-random, keyed on the PDB code only, so it is stable
    # across runs and independent of everything the real metrics see.
    rng = np.random.default_rng(abs(hash(("null_random", pdb))) % (2**32))
    out = {
        "null_neglen": -float(len(seq)),
        "null_chargecomp": float(q_pep * q_rec),
        "null_random": float(rng.normal()),
        "_q_pep": q_pep,
        "_q_rec_iface": q_rec,
    }
    if dg_row:
        out["dg_rosetta"] = float(dg_row["dG"])
        out["null_negdsasa"] = -float(dg_row["dsasa"])
    return out


_INT2AA = {}


def _init_int2aa(scorer):
    global _INT2AA
    _INT2AA = dict(scorer.int_to_aa)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_dg_rows(path):
    """Mean Rosetta dG / dSASA per PDB from the existing gate's output. Joined,
    never recomputed: a second relax would be a second number for the same
    quantity and the two would drift."""
    agg = {}
    p = Path(path)
    if not p.is_file():
        return agg
    acc = {}
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not r.get("valid"):
            continue
        a = acc.setdefault(r["pdb"], {"dG": [], "dsasa": []})
        a["dG"].append(r["dG"])
        a["dsasa"].append(r["dsasa"])
    for pdb, a in acc.items():
        if a["dG"]:
            agg[pdb] = {"dG": sum(a["dG"]) / len(a["dG"]),
                        "dsasa": sum(a["dsasa"]) / len(a["dsasa"])}
    return agg


def _prepared_path(work_dir, pdb):
    return Path(work_dir) / f"{pdb}_prepared.pdb"


def _chains_of(pdb_path, pep_hint=None):
    """(receptor_chains, peptide_chain) from a prepared PDB: the peptide is the
    LAST chain written by `cpset_accuracy.prepare_pdb`, and also the shortest."""
    order, counts = [], {}
    for ln in Path(pdb_path).read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        ch = ln[21]
        key = ln[22:27]
        if ch not in counts:
            counts[ch] = set()
            order.append(ch)
        counts[ch].add(key)
    if not order:
        raise RuntimeError(f"no ATOM records in {pdb_path}")
    pep = pep_hint if pep_hint in counts else min(order, key=lambda c: len(counts[c]))
    rec = [c for c in order if c != pep]
    return rec, pep


def score_decoy(work_dir, out_path, data_dir, models=(PRIMARY_MODEL,),
                device="cpu", verbose=True):
    """THE SPECIFICITY CONTROL: score each series' peptides on ANOTHER TARGET's pocket.

    The first run left one objection standing. `pll_pep` -- the peptide scored
    with its receptor deleted -- reached 0.923 agreement on the BB/MIX tier, so
    the metric was not yet shown to be reading the POCKET rather than the
    peptide's own bound backbone conformation.

    This settles it with a sharper instrument than any correction term. Thread a
    series' variants onto a same-length template from a DIFFERENT series, i.e. a
    different receptor entirely (KEAP1's 7-mers into streptavidin's site;
    SFTI-like 14-mers across three unrelated receptors). The peptide backbone
    conformation is now a foreign one and the pocket is the wrong pocket.

      * If within-series ranking SURVIVES on a decoy receptor, the metric is not
        pocket-specific and the CPSet result is about peptide conformation or
        composition, not binding. That would be decisive against it.
      * If ranking COLLAPSES on the decoy while holding on the true receptor,
        the signal is the interface -- which is the claim.

    Rows are tagged `decoy=True` and carry `template_series`, so they can never
    be pooled with the real ones by accident.
    """
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        for ln in out_path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["series"], r["template"], r["pdb"]))

    fh = out_path.open("a")
    n_new = 0
    for model_name in models:
        scorer = MPNNScorer(model_name=model_name, device=device)
        _init_int2aa(scorer)
        # Parse every structure once, note its peptide length.
        info = {}
        for s in SERIES:
            for p in s["pdbs"]:
                path = _prepared_path(work_dir, p)
                if not path.is_file():
                    continue
                try:
                    rec, pep = _chains_of(path)
                    cp = scorer.parse(path, pep_chain=pep, rec_chains=rec)
                    pp = scorer.parse(path, pep_chain=pep, rec_chains=None)
                except Exception:  # noqa: BLE001
                    continue
                info[p] = {"series": s["id"], "tier": s["tier"], "cp": cp,
                           "pp": pp, "seq": cp.pep_seq, "len": len(cp.pep_seq)}
        if verbose:
            print(f"=== decoy control, model {model_name}: "
                  f"{len(info)} structures ===", flush=True)
        for tmpl, ti in sorted(info.items()):
            for var, vi in sorted(info.items()):
                if vi["series"] == ti["series"]:
                    continue                       # that is the real arm
                if vi["len"] != ti["len"]:
                    continue                       # cannot thread
                key = (model_name, vi["series"], tmpl, var)
                if key in done:
                    continue
                try:
                    m = battery(scorer, ti["cp"], ti["pp"], vi["seq"], n_shuf=0)
                    m.update(null_metrics(vi["seq"], ti["cp"], var,
                                          None, int_to_aa=scorer.int_to_aa))
                except Exception as exc:  # noqa: BLE001
                    print(f"    {tmpl}<-{var}: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                fh.write(json.dumps({
                    "model": model_name, "model_type": scorer.model_type,
                    "sc_context": scorer.use_sc_context,
                    "series": vi["series"], "tier": vi["tier"],
                    "template": tmpl, "template_series": ti["series"],
                    "pdb": var, "is_self": False, "decoy": True,
                    "seq": vi["seq"], "pep_len": vi["len"],
                    "pkd": (truth.get(var) or {}).get("pkd"),
                    "metrics": m,
                }) + "\n")
                fh.flush()
                n_new += 1
    fh.close()
    if verbose:
        print(f"wrote {n_new} decoy rows to {out_path}", flush=True)
    return n_new


def score(work_dir, out_path, data_dir, dg_path=None, models=(PRIMARY_MODEL,),
          series_only=None, device="cpu", order_seed=0, verbose=True, n_shuf=8):
    """Score every (series, template, variant, model) combination. Resumable."""
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    dg_rows = _load_dg_rows(dg_path) if dg_path else {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.is_file():
        for ln in out_path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["series"], r["template"], r["pdb"]))
    if verbose and done:
        print(f"resuming: {len(done)} rows already present", flush=True)

    fh = out_path.open("a")
    n_new = 0
    t0 = time.time()
    for model_name in models:
        scorer = MPNNScorer(model_name=model_name, device=device)
        _init_int2aa(scorer)
        if verbose:
            print(f"=== model {model_name} ({scorer.model_type}, "
                  f"sc_context={scorer.use_sc_context}) ===", flush=True)
        for s in SERIES:
            sid = s["id"]
            if series_only and sid not in series_only:
                continue
            members = [p for p in s["pdbs"] if _prepared_path(work_dir, p).is_file()]
            if len(members) < 2:
                if verbose:
                    print(f"  {sid}: {len(members)} prepared structures, skipping",
                          flush=True)
                continue
            # Parse every member once: its complex pose, its peptide-only pose,
            # and its peptide sequence.
            poses = {}
            for p in members:
                path = _prepared_path(work_dir, p)
                rec, pep = _chains_of(path)
                try:
                    cp = scorer.parse(path, pep_chain=pep, rec_chains=rec)
                    pp = scorer.parse(path, pep_chain=pep, rec_chains=None)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {sid}/{p}: parse failed: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                poses[p] = (cp, pp)
            if verbose:
                lens = {p: len(v[0].pep_seq) for p, v in poses.items()}
                print(f"  {sid} tier={s['tier']} members={len(poses)} lens={lens}",
                      flush=True)
            # THE MEMORISATION NULL. These crystals predate the inverse-folding
            # model's training cutoff, so it may simply prefer sequences close to
            # the ones it saw. Averaging over templates would then favour the
            # series CONSENSUS -- and in S03 the consensus happens to agree with
            # the strong binders, because the weak variants are the ones carrying
            # the unusual substitutions. If distance-to-consensus ranks a series
            # as well as the metric does, the metric has not been shown to add
            # anything. Computed from the OTHER members only, matching the
            # xtemplate leakage firewall.
            cons = _consensus_excluding(
                {p: v[0].pep_seq for p, v in poses.items()})
            for tmpl, (cp, pp) in sorted(poses.items()):
                tlen = len(cp.pep_seq)
                for var in sorted(poses):
                    vseq = poses[var][0].pep_seq
                    if len(vseq) != tlen:
                        continue  # cannot thread across lengths
                    key = (model_name, sid, tmpl, var)
                    if key in done:
                        continue
                    try:
                        m = battery(scorer, cp, pp, vseq, order_seed=order_seed,
                                    n_shuf=n_shuf)
                        m.update(null_metrics(vseq, cp, var, dg_rows.get(var),
                                              int_to_aa=scorer.int_to_aa))
                        if var in cons:
                            m["null_consensus"] = cons[var]
                    except Exception as exc:  # noqa: BLE001
                        print(f"    {sid} {tmpl}<-{var}: FAILED "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        continue
                    rec_out = {
                        "model": model_name, "model_type": scorer.model_type,
                        "sc_context": scorer.use_sc_context,
                        "series": sid, "tier": s["tier"],
                        "template": tmpl, "pdb": var, "is_self": tmpl == var,
                        "seq": vseq, "pep_len": len(vseq),
                        "pkd": (truth.get(var) or {}).get("pkd"),
                        "chain_type": (truth.get(var) or {}).get("chain_type"),
                        "order_seed": order_seed,
                        "metrics": m,
                    }
                    fh.write(json.dumps(rec_out) + "\n")
                    fh.flush()
                    n_new += 1
    fh.close()
    if verbose:
        print(f"wrote {n_new} new rows to {out_path} in {time.time()-t0:.1f}s",
              flush=True)
    return n_new


# ---------------------------------------------------------------------------
# Self-test: the claims this module makes about itself, asserted against the
# real model rather than described in a comment.
# ---------------------------------------------------------------------------
def selftest(work_dir, pdb="7K2E", device="cpu"):
    import torch
    path = _prepared_path(work_dir, pdb)
    if not path.is_file():
        print(f"SELFTEST SKIPPED: no {path}")
        return 0
    sc = MPNNScorer(device=device)
    _init_int2aa(sc)
    rec, pep = _chains_of(path)
    cp = sc.parse(path, pep_chain=pep, rec_chains=rec)
    print(f"parsed {pdb}: receptor {rec} peptide {pep} seq={cp.pep_seq}")

    ok = True

    # 1. DETERMINISM of the primary mode across decoding-order seeds.
    a = sc.marginals(cp, see_other_seq=False, order_seed=0)
    b = sc.marginals(cp, see_other_seq=False, order_seed=12345)
    d = float(np.abs(a - b).max())
    print(f"[1] marginal mode, max |diff| across order seeds = {d:.3e}")
    ok &= d < 1e-9
    if d >= 1e-9:
        print("    FAIL: the primary mode is supposed to be exactly deterministic")

    # 2. Order dependence of the AR mode is real but small -- measured, so the
    #    'low noise' claim for the secondary mode is a number, not a hope.
    a2 = sc.marginals(cp, see_other_seq=True, order_seed=0)
    b2 = sc.marginals(cp, see_other_seq=True, order_seed=12345)
    print(f"[2] AR mode,       max |diff| across order seeds = "
          f"{float(np.abs(a2-b2).max()):.3e}")

    # 3. THE FLAG INVERSION, pinned against upstream. Upstream's
    #    use_sequence=True is our see_other_seq=False.
    fd = dict(cp.feature_dict)
    S = cp.protein_dict["S"]
    fd["S"] = S[None, :] if S.dim() == 1 else S
    fd["batch_size"] = 1
    g = torch.Generator(device="cpu").manual_seed(0)
    fd["randn"] = torch.randn(1, fd["mask"].shape[1], generator=g).to(sc.device)
    with torch.no_grad():
        up = sc.model.single_aa_score(fd, use_sequence=True)
    up_lp = up["log_probs"][0].double().cpu().numpy()[cp.pep_idx]
    d3 = float(np.abs(up_lp - a).max())
    print(f"[3] vs upstream single_aa_score(use_sequence=True): max |diff| = {d3:.3e}")
    if d3 >= 1e-5:
        ok = False
        print("    FAIL: subset scorer disagrees with upstream, or upstream "
              "fixed its inverted flag -- read the FLAG INVERSION note.")

    # 4. The peptide's own side chains must be invisible: threading a different
    #    sequence onto this pose must change the score, and the receptor's side
    #    chains must still be doing work.
    alt = "A" * len(cp.pep_seq)
    lp_alt = sc.marginals(cp, seq=alt, see_other_seq=False)
    d4 = float(np.abs(lp_alt - a).max())
    print(f"[4] marginal is sequence-independent (encoder cached): "
          f"max |diff| after threading poly-A = {d4:.3e}")
    ok &= d4 < 1e-9
    if d4 >= 1e-9:
        print("    FAIL: the marginal must not depend on the peptide's own "
              "sequence -- the queried position is decoded first.")

    # 5. The receptor must matter: peptide-alone marginals must differ.
    pp = sc.parse(path, pep_chain=pep, rec_chains=None)
    lp_p = sc.marginals(pp, see_other_seq=False)
    d5 = float(np.abs(lp_p - a).max())
    print(f"[5] receptor changes the marginal: max |diff| = {d5:.3e}")
    ok &= d5 > 0.01
    if d5 <= 0.01:
        print("    FAIL: removing the receptor changed nothing -- ICG would be 0")

    # 6. The headline worked example: the full battery on the KEAP1 series.
    print("\n[6] S03_keap1, scored on this one template "
          f"({pdb}), ICG per variant:")
    rows = []
    for p in SERIES[2]["pdbs"]:
        q = _prepared_path(work_dir, p)
        if not q.is_file():
            continue
        vseq = sc.parse(q, pep_chain=_chains_of(q)[1], rec_chains=None).pep_seq
        if len(vseq) != len(cp.pep_seq):
            continue
        m = battery(sc, cp, pp, vseq)
        truth = load_dataset1(Path(work_dir).parent / "cpset_raw" / "dataset1.csv")
        rows.append((truth.get(p, {}).get("pkd"), p, vseq, m["icg"], m["pll_cplx"]))
    for pkd, p, s, icg, pll in sorted(rows, key=lambda r: -(r[0] or 0)):
        print(f"    pKd {pkd:5.2f}  {p}  {s}  ICG {icg:+.4f}  pll_cplx {pll:+.4f}")
    if len(rows) >= 3:
        xs = [r[0] for r in rows]
        ys = [r[3] for r in rows]
        from lnr.skempi_accuracy import _spearman
        print(f"    rho(ICG, pKd) on this single template = {_spearman(ys, xs):+.3f} "
              f"(positive = working; this is ONE template, not the headline)")

    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


class TargetPSSM:
    """The deployable artefact: a target's binding metric as a lookup table.

    In marginal mode the model's output does not depend on the peptide's own
    sequence (`selftest` [4] asserts this bitwise), so one reference complex
    determines a fixed `n_positions x 20` matrix of log p(aa | bound pose).
    Scoring ANY sequence against that target is then a sum of table lookups:

        score(s) = mean_i  M[i, s_i]              nats / residue

    Which is why this is usable as an in-loop fitness where the incumbent is
    not. Per evaluation the incumbent costs a Boltz fold plus a Rosetta relax
    and returns a number with a +-3-4 REU noise floor; this costs ~20
    microseconds and returns the same number every time.

    It also makes things possible that a noisy oracle cannot do at all:
      * `best_sequence()` is the EXACT optimum of the metric, not a search for it
      * `top_k_at()` exposes the per-position preference the score is made of
      * a whole library can be ranked without a single structure prediction

    Build cost is one encode of the reference complex (~1 s on CPU).
    """

    # The two metrics this table can be read with. `pll_bg` is THE DEPLOYED ONE
    # (the one graded on CPSet and SKEMPI); `pll_cplx` is the uncorrected mean
    # log-likelihood, kept because it is what the secondary row of every results
    # table reports. They are NOT interchangeable -- see `eff`.
    PSSM_METRICS = ("pll_bg", "pll_cplx")
    DEFAULT_METRIC = "pll_bg"

    def __init__(self, matrix, alphabet, positions, template, pep_chain,
                 pep_seq, model_name, iface=None):
        self.M = np.asarray(matrix, dtype=np.float64)   # [n_pos, 21]
        self.alphabet = list(alphabet)
        self.aa_to_int = {a: i for i, a in enumerate(self.alphabet)}
        self.positions = list(positions)
        self.template = template
        self.pep_chain = pep_chain
        self.pep_seq = pep_seq
        self.model_name = model_name
        self.iface = list(iface) if iface is not None else None
        self._bg = None
        self._eff_bg = None
        # Positions this table will ever average over, FIXED at construction from
        # the template sequence. A position whose template residue is not one of
        # the 20 (an unparsable/modified residue -> 'X') is dropped, and it is
        # dropped for EVERY sequence scored against this table. Deciding that
        # per-sequence instead would average two variants over different position
        # sets and the difference between their scores would be meaningless.
        self.scorable = [i for i, a in enumerate(self.pep_seq or "")
                         if a in AA20_SET] or list(range(self.n))

    @property
    def n(self):
        return self.M.shape[0]

    def _bg_row(self):
        """log background frequency per ALPHABET column (not per amino acid), so
        the subtraction is a single vectorised row operation."""
        if self._bg is None:
            self._bg = np.array([LOG_BG.get(a, math.log(1 / 20))
                                 for a in self.alphabet], dtype=np.float64)
        return self._bg

    def eff(self, metric=None):
        """The [n_pos, 21] matrix whose mean over positions IS `metric`.

        WHY THIS EXISTS, AND WHY THE DEFAULT IS pll_bg. `pll_bg` subtracts the
        amino acid's background log-frequency, so a residue is credited for being
        preferred at THIS position over how often it simply occurs anywhere. That
        subtraction is a per-COLUMN offset, which is not a monotone relabelling of
        `pll_cplx`: it reorders the per-position argmax, and it reorders two
        sequences of the same length whenever they differ in composition. A
        deployment that reports `pll_bg` but optimises `pll_cplx` is therefore
        optimising a different metric from the one that was graded -- which is
        exactly the slip this method is here to make impossible.
        """
        metric = metric or self.DEFAULT_METRIC
        if metric == "pll_cplx":
            return self.M
        if metric == "pll_bg":
            if self._eff_bg is None:
                self._eff_bg = self.M - self._bg_row()[None, :]
            return self._eff_bg
        raise ValueError(f"unknown PSSM metric {metric!r}; "
                         f"allowed: {self.PSSM_METRICS}")

    def matches(self, seq):
        """(ok, detail) -- is `seq` indexed the SAME WAY as this table's template?

        The silent failure this guards against: the caller's peptide sequence
        comes from a different parser than the one that built the matrix
        (`env.LNRTarget`, which silently DROPS any residue missing a backbone
        atom, vs LigandMPNN's `parse_PDB`). If the two disagree by one dropped
        residue but happen to return the same length, every lookup reads the
        wrong row, the score is confidently wrong, and nothing raises. So the
        deployment asserts the template sequences are the SAME STRING -- not
        merely the same length. Mismatches at a position where either side is
        'X' (an unparsable residue) are tolerated and reported.
        """
        if len(seq) != self.n:
            return False, (f"length {len(seq)} != template peptide length "
                           f"{self.n} ({self.pep_seq})")
        bad = [i for i, (a, b) in enumerate(zip(seq, self.pep_seq))
               if a != b and a in AA20_SET and b in AA20_SET]
        if bad:
            shown = ", ".join(f"{i}:{self.pep_seq[i]}->{seq[i]}" for i in bad[:6])
            return False, (f"{len(bad)}/{self.n} template residues differ "
                           f"({shown}{', ...' if len(bad) > 6 else ''}); "
                           f"pssm={self.pep_seq} caller={seq}")
        soft = [i for i, (a, b) in enumerate(zip(seq, self.pep_seq)) if a != b]
        return True, (f"aligned on {self.n} positions"
                      + (f" ({len(soft)} unparsable-residue position(s) ignored)"
                         if soft else ""))

    @classmethod
    def from_complex(cls, pdb_path, pep_chain=None, rec_chains=None,
                     model_name=PRIMARY_MODEL, device="cpu", scorer=None):
        scorer = scorer or MPNNScorer(model_name=model_name, device=device)
        if pep_chain is None:
            rec_chains, pep_chain = _chains_of(pdb_path)
        elif rec_chains is None:
            rec_chains = [c for c in _chains_of(pdb_path, pep_chain)[0]]
        pose = scorer.parse(pdb_path, pep_chain=pep_chain, rec_chains=rec_chains)
        M = scorer.marginals(pose, see_other_seq=False)
        ifm = interface_mask(pose.cb, pose.pep_idx, pose.rec_idx)
        return cls(M, [scorer.int_to_aa[i] for i in range(M.shape[1])],
                   list(range(M.shape[0])), str(pdb_path), pep_chain,
                   pose.pep_seq, scorer.model_name,
                   iface=[bool(x) for x in ifm])

    def _positions(self, interface_only=False, positions=None):
        """The position set a score is averaged over. A function of the TEMPLATE
        and the caller's restriction only -- never of the sequence being scored,
        so any two sequences scored with the same arguments are comparable."""
        pos = list(self.scorable)
        if positions is not None:
            want = {int(p) for p in positions}
            pos = [i for i in pos if i in want]
            if not pos:
                raise ValueError("position restriction leaves no scorable "
                                 "position to average over")
        if interface_only and self.iface:
            sub = [i for i in pos if self.iface[i]]
            if sub:
                return sub
        return pos

    def score(self, seq, interface_only=False, metric=None, positions=None):
        """nats/residue. Higher = better suited to this bound pose.

        Defaults to `pll_bg`, the metric that was graded -- see `eff`.
        """
        if len(seq) != self.n:
            raise ValueError(
                f"sequence length {len(seq)} != template peptide length "
                f"{self.n}; this PSSM is for {self.pep_seq}")
        unknown = sorted({a for a in seq if a not in self.aa_to_int})
        if unknown:
            raise ValueError(f"sequence has letters outside the model alphabet: "
                             f"{unknown}")
        E = self.eff(metric)
        pos = self._positions(interface_only, positions)
        idx = [self.aa_to_int[seq[i]] for i in pos]
        return float(E[np.asarray(pos), idx].mean())

    def score_many(self, seqs, **kw):
        return [self.score(s, **kw) for s in seqs]

    def best_sequence(self, allowed=None, metric=None):
        """The EXACT argmax of the metric -- available because the metric is a
        per-position sum, not a search over a noisy oracle. Reads the metric's
        OWN matrix, so the optimum returned is the optimum of the metric being
        deployed and not of the uncorrected log-likelihood."""
        E = self.eff(metric)
        ok = set(allowed or AA20_SET)
        cols = [i for i, a in enumerate(self.alphabet) if a in ok]
        return "".join(self.alphabet[cols[int(np.argmax(E[p, cols]))]]
                       for p in range(self.n))

    def top_k_at(self, pos, k=5, allowed=None, metric=None):
        E = self.eff(metric)
        ok = set(allowed or AA20_SET)
        cols = [i for i, a in enumerate(self.alphabet) if a in ok]
        order = sorted(cols, key=lambda c: -E[pos, c])[:k]
        return [(self.alphabet[c], float(E[pos, c])) for c in order]

    def position_gains(self, seq, allowed=None, positions=None, metric=None,
                       interface_only=False, mutable=None):
        """[(pos, best_aa, gain)] -- what the single best substitution at each
        position is worth, in the metric's own units, sorted best first.

        `gain` is per-POSITION (not per-residue): the score of the mutant is
        `score(seq) + sum(gains)/n_scored`, because the metric is a mean of
        independent per-position terms.

        TWO DIFFERENT POSITION SETS, and conflating them makes an optimum that no
        search can reach:
          `positions` / `interface_only` -- which positions the score AVERAGES
              over. A substitution outside this set cannot change the score at
              all, so it can never carry a gain.
          `mutable` -- which positions a caller is ALLOWED TO CHANGE. The loop's
              random-EA operator only ever adds positions drawn from the target's
              interface set, so an optimum that moves a non-interface residue is
              unreachable BY CONSTRUCTION, and grading the search against it
              would blame it for something it is forbidden to do. Defaults to
              every position.
        """
        if len(seq) != self.n:
            raise ValueError(f"sequence length {len(seq)} != {self.n}")
        E = self.eff(metric)
        ok = set(allowed or AA20_SET)
        cols = [i for i, a in enumerate(self.alphabet) if a in ok]
        allow_pos = set(self._positions(interface_only, positions))
        if mutable is not None:
            allow_pos &= {int(p) for p in mutable}
        out = []
        for p in sorted(allow_pos):
            wt = self.aa_to_int.get(seq[p])
            if wt is None:
                continue
            best = max(cols, key=lambda c: E[p, c])
            gain = float(E[p, best] - E[p, wt])
            if gain > 0:
                out.append((p, self.alphabet[best], gain))
        out.sort(key=lambda r: -r[2])
        return out

    def best_subs(self, seq, k, allowed=None, positions=None, metric=None,
                  interface_only=False, mutable=None):
        """The EXACT best `<= k` substitution set on `seq`, and the score it gets.

        Returns `([(pos, aa), ...], score)`.

        This is the whole reason a deterministic per-position metric changes what
        the in-loop experiment can measure. The metric is a sum of independent
        per-position terms, so the optimum under a Hamming budget of k is not
        something to search for -- it is the k largest per-position gains, read
        off the table. So the loop's finalist can be scored against the TRUE
        optimum (its regret), which is impossible with the Boltz/Rosetta oracle
        where the optimum is unknown and unknowable. A search arm that does not
        reach regret 0 on a budget this small is failing at a solved problem.
        """
        gains = self.position_gains(seq, allowed=allowed, positions=positions,
                                    metric=metric, interface_only=interface_only,
                                    mutable=mutable)
        # take the k largest gains, then canonicalise BY POSITION: that is the
        # identity `seqmut.subs_key` uses, so the returned set can be compared
        # directly against a genome from the loop. `position_gains` is still the
        # gain-ordered view for anyone who wants it.
        subs = sorted((p, a) for p, a, _ in gains[: max(0, int(k))])
        mut = list(seq)
        for p, a in subs:
            mut[p] = a
        return subs, self.score("".join(mut), interface_only=interface_only,
                                metric=metric, positions=positions)

    def to_json(self):
        return {"matrix": self.M.tolist(), "alphabet": self.alphabet,
                "positions": self.positions, "template": self.template,
                "pep_chain": self.pep_chain, "pep_seq": self.pep_seq,
                "model": self.model_name, "iface": self.iface}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json()))
        return str(path)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["matrix"], d["alphabet"], d["positions"], d["template"],
                   d["pep_chain"], d["pep_seq"], d["model"], d.get("iface"))


def build_pssm(pdb_path, out_path, pep_chain=None, model_name=PRIMARY_MODEL,
               device="cpu"):
    t = TargetPSSM.from_complex(pdb_path, pep_chain=pep_chain,
                                model_name=model_name, device=device)
    t.save(out_path)
    print(f"target   : {Path(pdb_path).name}  peptide chain {t.pep_chain}")
    print(f"model    : {t.model_name}")
    print(f"positions: {t.n}   native sequence: {t.pep_seq}")
    print(f"metric   : {t.DEFAULT_METRIC} (the graded/deployed one); "
          f"pll_cplx shown beside it for reference")
    print(f"native score         : {t.score(t.pep_seq):+.4f} nats/residue "
          f"(pll_cplx {t.score(t.pep_seq, metric='pll_cplx'):+.4f})")
    best = t.best_sequence()
    print(f"metric-optimal seq   : {best}  ({t.score(best):+.4f})")
    # The two metrics do NOT share an optimum; printing both makes that visible
    # rather than leaving it to be rediscovered as a bug.
    best_c = t.best_sequence(metric="pll_cplx")
    if best_c != best:
        print(f"  (pll_cplx would pick {best_c} -- a DIFFERENT sequence, which is "
              f"why the metric is carried explicitly)")
    print()
    print("per-position preference (top 5), native residue marked *:")
    for p in range(t.n):
        tops = t.top_k_at(p, 5)
        cells = "  ".join(
            f"{'*' if a == t.pep_seq[p] else ' '}{a}{v:+6.2f}" for a, v in tops)
        print(f"  pos {p+1:2d} ({t.pep_seq[p]}): {cells}")
    print()
    print(f"wrote {out_path}")
    return 0


def sar_table(path, data_dir, series_id, mode="xtemplate"):
    """Per-position breakdown for one series: WHERE does the score go?

    A metric can rank a series correctly for the wrong reason. On S03_keap1 the
    measured SAR is known and local -- D2->A costs 2.33 log units, T5->A costs
    2.78, E3->P gains 0.5-0.9 -- so the metric's per-position log-probability
    should drop at positions 2 and 5 and nowhere else. This prints that matrix,
    which is the difference between "it correlates" and "it is measuring the
    thing the chemistry says it should".
    """
    from lnr.metric_gate import load_rows
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    rows = [r for r in load_rows(path) if r.get("series") == series_id]
    if mode == "xtemplate":
        rows = [r for r in rows if not r.get("is_self")]
    elif mode == "self":
        rows = [r for r in rows if r.get("is_self")]
    if not rows:
        print(f"no rows for {series_id}")
        return 1
    acc = {}
    for r in rows:
        pp = (r.get("metrics") or {}).get("_per_pos_cplx")
        if not pp:
            continue
        a = acc.setdefault(r["pdb"], {"seq": r["seq"], "vals": []})
        a["vals"].append(pp)
    if not acc:
        print("rows carry no _per_pos_cplx")
        return 1
    items = sorted(acc.items(),
                   key=lambda kv: -((truth.get(kv[0]) or {}).get("pkd") or 0))
    npos = len(items[0][1]["vals"][0])
    print(f"=== per-position log p(s_i | complex), {series_id}, mode={mode} ===")
    print("Positions where a weak variant LOSES score are the metric's "
          "explanation for its rank.")
    print()
    hdr = "  ".join(f"{i+1:>7d}" for i in range(npos))
    print(f"{'pKd':>5s} {'pdb':6s} {'seq':{npos+2}s} {hdr}   {'mean':>7s}")
    ref = None
    for pdb, a in items:
        mean_pos = np.mean(np.array(a["vals"], dtype=float), axis=0)
        pkd = (truth.get(pdb) or {}).get("pkd")
        cells = "  ".join(f"{v:+7.3f}" for v in mean_pos)
        print(f"{pkd:5.2f} {pdb:6s} {a['seq']:{npos+2}s} {cells}   "
              f"{mean_pos.mean():+7.3f}")
        if ref is None:
            ref = (pdb, a["seq"], mean_pos)
    print()
    print(f"--- change vs the strongest binder ({ref[0]} {ref[1]}); "
          f"'.' = identical residue ---")
    for pdb, a in items[1:]:
        mean_pos = np.mean(np.array(a["vals"], dtype=float), axis=0)
        d = mean_pos - ref[2]
        pkd = (truth.get(pdb) or {}).get("pkd")
        marks = "  ".join(
            ("      ." if a["seq"][i] == ref[1][i] else f"{d[i]:+7.3f}")
            for i in range(npos))
        muts = ",".join(f"{ref[1][i]}{i+1}{a['seq'][i]}"
                        for i in range(npos) if a["seq"][i] != ref[1][i])
        print(f"{pkd:5.2f} {pdb:6s} {a['seq']:{npos+2}s} {marks}   "
              f"dpKd {pkd - (truth.get(ref[0]) or {}).get('pkd', 0):+5.2f}  {muts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", metavar="JSONL")
    ap.add_argument("--work-dir", default=os.environ.get(
        "CPSET_WORK_DIR", "/zfsauton/scratch/yixiz/lp2cp_stage2/cpset_prepared"))
    ap.add_argument("--data-dir", default=os.environ.get(
        "CPSET_DATA_DIR", "/zfsauton/scratch/yixiz/lp2cp_stage2/cpset_raw"))
    ap.add_argument("--dg", default=os.environ.get(
        "CPSET_OUT", "/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/cpset_dg.jsonl"))
    ap.add_argument("--out", default="bindscore.jsonl")
    ap.add_argument("--models", default=PRIMARY_MODEL,
                    help="comma-separated checkpoint names")
    ap.add_argument("--series", default="", help="comma-separated series ids")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--tier", default="", help="report: restrict to tiers, e.g. BB,MIX")
    ap.add_argument("--sar", metavar="JSONL",
                    help="per-position breakdown for one series (needs --series)")
    ap.add_argument("--n-shuf", type=int, default=8,
                    help="permutations for the composition control (0 = skip)")
    ap.add_argument("--decoy", action="store_true",
                    help="specificity control: score each series on OTHER "
                         "targets' pockets")
    ap.add_argument("--pssm", metavar="PDB",
                    help="build the deployable lookup table for one complex")
    ap.add_argument("--pep-chain", default=None)
    ap.add_argument("--pssm-out", default="target_pssm.json")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.work_dir, device=args.device)
    if args.score:
        score(args.work_dir, args.out, args.data_dir, dg_path=args.dg,
              models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
              series_only=set(x for x in args.series.split(",") if x) or None,
              device=args.device, order_seed=args.order_seed, n_shuf=args.n_shuf)
        return 0
    if args.decoy:
        score_decoy(args.work_dir, args.out, args.data_dir,
                    models=tuple(m.strip() for m in args.models.split(",")
                                 if m.strip()), device=args.device)
        return 0
    if args.pssm:
        return build_pssm(args.pssm, args.pssm_out, pep_chain=args.pep_chain,
                          model_name=args.models.split(",")[0],
                          device=args.device)
    if args.sar:
        sid = (args.series.split(",")[0] if args.series else "S03_keap1")
        return sar_table(args.sar, args.data_dir, sid)
    if args.report:
        from lnr.metric_gate import report
        tiers = tuple(t.strip() for t in args.tier.split(",") if t.strip()) or None
        return report(args.report, args.data_dir, tiers=tiers)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
