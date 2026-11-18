"""Plot example LNR cases: native bound pose vs baseline (rule) ring vs ours (LLM).

For a handful of targets, render the backbone geometry of three conformations:
  NATIVE   -- the deposited LINEAR peptide in its bound pose (what binds).
  BASELINE -- placement_search's committed ring (best geometric closer), re-solved
              in torsion space. It closes on paper but CLASHES (shown).
  OURS     -- gpt-6-luna mask's committed CLASH-FREE ring, re-solved.

Held-window residues keep their native bound coordinates in both rings, so all
three conformations live in ONE coordinate frame and the held region overlays
exactly; only the released/closed portion differs. Binding is annotated as a
native-scaled interface dG (REU, negative = favorable): retained_dG =
(fraction of native interface energy the held residues carry) x native_dG.

Two modes, so figures are redrawable without re-solving a closure:
  compute : solve the closed backbones for the selected cases -> cases.json
  render  : read cases.json -> cases.png   (no solve)
  all     : both (default)

Usage:
  python -m lnr.plot_cases --manifest M --pdb-dir P --out-dir O \
      [--targets 5l9b,1d4t,4w50,5njx] [--mode all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from lnr import env as E  # noqa: E402

DEFAULT_TARGETS = ["5l9b", "1d4t", "4w50", "5njx"]


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def self_clash_pairs(CA, length, cut=3.0):
    """Non-adjacent CA pairs closer than cut (cyclic: 0 and L-1 are bonded)."""
    pairs = []
    d = np.linalg.norm(CA[:, None, :] - CA[None, :, :], axis=-1)
    for i in range(length):
        for j in range(i + 2, length):
            if j - i == length - 1:      # 0<->L-1 ring closure bond, not a clash
                continue
            if d[i, j] < cut:
                pairs.append([i, j])
    return pairs


def rec_clash_pairs(CA, rec, released, cut=3.0):
    pairs = []
    for i in released:
        dr = np.linalg.norm(CA[i][None, :] - rec, axis=-1)
        j = int(dr.argmin())
        if dr[j] < cut:
            pairs.append([i, j])
    return pairs


def compute(args):
    targets = {t.target_id: t for t in E.load_manifest(args.manifest, args.pdb_dir)}
    d = Path(args.out_dir)
    ca_clash = {(c["target_id"], c["a"], c["b"]): c
                for c in load_jsonl(d / "clashes.jsonl")}
    bind = {}
    for b in load_jsonl(d / "rosetta_interface.jsonl"):
        if "per_residue_ddG_ala" in b:
            pr = {int(k): (v or 0.0) for k, v in b["per_residue_ddG_ala"].items()}
            tot = b.get("total_positive_ddG") or sum(v for v in pr.values() if v > 0) or 1.0
            bind[b["target_id"]] = (pr, tot, b.get("native_dG"))

    rows = []
    for f in sorted(d.glob("rules*.jsonl")) + sorted(d.glob("llm_*.jsonl")):
        rows += load_jsonl(f)
    arms = {}
    for r in rows:
        nm = r["arm"]
        if nm.startswith("llm-"):
            nm = f"{r.get('model_id','?')}|{nm}"
        arms.setdefault(nm, {})[r["target_id"]] = r
    lm = arms["gpt-6-luna|llm-mask"]
    ps = arms["placement_search"]
    boa = arms.get("bo", {})

    def best_clean_win(r):
        best = (-1.0, None)
        for h in r.get("history", []):
            if (h.get("closes")
                    and ca_clash.get((r["target_id"], h["a"], h["b"]), {}).get("clash_free")
                    and h["retention"] > best[0]):
                best = (h["retention"], (h["a"], h["b"]))
        return best[1]

    def binding_ret(tid, w):
        if not w or tid not in bind:
            return 0.0
        pr, tot, _ = bind[tid]
        a, b = w
        return sum(pr[k] for k in range(a, b + 1) if k in pr and pr[k] > 0) / tot

    cases = []
    for tid in args.targets:
        if tid not in targets:
            print(f"[plot] skip {tid}: not in manifest")
            continue
        t = targets[tid]
        L = t.length
        ndg = bind.get(tid, (None, None, None))[2]

        ours_w = best_clean_win(lm[tid]) if tid in lm else None
        rule_w = tuple(ps[tid]["best_window"]) if (tid in ps and ps[tid].get("best_window")) else None
        bo_w = tuple(boa[tid]["best_window"]) if (tid in boa and boa[tid].get("best_window")) else None

        def solve(w):
            if not w:
                return None
            ok, resid, Nn, CAn, Cn = E._solve_closed(t, w[0], w[1], n_restarts=12,
                                                     seed=0, max_seconds=12.0)
            return (ok, CAn) if ok else (False, None)

        ours = solve(ours_w)
        rule = solve(rule_w)
        boo = solve(bo_w)

        # nearby receptor CAs (within 12 A of any native peptide CA)
        drec = np.linalg.norm(t.CA[:, None, :] - t.rec_ca[None, :, :], axis=-1)
        near = t.rec_ca[drec.min(axis=0) < 12.0]

        def pack(w, solved, is_ours):
            if not w or not solved or solved[1] is None:
                return None
            CAn = solved[1]
            released = [k for k in range(L) if k < w[0] or k > w[1]]
            ret = binding_ret(tid, w)
            return {
                "window": list(w), "release": L - (w[1] - w[0] + 1),
                "CA": CAn.tolist(),
                "retention_binding": ret,
                "retention_contact": E.retention_of(t, w[0], w[1]),
                "binding_dg": (ret * ndg) if ndg is not None else None,
                "self_clash_pairs": self_clash_pairs(CAn, L),
                "rec_clash_pairs": rec_clash_pairs(CAn, near, released),
                "clean": (not self_clash_pairs(CAn, L)
                          and not rec_clash_pairs(CAn, near, released)),
            }

        cases.append({
            "target_id": tid, "length": L, "native_dg": ndg,
            "native_CA": t.CA.tolist(),
            "receptor_CA": near.tolist(),
            "held_window_ours": list(ours_w) if ours_w else None,
            "ours": pack(ours_w, ours, True),
            "rule": pack(rule_w, rule, False),
            "bo": pack(bo_w, boo, False),
        })
        print(f"[plot] {tid} L={L} ours={ours_w} rule={rule_w} bo={bo_w} native_dG={ndg}")

    out = Path(args.out_dir) / "cases.json"
    out.write_text(json.dumps(cases))
    print(f"[plot] wrote {out} ({len(cases)} cases)")
    return cases


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    d = Path(args.out_dir)
    cases = json.loads((d / "cases.json").read_text())
    n = len(cases)
    ncol = 4
    fig = plt.figure(figsize=(21, 5.2 * n))

    def trace(ax, CA, color, lw=2.2, label=None, alpha=1.0, ms=28):
        CA = np.asarray(CA)
        ax.plot(CA[:, 0], CA[:, 1], CA[:, 2], "-", color=color, lw=lw,
                alpha=alpha, label=label)
        ax.scatter(CA[:, 0], CA[:, 1], CA[:, 2], color=color, s=ms, alpha=alpha,
                   depthshade=False)

    def ring_bond(ax, CA, color):
        CA = np.asarray(CA)
        ax.plot([CA[0, 0], CA[-1, 0]], [CA[0, 1], CA[-1, 1]],
                [CA[0, 2], CA[-1, 2]], "--", color=color, lw=1.6, alpha=0.9)

    for ri, c in enumerate(cases):
        tid = c["target_id"]
        L = c["length"]
        nat = np.asarray(c["native_CA"])
        rec = np.asarray(c["receptor_CA"]) if c["receptor_CA"] else np.zeros((0, 3))
        hw = c["held_window_ours"]
        allpts = [nat]
        for k in ("ours", "rule", "bo"):
            if c.get(k):
                allpts.append(np.asarray(c[k]["CA"]))
        if len(rec):
            allpts.append(rec)
        P = np.concatenate(allpts, axis=0)
        lo, hi = P.min(0), P.max(0)
        ctr = (lo + hi) / 2
        rng = (hi - lo).max() / 2 * 1.1

        panels = [
            ("NATIVE  (bound linear peptide)", "native", "0.35"),
            ("BASELINE  placement_search ring", "rule", "#d1495b"),
            ("BASELINE  bo / GP-UCB ring", "bo", "#e8862a"),
            ("OURS  gpt-6-luna mask ring", "ours", "#2e6fb7"),
        ]
        for ci, (title, key, col) in enumerate(panels):
            ax = fig.add_subplot(n, ncol, ri * ncol + ci + 1, projection="3d")
            if len(rec):
                ax.scatter(rec[:, 0], rec[:, 1], rec[:, 2], color="0.7", s=8,
                           alpha=0.18, depthshade=False)
            if key == "native":
                trace(ax, nat, "0.4", lw=2.0)
                # orient: mark the residues OURS holds, on the native
                if hw:
                    seg = nat[hw[0]:hw[1] + 1]
                    trace(ax, seg, "#2ca25f", lw=3.4, ms=42)
                sub = f"native $\\Delta G$ = {c['native_dg']:.0f} REU" if c["native_dg"] is not None else ""
            else:
                cc = c[key]
                if cc is None:
                    ax.text2D(0.5, 0.5, "(no closing window)", ha="center",
                              transform=ax.transAxes)
                    ax.set_title(f"{tid}  ·  {title}", fontsize=10)
                    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
                    continue
                CA = np.asarray(cc["CA"])
                trace(ax, CA, col, lw=2.2)
                ring_bond(ax, CA, col)
                a, b = cc["window"]
                seg = CA[a:b + 1]
                trace(ax, seg, "#2ca25f", lw=3.4, ms=42)   # held window
                # clashes
                for i, j in cc["self_clash_pairs"]:
                    ax.plot([CA[i, 0], CA[j, 0]], [CA[i, 1], CA[j, 1]],
                            [CA[i, 2], CA[j, 2]], color="black", lw=2.4, zorder=5)
                    ax.scatter(*CA[[i, j]].T, c="red", s=95, marker="X",
                               edgecolors="black", linewidths=0.8, zorder=6,
                               depthshade=False)
                for i, jrec in cc["rec_clash_pairs"]:
                    ax.scatter(*CA[i], c="red", s=105, marker="X",
                               edgecolors="black", linewidths=0.8, zorder=6,
                               depthshade=False)
                clean = cc["clean"]
                dg = cc["binding_dg"]
                nclash = len(cc["self_clash_pairs"]) + len(cc["rec_clash_pairs"])
                dgtxt = f"{dg:.0f}" if dg is not None else "n/a"
                if clean:
                    sub = (f"hold[{a},{b}] rel {cc['release']} · clash-free ✓ "
                           f"(VALID ring)\n"
                           f"binding $\\Delta G$ = {dgtxt} REU")
                else:
                    sub = (f"hold[{a},{b}] rel {cc['release']} · CLASHES ✗ "
                           f"({nclash}) — INVALID ring\n"
                           f"binding $\\Delta G$ = {dgtxt} REU (not usable)")
            ax.set_title(f"{tid} (L={L})  ·  {title}", fontsize=10)
            ax.text2D(0.5, -0.02, sub, ha="center", va="top",
                      transform=ax.transAxes, fontsize=9)
            ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
            ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
            ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=ri * 25 + 30)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.grid(False)

    fig.suptitle(
        "LNR ring closure: native bound pose vs baselines (placement_search, bo) "
        "vs gpt-6-luna mask (ours)\n"
        "green = held window (keeps bound coords) · dashed = head-to-tail closure "
        "bond · red ✗ = steric clash · $\\Delta G$ = RAW native-scaled interface "
        "binding energy (REU, absolute; lower = stronger binding)",
        fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = d / "cases.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--mode", choices=["compute", "render", "all"], default="all")
    args = ap.parse_args()
    args.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.mode in ("compute", "all"):
        compute(args)
    if args.mode in ("render", "all"):
        render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
