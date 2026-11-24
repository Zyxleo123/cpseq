# Publication figures — `pll_bg` vs measured pKd on CPSet Dataset I

Rendered artefacts, committed deliberately. They are the figure set for the
binding-metric result written up in [`../binding_metric_results.md`](../binding_metric_results.md),
kept in-repo so a reader of that doc sees the evidence without needing access to
`/zfsauton/scratch`. This is the one carve-out from the repo's "compute outputs
never in-repo" rule — see the note in `.gitignore`.

Each file is one plot answering one question; nothing is a multi-panel composite.

| File | Question it answers |
| --- | --- |
| `pll_cpset_agreement` | Does the proxy agree, across everything we scored? Every variant of every series, each series centered on its own mean. |
| `pll_cpset_tracking` | Per-series, does predicted track measured? z-scored pKd and `pll_bg` side by side, variants ordered by measured affinity. |
| `pll_cpset_s03` | The showcase series (S03) on raw axes, with the error bar that template choice contributes. |
| `pll_cpset_concordance` | Is it chance? Pooled within-series pair concordance against a within-series permutation null, plus the wrong-receptor specificity control. |
| `pll_cpset_sar` | Is it right for the right reason? Per-position log p for the series whose measured SAR is known and local. |
| `pll_cpset_series_rho` | Per-series rho — the unit of evidence that needs no independence assumption — for candidate, incumbent and composition control. |

Both `.pdf` (vector, for print) and `.png` (600 dpi) are committed.
`pll_cpset_values.json` carries every plotted value, so a caption can cite a
number without retyping it from the doc.

## Provenance

```
model   ligandmpnn_v_32_020_25
rows    /zfsauton/scratch/yixiz/lp2cp_stage2/lnr/bindscore_v2.jsonl
decoy   /zfsauton/scratch/yixiz/lp2cp_stage2/lnr/bindscore_v2_decoy.jsonl
n_perm  20000
```

Headline: rho = +0.52, P = 0.005 pooled over 32 peptides / 9 series; head-to-tail
only rho = +0.80; Rosetta dG rho = -0.06 on the same points.

## Regenerating

Do not hand-edit these. They are redrawn with no recomputation — every statistic
is computed by `lnr.metric_gate`, the same referee module behind the table in the
results doc, so a figure cannot drift from the result:

```
sbatch/submit_lnr_fig_pll_cpset.sh          # CPU partition; writes $RUN/lnr/figs
```

then copy the refreshed `pll_cpset_*` files over the ones here. Source:
[`../../agent/lnr/fig_pll_cpset.py`](../../agent/lnr/fig_pll_cpset.py).
