#!/usr/bin/env bash
#SBATCH --job-name=s2_probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_probe_%j.out
set -euo pipefail
. /zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh "$@"
banner
for M in claude-opus-5-5 claude-sonnet-5-5; do
  echo "=== $M ==="
  "$PY_API" "$AGENT/refusal_probe.py" --model "$M" \
    --parents "$BENCH_DIR/parents_public.json" \
    --out "$RUN/results/refusal_probe__${M}.json" --max-spend-usd 0.15
done
echo "DONE probe"
