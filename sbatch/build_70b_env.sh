#!/usr/bin/env bash
# Overlay venv for the 4-bit 70B arm: adds `accelerate` and `bitsandbytes` on
# top of the project venv without writing to it.
#
# Why an overlay rather than installing into the project venv, or building a
# standalone one: the project venv is full of deliberate pins and every other
# stage of this project runs in it, so a resolver that decided to move torch
# would be discovered by a training run. A standalone venv would have to
# replicate the real dependency set (transformers, tokenizers, safetensors,
# the editable installs) -- the project env with extra steps.
#
# The mechanism is a `.pth` file in the overlay's site-packages listing the
# paths the project venv adds. Not `--system-site-packages`: measured on this
# cluster, that flag does not chain venv to venv -- `uv venv --python <project
# venv>/bin/python` resolves through the symlink to the uv-managed CPython
# underneath, so "system" site-packages means THAT interpreter's, which holds
# nothing. A `.pth` also gets the search ORDER right: the paths it adds are
# appended, so the overlay's own packages win while everything else falls
# through to the project's known-good copies.
#
# This is a SEPARATE overlay from `envs/llmprop`, which belongs to the sibling
# cpseq work. Adding bitsandbytes there would change an environment other
# experiments run in.
#
#   bash build_70b_env.sh            # build (idempotent)
#   bash build_70b_env.sh --verify   # check only

set -euo pipefail

PROJ="${PROJ:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ENV_DIR="${ENV_DIR:-/zfsauton/scratch/yixiz/envs/lp2cp_70b}"
VENV="${ENV_DIR}/.venv"
BASE_PY="${BASE_PY:-${PROJ}/.venv/bin/python}"
UV="${UV:-${HOME}/.local/bin/uv}"
command -v "$UV" >/dev/null 2>&1 || UV="$(command -v uv)"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

[ -x "$BASE_PY" ] || { echo "FATAL: no project venv python at $BASE_PY" >&2; exit 1; }

# Every path the project venv ADDS over a bare interpreter, in its own order.
# Asked of the interpreter rather than assembled by hand: editable installs
# contribute several entries via .pth import hooks that a chained path does not
# re-run, so they have to be copied out explicitly. `env -u PYTHONPATH` so that
# whatever the caller happened to have set is not baked into the overlay.
BASE_PATHS="$(env -u PYTHONPATH "$BASE_PY" -c '
import sys
base = sys.base_prefix
for p in sys.path:
    if not p or p.endswith(".zip") or p.endswith("lib-dynload"):
        continue
    if p.startswith(base):
        continue
    print(p)
')"
[ -n "$BASE_PATHS" ] || { echo "FATAL: project venv contributed no paths" >&2; exit 1; }

if [ "$VERIFY_ONLY" = "0" ]; then
  mkdir -p "$ENV_DIR"
  echo "=== building overlay venv at $VENV"
  "$UV" venv --python "$BASE_PY" "$VENV"
  # --no-deps is deliberate: every requirement of both packages (torch, numpy,
  # packaging, psutil, pyyaml, huggingface_hub, safetensors) is already
  # satisfied by the inherited site-packages, and letting the resolver see them
  # risks it pulling a different torch into the overlay, which would shadow the
  # project's known-good 2.7.0+cu126 for exactly the job that needs the GPU.
  echo "=== installing accelerate + bitsandbytes (no deps)"
  "$UV" pip install --python "${VENV}/bin/python" --no-deps accelerate bitsandbytes
  OVERLAY_SITE="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "=== chaining to the project venv:"
  printf '%s\n' "$BASE_PATHS" | sed 's/^/      /'
  # `_zz_` so it sorts last: .pth files run in directory order and this one
  # must not run before the overlay's own.
  printf '%s\n' "$BASE_PATHS" > "${OVERLAY_SITE}/_zz_project_venv.pth"
fi

echo "=== verifying"
"${VENV}/bin/python" - "$BASE_PY" <<'PY'
import json, subprocess, sys
from pathlib import Path

base_py = sys.argv[1]
probe = (
    "import json, torch, transformers, numpy;"
    "print(json.dumps({'torch': torch.__version__,"
    " 'transformers': transformers.__version__, 'numpy': numpy.__version__}))"
)
mine = json.loads(subprocess.run([sys.executable, "-c", probe],
                                 capture_output=True, text=True, check=True).stdout)
theirs = json.loads(subprocess.run([base_py, "-c", probe],
                                   capture_output=True, text=True, check=True).stdout)
bad = {k: (theirs[k], mine[k]) for k in theirs if theirs[k] != mine[k]}
if bad:
    print("FATAL: the overlay shadowed the project's stack:")
    for k, (t, m) in bad.items():
        print(f"  {k}: project={t}  overlay={m}")
    raise SystemExit(1)

import accelerate            # noqa: E402
import bitsandbytes as bnb   # noqa: E402

print(f"OK  torch={mine['torch']}  transformers={mine['transformers']}  "
      f"numpy={mine['numpy']} (unchanged from the project venv)")
print(f"OK  accelerate={accelerate.__version__} (overlay only)")
print(f"OK  bitsandbytes={bnb.__version__} (overlay only)")

prov = {
    "python": sys.version.split()[0],
    "torch": mine["torch"],
    "transformers": mine["transformers"],
    "numpy": mine["numpy"],
    "accelerate": accelerate.__version__,
    "bitsandbytes": bnb.__version__,
    "base_venv": str(Path(base_py).parent.parent),
    "rule": ".pth chain to project venv + accelerate/bitsandbytes --no-deps",
}
Path("/tmp/_lp2cp70b_prov.json").write_text(json.dumps(prov))
PY

cp /tmp/_lp2cp70b_prov.json "${ENV_DIR}/env_provenance.json"
rm -f /tmp/_lp2cp70b_prov.json
echo "=== wrote ${ENV_DIR}/env_provenance.json"
echo "=== use: PY=${VENV}/bin/python"
