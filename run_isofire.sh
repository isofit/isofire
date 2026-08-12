#!/bin/bash
#SBATCH --job-name=isofire_av3_onboard
#SBATCH --output=isofire_av3_onboard_%A.out
#SBATCH --error=isofire_av3_onboard_%A.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --partition=gpuq
#SBATCH --time=00:30:00

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO"

INPUT_DIR=${INPUT_DIR:-tests/data/rtsp/l1b}

# Products are written unwarped as ENVI pairs for a downstream ortho step:
# <scene_id>_fireTemp_toOrtho[.hdr] and <scene_id>_fireTempUnc_toOrtho[.hdr].
TO_ORTHO_DIR=${TO_ORTHO_DIR:-results/to_ortho}

POLL_SECONDS=${POLL_SECONDS:-10}
SCENE_TIMEOUT=${SCENE_TIMEOUT:-300}
SETTLE_SECONDS=${SETTLE_SECONDS:-5}

mkdir -p "$TO_ORTHO_DIR"

echo "===== Host ====="
hostname; date

echo "===== GPU ====="
nvidia-smi || { echo "ERROR: no GPU on this allocation"; exit 1; }

echo "===== JAX devices ====="
uv run python -c "import jax; print('JAX devices:', jax.devices())"

echo "===== Discovery preview ====="
# Onboard requires rdn/loc/mask/obs only; GLT is not needed.
uv run python -c "
from isofire.onboard import discover_ready_scenes
ready, partial = discover_ready_scenes(input_dir='${INPUT_DIR}', settle_seconds=${SETTLE_SECONDS})
print(f'ready: {list(ready)}')
print(f'partial: {dict(partial)}')"

echo "===== Running retrieval (pre-ortho, ENVI out) ====="
# -1 marks in-scene pixels that were not retrieved, leaving 0 for the ortho step
# to mark out-of-swath.
uv run python -m isofire.onboard \
    --input_dir     "$INPUT_DIR" \
    --to_ortho_dir  "$TO_ORTHO_DIR" \
    --poll_seconds  "$POLL_SECONDS" \
    --scene_timeout_seconds "$SCENE_TIMEOUT" \
    --settle_seconds        "$SETTLE_SECONDS" \
    --no_exit_on_timeout \
    --fill_value -1.0 \
    --alpha_tuning 0.25 \
    --hfdi_filter \
    --hfdi_lo_nm 2061.0 \
    --hfdi_hi_nm 2430.0 \
    --hfdi_min -0.1 \
    --eta_sat_frac_low 0.02 \
    --eta_sat_frac_high 0.08 \
    --eta_floor_swir2_only_sat 0.75 \
    --eta_floor_swir1_and_swir2_sat 0.85 \
    --eta_ceiling_low_sat 1.0 \
    --eta_floor_max 0.9 \
    || EXIT_CODE=$?

echo "===== Outputs ====="
ls -lh "$TO_ORTHO_DIR"
date

echo "--- inputs retained ($INPUT_DIR) ---"; ls -1 "$INPUT_DIR" | wc -l

# 0 = clean exit, 1 = a scene failed or timed out, 143 = SIGTERM at the time limit.
exit "${EXIT_CODE:-0}"