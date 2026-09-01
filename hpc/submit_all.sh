#!/bin/bash
# Submit the campaign: one array job running every (task, dataset, pipeline)
# once, throttled to CONCURRENCY simultaneous tasks. Each worker calls GWDG
# directly with one key — no router.
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HPC_DIR/env.sh"

# ---- Preflight -----------------------------------------------------------
[ -f "$SIF" ] || { echo "ERROR: image not found: $SIF" >&2
    echo "  build it from the repo root:  singularity build hpc/mle_star.sif hpc/mle_star.def" >&2; exit 1; }
: "${GWDG_API_KEY:?ERROR: set GWDG_API_KEY_1 (copy hpc/secrets.env.example -> hpc/secrets.env)}"
command -v sbatch >/dev/null || { echo "ERROR: sbatch not on PATH (not a SLURM login node?)" >&2; exit 1; }
module load singularity 2>/dev/null || true   # needed for the manifest-gen exec below
if [ -f "$REPO/.env" ]; then
    echo "WARNING: $REPO/.env exists; if ADK auto-loads it, it could override OPENAI_API_BASE/KEY." >&2
    echo "         Remove it or make sure it points at GWDG with the intended key." >&2
fi

mkdir -p "$RUN_DIR/logs"

# ---- Manifest + array size ----------------------------------------------
singularity exec --bind "$REPO" "$SIF" python "$HPC_DIR/gen_run_manifest.py" > "$MANIFEST"
NTASKS=$(grep -c . "$MANIFEST")
[ "$NTASKS" -gt 0 ] || { echo "ERROR: empty manifest ($MANIFEST)" >&2; exit 1; }
ARRAY_MAX=$((NTASKS - 1))
echo "Campaign: ${NTASKS} runs ($((NTASKS / 4)) tasks x 2 datasets x 2 pipelines [baseline + baseline_xai]), ${CONCURRENCY} at a time"

# Propagate everything the job scripts need (sbatch --export=ALL is the default).
export REPO SIF MANIFEST RUN_DIR MODEL GWDG_API_BASE GWDG_API_KEY HPC_DIR

# Optional flags only added when set (empty ACCOUNT/PARTITION => omitted).
EXTRA=()
[ -n "$ACCOUNT" ]   && EXTRA+=(--account="$ACCOUNT")
[ -n "$PARTITION" ] && EXTRA+=(--partition="$PARTITION")

# ---- Worker array --------------------------------------------------------
AJOB=$(sbatch --parsable "${EXTRA[@]}" \
    --array="0-${ARRAY_MAX}%${CONCURRENCY}" --time="$TIME_WORKER" \
    --cpus-per-task="$WORKER_CPUS" --mem="$WORKER_MEM" \
    "$HPC_DIR/run_task.sbatch")
echo "Array job:   $AJOB"

echo
echo "Submitted. Monitor with:  squeue --me    |    tail -f $RUN_DIR/logs/*.out"
echo "Outputs:   $REPO/machine_learning_engineering/workspace/{ws_baseline,ws_baseline_xai}/<task>[_proxy]/"
