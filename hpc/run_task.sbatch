#!/bin/bash
#SBATCH --job-name=mle-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=hpc/_run/logs/%x-%A_%a.out
#SBATCH --error=hpc/_run/logs/%x-%A_%a.err
# NOTE: log paths above are relative to the submit dir, so submit from the project
#   root:  cd /beegfs/scratch/<username>/mle-star && bash hpc/submit_all.sh
# One array element = one (task, dataset, pipeline) run, where pipeline is
# baseline (XAI off) or baseline_xai (XAI on). Each combination runs exactly
# once. --array, resources, and time are supplied by submit_all.sh.
# The array is throttled with %CONCURRENCY so at most that many run at once.
# Each worker calls GWDG directly with one key — no router.
set -euo pipefail

module load singularity

: "${REPO:?}" "${SIF:?}" "${MANIFEST:?}" "${RUN_DIR:?}" \
  "${MODEL:?}" "${GWDG_API_BASE:?}" "${GWDG_API_KEY:?}"

# ---- 1. Resolve this element's task, dataset and pipeline from manifest ---
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
read -r TASK DATASET PIPELINE < <(sed -n "${LINE_NO}p" "$MANIFEST")
if [ -z "${TASK:-}" ] || [ -z "${DATASET:-}" ] || [ -z "${PIPELINE:-}" ]; then
    echo "[worker] bad/empty manifest line ${LINE_NO} in $MANIFEST" >&2
    exit 1
fi
case "$DATASET" in
    clean)     FLAG="--clean" ;;
    corrupted) FLAG="--corrupted" ;;
    *) echo "[worker] bad dataset '$DATASET' on line ${LINE_NO}" >&2; exit 1 ;;
esac
# Pipeline variant -> XAI flag + workspace prefix, matching auto_run_all_tasks'
# VARIANTS convention (baseline=off -> ws_baseline, baseline_xai=on -> ws_baseline_xai).
case "$PIPELINE" in
    baseline)     XAI="False"; WS_PREFIX="ws_baseline" ;;
    baseline_xai) XAI="True";  WS_PREFIX="ws_baseline_xai" ;;
    *) echo "[worker] bad pipeline '$PIPELINE' on line ${LINE_NO}" >&2; exit 1 ;;
esac
echo "[worker] idx=${SLURM_ARRAY_TASK_ID} task=${TASK} dataset=${DATASET} pipeline=${PIPELINE} node=$(hostname)"

# ---- 1b. Idempotency guard ----------------------------------------------
# The pipeline writes final_state.json only on successful completion (its
# after_agent_callback). If it already exists for this exact (pipeline, task,
# dataset), this run is done — skip it (safe for reruns / requeues). A crashed
# run leaves no final_state.json, so it will be retried.
WS="machine_learning_engineering/workspace/${WS_PREFIX}"
TASK_DIR="$TASK"; [ "$DATASET" = corrupted ] && TASK_DIR="${TASK}_proxy"
FINAL_STATE="$REPO/$WS/${TASK_DIR}/final_state.json"
if [ -f "$FINAL_STATE" ]; then
    echo "[worker] SKIP: ${FINAL_STATE} already exists (idempotent)."
    exit 0
fi

# ---- 2. Per-element scratch + isolated ADK dir ---------------------------
# Workspace outputs go to the shared FS (under the bind-mounted repo) where the
# eval scripts look; everything else (pip installs at runtime, caches, home)
# lives in the per-instance --writable-tmpfs overlay. The one shared path inside
# the repo that ADK might write, .adk, is shadowed by a per-element bind so two
# concurrent runs can never collide on it.
JOB_TMP="$RUN_DIR/jobtmp/idx${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMP/adk"
# WS was set by the idempotency guard above.

# ---- 3. Route + config, delivered into the container via SINGULARITYENV_ --
# config.py / llm.build_llm read these straight from os.environ, so no .env is
# needed (and none is present in a fresh checkout). Direct to GWDG, one key.
export SINGULARITYENV_OPENAI_API_BASE="${GWDG_API_BASE}"
export SINGULARITYENV_OPENAI_API_KEY="${GWDG_API_KEY}"
export SINGULARITYENV_ROOT_AGENT_MODEL="${MODEL}"
export SINGULARITYENV_MLE_PROFILE="google"          # 2 solutions, 2 candidates, ...
export SINGULARITYENV_USE_XAI_CORRECTION="${XAI}"   # False for baseline, True for baseline_xai
export SINGULARITYENV_USE_XAI_REFINEMENT="False"
export SINGULARITYENV_MLE_USE_RAG="0"
export SINGULARITYENV_MLE_WORKSPACE_DIR="${WS}"

# ---- 4. Run exactly one (task, dataset) into the pipeline's workspace ----
# No --variant/--matrix flag: single-pass mode uses the ambient environment, so
# MLE_WORKSPACE_DIR and USE_XAI_CORRECTION above take effect. --task selects the
# single task; --clean/--corrupted selects the dataset variant.
singularity exec --writable-tmpfs \
    --bind "$REPO" \
    --bind "$JOB_TMP/adk:$REPO/machine_learning_engineering/.adk" \
    --pwd "$REPO" \
    "$SIF" \
    python machine_learning_engineering/auto_run_all_tasks.py --task "$TASK" $FLAG

echo "[worker] done task=${TASK} dataset=${DATASET} pipeline=${PIPELINE} -> ${WS}/"
