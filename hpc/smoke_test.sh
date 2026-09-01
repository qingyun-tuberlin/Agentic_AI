#!/bin/bash
# Pre-flight smoke test: validate the container, the GWDG key, the chosen model,
# and one full task run BEFORE submitting the campaign. Calls GWDG directly (no
# router) with GWDG_API_KEY, exactly like the real workers.
#
# Run where there is internet + some compute (the full task run is heavy):
#   srun --pty --time=01:00:00 [--partition=... --account=...] bash hpc/smoke_test.sh
# The fast key/model check alone is fine on any node with internet:
#   bash hpc/smoke_test.sh --quick
#
# Options:
#   --quick          only ping the model (seconds), skip the full task
#   --task NAME      task to run end-to-end (default: titanic)
#   --variant V      clean | corrupted (default: clean)
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HPC_DIR/env.sh"

QUICK=0
SMOKE_TASK="${SMOKE_TASK:-titanic}"
SMOKE_VARIANT="clean"
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)   QUICK=1 ;;
        --task)    SMOKE_TASK="${2:?--task needs a value}"; shift ;;
        --variant) SMOKE_VARIANT="${2:?--variant needs a value}"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---- Preflight -----------------------------------------------------------
[ -f "$SIF" ] || { echo "ERROR: image missing: $SIF (build it first)" >&2; exit 1; }
module load singularity 2>/dev/null || true
: "${GWDG_API_KEY:?set GWDG_API_KEY_1 in hpc/secrets.env}"

SMOKE_DIR="$RUN_DIR/smoke"
mkdir -p "$SMOKE_DIR"

# ---- 1. Validate the key + model with a tiny completion (direct to GWDG) --
echo "[smoke] pinging model '${MODEL}' at ${GWDG_API_BASE} ..."
HTTP=$(curl -sS -o "$SMOKE_DIR/ping.json" -w '%{http_code}' \
    "${GWDG_API_BASE}/chat/completions" \
    -H "Authorization: Bearer ${GWDG_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":5}")
echo "[smoke] HTTP ${HTTP}; response:"
cat "$SMOKE_DIR/ping.json"; echo
if [ "$HTTP" != "200" ]; then
    echo "[smoke] FAIL: model/key did not return 200 (see response above)" >&2
    exit 1
fi
echo "[smoke] model + key OK."

if [ "$QUICK" = 1 ]; then
    echo "[smoke] --quick: skipping the full task run. PASS."
    exit 0
fi

# ---- 2. One full task end-to-end (direct to GWDG, like a real worker) ----
case "$SMOKE_VARIANT" in
    clean)     FLAG=--clean ;;
    corrupted) FLAG=--corrupted ;;
    *) echo "[smoke] bad --variant '$SMOKE_VARIANT'" >&2; exit 1 ;;
esac
JOB_TMP="$SMOKE_DIR/jobtmp"
mkdir -p "$JOB_TMP/adk"
WS="machine_learning_engineering/workspace/ws_smoke"
TASK_DIR="$SMOKE_TASK"; [ "$SMOKE_VARIANT" = corrupted ] && TASK_DIR="${SMOKE_TASK}_proxy"
echo "[smoke] running task '${SMOKE_TASK}' (${SMOKE_VARIANT}) -> ${WS}/ ..."

# MLE_PROFILE=test keeps this to a single solution/candidate so the smoke run is
# fast; the real campaign uses the larger google profile.
export SINGULARITYENV_OPENAI_API_BASE="${GWDG_API_BASE}"
export SINGULARITYENV_OPENAI_API_KEY="${GWDG_API_KEY}"
export SINGULARITYENV_ROOT_AGENT_MODEL="${MODEL}"
export SINGULARITYENV_MLE_PROFILE="test"
export SINGULARITYENV_USE_XAI_CORRECTION="True"
export SINGULARITYENV_USE_XAI_REFINEMENT="False"
export SINGULARITYENV_MLE_USE_RAG="0"
export SINGULARITYENV_MLE_WORKSPACE_DIR="${WS}"

singularity exec --writable-tmpfs \
    --bind "$REPO" \
    --bind "$JOB_TMP/adk:$REPO/machine_learning_engineering/.adk" \
    --pwd "$REPO" \
    "$SIF" \
    python machine_learning_engineering/auto_run_all_tasks.py --task "$SMOKE_TASK" $FLAG

FS="$REPO/$WS/${TASK_DIR}/final_state.json"
if [ -f "$FS" ]; then
    echo "[smoke] PASS: task completed; final_state.json written at:"
    echo "        $FS"
    echo "[smoke] You're clear to run the full campaign:  bash hpc/submit_all.sh"
else
    echo "[smoke] FAIL: task finished but no final_state.json at $FS (check output above)" >&2
    exit 1
fi
