#!/bin/bash
# Shared configuration for the SLURM run. Sourced by submit_all.sh and, via the
# inherited job environment (sbatch --export=ALL, the default), by the job
# scripts too. Edit the CLUSTER SETTINGS block for your site.

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HPC_DIR
# Repo root = project dir. Keep the checkout on BeeGFS scratch (home is only
# ~20G), e.g. /beegfs/scratch/<username>/mle-star — this resolves to wherever the
# repo actually lives, so submit from the project root.
export REPO="$(cd "$HPC_DIR/.." && pwd)"
export PROJECT="$REPO"

# ---- Artifacts / coordination (must live on the shared filesystem) ----
export RUN_DIR="$HPC_DIR/_run"                       # logs, manifest
export MANIFEST="$RUN_DIR/run_manifest.txt"          # ordered (task, dataset, pipeline) list
export SIF="${SIF:-$HPC_DIR/mle_star.sif}"           # built container image

# ---- Model / LLM endpoint (workers call GWDG directly, one key, no router) ----
export MODEL="${MODEL:-devstral-2-123b-instruct-2512}"  # ROOT_AGENT_MODEL for workers
export GWDG_API_BASE="${GWDG_API_BASE:-https://chat-ai.academiccloud.de/v1}"

# ---- Run shape ----
export CONCURRENCY="${CONCURRENCY:-1}"               # simultaneous tasks (rate-bound by GWDG limits, so 1 is as fast as more)

# ============================ CLUSTER SETTINGS ============================
# Defaults match the GWDG/BeeGFS cluster (partition=standard, single `cit`
# association used by default so --account is omitted). Override per site.
export ACCOUNT="${ACCOUNT:-}"                        # empty => omit (cit is the default association)
export PARTITION="${PARTITION:-standard}"           # SLURM partition
export TIME_WORKER="${TIME_WORKER:-08:00:00}"       # walltime per task
export WORKER_CPUS="${WORKER_CPUS:-8}"
export WORKER_MEM="${WORKER_MEM:-32G}"
# =========================================================================

# ---- Secrets (never committed; see hpc/secrets.env.example) ----
if [ -f "$HPC_DIR/secrets.env" ]; then
    # shellcheck disable=SC1091
    source "$HPC_DIR/secrets.env"
fi
# The single key used for all runs. Prefer an explicit GWDG_API_KEY, else fall
# back to GWDG_API_KEY_1 (key 1 has the highest daily budget: 1200/day).
export GWDG_API_KEY="${GWDG_API_KEY:-${GWDG_API_KEY_1:-}}"
