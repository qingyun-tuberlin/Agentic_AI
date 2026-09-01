# HPC batch runs (SLURM + Singularity)

Runs both pipelines — **baseline (XAI off)** and **baseline_xai (XAI on)** —
over **all tasks × {clean, corrupted}** on a SLURM cluster, **each combination
once**, one at a time. Each worker calls the **GWDG endpoint directly with one
key** — no router. The GWDG request limits make the campaign rate-bound, so
`CONCURRENCY` defaults to 1.

With the 10 tasks currently in `datasets.yaml` that gives
`10 tasks × 2 datasets × 2 pipelines = 40` runs.

## Architecture

```
                 ┌─────────────────────────┐  one array, 0-39 %1:
                 │  run_task.sbatch (array) │  10 tasks × {clean,corrupt}
                 │      %1  (40 runs)       │  × {baseline, baseline_xai}
                 └────────────┬────────────┘
                              │ OpenAI-compatible, OPENAI_API_KEY = GWDG key
                              ▼
                   https://chat-ai.academiccloud.de/v1
```

* **Direct to GWDG, one key** — each worker sets `OPENAI_API_BASE` +
  `OPENAI_API_KEY` (`GWDG_API_KEY`, default key 1) via `SINGULARITYENV_`, and the
  agent's client-side litellm calls GWDG. No proxy, no endpoint file, no
  inter-node dependency.
* **Both pipelines in one array** — every element carries its pipeline in the
  manifest and sets `USE_XAI_CORRECTION` + the `ws_baseline` / `ws_baseline_xai`
  workspace accordingly, so a single campaign produces the matched A/B pairs.
* **One-at-a-time** — the array throttle `%1` (`CONCURRENCY` in `env.sh`); raise
  it only if you get more request headroom, since the campaign is rate-bound.
* **Isolation** — workers run with `--writable-tmpfs` (so runtime `pip install`s
  by generated code, caches, and `$HOME` go to a per-instance overlay) while
  workspace outputs land on the shared filesystem. `.adk` is shadowed per
  element so concurrent runs never collide.

## Deploy to the cluster (one-time, then re-sync on changes)

Everything lives on **BeeGFS scratch** (home is only ~20G) at
`/beegfs/scratch/<username>/mle-star`, and you run **from the project root**. The
jobs `module load singularity` and default to `--partition=standard` (the single
`cit` association is used automatically, so `--account` is omitted). These
commands assume an `hpc` alias in `~/.ssh/config` (→ TU Berlin gateway, user
`<username>`).

> [!NOTE]
> Make sure to replace the `<username>` placeholder in `hpc/env.sh`, `hpc/run_task.sbatch`, and this README with your actual TU Berlin cluster username before deploying or running the scripts.

Build the `.sif` **locally** and copy it over — a `.sif` is one portable file,
and the cluster may not allow image builds. Its architecture must match
(build on **x86-64** for an x86-64 cluster).

1. **Secrets** (gitignored — never committed; rsync carries it up in step 4):
   ```bash
   cp hpc/secrets.env.example hpc/secrets.env && chmod 600 hpc/secrets.env
   # edit hpc/secrets.env: GWDG_API_KEY_1 (the single key used for all runs;
   #                       key 1 has the highest daily budget, 1200/day)
   ```

2. **Build the image locally** (needs internet for the base image + `uv sync`;
   run from the repo root so the `.def`'s `%files` — `pyproject.toml`, `uv.lock`,
   `README.md` — resolve). The lockfile pins CPU torch, so the image is CPU-only:
   ```bash
   singularity build --fakeroot hpc/mle_star.sif hpc/mle_star.def
   #   ...if --fakeroot isn't configured:  sudo singularity build hpc/mle_star.sif hpc/mle_star.def
   ```

3. **Create the target dir on BeeGFS:**
   ```bash
   ssh hpc 'mkdir -p /beegfs/scratch/<username>/mle-star/hpc'
   ```

4. **Sync the source tree** (skip venv/git/caches, the big `.sif`, RAG, and
   remote-generated outputs; `tasks/` — ~2.3G incl. the local-only `_proxy`
   variants — DOES ship). `--delete` won't touch the excluded output dirs:
   ```bash
   rsync -avz --delete \
     --exclude='.git/' --exclude='.venv/' \
     --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
     --exclude='*.sif' \
     --exclude='/RAG/' \
     --exclude='/machine_learning_engineering/workspace/' \
     --exclude='/hpc/_run/' \
     ./ hpc:/beegfs/scratch/<username>/mle-star/
   ```

5. **Copy the image** (separate — it's large, and rsync skips `*.sif`):
   ```bash
   scp hpc/mle_star.sif hpc:/beegfs/scratch/<username>/mle-star/hpc/mle_star.sif
   ```

**Re-run loop:** after code/prompt edits, just re-run step 4 — the repo is
bind-mounted into the container at run time, so no rebuild is needed. Rebuild
(step 2) + re-copy (step 5) only when **dependencies** change (`pyproject.toml` /
`uv.lock`).

> **Gateway vs. frontend.** `ssh hpc` may land on a gateway that doesn't mount
> `/beegfs`. Check: `ssh hpc 'ls -d /beegfs/scratch/<username> && echo OK'`. If it
> fails, add a frontend alias with `ProxyJump hpc` in `~/.ssh/config` and target
> that instead.

The `CLUSTER SETTINGS` block in `hpc/env.sh` already defaults to this cluster
(`PARTITION=standard`, empty `ACCOUNT`, walltimes, CPUs/mem); override inline or
edit the block.

## Smoke test first (recommended)

Now on the cluster, from the project root. Validate the image, your key, the
model, and one full task run — direct to GWDG, exactly like a real worker:

```bash
ssh hpc
cd /beegfs/scratch/<username>/mle-star

bash hpc/smoke_test.sh --quick                          # seconds: ping model+key
srun --partition=standard --cpus-per-task=8 --mem=32G --time=01:00:00 \
     --pty bash hpc/smoke_test.sh                       # one full task end-to-end
```
It sends a tiny completion to `MODEL` with your key, then (unless `--quick`)
runs one task (`--task`, default `titanic`, single-solution `test` profile) into
`ws_smoke/`. A non-zero exit means don't submit yet — read the printed log.

## Submit

From the project root on the cluster:
```bash
cd /beegfs/scratch/<username>/mle-star
bash hpc/submit_all.sh
```
It prints the array job ID. Knobs (override inline):
```bash
CONCURRENCY=1 MODEL=devstral-2-123b-instruct-2512 bash hpc/submit_all.sh
```

## Monitor

```bash
squeue --me
tail -f hpc/_run/logs/mle-run-*.out            # a worker
```

## Outputs & evaluation

Each run writes to a per-pipeline workspace:
```
machine_learning_engineering/workspace/ws_baseline/<task>[_proxy]/final_state.json      # XAI off
machine_learning_engineering/workspace/ws_baseline_xai/<task>[_proxy]/final_state.json  # XAI on
```
The eval scripts read `ws_baseline` (→ "vanilla") and `ws_baseline_xai`
(→ "ours") and report the A/B deltas between the two pipelines:
```bash
python eval2.py
python scripts/scan_tasks_status.py     # completion grid
python scripts/print_xai_overhead.py    # token/cost overhead per run
```

## Notes / tuning

* **Re-running** — finished cells are skipped (the worker's idempotency guard
  checks for `final_state.json`), so resubmitting `submit_all.sh` only fills
  gaps left by failed runs. Key 1's ~1200 requests/day cap likely spreads the
  campaign over multiple days; just resubmit each day and it resumes cleanly.
* **Rate limits** — one key (10/min, 600/hour, 1200/day). If GWDG 429s, the
  agent's own `num_retries` retries; if it's persistent, lower the load
  (`CONCURRENCY` is already 1) or wait out the hour/day window.
* **Using a different / higher-budget key** — set `GWDG_API_KEY` in
  `hpc/secrets.env` (overrides the key-1 default), or run with
  `GWDG_API_KEY=... bash hpc/submit_all.sh`.
* **`--writable-tmpfs` disabled on your site?** Drop it from `run_task.sbatch`
  and instead point `SINGULARITYENV_PYTHONUSERBASE` at a writable per-element
  dir so generated-code `pip install`s have somewhere to go.
