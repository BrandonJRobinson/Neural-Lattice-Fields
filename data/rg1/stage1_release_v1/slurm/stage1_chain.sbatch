#!/bin/bash
#RG-1 Stage 1: one independent single-core chain per array task (PLAN.json task index = array index).
#The time limit is set per submission group by submit_stage1.sh (--time), never above the plan's value.
#SBATCH --job-name=rg1-s1
#SBATCH --partition=capacity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/rg1_logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/rg1_logs/%x_%A_%a.err
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
: "${RG1_RELEASE:?}" "${RG1_DATA:?}"
PY="$(cat "$RG1_RELEASE/python_bin.txt")"
SCR="${TMPDIR:-/local_scratch/$USER}"
mkdir -p "$SCR"
cd "$RG1_RELEASE"
exec "$PY" -m rg1.run_chain --plan "$RG1_RELEASE/PLAN.json" --task "${SLURM_ARRAY_TASK_ID:?}" \
  --binary "$RG1_RELEASE/../bin/phi4_local2" --data-root "$RG1_DATA" --scratch "$SCR"
