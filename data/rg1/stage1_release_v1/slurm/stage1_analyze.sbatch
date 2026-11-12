#!/bin/bash
#RG-1 Stage 1 adjudication (afterany): records incomplete evidence too. 8 CPUs x 1.5 h = 12 core-h (in plan).
#SBATCH --job-name=rg1-s1-adj
#SBATCH --partition=capacity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/rg1_logs/%x_%j.out
#SBATCH --error=/scratch/%u/rg1_logs/%x_%j.err
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
: "${RG1_RELEASE:?}" "${RG1_DATA:?}" "${RG1_ARRAYS:?}"
PY="$(cat "$RG1_RELEASE/python_bin.txt")"
cd "$RG1_RELEASE"
OUT="$RG1_DATA/adjudication_${SLURM_JOB_ID}"
mkdir -p "$OUT"
sacct -j "$RG1_ARRAYS" -P --format=JobID,JobIDRaw,JobName%30,State,ExitCode,TotalCPU,AllocCPUS,ElapsedRaw,NodeList \
  </dev/null > "$OUT/sacct.txt"
"$PY" -m rg1.analyze_stage1 --plan "$RG1_RELEASE/PLAN.json" --data "$RG1_DATA" --sacct "$OUT/sacct.txt" \
  --out "$OUT" --workers 8
