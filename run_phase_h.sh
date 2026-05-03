#!/bin/bash
#SBATCH --job-name=phase_h
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Phase H / B.2 — Orthogonalization control for carry helix superposition
# ============================================================================
# Runs Phase G's Fourier test again on the curated 8,264-problem set, then runs
# the same test after projecting out known algebraic correlate subspaces.
#
# Reads:
#   - curated/curated_set_v1.json
#   - phase_b/classified_pairs.csv
#   - phase_c/residualized/*.npy
#   - phase_c/subspaces/**/basis.npy and projected_all.npy
#   - phase_d/subspaces/**/merged_basis.npy
#   - phase_g/summary/phase_g_helices.csv
#
# Outputs:
#   Data: /data/user_data/anshulk/arithmetic-geometry/phase_h/
#   Logs: /home/anshulk/arithmetic-geometry/logs/phase_h_orthogonalize.log
# ============================================================================

set -euo pipefail

echo "============================================================"
echo "Phase H / B.2 — Orthogonalization Control"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================================"

cd /home/anshulk/arithmetic-geometry
mkdir -p logs

echo "Activating conda environment..."
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

echo ""
echo "Running pre-flight checks..."

for path in \
    config.yaml \
    /data/user_data/anshulk/arithmetic-geometry/curated/curated_set_v1.json \
    /data/user_data/anshulk/arithmetic-geometry/phase_b/classified_pairs.csv \
    /data/user_data/anshulk/arithmetic-geometry/phase_g/summary/phase_g_helices.csv
do
    if [ ! -f "$path" ]; then
        echo "ERROR: Missing required file: $path"
        exit 1
    fi
done

RESID_COUNT=$(find /data/user_data/anshulk/arithmetic-geometry/phase_c/residualized -name '*.npy' 2>/dev/null | wc -l)
echo "  Residualized activation files: $RESID_COUNT"
if [ "$RESID_COUNT" -lt 27 ]; then
    echo "ERROR: Expected residualized activation files for L3-L5/layers."
    exit 1
fi

PHASE_C_BASIS_COUNT=$(find /data/user_data/anshulk/arithmetic-geometry/phase_c/subspaces -name 'basis.npy' 2>/dev/null | wc -l)
PHASE_D_BASIS_COUNT=$(find /data/user_data/anshulk/arithmetic-geometry/phase_d/subspaces -name 'merged_basis.npy' 2>/dev/null | wc -l)
echo "  Phase C bases: $PHASE_C_BASIS_COUNT"
echo "  Phase D merged bases: $PHASE_D_BASIS_COUNT"
if [ "$PHASE_C_BASIS_COUNT" -lt 100 ]; then
    echo "ERROR: Too few Phase C bases found."
    exit 1
fi

mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_h/summary
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_h/orthogonalize

echo ""
echo "Step 1: Toy math validation..."
python phase_h_orthogonalize.py --config config.yaml --toy

echo ""
echo "Step 2: B.2 curated-set orthogonalization run..."
python phase_h_orthogonalize.py --config config.yaml --n-perms 1000

echo ""
echo "============================================================"
echo "Phase H / B.2 COMPLETE"
echo "End: $(date)"
echo "============================================================"
