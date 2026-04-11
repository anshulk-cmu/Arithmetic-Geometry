#!/bin/bash
#SBATCH --job-name=phase_g
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#
# Phase G full pipeline: extraction (GPU) then analysis (CPU-only).
#
# Step 1: K&T pilot         — GPU, ~30 min
# Step 2: Number-token ext  — GPU, ~4 hours
# Step 3: Synthetic pilot   — CPU, ~1 min
# Step 4: Pilot 0b          — CPU, ~2 min
# Step 5: Full Fourier run  — CPU, ~6-7 hours

set -euo pipefail

echo "========================================="
echo "Phase G: Full Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================="

cd /home/anshulk/arithmetic-geometry
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

# ── Step 1: K&T replication pilot (GPU) ──
echo ""
echo "Step 1: K&T replication pilot..."
python phase_g_kt_pilot.py --config config.yaml
KT_EXIT=$?
if [ $KT_EXIT -ne 0 ]; then
    echo "ERROR: K&T pilot failed (exit code $KT_EXIT). Aborting."
    exit 1
fi
echo "K&T pilot complete."

# ── Step 2: Number-token activation extraction (GPU) ──
echo ""
echo "Step 2: Number-token activation extraction..."
python extract_number_token_acts.py --config config.yaml
NT_EXIT=$?
if [ $NT_EXIT -ne 0 ]; then
    echo "ERROR: Number-token extraction failed (exit code $NT_EXIT). Aborting."
    exit 1
fi
echo "Number-token extraction complete."

# ── Step 3: Synthetic pilot (CPU) ──
echo ""
echo "Step 3: Synthetic pilot tests..."
python phase_g_fourier.py --config config.yaml --pilot
PILOT_EXIT=$?
if [ $PILOT_EXIT -ne 0 ]; then
    echo "ERROR: Synthetic pilot failed (exit code $PILOT_EXIT). Aborting."
    exit 1
fi
echo "Synthetic pilot PASSED."

# ── Step 4: Pilot 0b — raw vs residualized spot check (CPU) ──
echo ""
echo "Step 4: Pilot 0b (raw vs residualized)..."
python phase_g_fourier.py --config config.yaml --pilot-0b
P0B_EXIT=$?
if [ $P0B_EXIT -ne 0 ]; then
    echo "ERROR: Pilot 0b failed (exit code $P0B_EXIT). Aborting."
    exit 1
fi
echo "Pilot 0b complete."

# ── Step 5: Full Fourier screening run (CPU) ──
echo ""
echo "Step 5: Full Phase G Fourier screening..."
python phase_g_fourier.py --config config.yaml --n-perms 1000
FULL_EXIT=$?
if [ $FULL_EXIT -ne 0 ]; then
    echo "ERROR: Full run failed (exit code $FULL_EXIT)."
    exit 1
fi

echo ""
echo "========================================="
echo "Phase G COMPLETE"
echo "Date: $(date)"
echo "========================================="
