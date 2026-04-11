#!/bin/bash
#SBATCH --job-name=numtok_fourier
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#
# Phase G Number-Token Fourier Screening
# CPU-only — activations already extracted. ~1-2 hours estimated.

set -euo pipefail

cd /home/anshulk/arithmetic-geometry

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geometry

echo "=== Number-Token Fourier Screening ==="
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo "CPUs:  ${SLURM_CPUS_PER_TASK:-24}"

python phase_g_numtok_fourier.py \
    --config config.yaml \
    --n-perms 1000 \
    --pca-dim 20

echo "End: $(date)"
echo "=== Done ==="
