#!/bin/bash
#SBATCH --job-name=phase_b
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Phase B — Concept Deconfounding
# ============================================================================
# CPU-only (no GPU). Label-level correlation diagnostics on coloring DFs.
# Runtime: under 1 minute for all levels.
#
# Inputs:
#   Coloring DFs: /data/user_data/anshulk/arithmetic-geometry/phase_a/coloring_dfs/
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_b/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_b/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase B — Concept Deconfounding"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/arithmetic-geometry || { echo "Failed to cd to workspace"; exit 1; }

echo "Activating conda environment..."
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Python packages
echo "  Checking Python packages..."
REQUIRED="numpy scipy yaml matplotlib pandas"
for pkg in $REQUIRED; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ERROR: Missing package: $pkg"
        exit 1
    fi
done
echo "  All packages available"

# Config
echo "  Checking config..."
if [ ! -f "config.yaml" ]; then
    echo "  ERROR: config.yaml not found"
    exit 1
fi

# Input data: coloring DFs
echo "  Checking coloring DataFrames..."
PKL_DIR="/data/user_data/anshulk/arithmetic-geometry/phase_a/coloring_dfs"
PKL_COUNT=$(ls "$PKL_DIR"/*.pkl 2>/dev/null | wc -l)
echo "  Coloring DFs: $PKL_COUNT .pkl files"
if [ "$PKL_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected 5 coloring DF files, found $PKL_COUNT"
    exit 1
fi

# Output directories
echo "  Creating output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_b
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_b
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PHASE B
# ============================================================================

echo "============================================================"
echo "Running Phase B"
echo "  Expected runtime: < 1 minute"
echo "============================================================"
echo ""

python phase_b_deconfounding.py --config config.yaml
PHASE_B_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_B_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_b"
PHASE_B_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_b"

echo "--- Data outputs ---"
for f in classified_pairs.csv spearman_comparison.csv deconfounding_plan.json summary.json; do
    path="$PHASE_B_DATA/$f"
    if [ -f "$path" ]; then
        SIZE=$(wc -c < "$path")
        echo "  $f: ${SIZE} bytes"
    else
        echo "  $f: MISSING"
    fi
done

CORR_COUNT=$(ls "$PHASE_B_DATA/correlation_matrices"/*.csv 2>/dev/null | wc -l)
echo "  Correlation matrix CSVs: $CORR_COUNT"

echo ""
echo "--- Plots ---"
PLOT_COUNT=$(ls "$PHASE_B_PLOTS"/*.png 2>/dev/null | wc -l)
echo "  Heatmap plots: $PLOT_COUNT"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $PHASE_B_EXIT"
echo "Total runtime: $SECONDS seconds"
echo ""

if [ $PHASE_B_EXIT -eq 0 ]; then
    echo "Phase B completed successfully"
    # Print decision from summary
    if [ -f "$PHASE_B_DATA/summary.json" ]; then
        echo ""
        echo "Decision: $(python -c "import json; print(json.load(open('$PHASE_B_DATA/summary.json'))['decision'])")"
    fi
    exit 0
else
    echo "PHASE B FAILED — check logs/phase_b_deconfounding.log"
    exit 1
fi
