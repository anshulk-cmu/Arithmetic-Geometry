#!/bin/bash
#SBATCH --job-name=phase_c
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Phase C — Concept Subspace Identification
# ============================================================================
# GPU-accelerated permutation null via CuPy on A6000.
# Expected runtime: ~15-30 min with GPU, ~3-4 hours CPU-only.
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_c/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_c/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase C — Concept Subspace Identification"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/arithmetic-geometry || { echo "Failed to cd to workspace"; exit 1; }

echo "Activating conda environment..."
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Python packages
echo "  Checking Python packages..."
REQUIRED="numpy scipy sklearn yaml matplotlib pandas"
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

# Input data: activations (45 .npy) + coloring DFs (5 .pkl)
echo "  Checking input data..."
ACT_DIR="/data/user_data/anshulk/arithmetic-geometry/activations"
PKL_DIR="/data/user_data/anshulk/arithmetic-geometry/phase_a/coloring_dfs"
ACT_COUNT=$(ls "$ACT_DIR"/*.npy 2>/dev/null | wc -l)
PKL_COUNT=$(ls "$PKL_DIR"/*.pkl 2>/dev/null | wc -l)
echo "  Activations: $ACT_COUNT .npy files"
echo "  Coloring DFs: $PKL_COUNT .pkl files"
if [ "$ACT_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 activation files (5 levels x 9 layers), found $ACT_COUNT"
    exit 1
fi
if [ "$PKL_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected 5 coloring DF files, found $PKL_COUNT"
    exit 1
fi

# Output directories
echo "  Creating output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_c
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_c
echo "  Output directories ready"

# Clean stale phase_c outputs from previous runs (different data sizes
# would cause the resume logic to load wrong cached results)
echo "  Cleaning stale phase_c outputs..."
PHASE_C_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_c"
if [ -d "$PHASE_C_DATA" ]; then
    rm -rf "$PHASE_C_DATA"
    echo "  Removed $PHASE_C_DATA"
fi
mkdir -p "$PHASE_C_DATA"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PHASE C
# ============================================================================

echo "============================================================"
echo "Running Phase C"
echo "  Expected runtime: ~15-30 min (GPU-accelerated)"
echo "============================================================"
echo ""

python phase_c_subspaces.py --config config.yaml --n-jobs 1
PHASE_C_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_C_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_c"
PHASE_C_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_c"

echo "--- Data outputs ---"
SUBSPACE_COUNT=$(find "$PHASE_C_DATA/subspaces" -name "metadata.json" 2>/dev/null | wc -l)
BASIS_COUNT=$(find "$PHASE_C_DATA/subspaces" -name "basis.npy" 2>/dev/null | wc -l)
RESID_COUNT=$(ls "$PHASE_C_DATA/residualized"/*.npy 2>/dev/null | wc -l)
echo "  Subspace metadata.json: $SUBSPACE_COUNT"
echo "  Basis .npy files: $BASIS_COUNT"
echo "  Residualized .npy files: $RESID_COUNT / 45"

echo ""
echo "--- Summary files ---"
for f in phase_c_results.csv correct_wrong_divergence.csv alignment_results.csv; do
    path="$PHASE_C_DATA/summary/$f"
    if [ -f "$path" ]; then
        ROWS=$(wc -l < "$path")
        echo "  $f: $ROWS rows"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Plots ---"
for subdir in eigenvalue_spectra dimensionality_heatmaps cross_layer_trajectories correct_wrong_comparison; do
    COUNT=$(find "$PHASE_C_PLOTS/$subdir" -name "*.png" 2>/dev/null | wc -l)
    echo "  $subdir: $COUNT plots"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $PHASE_C_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

if [ $PHASE_C_EXIT -eq 0 ]; then
    echo "Phase C completed successfully"
    exit 0
else
    echo "PHASE C FAILED — check logs/phase_c_subspaces.log"
    exit 1
fi
