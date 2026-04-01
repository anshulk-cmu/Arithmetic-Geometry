#!/bin/bash
#SBATCH --job-name=phase_d
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Phase D — LDA Refinement of Phase C Concept Subspaces
# ============================================================================
# GPU-accelerated Fisher LDA with permutation null via CuPy on 4× A6000.
# L5-only resume run — L1-L4 already completed (job 6893354).
# Reads Phase C outputs (residualized activations + subspace bases).
# DOES NOT clean Phase C outputs — Phase D depends on them.
#
# Expected runtime: ~20 hours (4 GPUs, all levels), ~18 hours (4 GPUs, L5 only).
#   L1-L4 are fast (~1h on 4 GPUs); L5 dominates due to N=122K samples.
#   Single GPU, all levels: ~80 hours.
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_d/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_d/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase D — LDA Refinement"
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

# Phase C outputs (Phase D's inputs — DO NOT clean these)
echo "  Checking Phase C outputs..."
PHASE_C_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_c"

RESID_COUNT=$(ls "$PHASE_C_DATA/residualized"/*.npy 2>/dev/null | wc -l)
echo "  Residualized activations: $RESID_COUNT / 45"
if [ "$RESID_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 residualized activation files, found $RESID_COUNT"
    echo "  Run Phase C first: sbatch run_phase_c.sh"
    exit 1
fi

SUBSPACE_COUNT=$(find "$PHASE_C_DATA/subspaces" -name "metadata.json" 2>/dev/null | wc -l)
BASIS_COUNT=$(find "$PHASE_C_DATA/subspaces" -name "basis.npy" 2>/dev/null | wc -l)
echo "  Phase C subspace metadata: $SUBSPACE_COUNT"
echo "  Phase C basis files: $BASIS_COUNT"
if [ "$SUBSPACE_COUNT" -lt 100 ]; then
    echo "  WARNING: Only $SUBSPACE_COUNT Phase C subspaces found (expected ~2800)"
    echo "  Phase D will proceed but novelty comparison will be limited"
fi

if [ -f "$PHASE_C_DATA/summary/phase_c_results.csv" ]; then
    PC_ROWS=$(wc -l < "$PHASE_C_DATA/summary/phase_c_results.csv")
    echo "  Phase C results CSV: $PC_ROWS rows"
else
    echo "  WARNING: phase_c_results.csv not found"
fi

# Raw activations (needed for product_binned)
ACT_DIR="/data/user_data/anshulk/arithmetic-geometry/activations"
ACT_COUNT=$(ls "$ACT_DIR"/*.npy 2>/dev/null | wc -l)
echo "  Raw activations: $ACT_COUNT .npy files"
if [ "$ACT_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 raw activation files, found $ACT_COUNT"
    exit 1
fi

# Coloring DFs
PKL_DIR="/data/user_data/anshulk/arithmetic-geometry/phase_a/coloring_dfs"
PKL_COUNT=$(ls "$PKL_DIR"/*.pkl 2>/dev/null | wc -l)
echo "  Coloring DFs: $PKL_COUNT .pkl files"
if [ "$PKL_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected 5 coloring DF files, found $PKL_COUNT"
    exit 1
fi

# Output directories (create, but DO NOT clean Phase C)
echo "  Creating Phase D output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_d
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_d
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PHASE D
# ============================================================================

echo "============================================================"
echo "Running Phase D — LDA Refinement (L5 resume, 4× GPU)"
echo "  S_T denominator (invariant to label permutation)"
echo "  Level: 5 only (L1-L4 complete)"
echo "  Layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]"
echo "  Populations: all, correct, wrong"
echo "  GPUs: 4× A6000 (parallel by layer)"
echo "  Expected runtime: ~18 hours (L5 only, 4 GPUs)"
echo "============================================================"
echo ""

python phase_d_lda.py --config config.yaml --level 5 --n-gpus 4
PHASE_D_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_D_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_d"
PHASE_D_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_d"

echo "--- Data outputs ---"
META_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "metadata.json" 2>/dev/null | wc -l)
LDA_BASIS_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "lda_basis.npy" 2>/dev/null | wc -l)
MERGED_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "merged_basis.npy" 2>/dev/null | wc -l)
NULL_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "null_lda_eigenvalues.npy" 2>/dev/null | wc -l)
echo "  metadata.json: $META_COUNT"
echo "  lda_basis.npy: $LDA_BASIS_COUNT"
echo "  merged_basis.npy: $MERGED_COUNT"
echo "  null_lda_eigenvalues.npy: $NULL_COUNT"

echo ""
echo "--- Summary files ---"
for f in phase_d_results.csv lda_novelty_summary.csv; do
    path="$PHASE_D_DATA/summary/$f"
    if [ -f "$path" ]; then
        ROWS=$(wc -l < "$path")
        echo "  $f: $ROWS rows"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Plots ---"
for subdir in eigenvalue_spectra novelty_heatmaps n_sig_heatmaps population_comparison; do
    COUNT=$(find "$PHASE_D_PLOTS/$subdir" -name "*.png" 2>/dev/null | wc -l)
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
echo "Exit code: $PHASE_D_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

if [ $PHASE_D_EXIT -eq 0 ]; then
    echo "Phase D completed successfully"
    exit 0
else
    echo "PHASE D FAILED — check logs/phase_d_lda.log"
    exit 1
fi
