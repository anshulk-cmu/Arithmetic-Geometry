#!/bin/bash
#SBATCH --job-name=phase_e
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# Handle preemption: SLURM sends SIGUSR1 120s before kill.
# Requeue the job so it restarts automatically.
handle_preempt() {
    echo ""
    echo "============================================================"
    echo "PREEMPTED — requeueing job $SLURM_JOB_ID at $(date)"
    echo "Resume logic will skip completed slices on restart."
    echo "============================================================"
    scontrol requeue "$SLURM_JOB_ID"
}
trap 'handle_preempt' USR1

# ============================================================================
# Phase E — Residual Hunting via PCA on Concept-Projected Residuals
# ============================================================================
# Projects out all known concept subspaces (Phase C+D merged bases),
# then runs PCA on the residual to detect unknown structure.
# Marchenko-Pastur distribution separates signal from noise eigenvalues.
#
# Reads Phase C outputs (residualized activations) and Phase D outputs
# (merged bases). DOES NOT clean Phase C or D outputs.
#
# Expected runtime: ~1–2 hours on single A6000 (99 slices).
#   L5 dominates (N=122K); L2–L4 are fast (~seconds each).
#   48h time limit is generous for preemption retries.
#
# Scope: L2–L5, all 9 layers, all viable populations = 99 slices
#   L1 skipped (only 64 samples, statistically useless for PCA in ~4000D)
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_e/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_e/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase E — Residual Hunting"
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

# Phase C outputs (residualized activations — Phase E's inputs)
echo "  Checking Phase C outputs..."
PHASE_C_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_c"

RESID_COUNT=$(ls "$PHASE_C_DATA/residualized"/*.npy 2>/dev/null | wc -l)
echo "  Residualized activations: $RESID_COUNT / 45"
if [ "$RESID_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 residualized activation files, found $RESID_COUNT"
    echo "  Run Phase C first: sbatch run_phase_c.sh"
    exit 1
fi

# Phase D outputs (merged bases — Phase E's inputs)
echo "  Checking Phase D outputs..."
PHASE_D_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_d"

META_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "metadata.json" 2>/dev/null | wc -l)
BASIS_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "merged_basis.npy" 2>/dev/null | wc -l)
echo "  Phase D metadata files: $META_COUNT"
echo "  Phase D merged basis files: $BASIS_COUNT"
if [ "$BASIS_COUNT" -lt 100 ]; then
    echo "  WARNING: Only $BASIS_COUNT merged basis files found (expected ~2844)"
    echo "  Phase E will proceed but union subspaces may be incomplete"
fi

if [ -f "$PHASE_D_DATA/summary/phase_d_results.csv" ]; then
    PD_ROWS=$(wc -l < "$PHASE_D_DATA/summary/phase_d_results.csv")
    echo "  Phase D results CSV: $PD_ROWS rows"
else
    echo "  WARNING: phase_d_results.csv not found"
fi

# Raw activations (needed for product β recomputation)
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

# Output directories (create, but DO NOT clean Phase C or D)
echo "  Creating Phase E output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_e
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_e
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PHASE E
# ============================================================================

echo "============================================================"
echo "Running Phase E — Residual Hunting"
echo "  Levels: 2, 3, 4, 5"
echo "  Layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]"
echo "  Populations: all, correct, wrong"
echo "  GPU: 1× A6000"
echo "  Expected: 99 slices, ~1–2 hours total"
echo "============================================================"
echo ""

python phase_e_residual_hunting.py --config config.yaml
PHASE_E_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_E_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_e"
PHASE_E_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_e"

echo "--- Data outputs ---"
UNION_COUNT=$(find "$PHASE_E_DATA/union_bases" -name "metadata.json" 2>/dev/null | wc -l)
UNION_BASIS_COUNT=$(find "$PHASE_E_DATA/union_bases" -name "union_basis.npy" 2>/dev/null | wc -l)
PCA_META_COUNT=$(find "$PHASE_E_DATA/pca" -name "metadata.json" 2>/dev/null | wc -l)
EIGEN_COUNT=$(find "$PHASE_E_DATA/pca" -name "eigenvalues.npy" 2>/dev/null | wc -l)
CORR_COUNT=$(find "$PHASE_E_DATA/correlations" -name "correlation_sweep.csv" 2>/dev/null | wc -l)
echo "  Union basis metadata.json: $UNION_COUNT / 99"
echo "  Union union_basis.npy: $UNION_BASIS_COUNT / 99"
echo "  PCA metadata.json: $PCA_META_COUNT / 99"
echo "  PCA eigenvalues.npy: $EIGEN_COUNT / 99"
echo "  Correlation sweep CSVs: $CORR_COUNT"

echo ""
echo "--- Summary files ---"
for f in phase_e_results.csv eigenvalue_cliff_summary.csv union_rank_by_layer.csv variance_explained.csv total_carry_sum_diagnostic.csv top_eigenvalues_all_slices.csv; do
    path="$PHASE_E_DATA/summary/$f"
    if [ -f "$path" ]; then
        ROWS=$(wc -l < "$path")
        echo "  $f: $ROWS rows"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Plots ---"
for subdir in eigenvalue_spectra mp_heatmaps variance_explained_heatmaps union_rank_trajectories correlation_heatmaps; do
    COUNT=$(find "$PHASE_E_PLOTS/$subdir" -name "*.png" 2>/dev/null | wc -l)
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
echo "Exit code: $PHASE_E_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

if [ $PHASE_E_EXIT -eq 0 ]; then
    echo "Phase E completed successfully"
    exit 0
else
    echo "PHASE E FAILED — check logs/phase_e_residual_hunting.log"
    exit 1
fi
