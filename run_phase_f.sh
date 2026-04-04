#!/bin/bash
#SBATCH --job-name=phase_f_jl
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
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
# Phase F/JL — Between-Concept Principal Angles & JL Distance Preservation
# ============================================================================
# Phase F: computes principal angles between every pair of concept subspaces
#   from Phase D merged bases. Detects superposition (shared dimensions).
# JL: checks whether pairwise distances are preserved under projection to
#   the union subspace from Phase E.
#
# Reads: Phase C residualized activations, Phase D merged bases,
#        Phase E union bases, Phase A coloring DataFrames.
#
# Expected runtime: Phase F ~5 min, JL ~6-12 hours on single A6000.
#   L5/all and L5/wrong use all 7.5B+ pairs (no subsampling).
#   7-day time limit for preemption retries. 256GB RAM for large distance arrays.
#
# Scope: L1-L5 for Phase F (108 slices), L2-L5 for JL (99 slices)
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_f/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_f/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase F/JL — Between-Concept Angles & JL Distance Check"
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

# Phase C outputs (residualized activations — JL inputs)
echo "  Checking Phase C outputs..."
PHASE_C_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_c"

RESID_COUNT=$(ls "$PHASE_C_DATA/residualized"/*.npy 2>/dev/null | wc -l)
echo "  Residualized activations: $RESID_COUNT / 45"
if [ "$RESID_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 residualized activation files, found $RESID_COUNT"
    echo "  Run Phase C first: sbatch run_phase_c.sh"
    exit 1
fi

# Phase D outputs (merged bases — Phase F inputs)
echo "  Checking Phase D outputs..."
PHASE_D_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_d"

META_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "metadata.json" 2>/dev/null | wc -l)
BASIS_COUNT=$(find "$PHASE_D_DATA/subspaces" -name "merged_basis.npy" 2>/dev/null | wc -l)
echo "  Phase D metadata files: $META_COUNT"
echo "  Phase D merged basis files: $BASIS_COUNT"
if [ "$BASIS_COUNT" -lt 100 ]; then
    echo "  WARNING: Only $BASIS_COUNT merged basis files found (expected ~2844)"
    echo "  Phase F will proceed but some concepts may be missing"
fi

# Phase E outputs (union bases — JL inputs)
echo "  Checking Phase E outputs..."
PHASE_E_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_e"

UNION_META_COUNT=$(find "$PHASE_E_DATA/union_bases" -name "metadata.json" 2>/dev/null | wc -l)
UNION_BASIS_COUNT=$(find "$PHASE_E_DATA/union_bases" -name "union_basis.npy" 2>/dev/null | wc -l)
echo "  Phase E union metadata.json: $UNION_META_COUNT / 99"
echo "  Phase E union_basis.npy: $UNION_BASIS_COUNT / 99"
if [ "$UNION_META_COUNT" -lt 99 ]; then
    echo "  ERROR: Expected 99 Phase E union metadata files, found $UNION_META_COUNT"
    echo "  Run Phase E first: sbatch run_phase_e.sh"
    exit 1
fi

# Spot-check Phase E data integrity
echo "  Spot-checking Phase E union basis shape..."
python -c "
import numpy as np
ub = np.load('$PHASE_E_DATA/union_bases/L3/layer_16/all/union_basis.npy')
assert ub.shape[0] > 0, f'Empty union basis: shape {ub.shape}'
assert ub.shape[1] == 4096, f'Wrong hidden dim: shape {ub.shape}'
print(f'  Phase E spot-check OK: shape {ub.shape}')
" || { echo "  ERROR: Phase E data integrity check failed"; exit 1; }

# Coloring DFs
PKL_DIR="/data/user_data/anshulk/arithmetic-geometry/phase_a/coloring_dfs"
PKL_COUNT=$(ls "$PKL_DIR"/*.pkl 2>/dev/null | wc -l)
echo "  Coloring DFs: $PKL_COUNT .pkl files"
if [ "$PKL_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected 5 coloring DF files, found $PKL_COUNT"
    exit 1
fi

# Output directories
echo "  Creating Phase F output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_f
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_f
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PHASE F/JL
# ============================================================================

echo "============================================================"
echo "Running Phase F/JL"
echo "  Phase F levels: 1, 2, 3, 4, 5  (108 slices)"
echo "  JL levels: 2, 3, 4, 5          (99 slices)"
echo "  Layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]"
echo "  Populations: all, correct, wrong"
echo "  GPU: 1× A6000"
echo "============================================================"
echo ""

python phase_f_jl.py --config config.yaml
PHASE_F_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_F_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_f"
PHASE_F_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_f"

echo "--- Phase F (Principal Angles) ---"
PA_META_COUNT=$(find "$PHASE_F_DATA/principal_angles" -name "metadata.json" 2>/dev/null | wc -l)
PA_CSV_COUNT=$(find "$PHASE_F_DATA/principal_angles" -name "pairwise_angles.csv" 2>/dev/null | wc -l)
echo "  Principal angles metadata.json: $PA_META_COUNT / 108"
echo "  Principal angles pairwise_angles.csv: $PA_CSV_COUNT"

echo ""
echo "--- JL (Distance Preservation) ---"
JL_META_COUNT=$(find "$PHASE_F_DATA/jl_check" -name "metadata.json" 2>/dev/null | wc -l)
JL_JSON_COUNT=$(find "$PHASE_F_DATA/jl_check" -name "jl_results.json" 2>/dev/null | wc -l)
echo "  JL metadata.json: $JL_META_COUNT / 99"
echo "  JL jl_results.json: $JL_JSON_COUNT"

echo ""
echo "--- Summary files ---"
for f in phase_f_principal_angles.csv superposition_summary.csv redundancy_decomposition.csv jl_distance_preservation.csv; do
    path="$PHASE_F_DATA/summary/$f"
    if [ -f "$path" ]; then
        ROWS=$(wc -l < "$path")
        echo "  $f: $ROWS rows"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Plots ---"
for subdir in superposition_heatmaps angle_distributions cross_layer_superposition tier_boxplots jl_scatter jl_trajectories variance_budget; do
    COUNT=$(find "$PHASE_F_PLOTS/$subdir" -name "*.png" 2>/dev/null | wc -l)
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
echo "Exit code: $PHASE_F_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

if [ $PHASE_F_EXIT -eq 0 ]; then
    echo "Phase F/JL completed successfully"
    exit 0
else
    echo "PHASE F/JL FAILED — check logs/phase_f_jl.log"
    exit 1
fi
