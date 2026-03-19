#!/bin/bash
#SBATCH --job-name=phase_a
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Phase A — Embeddings + Pre-flight Diagnostics
# ============================================================================
# GPU-accelerated via cuML (RAPIDS). Falls back to CPU if cuML unavailable.
#
# Step 1: phase_a_analysis.py  — norm profile + CKA matrices (~2 min, CPU)
# Step 2: phase_a_embeddings.py — UMAP/t-SNE embeddings, interestingness
#          scoring, heatmaps, comparison tables, selective plots
#          GPU: ~15-30 min | CPU fallback: ~3-6 hrs
#
# Both scripts checkpoint to disk and resume where they left off.
# Safe to resubmit if the job times out.
#
# Outputs:
#   Data:  /data/user_data/anshulk/arithmetic-geometry/phase_a/
#   Plots: /home/anshulk/arithmetic-geometry/plots/phase_a/
#   Logs:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Phase A — Embeddings + Pre-flight Diagnostics"
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

# GPU
echo "  Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "  WARNING: nvidia-smi not available — will run on CPU"
else
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

# cuML (GPU UMAP/t-SNE)
echo "  Checking cuML..."
if python -c "from cuml.manifold import UMAP; from cuml.manifold import TSNE; print('  cuML available')" 2>/dev/null; then
    :
else
    echo "  WARNING: cuML not available — will fall back to CPU UMAP/t-SNE"
fi

# Python packages
echo "  Checking Python packages..."
REQUIRED="numpy scipy sklearn yaml matplotlib pandas"
for pkg in $REQUIRED; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ERROR: Missing package: $pkg"
        exit 1
    fi
done
# umap-learn only needed if cuML is missing
if ! python -c "import cuml" 2>/dev/null; then
    if ! python -c "import umap" 2>/dev/null; then
        echo "  ERROR: Neither cuml nor umap-learn available"
        exit 1
    fi
fi
echo "  All packages available"

# Config
echo "  Checking config..."
if [ ! -f "config.yaml" ]; then
    echo "  ERROR: config.yaml not found"
    exit 1
fi

# Input data (activations + answers from pipeline.py)
echo "  Checking input data..."
ACT_DIR="/data/user_data/anshulk/arithmetic-geometry/activations"
ANS_DIR="/data/user_data/anshulk/arithmetic-geometry/answers"
ACT_COUNT=$(ls "$ACT_DIR"/*.npy 2>/dev/null | wc -l)
ANS_COUNT=$(ls "$ANS_DIR"/*.json 2>/dev/null | wc -l)
echo "  Activations: $ACT_COUNT .npy files"
echo "  Answers: $ANS_COUNT .json files"
if [ "$ACT_COUNT" -lt 45 ]; then
    echo "  ERROR: Expected 45 activation files (5 levels x 9 layers), found $ACT_COUNT"
    exit 1
fi
if [ "$ANS_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected 5 answer files, found $ANS_COUNT"
    exit 1
fi

# Output directories
echo "  Creating output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots/phase_a
mkdir -p /data/user_data/anshulk/arithmetic-geometry/phase_a
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# STEP 1: PRE-FLIGHT DIAGNOSTICS (norm profile + CKA)
# ============================================================================

echo "============================================================"
echo "Step 1: phase_a_analysis.py"
echo "  Activation norm profile + cross-layer CKA matrices"
echo "  Expected runtime: ~2 minutes"
echo "============================================================"
echo ""

python phase_a_analysis.py --config config.yaml
ANALYSIS_EXIT=$?

if [ $ANALYSIS_EXIT -ne 0 ]; then
    echo ""
    echo "phase_a_analysis.py FAILED (exit code $ANALYSIS_EXIT)"
    echo "Continuing to embeddings anyway..."
fi

echo ""
echo "Step 1 done ($SECONDS seconds elapsed)"
echo ""

# ============================================================================
# STEP 2: EMBEDDINGS + SCORING + PLOTS
# ============================================================================

echo "============================================================"
echo "Step 2: phase_a_embeddings.py"
echo "  UMAP/t-SNE embeddings, interestingness scoring, plots"
echo "  Expected runtime: ~15-30 min (GPU) / ~3-6 hrs (CPU)"
echo "============================================================"
echo ""

python phase_a_embeddings.py --config config.yaml
EMBEDDINGS_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

PHASE_A_DATA="/data/user_data/anshulk/arithmetic-geometry/phase_a"
PHASE_A_PLOTS="/home/anshulk/arithmetic-geometry/plots/phase_a"

echo "--- Data outputs ---"
CSV_COUNT=$(ls "$PHASE_A_DATA"/csvs/*.csv 2>/dev/null | wc -l)
EMB_COUNT=$(find "$PHASE_A_DATA"/embeddings -name "*.npy" 2>/dev/null | wc -l)
echo "  CSVs: $CSV_COUNT (expected ~117)"
echo "  Embedding .npy files: $EMB_COUNT"

echo ""
echo "--- Analysis outputs ---"
for f in norm_profile.json cka_matrices.json; do
    path="$PHASE_A_DATA/analysis/$f"
    if [ -f "$path" ]; then
        echo "  $f: OK ($(du -h "$path" | cut -f1))"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Scores ---"
SCORES="$PHASE_A_DATA/scores/interestingness_scores.csv"
if [ -f "$SCORES" ]; then
    SCORE_LINES=$(wc -l < "$SCORES")
    echo "  interestingness_scores.csv: $SCORE_LINES rows"
else
    echo "  interestingness_scores.csv: MISSING"
fi
for f in top_50_findings.md correct_wrong_comparison.md correct_wrong_comparison.csv l5_delta_interestingness.md; do
    path="$PHASE_A_DATA/scores/$f"
    if [ -f "$path" ]; then
        echo "  $f: OK"
    else
        echo "  $f: MISSING"
    fi
done

# L5 subsample metadata
L5_META="$PHASE_A_DATA/l5_subsample_meta.json"
if [ -f "$L5_META" ]; then
    echo "  l5_subsample_meta.json: OK ($(du -h "$L5_META" | cut -f1))"
else
    echo "  l5_subsample_meta.json: MISSING"
fi

echo ""
echo "--- Plots ---"
HEATMAP_COUNT=$(ls "$PHASE_A_PLOTS"/heatmaps/*.png 2>/dev/null | wc -l)
ANALYSIS_PLOT_COUNT=$(ls "$PHASE_A_PLOTS"/analysis/*.png 2>/dev/null | wc -l)
SCATTER_COUNT=$(find "$PHASE_A_PLOTS" -name "*.png" -not -path "*/heatmaps/*" -not -path "*/analysis/*" 2>/dev/null | wc -l)
echo "  Heatmaps: $HEATMAP_COUNT"
echo "  Analysis plots: $ANALYSIS_PLOT_COUNT"
echo "  Scatter plots: $SCATTER_COUNT"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Analysis exit code: $ANALYSIS_EXIT"
echo "Embeddings exit code: $EMBEDDINGS_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || echo "  (no GPU)"

echo ""
if [ $ANALYSIS_EXIT -eq 0 ] && [ $EMBEDDINGS_EXIT -eq 0 ]; then
    echo "All stages completed successfully"
    exit 0
else
    echo "SOME STAGES FAILED — check logs/"
    echo "  phase_a_analysis.log and phase_a_embeddings.log for details"
    exit 1
fi
