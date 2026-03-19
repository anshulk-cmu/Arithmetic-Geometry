#!/bin/bash
#SBATCH --job-name=l5_screen
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# L5 Two-Phase Screening
# ============================================================================
# Evaluates all 810,000 3-digit × 3-digit multiplication problems for
# correctness, then selects a carry-balanced subset for the main pipeline.
#
# Model: Meta-Llama-3.1-8B (local, bfloat16 ~16 GB VRAM)
# GPU: A6000 (48 GB)
# Expected runtime: 60-80 minutes (810K greedy generations at batch_size=256)
#
# Outputs:
#   Cache:    /data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_evaluation_cache.npz
#   Selected: /data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_selected_problems.json
# ============================================================================

echo "============================================================"
echo "L5 Two-Phase Screening"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/arithmetic-geometry || { echo "Failed to cd to workspace"; exit 1; }

eval "$(conda shell.bash hook)"
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# GPU
echo "  Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Python packages
echo "  Checking Python packages..."
for pkg in torch numpy yaml transformers; do
    mod=$pkg
    if ! python -c "import $mod" 2>/dev/null; then
        echo "  ERROR: Missing package: $pkg"
        exit 1
    fi
done
echo "  All packages available"

# Model
echo "  Checking local model..."
MODEL_DIR="/data/user_data/anshulk/arithmetic-geometry/model"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "  ERROR: Model not found at $MODEL_DIR"
    exit 1
fi
echo "  Model found: $(du -sh "$MODEL_DIR" | cut -f1)"

# Config
if [ ! -f "config.yaml" ]; then
    echo "  ERROR: config.yaml not found"
    exit 1
fi
echo "  config.yaml found"

# Output directory
mkdir -p /data/user_data/anshulk/arithmetic-geometry/l5_screening
mkdir -p /home/anshulk/arithmetic-geometry/logs
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN SCREENING
# ============================================================================

echo "============================================================"
echo "Running generate_l5_problems.py"
echo "  Problems: 810,000 (all 3-digit × 3-digit pairs)"
echo "  Batch size: 256"
echo "  Model: Llama 3.1 8B (bfloat16)"
echo "============================================================"
echo ""

python generate_l5_problems.py --config config.yaml
EXIT_CODE=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Screening Validation"
echo "============================================================"

SELECTED="/data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_selected_problems.json"
CACHE="/data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_evaluation_cache.npz"

if [ -f "$SELECTED" ]; then
    python -c "
import json
with open('$SELECTED') as f:
    d = json.load(f)
m = d['metadata']
print(f'Selected: {d[\"n_selected\"]:,} problems ({d[\"n_correct\"]:,} correct)')
print(f'Screened: {m[\"total_screened\"]:,} total, {m[\"total_correct_in_space\"]:,} correct ({m[\"accuracy\"]*100:.3f}%)')
print(f'Hard ceilings: {len(m.get(\"hard_ceilings\", {}))} carry values below floor')
"
else
    echo "  ERROR: Selected problems file not found"
fi

if [ -f "$CACHE" ]; then
    echo "  Cache: $(du -sh "$CACHE" | cut -f1)"
else
    echo "  WARNING: Evaluation cache not found"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Screening Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS — run 'sbatch run.sh' next for the main pipeline"
    exit 0
else
    echo "FAILED — check logs/generate_l5_problems.log"
    exit 1
fi
