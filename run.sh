#!/bin/bash
#SBATCH --job-name=arith_geom
#SBATCH --output=/home/anshulk/arithmetic-geometry/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/arithmetic-geometry/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Arithmetic Geometry Pipeline
# ============================================================================
# Extracts residual stream activations from Llama 3.1 8B (base) across 9 layers
# for 16,064 multiplication problems at 5 difficulty levels.
#
# Model: Meta-Llama-3.1-8B (local, bfloat16 ~16 GB VRAM)
# GPU: A6000 (48 GB) — plenty of headroom
# Expected runtime: ~2-3 hours
#
# Outputs:
#   Activations: /data/user_data/anshulk/arithmetic-geometry/activations/ (~2.9 GB)
#   Answers:     /data/user_data/anshulk/arithmetic-geometry/answers/
#   Labels:      /home/anshulk/arithmetic-geometry/data/
#   Logs+Plots:  /home/anshulk/arithmetic-geometry/logs/
# ============================================================================

echo "============================================================"
echo "Arithmetic Geometry Pipeline"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/arithmetic-geometry || { echo "Failed to cd to workspace"; exit 1; }

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate geometry || { echo "Failed to activate geometry environment"; exit 1; }

# Environment variables
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
REQUIRED="torch numpy matplotlib yaml transformers accelerate"
for pkg in $REQUIRED; do
    mod=$pkg
    [ "$pkg" = "yaml" ] && mod="yaml"
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
MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
echo "  Model found: $MODEL_SIZE"

# Config
echo "  Checking config..."
if [ ! -f "config.yaml" ]; then
    echo "  ERROR: config.yaml not found"
    exit 1
fi
echo "  config.yaml found"

# Output directories
echo "  Checking output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/data
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /data/user_data/anshulk/arithmetic-geometry/activations
mkdir -p /data/user_data/anshulk/arithmetic-geometry/answers
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN PIPELINE
# ============================================================================

echo "============================================================"
echo "Running pipeline.py"
echo "  Model: Llama 3.1 8B (local, bfloat16)"
echo "  Levels: 5 (16,064 problems total)"
echo "  Layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]"
echo "  Batch size: 32"
echo "============================================================"
echo ""

python pipeline.py --config config.yaml

PIPELINE_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

python -c "
import numpy as np
from pathlib import Path
import json

act_dir = Path('/data/user_data/anshulk/arithmetic-geometry/activations')
ans_dir = Path('/data/user_data/anshulk/arithmetic-geometry/answers')
layers = [4, 6, 8, 12, 16, 20, 24, 28, 31]
levels = [1, 2, 3, 4, 5]

print('--- Activation Files ---')
total_bytes = 0
for lvl in levels:
    for layer in layers:
        f = act_dir / f'activations_level{lvl}_layer{layer}.npy'
        if f.exists():
            arr = np.load(f, mmap_mode='r')
            total_bytes += f.stat().st_size
            if layer == layers[0]:
                print(f'  Level {lvl}: shape={arr.shape}, ', end='')
            if layer == layers[-1]:
                norms = np.linalg.norm(np.load(f), axis=1)
                print(f'norms=[{norms.min():.0f}, {norms.mean():.0f}, {norms.max():.0f}]')
        else:
            print(f'  MISSING: {f.name}')
print(f'  Total: {total_bytes / 1024**3:.2f} GB')

print()
print('--- Answer Files ---')
for lvl in levels:
    f = ans_dir / f'answers_level_{lvl}.json'
    if f.exists():
        data = json.load(open(f))
        print(f'  Level {lvl}: accuracy={data[\"accuracy\"]:.1%} ({data[\"n_correct\"]}/{data[\"n_problems\"]})')
    else:
        print(f'  MISSING: {f.name}')

print()
print('--- Diagnostic Plots ---')
logs = Path('/home/anshulk/arithmetic-geometry/logs')
for name in ['accuracy_by_level.png', 'activation_norm_profile.png', 'digit_coverage.png']:
    f = logs / name
    status = f'exists ({f.stat().st_size / 1024:.0f} KB)' if f.exists() else 'MISSING'
    print(f'  {name}: {status}')
" 2>/dev/null || echo "Validation script failed"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Pipeline exit code: $PIPELINE_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo ""
echo "Generated files:"
echo "  Activations:"
ls -lh /data/user_data/anshulk/arithmetic-geometry/activations/*.npy 2>/dev/null | head -5
ACT_COUNT=$(ls /data/user_data/anshulk/arithmetic-geometry/activations/*.npy 2>/dev/null | wc -l)
echo "  ... ($ACT_COUNT files total)"
echo "  Answers:"
ls -lh /data/user_data/anshulk/arithmetic-geometry/answers/*.json 2>/dev/null
echo "  Plots:"
ls -lh /home/anshulk/arithmetic-geometry/logs/*.png 2>/dev/null

echo ""
echo "============================================================"
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "Pipeline completed successfully"
    exit 0
else
    echo "Pipeline FAILED (exit code $PIPELINE_EXIT) — check logs/pipeline.log"
    exit 1
fi
