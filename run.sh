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
# Extracts residual stream activations from Llama 3.1 8B (base) across 9 layers.
#
# Dataset:
#   L1: 64 problems (exhaustive 1x1 grid)
#   L2: 4,000 problems (2x1 digit, random)
#   L3: 10,000 problems (2x2 digit, random)
#   L4: 10,000 problems (3x2 digit, random)
#   L5: ~35,000-40,000 problems (3x3 digit, carry-balanced from screening)
#
# PREREQUISITE: run_l5_screen.sh must complete first to produce L5 problem set.
#
# Model: Meta-Llama-3.1-8B (local, bfloat16 ~16 GB VRAM)
# GPU: A6000 (48 GB), batch_size=256
# Expected runtime: ~30-40 minutes
#
# Outputs:
#   Activations: /data/user_data/anshulk/arithmetic-geometry/activations/ (~10 GB)
#   Answers:     /data/user_data/anshulk/arithmetic-geometry/answers/
#   Labels:      /home/anshulk/arithmetic-geometry/labels/
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

# L5 screening output (prerequisite)
echo "  Checking L5 screening output..."
L5_SELECTED="/data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_selected_problems.json"
if [ ! -f "$L5_SELECTED" ]; then
    echo "  ERROR: L5 screening output not found at $L5_SELECTED"
    echo "  Run 'sbatch run_l5_screen.sh' first."
    exit 1
fi
echo "  L5 screening output found"

# Output directories
echo "  Checking output directories..."
mkdir -p /home/anshulk/arithmetic-geometry/labels
mkdir -p /home/anshulk/arithmetic-geometry/logs
mkdir -p /home/anshulk/arithmetic-geometry/plots
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
echo "  Levels: L1(64) + L2(4K) + L3(10K) + L4(10K) + L5(~35-40K from screening)"
echo "  Layers: [4, 6, 8, 12, 16, 20, 24, 28, 31]"
echo "  Batch size: 256"
echo "============================================================"
echo ""

python pipeline.py --config config.yaml
PIPELINE_EXIT=$?

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo ""
    echo "Pipeline FAILED (exit code $PIPELINE_EXIT) — check logs/pipeline.log"
    echo "Skipping analysis."
    echo "End time: $(date)"
    echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
    exit 1
fi

# ============================================================================
# ANALYSIS
# ============================================================================

echo ""
echo "============================================================"
echo "Running analysis.py"
echo "  Error classification, per-digit accuracy, carry correlation"
echo "  Error structure, input difficulty, 6 diagnostic plots"
echo "============================================================"
echo ""

python analysis.py --config config.yaml
ANALYSIS_EXIT=$?

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
        f = act_dir / f'level{lvl}_layer{layer}.npy'
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
    f = ans_dir / f'level_{lvl}.json'
    if f.exists():
        data = json.load(open(f))
        print(f'  Level {lvl}: accuracy={data[\"accuracy\"]:.1%} ({data[\"n_correct\"]}/{data[\"n_problems\"]})')
    else:
        print(f'  MISSING: {f.name}')

print()
print('--- L5 Screening Match Check ---')
l5_sel = Path('/data/user_data/anshulk/arithmetic-geometry/l5_screening/l5_selected_problems.json')
l5_ans = ans_dir / 'level_5.json'
if l5_sel.exists() and l5_ans.exists():
    sel = json.load(open(l5_sel))
    ans = json.load(open(l5_ans))
    expected = sel['n_correct']
    actual = ans['n_correct']
    if expected == actual:
        print(f'  PASS: screening n_correct={expected} matches pipeline n_correct={actual}')
    else:
        print(f'  MISMATCH: screening n_correct={expected} vs pipeline n_correct={actual}')
else:
    print('  SKIP: missing files')

print()
print('--- Analysis Summary ---')
summary_f = ans_dir / 'analysis_summary.json'
if summary_f.exists():
    print(f'  analysis_summary.json: exists ({summary_f.stat().st_size / 1024:.0f} KB)')
else:
    print(f'  analysis_summary.json: MISSING')

print()
print('--- Plots ---')
plots = Path('/home/anshulk/arithmetic-geometry/plots')
pipeline_plots = ['accuracy_by_level.png', 'activation_norm_profile.png', 'digit_coverage.png']
analysis_plots = ['per_digit_accuracy_heatmap.png', 'error_distributions.png',
                  'accuracy_vs_carries.png', 'accuracy_vs_magnitude.png',
                  'error_categories.png', 'digit_accuracy_by_carry.png']
for name in pipeline_plots + analysis_plots:
    f = plots / name
    status = f'OK ({f.stat().st_size / 1024:.0f} KB)' if f.exists() else 'MISSING'
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
echo "Analysis exit code: $ANALYSIS_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo ""
echo "Generated files:"
echo "  Activations:"
ACT_COUNT=$(ls /data/user_data/anshulk/arithmetic-geometry/activations/*.npy 2>/dev/null | wc -l)
echo "  $ACT_COUNT .npy files"
echo "  Answers:"
ls /data/user_data/anshulk/arithmetic-geometry/answers/*.json 2>/dev/null | wc -l | xargs -I{} echo "  {} .json files"
echo "  Plots:"
ls /home/anshulk/arithmetic-geometry/plots/*.png 2>/dev/null | wc -l | xargs -I{} echo "  {} .png files"

echo ""
echo "============================================================"
if [ $PIPELINE_EXIT -eq 0 ] && [ $ANALYSIS_EXIT -eq 0 ]; then
    echo "All stages completed successfully"
    exit 0
else
    echo "FAILED — check logs/pipeline.log and logs/analysis.log"
    exit 1
fi
