# Running in Google Colab (instructions only)

Use the **existing** `config.yaml`; override only paths (and optionally the model) inside Colab so the repo stays unchanged.

## 1. Clone and install

```python
!git clone https://github.com/YOUR_ORG/arithmetic-geometry.git /content/arithmetic-geometry
%cd /content/arithmetic-geometry
!pip install -q torch numpy matplotlib pyyaml transformers accelerate
```

Replace `YOUR_ORG/arithmetic-geometry` with your repo URL.

## 2. Point config at Colab paths (minimal change in Colab)

In Colab, create a copy of the config and set paths to `/content/` so everything lives in the runtime:

```python
import yaml
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["workspace"] = "/content/arithmetic-geometry"
cfg["paths"]["data_root"] = "/content/arithmetic-geometry/data"

# Optional: use HuggingFace ID so Colab downloads the model (no local snapshot)
# cfg["model"]["name"] = "meta-llama/Meta-Llama-3.1-8B"

with open("config_colab.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

If you use the Hugging Face model ID, log in first:

```python
from huggingface_hub import login
login()
```

## 3. Run pipeline (GPU on)

```python
!python pipeline.py --config config_colab.yaml
```

## 4. Run analysis

```python
!python analysis.py --config config_colab.yaml
```

## 5. View plots

```python
from IPython.display import Image, display
import os
for name in sorted(os.listdir("plots")):
    if name.endswith(".png"):
        display(Image(f"plots/{name}", width=500))
```

---

**Requirements:** Runtime with GPU (e.g. T4) for the pipeline; ~16 GB VRAM for Llama 3.1 8B. If you hit OOM, reduce `batch_size` or `problems_per_level` in the copied config in Colab.
