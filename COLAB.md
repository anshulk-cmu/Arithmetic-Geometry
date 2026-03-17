# Run on Google Colab

Short setup. **Model** = runtime (re-download each session). **Results** = your Drive (saved).

---

## Before you start

- **Runtime:** Turn on GPU (Runtime → Change runtime type → GPU).
- **Hugging Face:** Accept Llama 3.1 8B license at [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (use a **classic** Read token, or enable gated-repo access on a fine-grained token).
- **Colab secret:** Left sidebar → key icon → Add secret: name `HF_TOKEN`, value = your HF token.

---

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Complete the browser auth.

---

## 2. Results folder on Drive

```python
import os
DRIVE_RESULTS = "/content/drive/MyDrive/arithmetic-geometry-results"
os.makedirs(DRIVE_RESULTS, exist_ok=True)
```

---

## 3. Clone repo and install

Replace with your repo URL and branch if needed.

```python
!git clone https://github.com/YOUR_ORG/Arithmetic-Geometry /content/arithmetic-geometry
%cd /content/arithmetic-geometry
# !git checkout your-branch-name   # optional
!pip install -q torch numpy matplotlib pyyaml transformers accelerate
```

---

## 4. HF login, download model to runtime, build config

One cell: log in with your secret, download model to runtime, write `config_colab.yaml` (model = runtime, results = Drive).

```python
import yaml
from huggingface_hub import snapshot_download, login
from google.colab import userdata

login(token=userdata.get("HF_TOKEN"))

RUNTIME_MODEL = "/content/arithmetic-geometry/model"
HF_REPO_ID = "meta-llama/Meta-Llama-3.1-8B"
os.makedirs(RUNTIME_MODEL, exist_ok=True)
snapshot_download(HF_REPO_ID, local_dir=RUNTIME_MODEL)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["paths"]["workspace"] = DRIVE_RESULTS
cfg["paths"]["data_root"] = os.path.join(DRIVE_RESULTS, "data")
cfg["model"]["name"] = RUNTIME_MODEL

with open("config_colab.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
```

---

## 5. Run pipeline then analysis

```python
!python pipeline.py --config config_colab.yaml
```

```python
!python analysis.py --config config_colab.yaml
```

---

## 6. View plots

```python
from IPython.display import Image, display
for name in sorted(os.listdir(os.path.join(DRIVE_RESULTS, "plots"))):
    if name.endswith(".png"):
        display(Image(os.path.join(DRIVE_RESULTS, "plots", name), width=500))
```

---

**If you want to quick run or OOM?** in step 4, add e.g. `cfg["generation"]["batch_size"] = 8` and `cfg["dataset"]["problems_per_level"] = 500` before the `yaml.dump` line, then run pipeline again.
