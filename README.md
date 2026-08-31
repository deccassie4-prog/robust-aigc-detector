# robust-aigc-detector

Robust detection of AI-generated images under real-world transformations like blur, compression, color adjustment, cropping, or rescaling.

## Project Overview

This project builds a robust image-level detector that distinguishes AI-generated images (AIGC) from authentic images under real-world image transformations, remaining reliable after JPEG compression, Gaussian blur, resizing, Gaussian noise, color jitter, and center cropping.

The detector is built on **MIRROR** (Manifold Ideal Reference ReconstructOR, [arXiv:2602.02222](http://arxiv.org/abs/2602.02222)) and used fully **zero-shot** — no training was performed on the competition validation data or on any competition-related data. MIRROR reframes detection as a *reference-comparison* process: a frozen DINOv3-Huge encoder plus a learnable, orthogonal memory bank models the manifold of real images; each input is projected onto that manifold via sparse top-k cross-attention, and the **comparison residual** between input and its "ideal reference" serves as a generator-agnostic detection signal.

On top of the upstream pipeline we added **multi-crop scoring strategies** (center / five-crop / 3×3–5×5 grids / multiscale) with configurable **score aggregation** (mean / median / top-k / max), a competition-compliant prediction script, a benchmark evaluation pipeline, three GUI tools, and a headless regression test suite.

The model uses **≈0.856B parameters**, verified against the competition limit of fewer than 2B (see `scripts/param_viewer.py`).

## Setup & Installation

### Requirements

- Python 3.10+ (development environment: 3.11.6)
- Git
- PyTorch + torchvision (**installed separately**, see below)
- Everything else is listed in `requirements.txt`

### Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/deccassie4-prog/robust-aigc-detector.git
cd robust-aigc-detector

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Git Bash:
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate
```

Install PyTorch for your hardware (the two packages must match; do not rely on `requirements.txt` for them):

```bash
# Recent GPUs (RTX 50 series / Blackwell sm_120 and newer) need cu128+ builds:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Older GPUs: pick the matching CUDA build from https://pytorch.org
# CPU-only:
pip install torch torchvision
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Key version constraint: `transformers` must be 5.x — the checkpoint-loading key remap in our code depends on the 5.x layer layout (`dino.model.layer`); 4.x silently drops the LoRA weights. This is pinned in `requirements.txt`.

## Model Weights

All weights are **not included in this repository** (size limits and licensing). Download the following three items into `data/weights/`:

| File / directory | Purpose | Source |
|---|---|---|
| `checkpoint-h-cur.pth` | Phase-2 detector checkpoint | [MIRROR release (Google Drive)](https://drive.google.com/file/d/1gos1QgZA4Xuj706oa5i5E6vsOAoaLyr3/view?usp=sharing) |
| `mirror_phase1.pth` | Phase-1 memory bank weights | [MIRROR release (Google Drive)](https://drive.google.com/file/d/1CpgltI-F7JN7hDyk2O16Ix3Zr_2d2-G0/view?usp=sharing) |
| `dinov3-huge/` | DINOv3-Huge backbone (HF format: `config.json` + `model.safetensors`) | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) — requires accepting Meta's DINOv3 license |

```text
data/weights/
├── checkpoint-h-cur.pth
├── mirror_phase1.pth
└── dinov3-huge/
    ├── config.json
    └── model.safetensors
```

All scripts resolve relative weight paths against the repository root, so the layout above works from any working directory. `data/` is git-ignored.

Note that the three files **overlap** and are not additive: the Phase-2 checkpoint already contains the DINOv3 backbone (initialized from `dinov3-huge/`, plus ~2M LoRA adapter parameters) and the memory bank (trained in Phase 1 and also stored separately in `mirror_phase1.pth`). At inference time the Phase-2 checkpoint overwrites all of them, so exactly **one** model instance participates in the forward pass and the deployed parameter count is ≈0.856B once — you can reproduce this with `scripts/param_viewer.py` on any of the files.

## Output Format (competition deliverable)

`scripts/predict.py` takes one or more image directories and writes a JSON file. The default `--json_format competition` produces the competition-required format — a top-level list with one `image_path` / `pred` entry per image:

```json
[
 {
  "image_path": "C:/data/images/001.jpg",
  "pred": 0.053976
 }
]
```

- `pred` is the probability that the image is AI-generated, in `[0, 1]`; the decision threshold is `0.5`.
- Images that fail to load are reported with `pred = -1` (never silently dropped).
- The optional `--json_format detailed` adds a metadata header and an `is_fake` boolean per image for experimentation.

## Reproduction

### 1. Prepare the Environment

Create and activate a virtual environment and install dependencies as described in [Setup & Installation](#setup--installation), then download the weights into `data/weights/` (see [Model Weights](#model-weights)). A quick sanity check:

```bash
python scripts/smoke_test.py            # loads the three weights, runs one forward pass, prints the parameter count
python scripts/smoke_test.py <IMAGE_DIR> 8   # optional: score 8 real images end-to-end
```

### 2. Prepare the Data

The evaluation set uses a `0_real` / `1_fake` leaf-folder layout (labels come from the folder names — folder names are never used as labels at inference time):

```text
data/datasets/techjam_val/
├── 0_real/    # real images (COCO val2017)
└── 1_fake/    # AI-generated images (DALL·E Advanced from WildFake)
```

`scripts/organize_dataset.py` builds this layout from the raw downloads (hardlinks by default, falls back to copying):

```bash
python scripts/organize_dataset.py --real_dir <COCO_val2017_DIR> --fake_dir <WILDFAKE_ROOT> --out_dir data/datasets/techjam_val
```

Training and development data must not include the validation data specified in the competition rules; this project performs zero-shot inference only.

### 3. Run Inference (competition deliverable; GUI recommended for batch inference)

```bash
# Competition format: per-image {"image_path","pred"} JSON
python scripts/predict.py --data_dir <IMAGE_DIRECTORY> --output_json predictions.json --json_format competition

# Multi-crop + top-k aggregation (catches images the center crop misses)
python scripts/predict.py --data_dir <IMAGE_DIRECTORY> --crop_mode five --aggregate topk

# Several folders in one run: the model loads once, each folder gets its own <folder>.json
python scripts/predict.py --data_dir <DIR_A> <DIR_B> --output_dir results/

# Quick debug run on the first N images
python scripts/predict.py --data_dir <IMAGE_DIRECTORY> --limit 8
```

Useful options: `--crop_mode {center,five,grid3x3,grid4x4,grid5x5,multiscale,single336}`, `--aggregate {mean,median,topk,max}`, `--use_amp` (fp16, recommended), `--batch_size` (counts **crops** per forward batch; default 32 ≈ 10 GB VRAM, 64 is the verified ceiling ≈ 15 GB), `--dump_scores` (per-crop score CSV for error analysis).

### 4. Run the Benchmark Evaluation

```bash
python evaluation/evaluate.py --base_data_path data/datasets --benchmarks TechjamVal --batch_size 64 --num_workers 2
```

This prints Acc / Bal_Acc / AUC / AP per sub-dataset and writes a timestamped CSV per benchmark into `./results/`. Weight paths default to `data/weights/` and are overridable via `--model_path / --memory_path / --backbone_path`.

### 5. GUI Tools

```bash
python scripts/predict_gui.py      # sub-batch inference frontend
python scripts/pred_filter_gui.py  # collect images by pred range (error-analysis picker)
python scripts/param_viewer.py     # parameter count + <2B compliance of any checkpoint
```

- **predict_gui** — one tab per sub-batch (its own input directories and output folder); sub-batches run sequentially as fully isolated `predict.py` subprocesses, so a failing batch never affects the rest. Parameters are generated dynamically from `scripts/gui_config.json`.
- **pred_filter_gui** — loads `predict.py` output JSONs (both formats), filters images by score range, copies them renamed to their pred value with a `manifest.csv` — built for error analysis.
- **param_viewer** — picks any checkpoint file (`.pth/.pt/.bin/.ckpt/.safetensors`) or HF-format weight directory and reports total parameter count, `<2B` compliance verdict, composition by top-level module, and the largest tensors. Pure metadata-based counting; no model is instantiated. This tool produced the ≈0.856B compliance figure.

### 6. Run the Tests

```bash
pytest tests/ -q
```

or without pytest installed:

```bash
python tests/test_crop_modes.py        # crop strategies: pixel equivalence, block counts, positioning
python tests/test_predict_units.py     # command building, progress parsing, result schemas
python tests/test_gui_hover.py         # GUI layout stability regression
python tests/test_pred_filter_gui.py   # pred_filter end-to-end (10 check groups)
```

All suites are headless (no GPU, no model weights needed). The GUI tests require a display; on a headless Linux CI run them under `xvfb-run`.

## Robustness Evaluation Summary

*This section is finalized with the clean-vs-transformed comparison table before submission; the underlying numbers live in [`evaluation/results/`](evaluation/results/).*

Transformations under evaluation (the competition subset): JPEG compression (Q90/70/50/30), Gaussian blur (σ 0.5/1.0/2.0), rescaling (0.5×/0.25× then upsample), Gaussian noise (σ 0.02/0.05/0.10), color jitter (brightness/contrast/saturation ±20%), center crop (80%).

## Error Analysis

*This section links the error-analysis report and residual-heatmap visualisations; the report lives in [`analysis/`](analysis/).*

## Demo Video

*The public YouTube demo link is added here upon submission (also referenced from the Devpost description).*

## Limitations

Our prototype is designed for a hackathon-scale setting and has several limitations:

- **Limited generalisation:** Performance may vary for AI-generated images from generators or distributions that are not represented in the training data.
- **Transformation coverage:** We evaluate a selected subset of real-world transformations rather than every possible type of image manipulation.
- **False positives and false negatives:** No image-level detector is perfectly reliable. Authentic images may be classified as AIGC, while some AI-generated images may be classified as authentic.
- **Zero-shot scope:** We deliberately did not fine-tune on any competition-related data, which keeps the pipeline honest but leaves accuracy on specific local generators on the table.
- **Compute footprint:** The DINOv3-Huge backbone needs a discrete GPU for practical throughput (≈3.4 GB of weights plus activation memory scaling with the crop batch size).
- **Evolving generative models:** Image generation techniques continue to improve, so detection performance may degrade as new generation methods emerge.

Given more time and resources, we would expand the diversity of training data, evaluate against a wider range of generators and transformations, and investigate stronger domain-generalisation and calibration methods.

## Team Contributions

This project was developed collaboratively by a five-member team.

### Zhang Yiran — Inference Pipeline & Model Execution

* Cloned the upstream MIRROR repository and verified the upstream MIT license.
* Set up the required environment and downloaded the required model weights.
* Organised the validation dataset into the required real and AI-generated image structure.
* Ran the original inference pipeline and established the clean-image baseline performance.
* Verified the total number of model parameters against the competition limit of fewer than 2 billion parameters, and built `scripts/param_viewer.py` to make the check reproducible for any checkpoint.
* Developed `predict.py`, which takes an image directory as input and outputs a JSON file containing `image_path` and `pred` for each image, plus `predict_gui.py` as its batch-oriented frontend.
* Documented the environment setup and implementation issues for the rest of the team.

### Liu Zhenning — Robustness Evaluation

* Developed image transformation scripts covering JPEG compression, Gaussian blur, resizing, Gaussian noise, colour jitter, and centre cropping.
* Generated clean and transformed versions of the validation images.
* Ran batch inference after the inference environment was established.
* Produced a comparison of performance on clean versus transformed images.
* Shared per-image prediction results for subsequent error analysis.

### Chen Yizhen — Repository & Project Management

* Set up and maintained the GitHub repository and project directory structure.
* Established the README structure, including project overview, installation, reproduction steps, limitations, and team contributions.
* Defined the team's Git collaboration workflow, including branches, commits, pull requests, and file organisation.
* Integrated the final deliverables, including the inference script, evaluation pipeline, robustness results, error analysis, and documentation materials.
* Preserved the upstream MIT license and included appropriate attribution to the MIRROR project.

### Zhou Shuyuan — Error Analysis & Explainability

* Studied the two-stage MIRROR approach, with particular attention to residual-based comparison as an interpretable signal.
* Selected representative false positives and false negatives from per-image prediction results.
* Produced residual heatmap visualisations to analyse regions contributing to model decisions.
* Prepared the error analysis report, including representative cases and discussions of trade-offs such as robustness versus false-positive rates.

### Wang Zihan — Devpost Documentation & Demo Video

* Prepared the written Devpost project description, including the tools, models, libraries, and datasets used.
* Documented that the project uses pretrained weights for zero-shot inference and does not use the competition validation data for training.
* Designed the end-to-end demonstration flow, including image inputs, prediction scores, robustness evaluation results, and explainability visualisations.
* Recorded and edited the demo video.
* Uploaded the demo video to YouTube with public visibility and added the link to the Devpost submission and repository README.

## Project Workflow

The deliverables were built in the following dependency order (team member in parentheses):

1. **Inference pipeline & model execution** (Zhang Yiran) — the base that everything else depends on.
2. **Robustness evaluation** (Liu Zhenning) — depends on the inference environment and prediction pipeline.
3. **Error analysis** (Zhou Shuyuan) — depends on the per-image prediction results from 1 and 2.
4. **Demo & Devpost** (Wang Zihan) — depends on the robustness comparison and explainability results.
5. **Repository integration & documentation** (Chen Yizhen) — continuous throughout; final integration of 1–4 into this repository and README.

## Repository Structure

```text
robust-aigc-detector/
├── analysis/               # error analysis, visualisations, and reports
├── assets/                 # images and demo-related assets
├── configs/                # configuration files
├── data/                   # local weights & datasets (git-ignored; see Model Weights)
├── evaluation/
│   ├── evaluate.py         # multi-benchmark evaluation (Acc/Bal_Acc/AUC/AP → CSV)
│   └── results/            # robustness summary tables
├── scripts/
│   ├── predict.py          # competition deliverable: image dir → per-image score JSON
│   ├── predict_gui.py      # sub-batch inference frontend
│   ├── pred_filter_gui.py  # collect images by pred range (error analysis)
│   ├── param_viewer.py     # parameter count / <2B compliance viewer
│   ├── smoke_test.py       # weights + forward-pass sanity check
│   ├── organize_dataset.py # build the 0_real/1_fake dataset layout
│   └── gui_config.json     # predict_gui parameter registry
├── src/mirror/             # MIRROR detector model definition
├── tests/                  # headless regression tests (pytest or direct run)
├── .gitignore / .gitattributes
├── CONTRIBUTING.md         # Git and collaboration guidelines
├── LICENSE                 # MIT License
├── README.md               # this file
└── requirements.txt        # Python dependencies
```

## Acknowledgements

This project builds upon the MIRROR (Manifold Ideal Reference ReconstructOR) project for generalizable AI-generated image detection.

* Upstream repository: <https://github.com/handsome-rich/MIRROR>
* Paper: Ruiqi Liu et al., "MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection", arXiv:2602.02222, 2026.
* The MIRROR codebase and pretrained models are used as an upstream component of our prototype under the MIT license; the files derived from it in `src/mirror/` and `evaluation/` retain attribution to the original authors. The DINOv3 backbone weights are provided by Meta AI under their own license and must be obtained from the [official repository](https://github.com/facebookresearch/dinov3).

Our work focuses on evaluating and improving the robustness of AI-generated image detection under realistic image transformations, including compression, blur, resizing, noise, color adjustment, and cropping.
