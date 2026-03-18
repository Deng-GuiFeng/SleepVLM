<p align="center">
  <img src="assets/SleepVLM_logo.png" alt="SleepVLM Logo" width="200">
</p>

<h1 align="center">SleepVLM</h1>

<p align="center">
  <strong>Explainable and Rule-Grounded Sleep Staging via a Vision-Language Model</strong>
</p>

<p align="center">
  <a href="#">Paper</a> &nbsp;|&nbsp;
  <a href="https://github.com/Deng-GuiFeng/MASS-EX">MASS-EX Dataset</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/collections/Feng613/sleepvlm">Model Weights</a>
</p>

---

## Overview

Automated sleep staging has approached expert-level accuracy, yet clinical trust remains limited because most models output stage labels without auditable reasoning. **SleepVLM** bridges this gap by coupling competitive classification performance with transparent, rule-grounded explanations.

SleepVLM is a vision-language model that stages sleep from rendered multi-channel polysomnography (PSG) waveform images — mimicking how human sleep technologists visually inspect PSG traces. It generates structured outputs comprising:

- A predicted **sleep stage** (W, N1, N2, N3, R)
- Cited **AASM rule identifiers** (from 15 operationalized rules)
- A clinician-readable **natural language rationale**

The model is built on [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and fine-tuned via LoRA with a two-phase training pipeline. We also release [MASS-EX](https://github.com/Deng-GuiFeng/MASS-EX), an expert-annotated dataset to support future interpretable sleep-staging research.

<p align="center">
  <img src="assets/pipeline.png" alt="SleepVLM Pipeline" width="800">
</p>

**Figure 1.** SleepVLM pipeline overview. PSG signals are rendered as standardized waveform images, then processed through a two-phase training framework: Phase 1 (waveform-perceptual pre-training) and Phase 2 (rule-grounded supervised fine-tuning).

## Highlights

- **Explainable by design**: Generates clinician-readable rationales grounded in AASM Version 3 sleep staging rules, not post-hoc saliency maps
- **Visual perception approach**: Renders 6-channel PSG signals (F4-M1, C4-M1, O2-M1, LOC, ROC, Chin EMG) as standardized 448 &times; 224 px images
- **Contextual staging**: Uses a 3-epoch sliding window (preceding, current, subsequent) for temporal context, same as human technologists
- **Competitive performance**: Cohen's kappa 0.767 on MASS-SS1; expert-rated reasoning quality > 4.0/5.0 across all dimensions
- **Efficient deployment**: Supports W4A16 quantization for single-GPU (24 GB) inference with minimal performance loss

## Training Pipeline

| Phase | Name | Training Data | Input | Output | Vision Encoder |
|-------|------|---------------|-------|--------|----------------|
| Phase 1 | Waveform-Perceptual Pre-training (WPT) | MASS-SS2/4/5 (85 subjects) | Single epoch image | Per-second spectral & amplitude features | Unfrozen |
| Phase 2 | Rule-Grounded Supervised Fine-tuning (SFT) | MASS-SS3 via MASS-EX (50 subjects) | 3-epoch window | Sleep stage + rules + reasoning | Frozen |

## Datasets

### MASS (Montreal Archive of Sleep Studies)

This project uses the following MASS subsets. Apply for access at the [MASS repository](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS).

| Subset | Subjects | Role in SleepVLM | Scoring Standard |
|--------|----------|-------------------|-----------------|
| MASS-SS1 | 53 | Held-out test set | AASM (30-s epochs) |
| MASS-SS2 | 19 | Phase 1 WPT | R&K (20-s epochs) |
| MASS-SS3 | 62 | Phase 2 SFT (5 fine + 45 coarse training, 12 validation) | AASM (30-s epochs) |
| MASS-SS4 | 40 | Phase 1 WPT | R&K (20-s epochs) |
| MASS-SS5 | 26 | Phase 1 WPT | R&K (20-s epochs) |

> **Note:** Phase 1 WPT does not use sleep stage labels — it trains the model to perceive waveform morphology via spectral/amplitude prediction. The R&K scoring standard and 20-s epoch duration of SS2/4/5 therefore do not affect WPT supervision targets.

### MASS-EX (Expert Annotations)

[MASS-EX](MASS-EX/) provides expert-annotated data for all 62 MASS-SS3 subjects, containing:

- **Fine annotations** (5 subjects, 5,006 epochs): sleep stage + AASM rule identifiers + expert-written rationale
- **Coarse annotations** (57 subjects, 54,187 epochs): sleep stage + AASM rule identifiers

See [MASS-EX/README.md](MASS-EX/README.md) for details on the annotation pipeline and format.

## Project Structure

```
SleepVLM/
├── README.md                          # This file
├── LICENSE                            # Apache License 2.0
├── CITATION.cff                       # Machine-readable citation metadata
├── split.json                         # Train/val/test subject split
├── .gitignore
├── .gitmodules                        # MASS-EX as git submodule
├── requirements/                      # Python dependencies (per environment)
│   ├── train.txt                      #   Training
│   ├── inference.txt                  #   Inference (vLLM)
│   └── quantize.txt                   #   Quantization (AutoRound)
│
├── assets/                            # Logo and figures for README
│   ├── SleepVLM_logo.png
│   └── pipeline.png
│
├── MASS-EX/                           # Git submodule: expert annotations for MASS-SS3
│   ├── annotations/{fine,coarse}/     # Sleep staging annotations
│   ├── scripts/                       # MASS-SS3 preprocessing utilities
│   └── sleep_staging_rules.md         # 15 AASM-based scoring rules
│
├── configs/                           # Reference configuration files (YAML)
│   ├── phase1_wpt.yaml
│   ├── phase2_sft.yaml
│   ├── quantization.yaml
│   └── inference.yaml
│
├── data/                              # Data directory (gitignored, user-generated)
│   └── MASS/
│       ├── SS1/                       # Held-out test set (53 subjects)
│       │   ├── *.edf                  #   Raw EDF files (from MASS)
│       │   └── images/                #   Rendered waveform images
│       ├── SS2/                       # Phase 1 WPT (19 subjects)
│       │   ├── *.edf, images/, band_power/
│       ├── SS3/                       # Phase 2 SFT (62 subjects)
│       │   ├── *.edf, images/
│       ├── SS4/                       # Phase 1 WPT (40 subjects)
│       │   ├── *.edf, images/, band_power/
│       └── SS5/                       # Phase 1 WPT (26 subjects)
│           ├── *.edf, images/, band_power/
│
├── sleepvlm/                          # Main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocess.py              # Unified MASS preprocessing (all subsets)
│   │   ├── renderer.py                # PSG waveform image rendering
│   │   └── wpt_targets.py             # Phase 1 spectral/amplitude target generation
│   ├── prompts/                       # System prompts for each training phase
│   │   ├── phase1_wpt.md              # WPT system prompt
│   │   ├── phase2_sft_fine.md         # SFT fine-track prompt (with reasoning)
│   │   └── phase2_sft_coarse.md       # SFT coarse-track prompt (rules only)
│   ├── training/
│   │   └── train.py                   # Unified LoRA training (Phase 1 & 2)
│   ├── inference/
│   │   └── predict.py                 # Batch inference via vLLM
│   └── evaluation/
│       ├── metrics.py                 # Acc, Macro-F1, Kappa, bootstrap CI
│       └── parse_output.py            # Parse structured model JSON output
│
└── scripts/                           # CLI scripts & launch helpers
    ├── preprocess_all.sh              # Preprocess all MASS subsets
    ├── prepare_wpt_data.py            # Prepare Phase 1 training JSONL
    ├── prepare_sft_data.py            # Prepare Phase 2 training JSONL
    ├── train_phase1.sh                # Launch Phase 1 WPT training
    ├── train_phase2.sh                # Launch Phase 2 SFT training
    ├── run_inference.sh               # Launch vLLM servers & run inference
    ├── merge_lora.py                  # Merge LoRA adapters into base model
    ├── quantize.py                    # W4A16 post-training quantization
    └── evaluate.py                    # Compute evaluation metrics
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+
- 8 &times; NVIDIA A100 80GB (training) or 1 &times; NVIDIA RTX 4090 24GB (inference with quantization)

### Setup

```bash
# Clone the repository with MASS-EX submodule
git clone --recurse-submodules https://github.com/Deng-GuiFeng/SleepVLM.git
cd SleepVLM

# Create conda environment for training
conda create -n SleepVLM python=3.10
conda activate SleepVLM
pip install -r requirements/train.txt

# (Optional) Create separate environment for inference with vLLM
conda create -n SleepVLM-infer python=3.10
conda activate SleepVLM-infer
pip install -r requirements/inference.txt
```

> **Note:** Training and inference may require separate environments due to vLLM version constraints. If your system supports a unified installation, you can install both `requirements/train.txt` and `requirements/inference.txt` into a single environment.

## Quick Start

### 1. Prepare MASS Data

Download MASS data from the [MASS repository](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS) and place EDF files in the following structure:

```
data/MASS/
├── SS1/               # 01-01-0001 PSG.edf, 01-01-0001 Base.edf, ...
├── SS2/               # 01-02-0001 PSG.edf, 01-02-0001 Base.edf, ...
├── SS3/               # 01-03-0001 PSG.edf, 01-03-0001 Base.edf, ...
├── SS4/               # 01-04-0001 PSG.edf, ...
└── SS5/               # 01-05-0001 PSG.edf, ...
```

Download the base model:
```bash
# Place Qwen2.5-VL-3B-Instruct in models/
# Option 1: HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/Qwen2.5-VL-3B-Instruct
```

### 2. Preprocess & Render Waveform Images

```bash
# Preprocess all MASS subsets (signal filtering, resampling, rendering)
bash scripts/preprocess_all.sh
```

### 3. Prepare Training Data

```bash
# Generate Phase 1 WPT training data (spectral/amplitude targets)
python scripts/prepare_wpt_data.py

# Generate Phase 2 SFT training data (using MASS-EX annotations)
python scripts/prepare_sft_data.py
```

### 4. Training

```bash
# Phase 1: Waveform-Perceptual Pre-training
bash scripts/train_phase1.sh

# Phase 2: Rule-Grounded Supervised Fine-tuning
bash scripts/train_phase2.sh
```

### 5. Inference

```bash
# Run inference on test set
bash scripts/run_inference.sh
```

### 6. Evaluation

```bash
# Compute classification metrics and generate reports
python scripts/evaluate.py --prediction_dir outputs/predictions --test_set MASS-SS1
```

## Pre-trained Models

| Model | Quantization | Size | HuggingFace |
|-------|-------------|------|-------------|
| SleepVLM (full precision) | BF16 | ~6 GB | [Feng613/SleepVLM](https://huggingface.co/Feng613/SleepVLM) |
| SleepVLM (quantized) | W4A16 | ~2 GB | [Feng613/SleepVLM-W4A16](https://huggingface.co/Feng613/SleepVLM-W4A16) |

## Citation

If you use SleepVLM or MASS-EX in your research, please cite:

```bibtex
@article{deng2026sleepvlm,
  author  = {Deng, Guifeng and Wang, Pan and Wang, Jiquan and Li, Tao and Jiang, Haiteng},
  title   = {{SleepVLM}: Explainable and Rule-Grounded Sleep Staging
             via a Vision-Language Model},
  journal = {},
  year    = {2026}
}

@dataset{deng2026massex,
  author    = {Deng, Guifeng and Wang, Pan and Li, Tao and Jiang, Haiteng},
  title     = {{MASS-EX}: Expert-Annotated Dataset for Interpretable Sleep Staging},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The [MASS-EX](MASS-EX/) dataset is licensed under [CC BY-NC 4.0](MASS-EX/LICENSE). Use of the underlying MASS PSG signals is subject to the [MASS data use agreement](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS).

## Acknowledgements

This work was supported by the National Science and Technology Major Project, National Natural Science Foundation of China, Key R&D Program of Zhejiang, and other funding agencies. See the paper for full acknowledgements.

We thank the developers of [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [vLLM](https://github.com/vllm-project/vllm), [Intel AutoRound](https://github.com/intel/auto-round), and the [MASS](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS) team for making their resources publicly available.
