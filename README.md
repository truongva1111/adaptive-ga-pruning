# Adaptive GA-based Structured Pruning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IJEECS%202026-brightgreen.svg)](#citation)
[![DOI](https://img.shields.io/badge/DOI-10.11591%2Fijeecs.v43.i1.pp314--324-blue.svg)](https://doi.org/10.11591/ijeecs.v43.i1.pp314-324)

Official implementation of **"Layer-wise Adaptive Structured Pruning via Genetic Algorithms with Taylor-based Proxy Fitness"** — published in *Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)*, Vol. 43, No. 1, pp. 314-324, 2026 ([DOI: 10.11591/ijeecs.v43.i1.pp314-324](https://doi.org/10.11591/ijeecs.v43.i1.pp314-324)).

## Highlights

- 🎯 **92.78 ± 0.28% accuracy** on CIFAR-10 (over 50 independent search runs) with **70.0 ± 3.2% MACs reduction**
- 🧬 Novel **layer-wise hybrid strategy** jointly optimizing pruning ratios AND strategies (Min-Importance / Median-Rank)
- ⚡ **Training-free** Taylor-based proxy fitness — entire NSGA-II search completes in **under 15 seconds** on RTX 4090
- 🛡️ Robust against **Layer Collapse** under aggressive compression

## Overview

<p align="center">
  <img src="figure1.png" width="90%">
</p>

We propose an adaptive structured pruning framework that jointly optimizes:
- **How much** to prune (layer-wise pruning ratios `r_l ∈ (0,1]`)
- **How** to prune (Min-Importance vs. Median-Rank strategy `s_l` per layer)

Unlike rigid global pruning heuristics, our method uses NSGA-II to discover Pareto-optimal solutions balancing accuracy, MACs, and model size.

## Results

### CIFAR-10 (VGG16)

| Method           | Accuracy (%)     | Params ↓ (%)   | MACs ↓ (%)     |
|------------------|------------------|----------------|----------------|
| Baseline         | 93.62            | 0              | 0              |
| L1-Norm          | 93.40            | 64.0           | 34.2           |
| HRank            | 93.43            | 82.9           | 53.5           |
| ABCPruner        | 93.08            | 88.68          | 73.68          |
| **Ours (GA-SO)** | 92.33            | 31.0           | 73.6           |
| **Ours (GA-MO)** | **92.78 ± 0.28** | **35.3 ± 2.4** | **70.0 ± 3.2** |

GA-MO results are reported as **mean ± std over 50 independent NSGA-II search runs** (see [Reproducibility](#reproducibility)).

### CIFAR-100 (VGG16)

| Method           | MACs (G)  | MACs ↓ (%) | Accuracy (%) | Acc Drop |
|------------------|-----------|------------|--------------|----------|
| Baseline         | 0.33      | 0          | 72.78        | -        |
| Global Min       | 0.12      | 63.6       | 69.91        | -2.87    |
| Global Median    | 0.14      | 57.6       | 71.02        | -1.76    |
| Global Max       | 0.10      | 69.7       | 68.57        | -4.21    |
| **Ours (GA-MO)** | **0.116** | **64.8**   | **71.82**    | **-0.96**|
| **Ours (GA-SO)** | 0.086     | 73.9       | 70.68        | -2.10    |

## Reproducibility

### Environment

- Python 3.10.12
- PyTorch 2.0+ with CUDA
- Hardware tested: NVIDIA RTX 4090 GPU (24 GB VRAM)

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy
pip install --upgrade torch_pruning
pip install matplotlib pandas
```

### Hyperparameters

**Data pipeline (VGG16 / CIFAR):**
| Parameter           | Value |
|---------------------|-------|
| Batch size          | 128   |
| Num workers         | 2     |
| Normalization       | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Train augmentation  | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` |
| Classifier type     | `B` — Linear(512, 4096) → Linear(4096, 4096) → Linear(4096, num_classes) |

**Taylor importance pre-computation:**
| Parameter           | Value |
|---------------------|-------|
| Number of epochs    | 1 (single ranking epoch) |
| Layer normalization | Per-layer L2 normalization of accumulated `|activation × gradient|` |

**Single-Objective GA (GA-SO):**
| Parameter                       | Value      |
|---------------------------------|------------|
| Population size                 | 50         |
| Generations                     | 30         |
| Elite ratio                     | 0.2        |
| Crossover                       | Uniform-mask (`rand > 0.5`) |
| Mutation probability            | 0.1        |
| Mutation (ratio gene)           | Gaussian noise σ=0.1, clipped to [0.05, 1.0] |
| Mutation (strategy gene)        | Bit flip   |
| Hard min retention ratio        | 0.10       |
| Channel alignment               | Multiples of 8 |
| MACs soft penalty               | `10·v + 20·v²` if `macs_ratio > target` |
| Structural penalty              | `3·(0.4 - ratio)²` if attenuation<0.2 AND ratio<0.4 |
| Params penalty weight           | 0.10       |
| Target MACs ratio (CIFAR-10)    | 0.20       |
| Random seed (reference run)     | 2001       |

**Multi-Objective NSGA-II (GA-MO):**
| Parameter                       | Value      |
|---------------------------------|------------|
| Population size                 | 50         |
| Generations                     | 30         |
| Crossover                       | Single-point |
| Mutation probability (per gene) | 0.1        |
| Mutation (ratio gene)           | Uniform reinit in [`min_ratio`, 1.0] |
| Mutation (strategy gene)        | Uniform reinit in [0, 1] |
| Hard min retention ratio        | 0.10       |
| Channel alignment               | Multiples of 8 |
| Objectives                      | Maximize proxy-acc; minimize MACs and params |
| Knee-point criterion            | Min Euclidean distance to ideal point in normalized objective space |
| Random seed (reference run)     | 2026       |

**Fine-tuning with Knowledge Distillation (KD):**
| Parameter             | Value |
|-----------------------|-------|
| Epochs                | 150   |
| Optimizer             | SGD   |
| Initial learning rate | 0.01  |
| Momentum              | 0.9   |
| Weight decay          | 5e-4  |
| LR scheduler          | Cosine annealing (`T_max = 150`) |
| Gradient clipping     | `max_norm = 5.0` |
| KD temperature `T`    | 4.0   |
| KD weight `α`         | 0.9 (soft loss) / 0.1 (hard CE) |

**Fine-tuning without KD (for reference):** identical to KD setup except `lr = 0.001` and pure cross-entropy loss.

### Pre-trained Baseline Checkpoints

| Model | Dataset | Accuracy | Link |
|-------|---------|----------|------|
| VGG16 | CIFAR-10 | 93.62% | [Download](https://drive.google.com/drive/folders/1l-xspOKsrGxSasfUpyZtyL3q4sP8iHBP?usp=sharing) |
| VGG16 | CIFAR-100 | 72.78% | [Download](https://drive.google.com/drive/folders/1l-xspOKsrGxSasfUpyZtyL3q4sP8iHBP?usp=sharing) |

### Quick Start

```bash
git clone https://github.com/truongva1111/adaptive-ga-pruning.git
cd adaptive-ga-pruning
jupyter notebook adaptive-ga-pruning.ipynb
```

## Experimental Pipeline

The pruning pipeline involves Taylor pre-computation, evolutionary search (GA-SO or NSGA-II), physical pruning, and fine-tuning. Detailed algorithm steps are available in the paper. Below are specific execution details regarding the code structure.

### Notebook Cell Execution Map

The interactive notebook (`adaptive-ga-pruning.ipynb`) is organized as follows:

| Notebook Section | Purpose | Dependency |
|---|---|---|
| 1. Dependencies | Install/import packages | — |
| 2. Setup VGG16 + Config + Common function | Load model, set hyperparameters | Section 1 |
| 3. Training mô hình gốc (Baseline) | Train or load pre-trained VGG16 | Section 1-2 |
| 4. Global pruning | Empirical analysis of global strategies (Min/Median/Max) | Section 1–3 |
| **5. GAPruner** | Define GAPruner class (fitness evaluation, evolution) | Section 1–3 |
| **5.1 Building Ranking Tables (Pre-computation)** | **Stage 1:** Taylor pre-computation | Section 5 |
| **5.2. Starting GA Evolution** | **Stage 2 (GA-SO):** Single-objective search | Section 5.1 |
| **5.3. Physical pruning GA Execution** | **Stage 3:** Apply GA-SO result | Section 5.2 |
| **5.4. Finetune** | **Stage 4:** Fine-tune GA-SO model (Normal / KD) | Section 5.3 |
| **6. NSGAPruner** | Define NSGAPruner class (multi-objective evolution) | Section 5 (GAPruner) |
| **6.1. NSGA execution** | **Stage 2 (NSGA-II):** Multi-objective search | Section 6 |
| **6.2. Physical pruning NSGA** | **Stage 3:** Apply NSGA-II knee point | Section 6.1 |
| **6.3. Finetune** | **Stage 4:** Fine-tune NSGA-II model (Normal / KD) | Section 6.2 |

> ⚠️ **Critical:** Section 6 (NSGA-II) depends on the `GAPruner` class in Section 5. You must run Cells 1–3 and first cells in Section 5 before jumping to Section 6.

> [!WARNING]
> **For Reproducers:** When experimenting with different models or datasets, please pay close attention to the `dataset_class`, checkpoint filenames, and `save_name` variables in the configuration. Ensure these are updated correctly to avoid mixing up checkpoints, ranking tables, and logs between different dataset/model combinations.

### Standalone Script (ResNet-56)

The `resnet56_ga_pruning_std.py` script runs the entire pipeline end-to-end with automatic checkpointing and resume support:

```bash
python resnet56_ga_pruning_std.py \
    --dataset CIFAR10 \
    --target_macs 0.30 \
    --pop_size 50 \
    --generations 30 \
    --kd_epochs 150 \
    --seed 42
```

The script outputs:
- `checkpoint_resnet56/` — baseline, pruned architecture, and fine-tuned student checkpoints
- `logs_resnet56/` — Pareto front JSON, knee point JSON, ranking tables (`.npz`), summary
- `figures_resnet56/` — Pareto front plot, training curve

### Physical Pruning (Stage 3)

*Note: This stage is an engineering step between search and fine-tuning and is only described implicitly in the paper.*

1. Decode the selected chromosome (GA-SO best or NSGA-II knee point) into per-layer channel counts and strategies.
2. For each layer, calculate the number of filters to prune and determine their specific indices based on the selected strategy (Min-Importance or Median-Rank).
3. Physically remove the selected filters. The implementation adapts to the architecture's complexity:
   - **For VGG16 (Interactive Notebook):** Performs sequential, layer-by-layer manual filter and batch normalization slicing. This approach is lightweight and efficient for standard feed-forward architectures.
   - **For ResNet (Standalone Script):** Utilizes [`torch_pruning`](https://github.com/VainF/Torch-Pruning) `DependencyGraph` to automatically propagate dimensional changes across residual skip connections and coupled layers. The script supports both `torch_pruning` v0.x (`get_pruning_plan`) and v1.x+ (`get_pruning_group`) APIs with an automatic fallback mechanism.
4. The result is a **structurally smaller model** with reduced Conv2d dimensions — no masking or sparse tensors.

### Reproducing the 50-Run Statistics

The reported `92.78 ± 0.28%` figure is the mean ± std of **50 independent runs of the full pipeline** (ranking → NSGA-II → knee selection → KD fine-tuning). Only the **baseline (teacher) checkpoint is shared** across runs; every run re-seeds RNGs, recomputes its own Taylor ranking tables, evolves its own Pareto front, picks its own knee point, and fine-tunes for 150 epochs with KD.

The full statistical pipeline is implemented as a standalone script for the residual-architecture validation (`resnet56_ga_pruning_std.py`) — the same pattern applies to VGG16. To launch multiple search runs:

```bash
python resnet56_ga_pruning_std.py \
    --dataset CIFAR10 \
    --target_macs 0.30 \
    --pop_size 50 \
    --generations 30 \
    --n_search_runs 50 \
    --seed 42
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{vo2026adaptive,
  title   = {Layer-wise Adaptive Structured Pruning via Genetic Algorithms with Taylor-based Proxy Fitness},
  author  = {Vo, Anh-Truong and Tran, Hoang-Loc and Phan, Dinh-Duy and Vu, Duc-Lung},
  journal = {Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)},
  volume  = {43},
  number  = {1},
  pages   = {314--324},
  year    = {2026},
  doi     = {10.11591/ijeecs.v43.i1.pp314-324},
  url     = {https://doi.org/10.11591/ijeecs.v43.i1.pp314-324}
}
```

**DOI:** [10.11591/ijeecs.v43.i1.pp314-324](https://doi.org/10.11591/ijeecs.v43.i1.pp314-324)  
**URL:** [https://ijeecs.iaescore.com/index.php/IJEECS/article/view/45839](https://ijeecs.iaescore.com/index.php/IJEECS/article/view/45839)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MMLab**, University of Information Technology, Vietnam National University Ho Chi Minh City — computational resources.
- [pytorch-pruning](https://github.com/jacobgil/pytorch-pruning) by Jacob Gildenblat — reference implementation for filter pruning utilities.

## Contact

For questions, please contact: **truongva.18@grad.uit.edu.vn** (Anh-Truong Vo)
or **lungvd@uit.edu.vn** (Duc-Lung Vu, corresponding author).
