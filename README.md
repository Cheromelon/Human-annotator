# Modeling Human Annotator Disagreement with Soft Labels on CIFAR-10H

## Overview

This project investigates whether deep learning models can learn not just *what* an image is, but *how uncertain* human annotators are about it. Using the CIFAR-10H dataset — which provides probabilistic human annotation distributions rather than hard labels — two neural network architectures are trained and evaluated on their ability to replicate annotator disagreement patterns across three different loss functions.

The central question: can a model learn to produce probability distributions that reflect genuine human perceptual ambiguity, and which combination of architecture and loss function does this best?

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models](#models)
- [Loss Functions](#loss-functions)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Visualization](#results-visualization)
- [Requirements](#requirements)

---

## Dataset

**CIFAR-10** (50,000 training / 10,000 test images, 32x32 RGB, 10 classes)

**CIFAR-10H** — a human annotation dataset overlaid on the CIFAR-10 test set. Each of the 10,000 test images has an associated probability vector (summing to 1.0) representing the proportion of human annotators who assigned each class label. This distribution captures genuine perceptual ambiguity — for example, an image that looks like both a cat and a dog will have non-zero probability mass across both classes.

Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

The CIFAR-10H soft label file is downloaded automatically from the [official repository](https://github.com/jcpeterson/cifar-10h) at runtime.

---

## Project Structure

```
human_annotator_project.ipynb   # Main notebook
resnet20_pretrained_weights.weights.h5
customcnn_pretrained_weights.weights.h5
resnet20_kl_best.keras
resnet20_ce_best.keras
resnet20_custom_best.keras
customcnn_kl_best.keras
customcnn_ce_best.keras
customcnn_custom_best.keras
```

---

## Methodology

The project follows a two-phase training strategy:

**Phase 1 — Pretraining on Hard Labels**

Both models are first pretrained on the full CIFAR-10 training set (50,000 images) using standard categorical cross-entropy with one-hot labels. This gives each model a strong feature representation before it encounters the soft label task. Pretrained weights are saved and reused as the initialization point for all fine-tuning experiments.

**Phase 2 — Fine-tuning on Soft Labels**

The CIFAR-10H test set (10,000 images) is split into three subsets:

| Split | Size |
|---|---|
| Soft Train | 6,000 images |
| Soft Validation | 2,000 images |
| Soft Test | 2,000 images |

Each model is then fine-tuned from its pretrained weights three separate times — once per loss function — using the soft label distributions as targets.

---

## Annotator Disagreement Analysis

Before model training, the CIFAR-10H soft labels are analyzed to characterize the nature and distribution of human disagreement:

- **Shannon entropy** is computed per image to quantify annotator uncertainty. High entropy indicates roughly equal probability mass spread across classes (high disagreement); low entropy indicates strong annotator consensus.
- **Distribution statistics**: mean, standard deviation, max, and min entropy across the test set.
- **Per-class entropy**: average annotator uncertainty broken down by true class, revealing which categories are most ambiguous to humans (e.g., cats and dogs tend to be confused more often than ships and airplanes).
- **Confusion-style matrix**: constructed from the average soft label distribution per true class. Unlike a standard prediction confusion matrix, this visualizes *where annotator probability mass flows* — e.g., how much probability annotators assign to "cat" when the true label is "dog".
- **Qualitative visualization**: images with the lowest and highest entropy are displayed side-by-side, illustrating that high-entropy images are genuinely ambiguous even to a human observer.

---

## Models

### Custom CNN

A sequential convolutional network designed for the 32x32 input resolution of CIFAR-10:

```
Conv2D(32) -> MaxPool
Conv2D(64) -> MaxPool
Conv2D(128) -> MaxPool
Flatten
Dense(256) -> Dropout(0.3)
Dense(128) -> Dropout(0.3)
Dense(10, softmax)
```

### ResNet-20

A custom implementation of the ResNet-20 architecture adapted for small images, built using the functional Keras API:

- Initial 16-filter convolution followed by Batch Normalization
- Three residual stages with 16, 32, and 64 filters respectively (3 blocks per stage)
- Downsampling via strided convolutions with 1x1 projection shortcuts to match dimensions
- Global Average Pooling followed by an MLP head (Dense 256 -> Dropout -> Dense 128 -> Dropout -> Dense 10)

The residual connection implementation handles both same-dimension shortcuts and dimension-changing shortcuts (when downsampling) via projection convolutions.

---

## Loss Functions

Three loss functions are compared for the soft label fine-tuning task:

**1. KL Divergence (`kl_loss`)**

Measures how much the predicted probability distribution diverges from the human annotator distribution. Minimizing KL divergence directly penalizes predictions that are confident where annotators are uncertain, and vice versa.

**2. Categorical Cross-Entropy (`ce_loss`)**

Standard cross-entropy applied to soft targets rather than one-hot vectors. Treats the soft labels as a weighted sum of per-class losses. A widely used baseline for label smoothing and soft label scenarios.

**3. Custom KL + Entropy Penalty (`kl_entropy_loss`)**

A novel composite loss designed for this project:

```
L = KL(y_true || y_pred) + lambda * MSE(H(y_true), H(y_pred))
```

Where `H(.)` is Shannon entropy (base 2) and `lambda = 0.5`. The entropy penalty explicitly forces the model to match not just the distribution shape but also the *degree of uncertainty* expressed by annotators. This is particularly relevant for distinguishing between distributions that have the same KL divergence from the true labels but different entropy levels.

---

## Training Pipeline

- **Optimizer**: Adam (pretraining at default LR; fine-tuning at 1e-4)
- **Batch size**: 64
- **Maximum epochs**: 50 per experiment
- **Data augmentation** (pretraining only): horizontal flips, width/height shifts (10%)
- **Callbacks**:
  - `EarlyStopping` — monitors validation loss, patience of 10 epochs, restores best weights
  - `ModelCheckpoint` — saves the best model weights per experiment

A total of six fine-tuning experiments are run: 2 architectures x 3 loss functions.

---

## Evaluation Metrics

Models are evaluated on the held-out soft test set (2,000 images) using a comprehensive suite of distribution-matching metrics:

**Distribution Distance Metrics**

| Metric | Description |
|---|---|
| KL Divergence | Mean KL divergence between true and predicted distributions |
| Jensen-Shannon Divergence (JSD) | Symmetric, bounded version of KL divergence |
| Cosine Similarity | Directional similarity between probability vectors |

**Entropy Correlation Metrics**

| Metric | Description |
|---|---|
| Pearson Correlation | Linear correlation between true and predicted per-image entropy |
| Spearman Correlation | Rank-order correlation between true and predicted entropy |

**Precision at K**

Measures whether the model correctly identifies the most uncertain images. For a given K, the top-K highest-entropy images in the ground truth are compared against the top-K predicted by the model:

- P@100, P@200, P@500

---

## Results Visualization

The notebook produces the following figures:

- Training vs. validation loss curves for all six experiments (ResNet-20 and Custom CNN, one plot per architecture)
- Validation KL divergence over training epochs, comparing all three loss functions per architecture
- Predicted vs. true entropy scatter plots for all six models, with Pearson correlation annotated
- Bar chart comparison of KL Divergence, JSD, and Cosine Similarity across all models
- Qualitative examples: five test images sampled evenly from low to high annotator entropy, with true and predicted distributions displayed

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
requests
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy
```

### 3. Launch the Notebook

```bash
jupyter notebook human_annotator_project.ipynb
```

Or if you prefer JupyterLab:

```bash
jupyter lab human_annotator_project.ipynb
```

### 4. Run the Cells

Run all cells in order from top to bottom. The notebook will:

- Automatically download the CIFAR-10 dataset via `tensorflow.keras.datasets`
- Automatically download the CIFAR-10H soft label file from GitHub
- Train both models through pretraining and fine-tuning phases
- Save model weights to disk after each experiment
- Generate all evaluation plots inline

> Note: Training all six fine-tuning experiments (2 models x 3 loss functions, up to 50 epochs each) can take a significant amount of time on CPU. A GPU is strongly recommended. If you are using Google Colab, go to Runtime > Change runtime type > GPU before running.
## Key Concepts

**Soft Labels** — probability distributions over classes rather than a single hard class assignment. They encode annotator disagreement and are a richer supervision signal than one-hot labels.

**Annotator Entropy** — the Shannon entropy of a soft label vector. An entropy of 0 means all annotators agreed; maximum entropy (log2(10) ≈ 3.32 bits for 10 classes) means annotators were completely split.

**Label Smoothing vs. Soft Labels** — label smoothing applies a fixed uniform perturbation to all labels. Soft labels from CIFAR-10H are image-specific and reflect real human perceptual uncertainty, making them a more meaningful target.
