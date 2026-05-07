# Machine-Learning-Final-Project

## Multi-Label Thoracic Disease Classification from Chest X-Rays

This repository contains the implementation and experimental pipeline for our study on **class imbalance handling in multi-label thoracic disease classification** using the CheXpert dataset.

The project systematically evaluates the impact of:

* Transfer learning vs. training from scratch
* Loss function design
* Threshold optimization
* Data augmentation strategies

on rare disease detection performance in chest X-ray classification.

---

# Overview

Class imbalance remains a major challenge in medical imaging. Rare but clinically important diseases are often underrepresented, causing deep learning models to favor majority classes.

In this work, we investigate how different modeling decisions affect performance under severe imbalance conditions using five thoracic diseases from the CheXpert dataset:

* Lung Opacity
* Pleural Effusion
* Atelectasis
* Pneumothorax
* Pneumonia

Our experiments demonstrate that:

* Square-root weighted BCE improves rare disease detection
* Threshold optimization substantially improves F1-score
* Transfer learning disproportionately benefits rare diseases
* Photometric augmentation outperforms geometric augmentation

---

# Key Results

| Configuration         | Macro AUROC | Macro F1 |
| --------------------- | ----------- | -------- |
| SimpleCNN Baseline    | 0.7626      | 0.340    |
| Final Optimized Model | 0.7806      | 0.506    |

## Major Findings

* **+48.8%** macro F1 improvement over baseline
* **8.7×** Pneumonia F1 improvement using sqrt-weighted BCE
* Per-disease threshold optimization improves macro F1 by **12%**
* Photometric augmentation improves macro F1 by **8.5%**

---

# Repository Structure

```text
├── data/
├── notebooks/
├── src/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── figures/
├── results/
├── requirements.txt
├── environment.yml
└── README.md
```

---

# Dataset

This project uses the **CheXpert** dataset:

> Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, AAAI 2019.

Dataset link:

```text
https://stanfordmlgroup.github.io/competitions/chexpert/
```

## Diseases Used

| Disease          | Approx. Prevalence |
| ---------------- | ------------------ |
| Lung Opacity     | 49%                |
| Pleural Effusion | 40%                |
| Atelectasis      | 16%                |
| Pneumothorax     | 9%                 |
| Pneumonia        | 2.4%               |

---

# Data Augmentation Experiments

Implemented augmentation strategies:

* Minimal augmentation
* Geometric augmentation
* Photometric augmentation

Photometric augmentation includes:

* Brightness adjustment
* Contrast adjustment

---

# Model Architectures

## SimpleCNN

* 4 convolutional blocks
* ~1.7M parameters
* Trained from scratch

## DenseNet-121

* ImageNet pretrained
* ~7M parameters
* Transfer learning baseline

---

# Loss Functions

Implemented losses:

* Standard BCE
* Square-root weighted BCE
* Focal Loss

## Square-Root Weighting

[
w = \sqrt{\frac{N}{n_{pos}}}
]

---

# Reproducibility

All experiments were conducted using:

* PyTorch
* CUDA GPU acceleration
* Adam optimizer
* Early stopping (patience = 5)

Random seeds were fixed where applicable for reproducibility.


