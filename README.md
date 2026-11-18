# Federated Learning for Marathi Sentiment Classification

NLP Mini Project (IA1) · M.Tech AI&DS · Dept. of Information Technology
K J Somaiya College of Engineering, Vidyavihar, Mumbai · 2026-27

## Group Members
| Name | Roll No. |
|------|----------|
| Zikra Siddiqui | 16034426013 |
| Khushi Beluse | 16034426014 |

Group No.: 6 · Paper (Overleaf): https://www.overleaf.com/project/6aa0ebae93da66992d11e99b/share#559474c353a72cda833d4a14b07b8419c5c308edaffecab0

## Problem Statement
Sentiment analysis for regional Indian languages is less developed than for
high-resource languages, and privacy concerns limit centralized collection of
user-generated text. This project fine-tunes a MuRIL transformer for
three-class Marathi sentiment classification (negative / neutral / positive)
using Federated Learning (FedAvg and FedProx), so raw text stays with the
clients. It compares the federated models with a centralized baseline and
runs a preliminary client-side differential privacy (DP-SGD) experiment.

## Dataset
- **L3Cube-MahaSent-MD** (L3Cube-Pune), multi-domain Marathi sentiment:
  60,864 samples (48,114 train / 6,000 val / 6,750 test).
- Source: https://github.com/l3cube-pune/MarathiNLP
- Cleaning: URLs/mentions removed, whitespace normalised, empty, duplicate,
  conflicting and cross-split duplicate texts removed.
  The cleaned test set has **6,675** sentences.
- Federated split: 3 simulated clients with IID partitions of the training set.

## Methods
- Model: MuRIL (`[google/muril-base-cased]`) with a classification head.
- FedAvg (μ = 0) and FedProx (μ = 0.01), 3 clients, 3 rounds × 1 local epoch,
  sample-count-weighted aggregation. The checkpoint with the best validation
  macro-F1 is tested once on the test set.
- Centralized baseline: MuRIL, 2 epochs on the full training set.
- DP pilot: Opacus DP-SGD, ε ≈ 8, δ = 1e-5, clipping norm 1.0, σ ≈ 0.64.

| Setting | Centralized | FedAvg / FedProx | DP pilot |
|---|---|---|---|
| Optimizer | AdamW | AdamW | DP-SGD (Opacus) |
| Learning rate | 2e-5 | 2e-5 | 1e-5 |
| Batch size | 8 | 8 | 16 (logical) |
| Max length | 128 | 128 | 64 |
| Epochs / rounds | 2 epochs | 3 rounds × 1 epoch | 2 local epochs |
| Seed | 42 | 42 | 42 |

## Repository Structure

README.md
requirements.txt
data/          # dataset or download script
src/           # preprocessing, training, FL, DP, evaluation
notebooks/     # Colab notebooks


## Setup

git clone [repo-url]
cd [repo]
pip install -r requirements.txt

Tested on Google Colab, Tesla T4 GPU, PyTorch + Hugging Face Transformers
([versions]). Random seed: 42.



## Results (held-out test set, 6,675 sentences)
| Model | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| Centralized MuRIL | 80.15 | 80.19 | 80.17 | 80.02 |
| FedAvg (3 clients) | 80.58 | 80.47 | 80.61 | 80.40 |
| FedProx (3 clients, μ=0.01) | 80.28 | 80.18 | 80.31 | 80.20 |

- Federated training matched centralized training. The differences are
  under 0.6 points and come from a single seed, so no method is claimed superior.
- Neutral is the hardest class (F1 0.73–0.74).
- DP pilot (ε ≈ 8, untuned) did not learn the task: 31.58% and 33.50% accuracy.

## Limitations
One dataset, 3 simulated IID clients, 3 rounds, one seed. FedNova, non-IID
partitions, communication cost and secure aggregation were not evaluated.
Federated models saw each record 3 times, the centralized model 2 times.


