# Federated Learning for Marathi Sentiment Classification

NLP Mini Project (IA1) · M.Tech AI&DS · Dept. of Information Technology
K J Somaiya College of Engineering, Vidyavihar, Mumbai · 2026-27

## Group Members
| Name | Roll No. |
|------|----------|
| Zikra Siddiqui | [roll] |
| Khushi Beluse | [roll] |

Group No.: [X] · Paper (Overleaf): [link]

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
- Download/prepare: `[python src/prepare_data.py]`

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
```
README.md
requirements.txt
data/          # dataset or download script
src/           # preprocessing, training, FL, DP, evaluation
notebooks/     # Colab notebooks
checkpoints/   # best models (or Drive link)
logs/          # training and console logs
results/       # plots, tables, metrics, results PDF
```

## Setup
```bash
git clone [repo-url]
cd [repo]
pip install -r requirements.txt
```
Tested on Google Colab, Tesla T4 GPU, PyTorch + Hugging Face Transformers
([versions]). Random seed: 42.

## How to Run
```bash
python src/prepare_data.py
python src/train_central.py  --seed 42
python src/train_federated.py --algo fedavg  --mu 0    --clients 3 --rounds 3 --seed 42
python src/train_federated.py --algo fedprox --mu 0.01 --clients 3 --rounds 3 --seed 42
python src/train_dp.py --records_per_client 500 --epochs 2 --seed 42
python src/evaluate.py        # tables and plots go to results/
```
(Or run `notebooks/[name].ipynb` in Colab, top to bottom.)
Replace the commands with your real file names.

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
- Full tables, class-wise results and plots are in `results/`
  and the Results PDF `results/[file].pdf`.

## Checkpoints and Logs
- Checkpoints: `checkpoints/` or [Drive / Git LFS link]
- Logs: `logs/`

## Limitations
One dataset, 3 simulated IID clients, 3 rounds, one seed. FedNova, non-IID
partitions, communication cost and secure aggregation were not evaluated.
Federated models saw each record 3 times, the centralized model 2 times.

## Paper
"Federated Learning for Marathi Sentiment Classification",
Z. Siddiqui, K. Beluse. [Overleaf link]
