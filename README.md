# SMILES-2026 Hallucination Detection

Detect whether a small language
model's answer is *hallucinated* (fabricated) or *truthful* using the model's
own internal representations (hidden states).

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Accuracy on the held-out `test.csv`.

## Repository Structure

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.py            # Main script - run to create a 
│
│   ── Files you implement ──────────────────────────────────────────────
├── aggregation.py         # Layer selection, token pooling, geometric features
├── probe.py               # HallucinationProbe — the binary classifier
├── splitting.py           # Train / validation / test split strategy
│
│   ── Fixed infrastructure (do not edit) ───────────────────────────────
├── model.py               # Loads Qwen2.5-0.5B and exposes get_model_and_tokenizer()
├── evaluate.py            # Evaluation loop, metrics, summary table, JSON output
│
├── requirements.txt       # Python dependencies
└── LICENSE
```


## Quick Start

### Google Colab

Open the terminal in Colab and run:

```python
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

### Local Setup

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

## Dataset

`data/dataset.csv` contains 689 labelled samples with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```


`data/test.csv` is structured identically but the `label` column is null - these are the samples you submit predictions for via a `predictions.csv` generated file.


## What You Implement

You are expected to edit **three files**:  
- `aggregation.py`
- `probe.py`
- `splitting.py`

The rest of the codebase shall remain untouched.

**Feature Engineering & Dimensionality Reduction**: Applicants are encouraged to experiment with adding hand-crafted features during the aggregation step, drawing on geometrical or topological methods to enrich the representation of probe outputs. Additionally, you may apply dimensionality reduction techniques within probe.py to compress or refine the feature space. 

## Evaluation

For each fold `evaluate.py` reports four numbers:

| # | Checkpoint | Metrics |
|---|-----------|---------|
| 1 | Majority-class baseline | Accuracy, F1 |
| 2 | `HallucinationProbe` on **training** split | Accuracy, F1, AUROC |
| 3 | `HallucinationProbe` on **validation** split | Accuracy, F1, AUROC |
| 4 | `HallucinationProbe` on **test** split | Accuracy, F1, AUROC |

**Accuracy on the `test.csv` is the primary competition metric.**

Results are averaged across folds (if using k-fold) and saved to
`results.json`.
