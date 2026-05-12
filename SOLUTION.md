# SMILES-2026 Hallucination Detection Solution

## Overview

This repository implements a reproducible hidden-state probe for detecting
hallucinated answers from `Qwen/Qwen2.5-0.5B`.  The fixed `solution.py` script
is left unchanged: it extracts hidden states, calls the three student modules,
writes `results.json`, and writes `predictions.csv`.

The solution modifies these components:

| File | Role |
| --- | --- |
| `aggregation.py` | Converts per-token, per-layer hidden states into one feature vector per sample. |
| `probe.py` | Trains a deterministic ensemble of regularized linear probes and tunes the classification threshold. |
| `splitting.py` | Builds reproducible stratified train, validation, and test folds for evaluation. |

## Required Components

The complete pipeline needs the following components:

1. Python 3.10 or newer.
2. The packages listed in `requirements.txt`: `torch`, `transformers`,
   `datasets`, `scikit-learn`, `numpy`, `pandas`, and `tqdm`.
3. Internet access for the first run, so Hugging Face can download
   `Qwen/Qwen2.5-0.5B`.
4. A CUDA GPU is recommended. CPU execution is supported by the code but is
   much slower because hidden-state extraction dominates runtime.
5. The provided files `data/dataset.csv` and `data/test.csv`.

## Reproducibility Instructions

From a fresh checkout, run:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the official entry point:

```bash
python solution.py
```

Expected generated artifacts:

| Artifact | Description |
| --- | --- |
| `results.json` | Cross-validation metrics produced by `evaluate.py`. |
| `predictions.csv` | Competition predictions for `data/test.csv`, with columns `id,label`. |

The implementation fixes all random seeds used by the split and probe
components (`random_state = 42`).  Hugging Face model downloads may be cached by
the local environment, but the model identifier remains fixed as
`Qwen/Qwen2.5-0.5B` through the provided `model.py`.

## Final Approach

### 1. Hidden-State Aggregation

The baseline used only the last real token from the final transformer layer.
This solution keeps that useful signal but expands it into a richer
representation:

- select five mid-to-late layers;
- collect the last-token vector from each selected layer;
- collect short-tail and medium-tail mean vectors from the end of the sequence;
- collect full-sequence mean vectors for prompt-plus-answer context;
- append scalar geometric statistics, including token norms, activation
  dispersion, and cosine drift between selected layers.

The fixed `solution.py` keeps `USE_GEOMETRIC = False`, so the final solution
places the important scalar statistics directly in `aggregate()`.  The optional
`extract_geometric_features()` function is still implemented for experiments.

### 2. Probe Classifier

The labelled dataset is small compared with the hidden-state feature dimension,
so the final classifier favors stable linear models over a large neural
network.  `HallucinationProbe` trains an averaged ensemble of:

- a balanced L2 logistic regression on all features;
- a balanced L2 logistic regression after univariate feature selection;
- a balanced L2 logistic regression after PCA whitening.

The probe estimates an out-of-fold threshold during `fit()` so the final
`predictions.csv` can be produced even when no separate validation set is
available.  During official fold evaluation, `evaluate.py` calls
`fit_hyperparameters()` and the threshold is retuned on that fold's validation
split.

### 3. Data Splitting

`splitting.py` returns five stratified folds.  Each fold reserves one outer
test split and then creates an inner stratified validation split from the
remaining rows.  This gives more stable metrics than a single holdout split and
still allows the final model to train on all labelled samples when
`solution.py` builds the competition predictor.

## Why These Choices

The largest expected gains come from aggregation.  Hallucination signals may
not be concentrated in one final hidden vector: answer endings, average answer
behavior, and representation drift across upper layers can all carry useful
information.  Concatenating several deterministic summaries gives the probe a
broader but still reproducible view of the model state.

The linear ensemble is intentionally conservative.  With only 689 labelled
examples, a flexible neural probe can memorize fold-specific noise.  Balanced
regularized logistic models are easier to validate, give calibrated
probability-like outputs, and support threshold tuning for the accuracy-focused
submission.

## Experiments and Failed Attempts

The following ideas were considered but not included in the final solution:

- A deeper MLP probe.  It increases training flexibility, but with this dataset
  size it is more sensitive to initialization, learning rate, and validation
  split noise.
- Using only the final layer.  It is fast, but it discards mid-layer and
  layer-drift information that often helps linear probes.
- A single train, validation, and test holdout.  It is simpler, but the reported
  score changes more with the random seed.  Stratified K-fold evaluation gives
  a more reliable estimate.
- Enabling `USE_GEOMETRIC` by editing `solution.py`.  The challenge statement
  asks the solution to remain runnable through the fixed infrastructure, so the
  final scalar statistics are included directly in `aggregate()`.

## Reproducing the Submitted Predictions

To recreate the submitted file, run:

```bash
python solution.py
```

The command will overwrite `results.json` and `predictions.csv` in the project
root.  No manual changes to `solution.py`, `evaluate.py`, or `model.py` are
required.
