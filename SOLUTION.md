# SOLUTION.md

## 1. Reproducibility Instructions

### Environment

This solution is designed to run within the official competition repository without modifying any restricted files.

Recommended setup:

* Python ≥ 3.10
* PyTorch (CUDA optional but recommended)
* GPU: Tested on NVIDIA T4 (Google Colab compatible)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the solution

```bash
python solution.py
```

This will automatically:

* Extract hidden states from the base LLM
* Train the hallucination probe
* Generate predictions on the test set

Outputs:

* `results.json` (metrics on validation/test split)
* `predictions.csv` (final submission file)

### Modified components

Only the allowed files were modified:

* `aggregation.py`
* `probe.py`
* `splitting.py`

Additionally:

* `USE_GEOMETRIC = True` was enabled in `solution.py`

No other files in the fixed infrastructure were changed.

---

## 2. Final Solution Description

### Problem framing

The task is to detect hallucinations in LLM-generated responses using internal hidden representations from **Qwen2.5-0.5B**.

Each sample consists of:

* Prompt (question/context)
* Model response
* Binary hallucination label

The key challenge is extracting a **robust representation** from token-level hidden states that captures:

* semantic correctness
* uncertainty
* internal inconsistency signals

---

### Representation strategy

#### Multi-layer feature extraction

Instead of relying solely on the final layer, the solution aggregates information from multiple depths:

* Final layer
* Layer -2
* Layer -4
* Layer -8

**Rationale:**

* Upper layers encode task-specific semantics
* Mid layers retain richer contextual and syntactic signals
* Combining them improves robustness

---

#### Multi-pooling per layer

For each selected layer, three complementary views are extracted:

1. **Last token representation**

   * Captures final answer signal
   * Strong indicator of generation confidence

2. **Mean pooling**

   * Represents overall sequence semantics

3. **Max pooling**

   * Captures salient activations (important tokens)

This results in a feature vector that encodes both **global** and **localized** information.

---

### Geometric / statistical features

Additional scalar features are appended to capture internal structure:

* L2 norm of last token
* L2 norm of mean-pooled representation
* Token-level variance (stability indicator)
* Cosine similarity between last token and mean
* Inter-layer cosine similarity (representation drift)
* Inter-layer distance (instability signal)
* Sequence length

**Insight:**
Hallucinated responses tend to exhibit:

* higher instability across layers
* weaker alignment between global and local representations
* abnormal norm/variance patterns

---

### Final classifier

The probe is implemented as a **classical ML pipeline**:

```text
StandardScaler → PCA → ExtraTreesClassifier
```

#### Why this works well:

* Dataset is relatively small (~689 samples)
* Tree ensembles capture nonlinear interactions
* PCA reduces noise from high-dimensional concatenated features
* More stable than neural probes under limited data

#### Model configuration:

* 600 trees
* `class_weight="balanced"`
* feature subsampling (`max_features="sqrt"`)

---

### Threshold optimization

Instead of using a fixed 0.5 threshold, the decision boundary is tuned on validation data:

* Evaluate F1 across candidate thresholds
* Select threshold that maximizes F1

**Reasoning:**

* Improves robustness under class imbalance
* Produces better calibrated predictions

---

## 3. Experiments and Failed Attempts

### Baseline: final-layer last-token only

* Simple and fast
* Misses sequence-level and cross-layer signals
* Underperformed significantly

---

### Mean pooling only

* Captures global semantics
* Loses critical signal from answer termination
* Worse than combined pooling

---

### Deep MLP probe

* Tried 2–3 layer neural network
* Observed:

  * high variance across runs
  * sensitivity to hyperparameters
  * overfitting on small dataset

Tree-based models were more stable.

---

### Using all layers

* Very high dimensionality
* Introduced noise
* Slower training
* No performance gain

Selective layer sampling performed better.

---

### Logistic Regression

* Strong baseline
* Too linear for feature interactions
* Underfit compared to tree ensemble

---

## 4. Key Takeaways

* Combining **multi-layer + multi-pooling** is critical
* Geometric features provide strong auxiliary signal
* Classical ML models outperform neural probes in low-data regimes
* Careful threshold tuning improves final performance
