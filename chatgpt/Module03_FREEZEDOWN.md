# tstamp: 2026-02-02 15:07

# FREEZEDOWN Snapshot — Deep Learning Study + Workflow Session

## Context

This session covered:

- D2L 3.6–3.7 (softmax, cross-entropy)
- D2L 12 (optimization)
- D2L 19 (CNNs / inductive bias)
- Implementation of core functions (sigmoid, softmax, NLL, linear regression, SGD variants)
- Numerical stability (log-sum-exp / max-shift trick)
- Study infrastructure (quiz sheet + short PyTorch exercises)
- Workflow instrumentation (tstamp, SITCHECK formatting fix)

This is an archival checkpoint.

---

## Conceptual Milestones

### 1) MLE → NLL → Cross-Entropy

- MLE = maximize probability of observed data.
- Log-likelihood converts products to sums.
- Minimizing negative log-likelihood (NLL) = maximizing likelihood.
- Binary: logits → sigmoid → NLL.
- Multiclass: logits → softmax → NLL.

Structural insight:
Sigmoid+NLL and Softmax+NLL are mathematically paired constructions.

---

### 2) Logits Clarified

- Logits live in (−∞, ∞).
- They are not probabilities.
- Sigmoid/softmax map logits → probabilities.

Mental model:
Logits = scores.
Probabilities = normalized beliefs.

---

### 3) Log-Sum-Exp Trick

Stability line analyzed:

```python
Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
```

Key idea:
- Subtract row-wise max before exponentiating.
- Prevents overflow.
- Does not change softmax outputs.

Softmax(z) = Softmax(z − c)

Correctly tied to quiz term:
Log-Sum-Exp Trick (D2L 3.7)

---

### 4) Optimization Insights

- Reverse-mode autodiff is ideal because we optimize a scalar loss with many parameters.
- One backward pass yields gradients for all weights.
- Momentum reduces oscillation in narrow valleys.
- SGD gradients are unbiased in expectation.
- Batch noise scales roughly as 1/√batch_size.
- Learning-rate decay that is too aggressive can stall convergence.

---

### 5) Architecture & Inductive Bias

MLP:
- Multilayer Perceptron.
- Fully connected layers + nonlinearities.
- No spatial inductive bias.

CNN:
- Local connectivity.
- Parameter sharing.
- Translation equivariance.
- Growing receptive field.

Normalization terms added:
- BatchNorm
- LayerNorm
- Internal covariate shift

---

## Study Infrastructure

### Printable Quiz Sheet

Includes:
- MLE
- Logits
- Softmax
- Cross-Entropy
- NLL
- Log-Sum-Exp
- SGD
- Momentum
- Learning-rate schedule
- Inductive bias
- MLP
- BatchNorm vs LayerNorm

All tied to D2L sections for quick lookup.

---

### Coding Confidence Plan

Short PyTorch reps:

- Softmax sanity check
- Numerical stability demo
- Verify gradient (p − y)
- SGD vs Momentum toy example
- Batch-size noise demo
- CNN parameter count comparison

Purpose:
Replace avoidance with bounded implementation wins.

---

## SITCHECK Protocol

Restored to original compact format:

D:T <days-emoji> <turns-emoji>

Single-line invariant preserved.

---

## State at Freeze

You now have:

- A coherent conceptual map of the week’s material.
- Stable understanding of logits, NLL, and numerical stability.
- A printable quiz scaffold with chapter anchors.
- A short-exercise PyTorch practice pipeline.
- A corrected archival workflow.

Conversation frozen.
