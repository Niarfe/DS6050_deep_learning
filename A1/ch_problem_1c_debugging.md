## Why the Loss Was “Off by a Factor of *n*”

### Per-sample definition
Cross-entropy (negative log-likelihood) is defined **for a single data point**.

- **Binary cross-entropy (per sample)**  
  \[
  \ell_i = -\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]
  \]

- **Multiclass cross-entropy (per sample)**  
  \[
  \ell_i = -\log p_{i, y_i}
  \]

These formulas describe how “surprised” the model is by **one observation**.

---

### What changes with a batch
Given \(n\) independent samples, there are two common conventions:

- **Summed loss**
  \[
  \mathcal{L}_{\text{sum}} = \sum_{i=1}^n \ell_i
  \]

- **Mean loss**
  \[
  \mathcal{L}_{\text{mean}} = \frac{1}{n}\sum_{i=1}^n \ell_i
  \]

Both are mathematically valid. They differ only by a constant scaling factor.

---

### What happened in this implementation
- The from-scratch code computed the **sum** over the batch.
- `sklearn.metrics.log_loss` returns the **mean** over the batch by default.
- With batch size \(n = 1000\), this produced a mismatch of exactly \(1000\times\).

---

### Why libraries use the mean
Machine learning libraries standardize on the mean loss because it:

1. Is **independent of batch size**, making runs comparable.
2. Keeps **gradient magnitudes stable** as batch size changes.
3. Produces losses with interpretable, consistent scale.

---

### Resolution
To match sklearn’s convention, the implementation was updated to return the **mean** loss instead of the sum:

- Replace `np.sum(...)` with `np.mean(...)`

This does not change the underlying model or likelihood — only the normalization of the reported value.

---

### Key takeaway
When a loss differs by a constant factor (often \(n\)), the cause is almost always a **reduction convention mismatch**: sum vs. mean.

