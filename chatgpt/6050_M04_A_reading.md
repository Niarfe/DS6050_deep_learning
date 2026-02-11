# Summary of D2L Readings: 5.6; 7; 8.1; 14.1

This set of readings moves from regularization, to deep convolutional scaling, to sequence modeling, and finally to attention mechanisms. Together, they trace the structural evolution of modern deep learning architectures.

---

## D2L 5.6 — Dropout

Dropout is a regularization technique based on stochastic thinning.

During training:
- Randomly zero out a fraction of hidden activations with probability $begin:math:text$ 1 \- p $end:math:text$.
- This prevents neurons from co-adapting.

During inference:
- Use the full network.
- Scale activations appropriately (or use inverted dropout during training).

Conceptually:
Dropout injects multiplicative noise into hidden representations. This forces robustness and effectively approximates training an ensemble of many thinned networks that share parameters.

Structural role:
Dropout combats overfitting by increasing redundancy and discouraging brittle internal dependencies.

---

## Chapter 7 — Modern Convolutional Neural Networks

This chapter traces architectural progress from early CNNs to deep residual networks.

Key evolutionary themes:

1. Increasing depth improves representational capacity.
2. Normalization techniques stabilize training.
3. Residual connections enable very deep networks.

### Residual Networks (ResNet)

Instead of directly learning $begin:math:text$ H\(x\) $end:math:text$, a residual block learns:

$begin:math:display$
F\(x\) \= H\(x\) \- x
$end:math:display$

The block outputs:

$begin:math:display$
x \+ F\(x\)
$end:math:display$

This skip connection:
- Allows gradients to flow directly.
- Reduces vanishing gradient problems.
- Prevents degradation when depth increases.

Core insight:
Depth becomes refinement rather than full transformation. Networks learn corrections to identity mappings.

---

## 8.1 — Recurrent Neural Networks (RNNs)

RNNs introduce sequence modeling.

Hidden state update:

$begin:math:display$
h\_t \= f\(x\_t\, h\_\{t\-1\}\)
$end:math:display$

Key properties:
- Parameters are shared across time.
- The network maintains memory.
- Backpropagation through time (BPTT) is required.

Important insight:
An unrolled RNN is effectively a very deep network whose depth equals sequence length.

This explains:
- Vanishing gradients.
- Exploding gradients.
- Difficulty modeling long-range dependencies.

RNNs introduce temporal inductive bias but suffer from memory bottlenecks.

---

## 14.1 — Attention Mechanisms

Attention addresses the fixed-memory limitation of RNNs.

Instead of compressing all information into a single hidden state, attention computes a weighted combination of relevant inputs.

Core components:
- Query (Q)
- Key (K)
- Value (V)

Attention computation:

$begin:math:display$
\\text\{softmax\}\\left\(\\frac\{QK\^T\}\{\\sqrt\{d\}\}\\right\)V
$end:math:display$

What changes:
- The model selectively focuses on relevant parts of the input.
- Long-range dependencies become easier to model.
- Information routing becomes dynamic rather than sequential.

Attention removes the bottleneck imposed by fixed hidden-state compression and forms the basis of Transformer architectures.

---

## Structural Arc Across the Readings

These chapters form a progression:

- Dropout → robustness through stochastic regularization.
- Modern CNNs → scaling depth safely with residual learning.
- RNNs → modeling sequences via recurrence.
- Attention → replacing recurrence with dynamic information routing.

Each innovation addresses a specific limitation:

- Overfitting.
- Vanishing gradients with depth.
- Memory constraints in sequential models.
- Long-range dependency bottlenecks.

Viewed structurally, architectural advances are gradient-routing improvements.

---

## Compressed Summary

This module traces the evolution from stabilizing feedforward networks, to enabling deep convolutional scaling, to modeling temporal dependence with recurrence, and finally to eliminating recurrence bottlenecks using attention.

Modern deep learning emerges from systematically removing structural pathologies in optimization, depth, and memory.
