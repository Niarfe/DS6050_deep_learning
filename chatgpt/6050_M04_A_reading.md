# Summary of D2L Readings: 5.6; 7; 8.1; 14.1

This module traces the structural evolution of deep learning: from regularization, to deep convolutional scaling, to sequence modeling, and finally to attention-based routing.

---

## D2L 5.6 — Dropout

Dropout is a regularization technique based on stochastic thinning.

During training:
- Randomly zero out hidden activations with probability $(1 - p)$.
- Prevent neurons from co-adapting.

During inference:
- Use the full network.
- Scale activations appropriately (or use inverted dropout during training).

Conceptually, dropout injects multiplicative noise into hidden representations. It behaves like training an ensemble of many thinned networks that share parameters.

Structural role:  
Dropout increases robustness and reduces overfitting.

---

## Chapter 7 — Modern Convolutional Neural Networks

This chapter tracks the progression from early CNNs (LeNet, AlexNet) to deep architectures like ResNet.

Key themes:

1. Increasing depth improves representational capacity.
2. Normalization stabilizes training.
3. Residual connections make extreme depth trainable.

### Residual Networks (ResNet)

Instead of directly learning $H(x)$, a residual block learns:

$$
F(x) = H(x) - x
$$

The block outputs:

$$
x + F(x)
$$

The skip connection:
- Improves gradient flow
- Reduces vanishing gradients
- Prevents degradation as depth increases

Core idea:  
Depth becomes refinement of identity rather than full transformation.

---

## 8.1 — Recurrent Neural Networks (RNNs)

RNNs introduce sequence modeling through a hidden state.

State update:

$$
h_t = f(x_t, h_{t-1})
$$

Key properties:

- Parameters shared across time
- Maintains memory
- Requires backpropagation through time (BPTT)

Structural insight:  
An unrolled RNN is effectively a deep network whose depth equals sequence length.

This explains:
- Vanishing gradients
- Exploding gradients
- Long-range dependency difficulty

RNNs introduce temporal inductive bias but suffer from memory bottlenecks.

---

## 14.1 — Attention Mechanisms

Attention removes the fixed-memory limitation of RNNs.

Instead of compressing all information into a single hidden state, attention computes a weighted combination of inputs.

Core components:
- Query (Q)
- Key (K)
- Value (V)

Attention computation:

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

What changes:

- The model dynamically focuses on relevant inputs
- Long-range dependencies become easier
- Information routing becomes direct rather than sequential

Attention forms the foundation of Transformer architectures.

---

## Structural Arc

The progression across the readings:

- Dropout → robustness  
- Deep CNNs → scalable depth  
- RNNs → temporal modeling  
- Attention → dynamic routing  

Each innovation removes a structural limitation:

- Overfitting  
- Vanishing gradients  
- Sequential memory bottlenecks  
- Long-range dependency constraints  

Architectural evolution can be understood as progressively improving gradient routing and information flow.
