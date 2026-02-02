# Weekly Reading Overview  
**D2L: 3.6–3.7; 12; 19**

This week looks scattered by chapter number, but conceptually it is very coherent. The unifying theme is a transition from *basic learning machinery* to *modern deep learning practice*.

At a high level, the readings answer three fundamental questions:

1. How do we turn raw model outputs into meaningful probabilities for classification?
2. How does optimization actually behave in high-dimensional, non-convex problems?
3. Why does model *structure* matter as much as optimization or data?

Together, these chapters explain why deep learning works in practice.

---

## Big Picture: What This Week Is About

Earlier material focused on:
- linear models  
- loss functions  
- gradients  
- basic stochastic gradient descent  

This week moves beyond that foundation and addresses reality:

- Classification requires probabilistic modeling, not just regression.
- Optimization is noisy, geometry-dependent, and biased by design choices.
- Architectures succeed because they encode assumptions about the data.

Modern deep learning works only when **loss, optimizer, and architecture are aligned**.

---

## D2L 3.6–3.7: Softmax Regression (Classification Done Properly)

These sections complete the story of classification.

The central idea is that **linear models do not produce probabilities**.  
Softmax is the mechanism that converts raw scores (logits) into a probability distribution.

Key concepts introduced:
- Logits as unnormalized scores
- Softmax as normalization across classes
- Cross-entropy as negative log-likelihood
- The probabilistic interpretation of classification

A critical insight:
> Softmax and cross-entropy are a *matched pair*.

This pairing leads to a remarkably simple gradient:
\[
\nabla = p - y
\]
which is why this combination appears everywhere in deep learning.

**Focus on:**
- The probabilistic interpretation
- Why cross-entropy is the “right” loss
- Shape and dimensional reasoning

**Skim:**
- Repetitive training-loop code
- Boilerplate examples once the idea is clear

This material will reappear constantly throughout the course.

---

## D2L 12: Optimization Algorithms (Why SGD Alone Is Not Enough)

This chapter reframes how you think about training.

Previously:
> Optimization was “compute gradient, step downhill.”

Now:
> Optimization is geometry, noise, and dynamics.

Major ideas:
- Loss landscapes contain ravines, plateaus, and saddle points
- Vanilla SGD oscillates and stalls in narrow valleys
- Momentum smooths updates over time
- Adaptive methods rescale dimensions differently
- Noise can help escape poor local minima

This chapter explains:
- Why momentum helps
- Why Adam converges fast but may generalize worse
- Why learning-rate schedules matter deeply

**Focus on:**
- Geometric intuition
- Failure modes of vanilla SGD
- The idea of *implicit bias* in optimization

**Skim:**
- Detailed derivations of update rules
- Hyperparameter “recipes”

The goal is understanding *behavior*, not memorizing optimizers.

---

## D2L 19: Convolutional Neural Networks (Structure Beats Scale)

This chapter introduces a conceptual leap: **inductive bias**.

Up to now, models have been:
- Fully connected
- Generic
- Structure-agnostic

CNNs encode assumptions about images:
- Locality
- Translation equivariance
- Parameter sharing
- Hierarchical feature learning

Key message:
> Architectures work because they assume something correct about the data.

CNNs succeed not just because they are deep, but because they are *appropriate*.

**Focus on:**
- Why MLPs fail on raw images
- How convolutions reduce parameters dramatically
- The idea of spatial hierarchies
- Pooling as controlled information loss

**Skim:**
- Historical architecture details
- Implementation minutiae

This chapter reshapes how you think about model design.

---

## How These Chapters Fit Together

Each chapter answers a different but essential question:

| Question | Chapter |
|--------|--------|
| How do we model uncertainty in classification? | 3.6–3.7 |
| How do we actually find good parameters? | 12 |
| What assumptions should the model make about data? | 19 |

Deep learning works only when:
- the loss matches the task,
- the optimizer matches the geometry,
- the architecture matches the data.

Break any one of these, and performance collapses.

---

## Recommended Reading Strategy

1. **Read 3.6–3.7 carefully**  
   This math will be reused constantly.

2. **Read Chapter 12 for intuition, not memorization**  
   Ask what goes wrong with SGD and why fixes help.

3. **Read Chapter 19 with architectural curiosity**  
   Ask what assumption each design choice encodes.

This week builds conceptual depth rather than surface techniques.
