# Deep Learning Quiz Sheet  
**Week: D2L 3.6–3.7, 12, 19 (+ MLP & Normalization Concepts)**

| Term | Definition |
|------|------------|
| **MLE (Maximum Likelihood Estimation)** | Choosing model parameters that make the observed data as probable as possible under the assumed probabilistic model. *(D2L Ch. 3)* |
| **Logits** | Raw, unnormalized real-valued scores produced by a model before applying sigmoid or softmax. *(D2L 3.6)* |
| **Softmax Function** | Converts a vector of logits into a probability distribution by exponentiating and normalizing so outputs are nonnegative and sum to 1. *(D2L 3.6)* |
| **Cross-Entropy Loss** | Measures the negative log-likelihood of the true class under the model’s predicted probability distribution. *(D2L 3.7)* |
| **Negative Log-Likelihood (NLL)** | The negative of the log probability assigned to the observed data; minimizing NLL is equivalent to maximizing likelihood. *(D2L 3.7)* |
| **Log-Sum-Exp Trick** | A numerical stability technique that subtracts the maximum logit before exponentiation to prevent overflow without changing results. *(D2L 3.7)* |
| **Gradient of Softmax + Cross-Entropy** | The gradient with respect to logits simplifies to \( p - y \), where \( p \) is predicted probabilities and \( y \) is the one-hot target. *(D2L 3.7)* |
| **Stochastic Gradient Descent (SGD)** | An optimization method that updates parameters using gradients computed from randomly sampled data points or minibatches. *(D2L Ch. 12)* |
| **Unbiased Gradient Estimate** | A stochastic gradient whose expected value equals the true full-batch gradient. *(D2L 12.2–12.3)* |
| **Momentum** | An optimization technique that accumulates a running average of past gradients to smooth updates and reduce oscillations. *(D2L 12.4)* |
| **Learning Rate Schedule** | A rule for adjusting the learning rate over training, often decreasing it to allow stable convergence. *(D2L 12.6)* |
| **Noise in SGD** | Random variation in gradient estimates due to minibatch sampling, which can help escape saddle points and poor local minima. *(D2L 12.4–12.5)* |
| **Inductive Bias** | The assumptions a model makes about the structure of the data that guide generalization from finite samples. *(D2L 19.1)* |
| **Fully Connected (Dense) Layer** | A layer where every input feature connects to every output unit with its own weight, ignoring spatial structure. *(D2L 19.1)* |
| **Multilayer Perceptron (MLP)** | A feedforward neural network composed of stacked fully connected layers with nonlinear activation functions between them. *(D2L Ch. 5)* |
| **Activation Function** | A nonlinear function applied after an affine transformation to enable neural networks to model non-linear relationships. *(D2L 5.1)* |
| **Batch Normalization (BatchNorm)** | A normalization technique that standardizes activations using batch-level statistics during training. *(D2L 8.5)* |
| **Layer Normalization (LayerNorm)** | A normalization technique that standardizes activations across features within each individual sample. *(D2L 8.5)* |
| **Internal Covariate Shift** | The change in the distribution of layer inputs during training as parameters update. *(Motivates normalization; D2L 8.5)* |
| **Convolutional Neural Network (CNN)** | A neural network architecture that exploits spatial locality and translation equivariance using convolutional layers. *(D2L Ch. 19)* |
| **Parameter Sharing** | Reusing the same parameters across different spatial locations in a convolutional layer. *(D2L 19.1–19.2)* |
| **Translation Equivariance** | A property where shifting the input leads to a corresponding shift in the output feature map. *(D2L 19.1)* |
| **Receptive Field** | The region of the input that influences a particular neuron’s output, which grows with depth in CNNs. *(D2L 19.2)* |
