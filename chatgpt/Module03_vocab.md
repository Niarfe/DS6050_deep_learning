# Deep Learning Quiz Sheet  
**Week: D2L 3.6–3.7, 12, 19**

| Term | Definition |
|------|------------|
| **MLE (Maximum Likelihood Estimation)** | Choosing model parameters that make the observed data as probable as possible under the assumed probabilistic model. |
| **Logits** | Raw, unnormalized scores produced by a model before applying a normalization function like softmax. |
| **Softmax Function** | Converts a vector of logits into a probability distribution by exponentiating and normalizing so outputs are nonnegative and sum to 1. |
| **Cross-Entropy Loss** | Measures the negative log-likelihood of the true class under the model’s predicted probability distribution. |
| **Negative Log-Likelihood (NLL)** | The negative of the log probability assigned to the observed data; minimizing NLL is equivalent to maximizing likelihood. |
| **Log-Sum-Exp Trick** | A numerical stability technique that subtracts a constant (usually the max logit) before exponentiation to prevent overflow. |
| **Gradient of Softmax + Cross-Entropy** | Simplifies to \( p - y \), where \( p \) is the predicted probability vector and \( y \) is the one-hot target. |
| **Stochastic Gradient Descent (SGD)** | An optimization method that updates parameters using gradients computed from randomly sampled data points or minibatches. |
| **Unbiased Gradient Estimate** | A stochastic gradient whose expected value equals the true full-batch gradient. |
| **Momentum** | An optimization technique that accumulates a running average of past gradients to smooth updates and reduce oscillations. |
| **Learning Rate Schedule** | A rule for changing the learning rate over time, typically decreasing it to allow fine convergence. |
| **Noise in SGD** | Random variation in gradient estimates due to minibatch sampling that can help escape saddle points or poor local minima. |
| **Inductive Bias** | The assumptions a model makes about the structure of the data that guide generalization from finite samples. |
| **Fully Connected (Dense) Layer** | A layer where every input feature connects to every output unit with a separate weight. |
| **Convolutional Neural Network (CNN)** | A neural network that exploits spatial locality and translation equivariance using convolutional layers. |
| **Parameter Sharing** | Reusing the same parameters across different spatial locations in a convolutional layer. |
| **Translation Equivariance** | Shifting the input results in a corresponding shift in the output. |
| **Receptive Field** | The region of the input that influences a particular neuron’s output. |
