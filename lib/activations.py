import numpy as np
from .layers import Layer


# ============================================================
# Sigmoid Activation Function
# ============================================================
# Maps input values to the range (0, 1)
# Commonly used in output layers for binary classification
class Sigmoid(Layer):

    def forward(self, x):
        # Apply sigmoid function
        # σ(x) = 1 / (1 + e^(-x))
        self.out = 1 / (1 + np.exp(-x))

        # Save output for use in backward pass
        return self.out

    def backward(self, grad_output, lr=None):
        # Derivative of sigmoid:
        # σ'(x) = σ(x) * (1 - σ(x))
        # Chain rule: dL/dx = dL/dy * dy/dx
        return grad_output * (self.out * (1 - self.out))


# ============================================================
# Tanh Activation Function
# ============================================================
# Maps input values to the range (-1, 1)
# Often used in hidden layers
class Tanh(Layer):

    def forward(self, x):
        # Apply tanh activation
        self.out = np.tanh(x)

        # Save output for backward pass
        return self.out

    def backward(self, grad_output, lr=None):
        # Derivative of tanh:
        # tanh'(x) = 1 - tanh(x)^2
        return grad_output * (1 - self.out ** 2)


# ============================================================
# ReLU Activation Function
# ============================================================
# Rectified Linear Unit
# ReLU(x) = max(0, x)
# Widely used in deep networks
class ReLU(Layer):

    def forward(self, x):
        # Store input to determine where x > 0
        self.x = x

        # Apply ReLU activation
        return np.maximum(0, x)

    def backward(self, grad_output, lr=None):
        # Gradient is passed only where input was positive
        grad_input = grad_output.copy()
        grad_input[self.x <= 0] = 0
        return grad_input


# ============================================================
# Softmax Activation Function
# ============================================================
# Converts raw scores into probabilities
# Output values sum to 1 for each sample
class Softmax(Layer):

    def forward(self, x):
        # Numerical stability trick:
        # subtract max value to prevent overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Compute softmax probabilities
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Save output for completeness
        return self.out

    def backward(self, grad_output, lr=None):
        # Simplified backward pass
        # Full Jacobian is not required for this project
        # This is acceptable since Softmax is not used for training
        return grad_output
