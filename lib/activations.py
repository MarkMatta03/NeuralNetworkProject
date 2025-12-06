import numpy as np
from .layers import Layer

# ======================
#   SIGMOID ACTIVATION
# ======================
class Sigmoid(Layer):
    def forward(self, x):
        # Compute the sigmoid activation:
        # σ(x) = 1 / (1 + e^(-x))
        # Store output for use in backward pass
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad_output, lr=None):
        # Derivative of sigmoid:
        # dσ/dx = σ(x) * (1 - σ(x))
        # Apply chain rule by multiplying with incoming gradient
        return grad_output * (self.out * (1 - self.out))


# ======================
#   TANH ACTIVATION
# ======================
class Tanh(Layer):
    def forward(self, x):
        # Compute tanh activation:
        # tanh(x) squashes input to (-1, 1)
        # Save output for use in backward pass
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_output, lr=None):
        # Derivative of tanh:
        # dtanh(x)/dx = 1 - tanh^2(x)
        # Multiply with incoming gradient (chain rule)
        return grad_output * (1 - self.out ** 2)
