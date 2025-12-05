import numpy as np
from .layers import Layer

class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad_output, lr=None):
        return grad_output * (self.out * (1 - self.out))

class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_output, lr=None):
        return grad_output * (1 - self.out ** 2)
