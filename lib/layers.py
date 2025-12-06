import numpy as np

# ==========================================
# Base Layer Class
# ==========================================
class Layer:
    def forward(self, inputs):
        # Every layer must implement forward propagation
        raise NotImplementedError

    def backward(self, grad_output):
        # Every layer must implement backward propagation
        raise NotImplementedError


# ==========================================
# Dense (Fully Connected) Layer
# ==========================================
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialize weights using Xavier initialization:
        # Helps keep gradients stable during training
        # W shape: (input_dim, output_dim)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

        # Bias vector initialized to zeros
        # b shape: (1, output_dim)
        self.b = np.zeros((1, output_size))

    def forward(self, inputs):
        # Save input for backpropagation
        self.inputs = inputs

        # Forward computation: Z = XW + b
        # Where:
        # X = inputs, W = weights, b = bias
        return np.dot(inputs, self.W) + self.b

    def backward(self, grad_output):
        # Compute gradient of loss w.r.t. weights:
        # dL/dW = X^T · dL/dZ
        self.grad_W = np.dot(self.inputs.T, grad_output)

        # Compute gradient of loss w.r.t. bias:
        # dL/db = sum(dL/dZ)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # Compute gradient w.r.t. the layer’s inputs:
        # dL/dX = dL/dZ · W^T
        # Return to propagate backward through previous layer
        return np.dot(grad_output, self.W.T)
