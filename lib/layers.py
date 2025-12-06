import numpy as np

# Base Layer class
class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


# Dense (Fully Connected) Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Xavier initialization
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, grad_output):
        # Compute gradients
        self.grad_W = np.dot(self.inputs.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # Propagate gradient to previous layer
        return np.dot(grad_output, self.W.T)
