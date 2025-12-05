import numpy as np

# Base Layer class
class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad_output, lr):
        raise NotImplementedError


# Dense (Fully Connected) Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Xavier-like initialization
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs   # Save input for backprop
        return np.dot(inputs, self.W) + self.b

    def backward(self, grad_output, lr):
        # Compute gradients
        grad_W = np.dot(self.inputs.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # Store gradients for gradient checking
        self.grad_W = grad_W
        self.grad_b = grad_b

        # Gradient to propagate back to previous layer
        grad_input = np.dot(grad_output, self.W.T)

        # Update weights (SGD)
        self.W -= lr * grad_W
        self.b -= lr * grad_b

        return grad_input
