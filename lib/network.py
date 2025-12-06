# lib/network.py

class Sequential:
    def __init__(self, layers):
        # Store a list of layers in the order they will be executed
        # Example: [Dense → Tanh → Dense → Sigmoid]
        self.layers = layers

    def forward(self, X):
        """
        Forward pass: Data flows from input → final output
        Each layer takes the output of the previous layer as input.
        
        X: input data (batch of samples)
        """
        for layer in self.layers:
            # Pass the data sequentially through each layer
            X = layer.forward(X)
        return X  # final prediction

    def backward(self, grad_output):
        """
        Backward pass: Gradients flow from output → input
        Using the chain rule, we call backward() on each layer in reverse order.
        
        grad_output: gradient coming from the loss function
        """
        for layer in reversed(self.layers):
            # Each layer computes gradients and passes gradient backward
            grad_output = layer.backward(grad_output)
        return grad_output  # gradient w.r.t input (not used further)
