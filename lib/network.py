class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, lr):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
        return grad_output
