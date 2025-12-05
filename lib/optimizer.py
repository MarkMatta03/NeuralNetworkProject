# lib/optimizer.py

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """
        params: list of parameters to update (e.g., [layer.W, layer.b])
        grads: list of corresponding gradients (e.g., [layer.grad_W, layer.grad_b])
        """
        for p, g in zip(params, grads):
            p -= self.lr * g