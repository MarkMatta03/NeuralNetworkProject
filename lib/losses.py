import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BCE:
    def forward(self, y_true, y_pred):
        eps = 1e-10
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    def backward(self, y_true, y_pred):
        eps = 1e-10
        return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + eps)
