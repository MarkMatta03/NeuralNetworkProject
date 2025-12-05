import numpy as np

class MSE:
    def forward(self, y_true, y_pred): # Compute mean squared error

        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):   # Gradient of MSE w.r.t predictions

        return 2 * (y_pred - y_true) / y_true.size
