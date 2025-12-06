import numpy as np

class MSE:
    # ======================================
    # Forward Pass
    # ======================================
    def forward(self, y_true, y_pred):
        """
        Computes the Mean Squared Error loss.

        y_true: Ground truth labels
        y_pred: Model predictions

        L = (1/N) * Σ (y_true - y_pred)^2
        """

        return np.mean((y_true - y_pred) ** 2)

    # ======================================
    # Backward Pass (Gradient Computation)
    # ======================================
    def backward(self, y_true, y_pred):
        """
        Computes the gradient of MSE loss w.r.t predictions.

        dL/dy_pred = (2/N) * (y_pred - y_true)

        This value tells us how much a small change in the
        predictions affects the loss—driving learning backward.
        """

        return 2 * (y_pred - y_true) / y_true.size
