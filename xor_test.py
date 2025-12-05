
import numpy as np
from lib.layers import Dense
from lib.activations import Sigmoid, Tanh
from lib.losses import MSE
from lib.network import Sequential

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Create model: 2 input neurons → 4 hidden → 1 output
model = Sequential([
    Dense(2, 4),
    Tanh(),
    Dense(4, 1),
    Sigmoid()
])

loss_fn = MSE()
lr = 1.0

# Training
for epoch in range(10000):
    output = model.forward(X)
    loss = loss_fn.forward(y, output)

    grad = loss_fn.backward(y, output)
    model.backward(grad, lr)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss}")

# Final Predictions
print("\nPredictions after training:")
print(model.forward(X))
