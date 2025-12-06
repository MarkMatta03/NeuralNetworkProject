
import sys, os
sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt

from lib.layers import Dense
from lib.activations import Sigmoid, Tanh
from lib.losses import MSE
from lib.network import Sequential
from lib.optimizer import SGD


# XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = Sequential([
    Dense(2, 4),
    Tanh(),
    Dense(4, 1),
    Sigmoid()
])

loss_fn = MSE()
lr = 1.0
epochs = 50000

loss_history = []

optimizer = SGD(lr=1.0)

for epoch in range(epochs + 1):
    out = model.forward(X)
    loss = loss_fn.forward(y, out)
    loss_history.append(loss)

    grad = loss_fn.backward(y, out)
    model.backward(grad)
    # Update weights using optimizer
    for layer in model.layers:
        if hasattr(layer, "W"):  # only Dense layers
            optimizer.step([layer.W, layer.b], [layer.grad_W, layer.grad_b])

print("Predictions:")
print(model.forward(X))
print("Final Loss:", loss)
# ======================
# TEST CUSTOM INPUTS
# ======================
print("\nTesting New Inputs:")
new_inputs = np.array([
    [0.2, 0.8],
    [0.8, 0.2],
    [0.0, 0.0],
    [1.0, 1.0]
], dtype=float)

print(model.forward(new_inputs))