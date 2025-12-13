# Neural Network From Scratch — Part 1

This project implements a complete neural network **from scratch using NumPy only**, without using deep learning frameworks such as TensorFlow or PyTorch.  
This work fulfills **Part 1** of the semester project requirements.

---

## 🎯 Objectives of Part 1

✔ Build a modular neural network library  
✔ Implement:
- Dense (Fully Connected) Layers  
- Activation Functions (Sigmoid, Tanh)  
- Mean Squared Error (MSE) Loss  
- Stochastic Gradient Descent (SGD) optimizer  

✔ Train the model to learn the XOR logic function  
✔ Perform Gradient Checking to verify correctness of backpropagation  
✔ Present training results in a Jupyter Notebook

---

## 🧠 XOR Problem Training

The XOR truth table:

| Input | Output |
|------|--------|
| (0,0) | 0 |
| (0,1) | 1 |
| (1,0) | 1 |
| (1,1) | 0 |

The neural network architecture used:

Input(2) → Dense(4) + Tanh → Dense(1) + Sigmoid

Training Configuration:

- Loss function: **MSE**
- Optimization: **SGD**
- Epochs: 50,000

### ✔ Final XOR Predictions

[[0.01]
[0.98]
[0.98]
[0.02]]

➡ The model successfully learns XOR 🎉

---

## 📈 Training Loss Curve

The loss smoothly approaches ~0 during training.

📍 Included inside:
notebooks/project_demo.ipynb
---

## 🧪 Gradient Checking

To ensure the correctness of backpropagation:

- Numerical gradients were calculated using finite difference
- Compared with analytical gradients from backward pass

Result:
Maximum difference ≈ 1e-5
✔ Confirms backpropagation implementation is correct

---

## 📁 Project Structure

NeuralNetworkProject/
│
├─ lib/
│ ├─ layers.py # Dense layer + SGD update
│ ├─ activations.py # Sigmoid & Tanh
│ ├─ losses.py # MSE + gradient
│ ├─ network.py # Sequential model container
│
├─ notebooks/
│ └─ project_demo.ipynb # Part 1 report & results
│
├─ xor_mse_test.py # Quick test script for XOR
└─ README.md
---

## ▶️ How to Run

Open Terminal in project root:

```bash
python -m xor_mse_test
Or open the notebook:
notebooks/project_demo.ipynb
