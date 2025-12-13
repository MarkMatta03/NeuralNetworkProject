

```markdown
# Neural Network From Scratch — Part 1 & Part 2

This project implements a complete neural network **from scratch using NumPy**, without relying on deep learning frameworks for the core implementation.  
The work is divided into **Part 1** (fundamentals and XOR) and **Part 2** (autoencoder on MNIST with latent-space classification).

---

## 🎯 Part 1 — Neural Network Fundamentals

### Objectives

✔ Build a modular neural network library  
✔ Implement:
- Dense (Fully Connected) layers  
- Activation functions (Sigmoid, Tanh)  
- Mean Squared Error (MSE) loss  
- Stochastic Gradient Descent (SGD) optimizer  

✔ Train a neural network to learn the XOR logic function  
✔ Perform gradient checking to verify backpropagation correctness  

---

### 🧠 XOR Problem

**XOR Truth Table**

| Input | Output |
|------|--------|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

**Network Architecture**
```

Input (2)
→ Dense (4) + Tanh
→ Dense (1) + Sigmoid

```

**Training Configuration**
- Loss: Mean Squared Error (MSE)
- Optimizer: SGD
- Epochs: 50,000

**Final Predictions**
```

[[0.01]
[0.98]
[0.98]
[0.02]]

```

✔ The network successfully learns the XOR function.

---

### 🧪 Gradient Checking

- Numerical gradients computed using finite differences  
- Compared with analytical gradients from backpropagation  

**Result:**  
Maximum difference ≈ **1e-5**, confirming correctness of the implementation.

---

## 🎯 Part 2 — Autoencoder on MNIST

### Objectives

✔ Apply the custom neural network library to a real dataset  
✔ Train an autoencoder for unsupervised representation learning  
✔ Visualize image reconstruction quality  
✔ Use latent features for classification  
✔ Validate results using a TensorFlow/Keras reference model  

---

### 🖼️ MNIST Autoencoder

**Dataset**
- MNIST handwritten digits
- Input dimension: 784
- Normalized to range [0, 1]

**Architecture**
```

Encoder: 784 → 256 → 64
Decoder:  64 → 256 → 784

```

- Activations: Tanh (hidden), Sigmoid (output)
- Loss: Mean Squared Error (MSE)
- Optimizer: SGD

✔ The autoencoder successfully reconstructs digit images.

---

### 🎯 Latent Space Classification

- Latent vectors (64-D) extracted from the encoder  
- Support Vector Machine (SVM) trained on:
  - Raw pixels (baseline)
  - Latent features (autoencoder output)

✔ Latent features achieve comparable or better accuracy with much lower dimensionality.

---

### 🔁 TensorFlow / Keras Comparison

A reference autoencoder is implemented using **TensorFlow/Keras** with the same architecture and loss function to validate the correctness of the custom implementation.

> TensorFlow is used **only for comparison**, not for the main implementation.

---

## 📁 Project Structure

```

NeuralNetworkProject/
│
├── lib/
│   ├── layers.py        # Dense layers
│   ├── activations.py  # Sigmoid & Tanh
│   ├── losses.py       # MSE loss
│   ├── optimizer.py    # SGD optimizer
│   └── network.py      # Sequential container
│
├── notebooks/
│   └── project_demo.ipynb   # Part 1 & Part 2 results
│
├── xor_mse_test.py     # XOR test script
├── requirements.txt
└── README.md

````

---

## ▶️ How to Run

**XOR Test**
```bash
python xor_mse_test.py
````

**Full Project Demo**
Open:

```
notebooks/project_demo.ipynb
```

Run all cells using the **Python 3.11** kernel.

---

## 🛠️ Environment & Dependencies

* Python **3.11**
* Required libraries:

```
numpy
matplotlib
scikit-learn
pandas
tensorflow
```

Install with:

```bash
pip install -r requirements.txt
```

