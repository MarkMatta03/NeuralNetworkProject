# Neural Network From Scratch — Part 1 & Part 2

This project implements a complete neural network **from scratch using NumPy only**, without using deep learning frameworks such as TensorFlow or PyTorch for the core implementation.  
This work fulfills **Part 1 and Part 2** of the semester project requirements.

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
- Epochs: **50,000**

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

- Numerical gradients were calculated using finite differences  
- Compared with analytical gradients from the backward pass  

Result:  
Maximum difference ≈ **1e-5**  
✔ Confirms backpropagation implementation is correct  

---

## 🎯 Objectives of Part 2

✔ Apply the custom neural network library to a real dataset  
✔ Train an autoencoder for unsupervised learning  
✔ Reconstruct input images  
✔ Extract latent features  
✔ Perform classification using latent representations  
✔ Compare results with a TensorFlow/Keras reference model  

---

## 🖼️ Autoencoder on MNIST Dataset

Dataset used:
- MNIST handwritten digits  
- Input size: 784  
- Pixel values normalized to range [0, 1]  

Autoencoder architecture:

Encoder: 784 → 256 → 64  
Decoder:  64 → 256 → 784  

Training Configuration:

- Loss function: **MSE**
- Optimization: **SGD**

➡ The autoencoder successfully reconstructs digit images.

---

## 🎯 Latent Space Classification

- Latent features extracted from the encoder (64 dimensions)  
- Support Vector Machine (SVM) trained using:
  - Raw pixels (baseline)
  - Latent features (autoencoder output)

✔ Latent features achieve comparable or better accuracy with much lower dimensionality.

---

## 🔁 TensorFlow / Keras Comparison

A reference autoencoder was implemented using **TensorFlow/Keras** with the same architecture and loss function.

➡ Used only for validation and comparison with the custom implementation.

---

## 📁 Project Structure

NeuralNetworkProject/
│
├─ lib/
│ ├─ layers.py # Dense layers
│ ├─ activations.py # Sigmoid & Tanh
│ ├─ losses.py # MSE + gradient
│ ├─ optimizer.py # SGD optimizer
│ └─ network.py # Sequential model container
│
├─ notebooks/
│ └─ project_demo.ipynb # Part 1 & Part 2 report & results
│
├─ xor_mse_test.py # XOR test script
├─ requirements.txt
└─ README.md

---

## ▶️ How to Run

Open terminal in project root:

```bash
python xor_mse_test.py
Or open the notebook:
notebooks/project_demo.ipynb
Run all cells from top to bottom using the Python 3.11 kernel.
