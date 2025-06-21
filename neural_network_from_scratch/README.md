# Neural Network from Scratch

Welcome to the `neural_network_from_scratch` folder of the [`build_from_scratch`](https://github.com/your-username/build_from_scratch) repository.  
This section is dedicated to implementing neural network components entirely from scratch — starting from the most basic neuron to full-fledged Multi-Layer Perceptrons (MLPs) — without relying on high-level libraries like TensorFlow or PyTorch.

The main goal is to **demystify the internals of neural networks** and build an intuitive understanding by coding each piece manually. Early implementations even avoid using NumPy, making it an excellent learning resource for beginners and enthusiasts alike.

---

## 📁 Folder Overview

### 1. [`neuron.ipynb`](./neuron.ipynb)
> **Concepts Covered**: Scalar inputs, input vectors, manual computation of outputs  
> **Tools Used**: Pure Python (No external libraries)

This notebook walks through:
- Implementing a **single neuron** from scratch
- Extending it to handle a **vector of inputs**
- Building a **layer of neurons**
  
All computations are done manually, helping you understand dot products, weights, and biases at the most granular level.

---

### 2. [`neuron_using_numpy.ipynb`](./neuron_using_numpy.ipynb)
> **Concepts Covered**: Vectorized computations, batch processing  
> **Tools Used**: Python with NumPy

This version enhances the earlier neuron implementations using NumPy to:
- Efficiently compute outputs for batches of inputs
- Handle multiple neurons in a layer
- Highlight performance and code readability improvements via vectorized operations

---

### 3. [`MLP_using_numpy.ipynb`](./MLP_using_numpy.ipynb)
> **Concepts Covered**: Feedforward networks, forward pass through multiple layers  
> **Tools Used**: Python with NumPy

This notebook implements:
- A customizable **Multi-Layer Perceptron (MLP)** with one or more hidden layers
- Forward propagation logic from input to output
- Layer-by-layer computation using matrix multiplications

It lays the foundation for deeper architectures and potential backpropagation implementation in the future.

---

## 🚧 Upcoming Additions

The following topics are planned to be added soon:
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions and their gradients
- Backpropagation and training loop
- Support for classification and regression tasks
- Visualization of decision boundaries

Stay tuned! Contributions are welcome 😊

---

## 🤝 Contributing

Interested in contributing? Here's how you can help:
- 🧠 Add new neural network architectures (e.g., CNNs, RNNs)
- 🛠️ Improve or refactor existing notebooks
- 🧾 Add explanatory markdown cells or visualizations
- 📚 Submit tutorials or beginner-friendly walkthroughs

To contribute:
1. Fork the repository
2. Clone your fork:  
   ```bash
   git clone https://github.com/Harper2123/build_from_scratch.git
   ```
3. Create a new branch and start hacking!
4. Submit a pull request for review

All contributions, big or small, are appreciated!

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## ⭐ Acknowledgements

Special thanks to all the contributors and learners exploring the core principles of deep learning — one neuron at a time!

---

> _This README will evolve as more implementations and tutorials are added to the folder._
