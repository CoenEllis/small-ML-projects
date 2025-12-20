# MNIST Digit Classification

Two implementations of handwritten digit classification on the MNIST dataset using PyTorch.

## Implementations

### MLP (Multi-Layer Perceptron)
A simple feedforward neural network with fully connected layers.

**Architecture:**
- Input: 784 (flattened 28×28 images)
- Hidden layers: 784 → 128 → 64 → 10
- Activation: ReLU

### CNN (Convolutional Neural Network)
A convolutional neural network that preserves spatial structure.

**Architecture:**
- Conv layers: 1 → 32 → 64 channels
- MaxPooling after each conv layer
- Fully connected: 3136 → 128 → 10
- Dropout: 25%