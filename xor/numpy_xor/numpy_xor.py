"""
Numpy XOR Module

This module trains a NN on XOR using Numpy.
"""

import numpy as np


class NumpyXOR:
    """A class for training a NN to learn XOR."""

    def __init__(self):
        """
        Initialize the input, and output matrices
        as well as the weights, biases, and learning rate.
        """
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input layer

        self.y = np.array([[0], [1], [1], [0]])

        np.random.seed(50)  # Set seed for reproducibility

        # 2 inputs -> 2 hidden neurons -> 1 output
        self.W1 = np.random.rand(2, 2)  # Weights from input to hidden layer
        self.W2 = np.random.rand(2, 1)  # Weights from hidden to output layer

        self.b1 = np.random.rand(1, 2)  # Bias for hidden layer
        self.b2 = np.random.rand(1, 1)  # Bias for output layer

        self.learning_rate = 0.1

    def sigmoid(self, x):
        """
        Squashes a number between 0 and 1.

        Args:
            x (float): The input matrix.

        Returns:
            float: The number squashed between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Computes the derivative of the sigmoid function for backpropagation.

        Args:
            x (float): The sigmoid-activated output.

        Returns:
            float: The derivative value used in gradient calculations.
        """
        return x * (1 - x)  # Assuming x is already sigmoid(x)

    def train(self, epochs, print_rate=1000):
        """
        Training loop for the ML model.

        Args:
            epochs (int): The amount of epochs to train the ML model.
            print_rate (int): How often to print loss and epoch.
                Default is 1000.
        """
        for i in range(epochs):
            # Forward pass for the hidden and final outputs
            hidden_output = self.sigmoid(np.dot(self.X, self.W1) + self.b1)
            final_output = self.sigmoid(np.dot(hidden_output, self.W2) + self.b2)

            # Backpropagation
            # Calculate output layer error and gradient
            error_output = self.y - final_output
            delta_output = error_output * self.sigmoid_derivative(final_output)

            # Calculate hidden layer error and gradient
            error_hidden = delta_output.dot(self.W2.T)
            delta_hidden = error_hidden * self.sigmoid_derivative(hidden_output)

            # Update weights and biases using gradients
            self.W2 += hidden_output.T.dot(delta_output) * self.learning_rate
            self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate

            self.W1 += self.X.T.dot(delta_hidden) * self.learning_rate
            self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

            if i % print_rate == 0:
                loss = np.mean(error_output**2)
                print(f"Epoch {i:,} Loss: {loss}")

        print("\nPredictions:")
        print(final_output)


if __name__ == "__main__":
    # Script to train Numpy XOR
    numpy_xor = NumpyXOR()
    numpy_xor.train(10000, 1000)
