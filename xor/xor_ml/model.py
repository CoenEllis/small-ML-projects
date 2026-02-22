"""
XOR Model Module

This module sets up the layers for the XOR neural network and
performs the forward pass.
"""

import torch.nn as nn


class XORModel(nn.Module):
    """A simple feedforward neural network for learning XOR logic."""

    def __init__(self, hidden_size=4):
        """
        Initialize the XORModel with two layers and an activation.

        Args:
            hidden_size (int): The amount of neurons in the hidden layer.
        """
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the XOR NN.

        Args:
            x (torch.Tensor): The input layer.

        Returns:
            torch.Tensor: The model's prediction (output after
                final activation).
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        return x
