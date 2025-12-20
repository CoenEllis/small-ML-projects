"""
MLP MNIST Model Module

This module sets up the layers for the model and
performs the forward pass.
"""

import torch.nn as nn


class MNISTModel(nn.Module):
    """A neural network for classifying digits."""
    def __init__(self):
        """Initialize the MNISTModel with three layers and an activation."""
        super().__init__()
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): The input layer.

        Returns:
            torch.Tensor: The model's prediction (output after
                final activation).
        """
        x = x.view(x.size(0), -1)  # Flatten image
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x
