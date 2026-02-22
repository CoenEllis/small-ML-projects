"""
CNN MNIST Model Module

This module sets up the layers for the model and
performs the forward pass.
"""

import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """A convolutional neural network for classifying digits."""

    def __init__(self):
        """Initialize the CNN with conv and fully connected layers."""
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Forward pass for the CNN model.

        Args:
            x (torch.Tensor): The input tensor (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: The model's prediction logits (batch_size, 10).
        """
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        # Flatten
        x = x.view(x.size(0), -1)  # 64*7*7 = 3136

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
