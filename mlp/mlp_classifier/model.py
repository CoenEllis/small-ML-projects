"""
MLP Classifier Model Module

This modules sets up the layers for the classifier and
performs the forward pass.
"""

import torch.nn as nn


class ClassifierModel(nn.Module):
    """A neural network for classifying plants."""
    def __init__(self, hidden_size=4):
        """
        Initialize the ClassifierModel with two layers and an activation.

        Args:
            hidden_size (int): The amount of neurons in the hidden layer.
        """
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the classifier.

        Args:
            x (torch.Tensor): The input layer.

        Returns:
            torch.Tensor: The model's prediction (output after
                final activation).
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
