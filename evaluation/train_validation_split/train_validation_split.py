"""
Train Validation Split Module

This module tests the accuracy of a model by splitting training data into
two parts, training on one, then testing on the other data which has not
been seen
"""

import torch
import torch.nn as nn
import torch.optim as optim


class TrainValidation:
    """A class to train and evaluate accuracy."""
    def __init__(self):
        """Initialize the tensors, model, criterion, and optimizer."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor([
            [1.0, 2.0],
            [1.5, 1.8],
            [3.0, 3.5],
            [3.2, 4.0]
        ]).to(self.device)

        self.y = torch.tensor([0, 0, 1, 1]).to(self.device)

        self.train_X = self.X[:3]
        self.train_y = self.y[:3]

        self.val_X = self.X[3:]
        self.val_y = self.y[3:]

        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):
        """Train model and data."""
        self.model.train()
        for _ in range(500):
            logits = self.model(self.train_X)
            loss = self.criterion(logits, self.train_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("Training complete.")

    def train_split(self):
        """
        Evaluate accuracy of model by testing on new data.

        Returns:
            torch.Tensor: The predictions of the data.
            float: The accuracy of the data.
        """
        with torch.no_grad():
            val_logits = self.model(self.val_X)
            val_predictions = torch.argmax(val_logits, dim=1)
            accuracy = (val_predictions == self.val_y).sum().item() / \
                len(self.val_y)
        return val_predictions, accuracy


if __name__ == "__main__":
    # Script to train the model and validate training accuracy
    train_validation = TrainValidation()
    print(f"Using device: {train_validation.device}")
    train_validation.train()
    val_predictions, accuracy = train_validation.train_split()
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
