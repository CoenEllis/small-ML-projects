"""
Evaluation Metrics Module

This module shows the accuracy of the given model and provides a
confusion matrix.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class EvaluationMetrics():
    """A class to train the model and evaluate metrics."""
    def __init__(self):
        """Initialize tensors, model, criterion, and optimizer."""
        self.X = torch.tensor([
            [1.0, 2.0],
            [1.5, 1.8],
            [3.0, 3.5],
            [3.2, 4.0]
        ])

        self.y = torch.tensor([0, 0, 1, 1])

        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):
        """Train model on data."""
        self.model.train()
        for _ in range(500):
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Training complete.")

    def evaluate(self):
        """
        Evaluate training data and create a confusion matrix.

        Returns:
            float: The accuracy of the model.
            torch.Tensor: The confusion matrix.
        """
        with torch.no_grad():
            logits = self.model(self.X)
            predictions = torch.argmax(logits, dim=1)

        correct = (predictions == self.y).sum().item()
        accuracy = correct / len(self.y)

        num_classes = 2
        confusion_matrix = torch.zeros(num_classes, num_classes)

        for true, pred in zip(self.y, predictions):
            confusion_matrix[true][pred] += 1

        return accuracy, confusion_matrix


if __name__ == "__main__":
    # Script to train the model and evaluate metrics
    evaluator = EvaluationMetrics()
    evaluator.train()
    accuracy, confusion_matrix = evaluator.evaluate()
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)
