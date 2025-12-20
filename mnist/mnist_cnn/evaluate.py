"""
CNN MNIST Evaluate Module

This module evaluates the MNIST model.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTModel


class MNISTEvaluator:
    """A class to evaluate the model."""
    def __init__(self, checkpoint_path, batch_size=64):
        """
        Initialize data and components to evaluate the neural network.

        Args:
            checkpoint_path (str): The path where the model exists.
            batch_size (int): The number of batches processed at once.
                Default is 64.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = MNISTModel().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {checkpoint_path}")

        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def evaluate(self):
        """
        Evaluate the model and return the accuracy and confusion matrix.

        Returns:
            tuple: (accuracy, confusion_matrix) where accuracy is a float
                and confusion_matrix is a torch.Tensor.
        """
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                # Calculate accuracy
                correct += (predictions == target).sum().item()
                total += target.size(0)

                # Update confusion matrix
                for true, pred in zip(target, predictions):
                    confusion_matrix[true][pred] += 1

        accuracy = 100 * correct / total
        return accuracy, confusion_matrix

    def print_results(self):
        """
        Print results of the evaluation.

        Returns:
            tuple: (accuracy, confusion_matrix) where accuracy is a float
                and confusion_matrix is a torch.Tensor.
        """
        accuracy, confusion_matrix = self.evaluate()
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix.numpy())
        return accuracy, confusion_matrix


if __name__ == "__main__":
    # Script to evaluate the MNIST model
    evaluator = MNISTEvaluator(checkpoint_path="mnist_cnn_model.pth")
    evaluator.print_results()
