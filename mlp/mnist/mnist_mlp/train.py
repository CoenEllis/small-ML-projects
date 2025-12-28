"""
MLP MNIST Train Module

This module trains the MNIST model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


class MNISTTrainer:
    """Trains the MNIST model."""
    def __init__(self, batch_size=64, learning_rate=0.001,
                 checkpoint_path=None):
        """
        Initialize the MNISTTrainer with data to load and train MNIST.

        Args:
            batch_size (int): The amount of batches trained before
                updating weights. Default is 64.
            learning_rate (float): Learning rate for the Adam optimizer.
                Default is 0.001.
            checkpoint_path (str or None): The path to load an existing
                model. Default is None.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # Transform to flatten 28x28 images to 784-dimensional vectors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        # Load MNIST dataset
        self.train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        self.test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.model = MNISTModel().to(self.device)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Model loaded from: {checkpoint_path}")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def train(self, epochs, print_rate=100, save_path=None):
        """
        Train the MLP MNIST model.

        Args:
            epochs (int): The number of times the model trains.
            print_rate (int): The number of batches trained before
                printing. Default is 100.
            save_path (str or None): The file path where the model
                will be saved. Default is None.
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if batch_idx % print_rate == 0:
                    print(f'Epoch {epoch+1}/{epochs}, '
                          f'Batch {batch_idx}: Loss: {loss.item():.4f}')

            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    # Script to train the model
    trainer = MNISTTrainer(batch_size=64, learning_rate=0.001)
    print(f"Using device: {trainer.device}")
    trainer.train(10)
