"""
XOR Trainer Module

This module trains and evaluates the XOR neural network,
including data preparation, model training, checkpointing,
and displaying predictions.
"""


import torch
import torch.nn as nn
import torch.optim as optim

from model import XORModel


class XORTrainer:
    """Trains and evaluates a neural network to learn XOR logic."""
    def __init__(self, hidden_size=4, learning_rate=0.001,
                 checkpoint_path=None):
        """
        Initialize the XORTrainer with model, data, and training components.

        Args:
            hidden_size (int): Number of neurons in the hidden layer.
                Default is 4.
            learning_rate (int): Learning rate for the Adam optimizer.
                Default is 0.001.
            checkpoint_path (str, optional): Path to load a saved model
                checkpoint. Default is None.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]],
                              dtype=torch.float32).to(self.device)

        self.y = torch.tensor([[0], [1], [1], [0]],
                              dtype=torch.float32).to(self.device)
        self.model = XORModel(hidden_size).to(self.device)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Model loaded from: {checkpoint_path}")
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate)

    def train(self, epochs, print_rate=1000, save_rate=10000,
              save_path=None):
        """
        Train the model on XOR data with periodic progress updates
        and checkpointing.

        Args:
            epochs (int): Number of training epochs.
            print_rate (int): Print loss every N checkpoints.
                Default is 1000.
            save_rate (int): Save model checkpoint every N epochs.
                Default is 10000.
            save_path (str, optional): Path to save model checkpoints.
                If None, no saving occurs.
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(self.X)
            loss = self.criterion(output, self.y)
            loss.backward()
            self.optimizer.step()
            if epoch % print_rate == 0:
                print(f'Epoch: {epoch}: Loss: {loss.item():.4f}')
            if save_path is not None:
                if epoch % save_rate == 0:
                    print(f'Saved model. Epoch {epoch}')
                    torch.save(self.model.state_dict(), save_path)

        print("\nPredictions:")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X)
            print(predictions)
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved at path: {save_path}")


if __name__ == "__main__":
    # Script to train and evaluate the XOR neural network
    trainer = XORTrainer()
    print(f"Using device: {trainer.device}")
    trainer.train(10000)
