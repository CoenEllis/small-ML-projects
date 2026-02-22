"""
MLP Classifier Trainer Module

This module trains and evaluates the classifier neural network,
including data preparation, model training, checkpointing,
and displaying predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model import ClassifierModel


class ClassifierTrainer:
    """Trains and evaluates a neural network to classify plants."""

    def __init__(self, hidden_size=4, learning_rate=0.001, checkpoint_path=None):
        """
        Initialize the ClassifierTrainer with model, data,
        and training components.

        Args:
            hidden_size (int): Number of neurons in the hidden layer.
                Default is 4.
            learning_rate (int): Learning rate for the Adam optimizer.
                Default is 0.001.
            checkpoint_path (str, optional): Path to load a saved model
                checkpoint. Default is None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = torch.tensor(
            [[2.2, 1.4], [4.9, 2.8], [5.7, 3.2], [4.5, 3.2], [0.2, 0.8], [1.3, 1.4]]
        ).to(self.device)
        # Each corresponding value is the class it belongs to
        self.y = torch.tensor([0, 0, 1, 1, 2, 2]).to(self.device)
        self.model = ClassifierModel(hidden_size).to(self.device)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Model loaded from: {checkpoint_path}")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs, print_rate=1000, save_rate=10000, save_path=None):
        """
        Train the model on plant classes with periodic progress updates
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
                print(f"Epoch: {epoch}: Loss: {loss.item():.4f}")
            if save_path is not None:
                if epoch % save_rate == 0:
                    print(f"Saved model. Epoch {epoch}")
                    torch.save(self.model.state_dict(), save_path)

        print("\nPredictions:")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X)
            predictions = torch.argmax(outputs, dim=1)
            print(predictions)
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved at path: {save_path}")

    def classify(self, stem_height, leaf_radius):
        new_plant = torch.tensor([[stem_height, leaf_radius]]).to(self.device)
        with torch.no_grad():
            logits = self.model(new_plant)
            prediction = torch.argmax(logits, dim=1)
        return prediction.item()


if __name__ == "__main__":
    # Script to train, evaluate, and test the plant classifier
    trainer = ClassifierTrainer()
    print(f"Using device: {trainer.device}")
    trainer.train(10000)
    print("Plant Classes:")
    print("Class A: stem height: 0-5, leaf radius: 0-2")
    print("Class B: stem height: 3-6, leaf radius: 2-4")
    print("Class C: stem height: 1-2, leaf radius: 0-2")
    stem_height = float(input("Stem height: "))
    leaf_radius = float(input("Leaf radius: "))
    plant_class = trainer.classify(stem_height, leaf_radius)
    print(f"Predicted plant class: {plant_class}")
