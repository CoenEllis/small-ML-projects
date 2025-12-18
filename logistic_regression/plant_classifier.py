"""
Plant Classifier Module

This module uses logistic regression to classify plants based on
their stem height and leaf radius into 3 groups.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class PlantClassifier:
    """A class for classifying a type of plant."""
    def __init__(self, learning_rate=0.1):
        """Initialize tensors, model, criterion, optimizer."""
        # Plant A: 0-5, 0-2
        # Plant B: 3-6, 2-4
        # Plant C: 1-2, 0-2
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # First value is stem height, second value is leaf radius
        self.X = torch.tensor([[2.2, 1.4],
                               [4.9, 2.8],
                               [5.7, 3.2],
                               [4.5, 3.2],
                               [0.2, 0.8],
                               [1.3, 1.4]]).to(self.device)
        # Each corresponding value is the class it belongs to
        self.y = torch.tensor([0, 0, 1, 1, 2, 2]).to(self.device)
        self.model = nn.Linear(2, 3).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs=1000, print_rate=100):
        """
        Train the model.

        Args:
            epochs (int): Amount of training iterations. Default is 1000.
            print_rate (int): Epochs before printing status. Default is 100.
        """
        self.model.train()
        for epoch in range(epochs):
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % print_rate == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

        print("\nPredictions:")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            predictions = torch.argmax(logits, dim=1)
            print(predictions)

    def classify(self, stem_height, leaf_radius):
        """
        Classify the type of plant

        Args:
            stem_height (float): The height of the stem.
            leaf_radius (float): The radius of the leaf.

        Returns:
            int: The group of plant.
        """
        new_plant = torch.tensor([[stem_height, leaf_radius]]).to(self.device)
        with torch.no_grad():
            logits = self.model(new_plant)
            prediction = torch.argmax(logits, dim=1)
        return prediction.item()


if __name__ == "__main__":
    # Script to train plant classifier and classify a plant
    classifier = PlantClassifier()
    print(f"Using device {classifier.device}")
    classifier.train()
    print("Plant Classes:")
    print("Class A: stem height: 0-5, leaf radius: 0-2")
    print("Class B: stem height: 3-6, leaf radius: 2-4")
    print("Class C: stem height: 1-2, leaf radius: 0-2")
    stem_height = float(input("Stem height: "))
    leaf_radius = float(input("Leaf radius: "))
    plant_class = classifier.classify(stem_height, leaf_radius)
    print(f"Predicted plant class: {plant_class}")
