"""
Plant Age Identifier Module

This module uses linear regression to identify the age of a plant given the
stem size and leaf radius.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class PlantAgeIdentifier:
    """A class for identifying the age of a plant."""
    def __init__(self, learning_rate=0.01):
        """Initialize tensors, model, criterion, optimizer."""
        # Input data: first value is stem height, second is leaf radius
        self.X = torch.tensor([[1.0, 0.3],
                               [1.5, 0.8],
                               [2.2, 1.1],
                               [2.9, 1.9],
                               [3.8, 2.2],
                               [4.3, 2.3],
                               [4.5, 2.5],
                               [4.6, 2.7]]).to(self.device)

        # Output data, each value is the corresponding age
        self.y = torch.tensor([[1.5], [2.0], [2.5], [3.0],
                               [3.5], [4.5], [5.0], [5.5]]).to(self.device)

        # Linear regression model: 2 inputs -> 1 output (age)
        self.model = nn.Linear(2, 1).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs=1000, print_rate=100):
        """
        Train the model.

        Args:
            epochs (int): Amount of training iterations. Default is 1000.
            print_rate (int): Epochs before printing status. Default is 100.
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model(self.X)
            loss = self.criterion(y_pred, self.y)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % print_rate == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        print("\nPredictions:")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X)
            print(predictions)

    def identify(self, stem_height, leaf_radius):
        """
        Identify the age of the plant.

        Args:
            stem_height (float): The height of the stem.
            leaf_radius (float): The radius of the leaf.

        Returns:
            float: The predicted age of the plant.
        """
        new_plant = torch.tensor([[stem_height, leaf_radius]])
        with torch.no_grad():
            prediction = self.model(new_plant)
        return prediction.item()


if __name__ == "__main__":
    # Script to train the model and predict the age of a plant
    age_identifier = PlantAgeIdentifier()
    print(f"Using device {age_identifier.device}")
    age_identifier.train()
    stem_height = float(input("Stem height: "))
    leaf_radius = float(input("Leaf radius: "))
    predicted_age = age_identifier.identify(stem_height, leaf_radius)
    print(f"Predicted age: {predicted_age}")
