"""
Model definitions migrated from federated-learning-master/models/Nets.py
Minimal, clean PyTorch models + factory.
"""

from typing import Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 29, hidden: int = 200, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)



class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(name: str = "mlp", input_dim: int = 784, num_classes: int = 10) -> nn.Module:
    """
    Factory to produce model used by fedselect.
    name: "mlp" | "cnn"
    """
    if name.lower() == "cnn":
        return SimpleCNN(num_classes=num_classes)
    return MLP(input_dim=input_dim, num_classes=num_classes)