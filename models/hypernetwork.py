import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    def __init__(self, z_dim=64, delta_dim=402):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, delta_dim)
        )

    def forward(self, z):
        return self.fc(z)
