import torch.nn as nn

class MultiOutputNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200), # Couche d'entrée
            nn.ReLU(),
            nn.Linear(200, 200),# Hidden layers 1
            nn.ReLU(),
            nn.Linear(200, 200),# Hidden layers 2
            nn.ReLU(),
            nn.Linear(200, out_dim),# Couche de sortie
        )

    def forward(self, x):
        return self.layers(x)