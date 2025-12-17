import torch.nn as nn


class Classification(nn.Module):
    def __init__(self, input_dim, hidden_layers=None):
        super(Classification, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        current_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.6))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
