import torch
from torch import nn


class Connect4o(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # (3 x 6 x 7)
            nn.Flatten(start_dim=1),

            nn.Linear(3 * 6 * 7, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.net(x)

