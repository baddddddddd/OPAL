import torch
from torch import nn


class Connect4o(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # (3, 6, 7)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # -> (32, 6, 7)
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # -> (32, 6, 7)
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # -> (32, 6, 7)
            nn.ReLU(),

            nn.Flatten(), 

            nn.Linear(32 * 6 * 7, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.net(x)

