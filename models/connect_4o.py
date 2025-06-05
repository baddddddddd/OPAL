import torch
from torch import nn


class Connect4o(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # (3, 6, 7)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            # (16, 6, 7)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            # (16, 6, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (16, 3, 3)
            nn.Flatten(),

            nn.Linear(16 * 3 * 3, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.net(x)

