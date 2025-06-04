import torch
from torch import nn

class TicTacAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            
            # (3 x 3 x 3)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(1 * 1 * 32, 32),
            nn.ReLU(),

            # nn.Linear(16, 16),
            # nn.ReLU(),

            nn.Linear(32, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.net(x)
