import torch
from torch import nn

class TicTacAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            
            # (3 x 3 x 3)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),

            # (3 x 3 x 3)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            #### works best
            nn.Flatten(),

            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Tanh(),
            ########
        )


    def forward(self, x):
        return self.net(x)
