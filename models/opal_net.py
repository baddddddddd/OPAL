import torch
import torch.functional as F
from torch import nn

from modules.resnet import ResidualBlock


class OPALNetv1(nn.Module):
    def __init__(self, input_shape, filters, blocks, fc_length, output_length):
        super().__init__()

        assert len(input_shape) == 3, "Invalid input shape"

        input_channels = input_shape[0]
        cell_count = input_shape[1] * input_shape[2]

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        tower_modules = []
        for _ in range(blocks):
            tower_modules.append(ResidualBlock(filters, filters))

        self.residual_tower = nn.Sequential(*tower_modules)

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(filters * cell_count, fc_length),
            nn.ReLU(inplace=True),

            nn.Linear(fc_length, output_length),
        )


    def forward(self, x):
        x = self.initial(x)
        x = self.residual_tower(x)
        x = self.fc(x)
        return x