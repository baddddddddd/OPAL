import torch
from torch import nn

class TicTacAIv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            
            # (3 x 3 x 3)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            # (3 x 3 x 3)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
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


# # Convolutional Layers + Batch Norm
# class TicTacAIv2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
            
#             # (3 x 3 x 3)
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             # (3 x 3 x 3)
#             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             #### works best
#             nn.Flatten(),

#             nn.Linear(16 * 3 * 3, 32),
#             nn.ReLU(),

#             nn.Linear(32, 32),
#             nn.ReLU(),

#             nn.Linear(32, 1),
#             nn.Tanh(),
#             ########
#         )


#     def forward(self, x):
#         return self.net(x)

from modules.resnet import ResidualBlock


# # ResNet-Style, Residual tower + Global Average Pool
# class TicTacAIv2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

#         self.residual_tower = nn.Sequential(
#             ResidualBlock(16, 16),
#             ResidualBlock(16, 16),
#             # ResidualBlock(16, 16),
#         )


#         self.gap = nn.AdaptiveAvgPool2d((1, 1))

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(16, 1),
#             nn.Tanh(),
#         )


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.residual_tower(x)
#         x = self.gap(x)
#         x = self.fc(x)
#         return x


# # Residual Tower + GAP + MLP
# class TicTacAIv2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             ResidualBlock(3, 16),
#             ResidualBlock(16, 16),
#             nn.AdaptiveAvgPool2d((1, 1)),

#             nn.Flatten(),

#             nn.Linear(16, 16),
#             nn.ReLU(),

#             nn.Linear(16, 1),
#             nn.Tanh(),
#         )


#     def forward(self, x):
#         return self.net(x)

# # Residual Tower + MLP
# class TicTacAIv2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.residual_tower = nn.Sequential(
#             ResidualBlock(3, 16),
#             ResidualBlock(16, 16),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(16 * 3 * 3, 16),
#             nn.ReLU(),

#             nn.Linear(16, 1),
#             nn.Tanh(),
#         )


#     def forward(self, x):
#         x = self.residual_tower(x)
#         x = self.fc(x)
#         return x


# ResNet + MLP, learned at 500 steps
# class TicTacAIv2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.initial = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#         )

#         self.residual_tower = nn.Sequential(
#             ResidualBlock(16, 16),
#             ResidualBlock(16, 16),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(16 * 3 * 3, 16),
#             nn.ReLU(inplace=True),

#             nn.Linear(16, 1),
#             nn.Tanh(),
#         )


#     def forward(self, x):
#         x = self.initial(x)
#         x = self.residual_tower(x)
#         x = self.fc(x)
#         return x


from models.opal_net import OPALNetv1
# class TicTacAIv2(OPALNetv1):
#     def __init__(self):
#         super().__init__(input_shape=(3, 3, 3), filters=32, blocks=2, fc_length=32, output_length=1)


#     def forward(self, x):
#         x = super().forward(x)
#         x = torch.tanh(x)
#         return x


class TicTacAIv3(OPALNetv1):
    def __init__(self):
        super().__init__(input_shape=(3, 3, 3), filters=32, blocks=2, fc_length=32, output_length=3)
