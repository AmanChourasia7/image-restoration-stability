import torch
import torch.nn as nn


class DnCNN(nn.Module):

    def __init__(self, channels=3, num_layers=17, features=64):
        super(DnCNN, self).__init__()

        layers = []

        layers.append(
            nn.Conv2d(channels, features, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(features, channels, kernel_size=3, padding=1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
