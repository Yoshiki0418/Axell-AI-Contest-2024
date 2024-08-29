import torch
from torch import nn, clip, tensor

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        return x * self.fc(avg_out)

class LiteRCAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.i = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ChannelAttention(in_channels=32) 
        )
        nn.init.normal_(self.residual_block[0].weight, mean=0, std=0.001)
        nn.init.zeros_(self.residual_block[0].bias)
        nn.init.normal_(self.residual_block[2].weight, mean=0, std=0.001)
        nn.init.zeros_(self.residual_block[2].bias)

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X_in: tensor) -> tensor:
        X = self.act(self.conv_1(X_in))

        for _ in range(self.i):
            X = self.residual_block(X) + X

        X = self.conv_2(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out