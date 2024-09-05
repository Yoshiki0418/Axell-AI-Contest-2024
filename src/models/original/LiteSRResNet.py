import torch
from torch import nn, clip, tensor

class LiteSRResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.i = 5
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.act = nn.ReLU()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
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
        X = self.act(self.conv_3(X))
        X_1 = X

        for _ in range(self.i):
            X = self.residual_block(X) + X
        X = X + X_1
        X = self.conv_2(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out
    
class LiteSRResNet_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.i = 5
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.act = nn.ReLU()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
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
        X_in_1 = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X = self.act(self.conv_1(X_in))
        X = self.act(self.conv_3(X))
        X_1 = X

        for _ in range(self.i):
            X = self.residual_block(X) + X
        X = X + X_1
        X = self.conv_2(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out
