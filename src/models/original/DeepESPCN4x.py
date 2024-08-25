import torch
from torch import nn, clip, tensor

class DeepESPCN4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        # self.conv_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        # nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        # nn.init.zeros_(self.conv_4.bias)

        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_5.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_5.bias)

        self.conv_6 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_6.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_6.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X_in: tensor) -> tensor:
        X1 = self.act(self.conv_1(X_in))
        X = self.act(self.conv_2(X1))
        X = self.act(self.conv_3(X))+ X1
        # X = self.act(self.conv_4(X)) + X1
        X = self.act(self.conv_5(X))
        X = self.conv_6(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out

class DeepESPCN4x_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, padding=4)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        # self.conv_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        # nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        # nn.init.zeros_(self.conv_4.bias)

        # self.conv_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # nn.init.normal_(self.conv_5.weight, mean=0, std=0.001)
        # nn.init.zeros_(self.conv_5.bias)

        self.conv_6 = nn.Conv2d(in_channels=32, out_channels=(8 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_6.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_6.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        self.conv_7 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=9, padding=4)

    def forward(self, X_in: tensor) -> tensor:
        X1 = self.act(self.conv_1(X_in))
        X = self.act(self.conv_2(X1))
        X = self.act(self.conv_3(X))
        # X = self.act(self.conv_4(X)) + X1
        # X = self.act(self.conv_5(X))
        X = self.conv_6(X)
        X = self.act(self.pixel_shuffle(X))
        X = self.conv_7(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out