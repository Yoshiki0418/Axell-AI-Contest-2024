import torch
from torch import nn, clip, tensor

class DeepESPCN4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_5.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_5.bias)

        self.conv_6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_6.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_6.bias)

        self.conv_7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_7.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_7.bias)

        self.conv_8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_8.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_8.bias)

        self.conv_9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_9.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_9.bias)

        self.conv_10 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_10.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_10.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X_in: tensor) -> tensor:
        X1 = self.act(self.conv_1(X_in))
        X = self.act(self.conv_2(X1))
        X2 = self.act(self.conv_3(X))+ X1
        X3 = self.act(self.conv_4(X2)) + X2
        X4 = self.act(self.conv_5(X3))+ X3
        X5 = self.act(self.conv_6(X4))+ X4
        X6 = self.act(self.conv_7(X5))+ X5
        X7 = self.act(self.conv_8(X6))+ X6
        X8 = self.act(self.conv_9(X7))+ X7
        X = self.conv_10(X8)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out

"""
No.1の精度
まだ推論時間に余裕あり
-> 層を深くする（model1にて実験）
-> 32チャン統一から64に変更
"""
class DeepESPCN4x_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

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
        X2 = self.act(self.conv_3(X))+ X1
        X3 = self.act(self.conv_4(X2)) + X2
        X = self.act(self.conv_5(X3))+ X3
        X = self.conv_6(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out
    
class DeepESPCN4x_v3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4)
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

        self.final_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, X_in: tensor) -> tensor:
        X1 = self.act(self.conv_1(X_in))
        X = self.act(self.conv_2(X1))
        X = self.act(self.conv_3(X))+ X1
        # X = self.act(self.conv_4(X)) + X1
        X = self.act(self.conv_5(X))
        X = self.conv_6(X)
        X = self.act(self.pixel_shuffle(X))
        X = self.final_conv(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out