import torch
from torch import nn, clip

"""
パラメータ数は少ないが、推論時間が遅い
ピクセルシャッフルを２回にしたことで、メモリが再配置されるため計算効率が悪くなったことが推測される。
"""

class StepShuffleSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.scale = 4
        self.i = 3
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

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

        # Upsampling layers
        self.upsampling_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        nn.init.normal_(self.upsampling_layer1[0].weight, mean=0, std=0.001)
        nn.init.zeros_(self.upsampling_layer1[0].bias)

        self.upsampling_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        nn.init.normal_(self.upsampling_layer2[0].weight, mean=0, std=0.001)
        nn.init.zeros_(self.upsampling_layer2[0].bias)

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.final_conv.weight, mean=0, std=0.001)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        X = self.act(self.conv_1(X_in))

        for _ in range(self.i):
            X = self.residual_block(X) + X
        
        X = self.upsampling_layer1(X) 
        X = self.upsampling_layer2(X)
        X = self.final_conv(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out