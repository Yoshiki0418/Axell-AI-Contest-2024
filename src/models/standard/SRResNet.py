import torch
from torch import nn

class SRResNet4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        # Residual Block (conv_block)
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )

        # Middle Block
        self.middle_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )

        # Upsampling layers
        self.upsampling_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.upsampling_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.prelu(self.conv1(X))
        residual = X

        for _ in range(5):  
            X = self.residual_block(X) + X

        X = self.middle_block(X) + residual  

        X = self.upsampling_layer1(X)
        X = self.upsampling_layer2(X)

        output = self.final_conv(X)
        return output
