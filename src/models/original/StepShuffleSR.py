import torch
from torch import nn, clip

class StepShuffleSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        # Upsampling layers
        self.upsampling_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        nn.init.normal_(self.upsampling_layer1[0].weight, mean=0, std=0.001)
        nn.init.zeros_(self.upsampling_layer1[0].bias)

        self.upsampling_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        nn.init.normal_(self.upsampling_layer2[0].weight, mean=0, std=0.001)
        nn.init.zeros_(self.upsampling_layer2[0].bias)

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.final_conv.weight, mean=0, std=0.001)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        X1 = self.act(self.conv_1(X_in))
        X = self.act(self.conv_2(X1)) + X1
        X2 = self.act(self.conv_3(X))
        X = self.act(self.conv_4(X2)) + X2
        X = self.upsampling_layer1(X) 
        X = self.upsampling_layer2(X)
        X = self.final_conv(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out