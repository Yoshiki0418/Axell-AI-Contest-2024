import torch
from torch import nn, clip

class FusionSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.scale = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        nn.init.normal_(self.conv1[0].weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv1[0].bias)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        nn.init.normal_(self.conv2[0].weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv2[0].bias)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv3.bias)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv4.bias)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv5.bias)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv6.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self.prelu = nn.PReLU()
    
    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        X_in_1 = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X_c1 = self.conv1(X_in_1)
        X_c3 = self.conv2(X_in)
        X = X_c1 + X_c3
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.prelu(self.pixel_shuffle(X))
        X_out = clip(X, 0.0, 1.0)
        return X_out


class LiteFusionSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.scale = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        nn.init.normal_(self.conv1[0].weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv1[0].bias)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        nn.init.normal_(self.conv2[0].weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv2[0].bias)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv3.bias)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv4.bias)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv5.bias)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.001) 
        nn.init.zeros_(self.conv6.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self.prelu = nn.PReLU()
    
    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        X_in_1 = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X_c1 = self.conv1(X_in_1)
        X_c3 = self.conv2(X_in)
        X = X_c1 + X_c3
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.prelu(self.pixel_shuffle(X))
        X_out = clip(X, 0.0, 1.0)
        return X_out     
