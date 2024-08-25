import torch
from torch import nn
from models import *

def calculate_flops(model, input_size):
    flops = 0
    def conv2d_hook(self, input, output):
        # Calculate FLOPs for convolution
        batch_size, in_channels, h, w = input[0].size()
        out_channels, _, kernel_h, kernel_w = self.weight.size()
        flops_per_instance = out_channels * in_channels * kernel_h * kernel_w * h * w
        flops_per_instance = flops_per_instance // self.groups
        flops_total = flops_per_instance * batch_size
        nonlocal flops
        flops += flops_total

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv2d_hook))

    # Forward pass to trigger hooks
    input = torch.randn(*input_size)
    model(input)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    return flops


# モデルのインスタンス化
model = StepShuffleSR()
# パラメータ数の計算
total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数:{total_params}")
# 入力サイズを指定 (バッチサイズ, チャンネル数, 高さ, 幅)
input_size = (1, 3, 64, 64)

# FLOPsを計算
total_flops = calculate_flops(model, input_size)

print(f"計算量：{total_flops}")