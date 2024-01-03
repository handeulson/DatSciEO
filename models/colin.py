import numpy as np
import torch
import torch.nn as nn


class TreeClassifConvNet(nn.Module):
    def __init__(self, n_classes=10, width=5, height=5, depth=30, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes
        self.width = width
        self.height = height
        self.depth = depth
    
        self.model = nn.Sequential(
            nn.Conv2d(depth, depth//2, kernel_size=3, padding=1),       # e.g. 5x5x30 -> 5x5x15
            nn.ReLU(),
            nn.Conv2d(depth//2, depth//4, kernel_size=3, padding=1),    # e.g. 5x5x15 -> 5x5x7
            nn.ReLU(),
            nn.Conv2d(depth//4, 5, kernel_size=3, padding=1),           # e.g. 5x5x7 -> 5x5x5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5*5*5, n_classes)
        )
        

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # input size: N, C_in, H, W
    # output size: N, C_out, H_out, W_out
    # H_out = (H_in + 2*pad - dil * (kernel-1) - 1) / stride  +  1

    depth, width, height = 30, 10, 10

    input = torch.randn(1, depth, height, width)
    model = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3)

    out = model(input)
    print("in shape: ", input.shape)
    print("out shape:", out.shape)