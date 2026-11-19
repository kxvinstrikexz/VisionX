"""
PatchGAN Discriminator for Conditional Adversarial Satellite Image Translation.
Evaluates local NxN patches to classify real vs synthetic satellite RGB images.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.
    Takes concatenated [IR, RGB] (e.g. 1+3=4 channels or 3+3=6 channels)
    and predicts whether each local patch is real or synthetic.
    """
    def __init__(self, in_channels: int = 4, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: No BatchNorm on first layer
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 1 channel prediction map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Concatenate condition (IR) and target (RGB) along channel dimension
        x = torch.cat([condition, target], dim=1)
        return self.model(x)


if __name__ == "__main__":
    disc = PatchGANDiscriminator(in_channels=1 + 3, ndf=32)
    ir = torch.randn(2, 1, 64, 64)
    rgb = torch.randn(2, 3, 64, 64)
    pred = disc(ir, rgb)
    print(f"Discriminator output shape: {pred.shape}")
    print("PatchGAN discriminator smoke test passed!")
