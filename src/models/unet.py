"""
Satellite U-Net Architecture for Cross-Spectral (IR to RGB) Translation.
Preserves high-frequency spatial structures (rivers, roads, fields, urban edges)
via skip connections while performing spectral radiometric color mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> LeakyReLU) * 2 with optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if in_channels != out_channels and residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity() if residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DownBlock(nn.Module):
    """Downsampling with MaxPool followed by DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling followed by feature concatenation and DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if odd spatial dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SatelliteUNet(nn.Module):
    """
    Satellite U-Net for Infrared to RGB Translation.
    
    Args:
        in_channels: 1 for single-band IR (NIR), 3 for multispectral IR (NIR, SWIR1, SWIR2).
        out_channels: 3 for RGB output.
        init_features: Base channel width (default 32).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 3, init_features: int = 32):
        super().__init__()
        f = init_features
        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = DownBlock(f, f * 2)        # 32 -> 64
        self.down2 = DownBlock(f * 2, f * 4)    # 64 -> 128
        self.down3 = DownBlock(f * 4, f * 8)    # 128 -> 256

        # Bottleneck refinement
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder with skip connections
        self.up1 = UpBlock(f * 8, f * 4)        # 256 -> 128
        self.up2 = UpBlock(f * 4, f * 2)        # 128 -> 64
        self.up3 = UpBlock(f * 2, f)            # 64 -> 32

        # Final head
        self.outc = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1),
            nn.Sigmoid()  # Output scaled strictly to normalized reflectance [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        x = self.up1(b, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        rgb = self.outc(x)
        return rgb


if __name__ == "__main__":
    # Smoke test model
    model = SatelliteUNet(in_channels=1, out_channels=3)
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    assert output.shape == (2, 3, 64, 64)
    assert 0.0 <= output.min() and output.max() <= 1.0
    print("SatelliteUNet smoke test passed!")
