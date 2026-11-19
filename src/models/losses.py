"""
Custom Loss Functions and Remote Sensing Evaluation Metrics.
Includes L1 Radiometric Loss, Differentiable SSIM Loss, and Spatial Sobel Gradient Loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel window for SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Index.
    Measures structural integrity, luminance, and contrast preservation.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        window = _gaussian_window(window_size, sigma, channels)
        self.register_buffer('window', window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if channel != self.channels:
            window = _gaussian_window(self.window_size, 1.5, channel).to(img1.device, dtype=img1.dtype)
        else:
            window = self.window.to(img1.device, dtype=img1.dtype)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        padding = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()


class GradientLoss(nn.Module):
    """
    Sobel Edge / Gradient Difference Loss.
    Enforces sharp geographic boundaries (coastlines, rivers, roads, urban blocks).
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.size()
        kx = self.sobel_x.repeat(c, 1, 1, 1).to(pred.device, dtype=pred.dtype)
        ky = self.sobel_y.repeat(c, 1, 1, 1).to(pred.device, dtype=pred.dtype)

        grad_x_pred = F.conv2d(pred, kx, padding=1, groups=c)
        grad_y_pred = F.conv2d(pred, ky, padding=1, groups=c)
        grad_x_tgt = F.conv2d(target, kx, padding=1, groups=c)
        grad_y_tgt = F.conv2d(target, ky, padding=1, groups=c)

        loss_x = F.l1_loss(grad_x_pred, grad_x_tgt)
        loss_y = F.l1_loss(grad_y_pred, grad_y_tgt)
        return loss_x + loss_y


class CompositeSatelliteLoss(nn.Module):
    """
    Combines Radiometric L1, Structural Similarity (SSIM), and Geographic Gradient Losses.
    """
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5, grad_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM()
        self.grad_loss = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = self.l1_loss(pred, target)
        ssim_val = self.ssim_loss(pred, target)
        ssim_loss = 1.0 - ssim_val
        grad = self.grad_loss(pred, target)

        total = (self.l1_weight * l1) + (self.ssim_weight * ssim_loss) + (self.grad_weight * grad)
        return total, {
            'total_loss': total.item(),
            'l1_loss': l1.item(),
            'ssim_val': ssim_val.item(),
            'ssim_loss': ssim_loss.item(),
            'grad_loss': grad.item()
        }


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


if __name__ == "__main__":
    # Smoke test losses
    loss_fn = CompositeSatelliteLoss()
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    loss, details = loss_fn(a, b)
    psnr = compute_psnr(a, b)
    print(f"Loss details: {details}")
    print(f"PSNR: {psnr:.2f} dB")
    print("Loss functions smoke test passed!")
