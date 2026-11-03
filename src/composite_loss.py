# src/composite_loss.py
"""
MUESUM — Multi-Scale Composite Restoration Loss
===============================================
Formulates the comprehensive multi-term training objective for historical artwork inpainting:
  L_total = lambda_l1 * L_l1_masked + lambda_edge * L_edge + lambda_tv * L_tv + lambda_perc * L_perc

Designed for:
  - Preserving fine craquelure, brushstroke textures, and high-frequency edges.
  - Prioritizing missing substrate inside damaged masks while enforcing boundary continuity.
  - Mitigating the over-smoothing and color desaturation observed under pure L1 loss.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MaskWeightedL1Loss(nn.Module):
    """
    Mask-weighted L1 loss. Applies higher penalty inside the damage mask
    to force the network to actively synthesize missing pictorial content.
    """
    def __init__(self, damage_weight: float = 4.0, clean_weight: float = 1.0):
        super().__init__()
        self.damage_weight = damage_weight
        self.clean_weight = clean_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(pred - target)
        if mask is None:
            return diff.mean()

        # mask shape: [B, 1, H, W] or [B, H, W], where > 0.5 is damaged
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != pred.shape[1]:
            mask = mask.repeat(1, pred.shape[1], 1, 1)

        weights = torch.where(mask > 0.5, self.damage_weight, self.clean_weight)
        weighted_diff = diff * weights
        return weighted_diff.sum() / (weights.sum() + 1e-6)


class SobelEdgeLoss(nn.Module):
    """
    Computes L1 difference of Sobel edge gradient magnitudes.
    Enforces sharp structural contours and brushwork boundaries.
    """
    def __init__(self):
        super().__init__()
        # Sobel horizontal & vertical filter kernels
        kx = torch.tensor([[1.0, 0.0, -1.0],
                           [2.0, 0.0, -2.0],
                           [1.0, 0.0, -1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1.0,  2.0,  1.0],
                           [0.0,  0.0,  0.0],
                           [-1.0, -2.0, -1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert RGB to Luminance
        pred_gray = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
        tgt_gray = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

        gx_pred = F.conv2d(pred_gray, self.kx, padding=1)
        gy_pred = F.conv2d(pred_gray, self.ky, padding=1)
        gx_tgt = F.conv2d(tgt_gray, self.kx, padding=1)
        gy_tgt = F.conv2d(tgt_gray, self.ky, padding=1)

        mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)
        mag_tgt = torch.sqrt(gx_tgt**2 + gy_tgt**2 + 1e-6)

        return F.l1_loss(mag_pred, mag_tgt)


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) loss penalizes high-frequency checkerboard artifacts and pixel noise.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw


class VGGPerceptualFeatureLoss(nn.Module):
    """
    Multi-scale perceptual feature loss using pre-trained VGG-19 feature activations.
    Computes feature distance across shallow (edges), intermediate (textures), and deep layers.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()

        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])     # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 18)])  # relu3_4

        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        # Standardize to ImageNet range
        p = (pred - mean) / std
        t = (target - mean) / std

        # Slice 1
        p_f1 = self.slice1(p)
        t_f1 = self.slice1(t)
        loss = F.l1_loss(p_f1, t_f1)

        # Slice 2
        p_f2 = self.slice2(p_f1)
        t_f2 = self.slice2(t_f1)
        loss += F.l1_loss(p_f2, t_f2)

        # Slice 3
        p_f3 = self.slice3(p_f2)
        t_f3 = self.slice3(t_f2)
        loss += F.l1_loss(p_f3, t_f3)

        return loss


class CompositeRestorationLoss(nn.Module):
    """
    Master Composite Loss combining Masked L1, Sobel Edge, Total Variation, and Perceptual Loss.
    """
    def __init__(
        self,
        lam_l1: float = 1.0,
        lam_edge: float = 0.10,
        lam_tv: float = 0.01,
        lam_perc: float = 0.05,
        damage_weight: float = 4.0,
        use_perceptual: bool = True
    ):
        super().__init__()
        self.lam_l1 = lam_l1
        self.lam_edge = lam_edge
        self.lam_tv = lam_tv
        self.lam_perc = lam_perc
        self.use_perceptual = use_perceptual

        self.l1_loss = MaskWeightedL1Loss(damage_weight=damage_weight, clean_weight=1.0)
        self.edge_loss = SobelEdgeLoss()
        self.tv_loss = TotalVariationLoss()
        self.perc_loss = VGGPerceptualFeatureLoss() if use_perceptual else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward computation returning total loss and breakdown metrics dictionary.
        """
        l1_val = self.l1_loss(pred, target, mask=mask)
        edge_val = self.edge_loss(pred, target)
        tv_val = self.tv_loss(pred)

        total = self.lam_l1 * l1_val + self.lam_edge * edge_val + self.lam_tv * tv_val

        perc_val_scalar = 0.0
        if self.use_perceptual and self.perc_loss is not None:
            perc_val = self.perc_loss(pred, target)
            total = total + self.lam_perc * perc_val
            perc_val_scalar = float(perc_val.detach().item())

        metrics = {
            "total_loss": float(total.detach().item()),
            "l1_loss": float(l1_val.detach().item()),
            "edge_loss": float(edge_val.detach().item()),
            "tv_loss": float(tv_val.detach().item()),
            "perc_loss": perc_val_scalar
        }

        return total, metrics
