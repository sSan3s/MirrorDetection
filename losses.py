"""
거울 탐지를 위한 손실 함수 모음
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCEDiceLoss(nn.Module):
    """BCE + Dice 복합 손실"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class IoULoss(nn.Module):
    """IoU (Intersection over Union) 손실"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class FocalLoss(nn.Module):
    """Focal Loss - 어려운 샘플에 집중"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - FP와 FN에 다른 가중치"""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class StructuralLoss(nn.Module):
    """구조적 유사성 손실 (거울 경계 보존)"""

    def __init__(self, kernel_size=11):
        super().__init__()
        self.kernel_size = kernel_size
        self.channel = 1
        self.window = self._create_window(kernel_size, self.channel)

    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                  for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        if pred.size(1) == 1:  # 단일 채널
            channel = self.channel
            window = self.window.to(pred.device)

            mu1 = F.conv2d(pred, window, padding=self.kernel_size // 2, groups=channel)
            mu2 = F.conv2d(target, window, padding=self.kernel_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(pred * pred, window, padding=self.kernel_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=self.kernel_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(pred * target, window, padding=self.kernel_size // 2, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return 1 - ssim_map.mean()


class MirrorLoss(nn.Module):
    """거울 탐지를 위한 종합 손실 함수"""

    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,
                 iou_weight=0.5,
                 ssim_weight=0.1,
                 focal_weight=0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BCEDiceLoss(bce_weight=0, dice_weight=1)
        self.iou = IoULoss()
        self.ssim = StructuralLoss()
        self.focal = FocalLoss()

        self.weights = {
            'bce': bce_weight,
            'dice': dice_weight,
            'iou': iou_weight,
            'ssim': ssim_weight,
            'focal': focal_weight
        }

    def forward(self, pred, target):
        losses = {}

        if self.weights['bce'] > 0:
            losses['bce'] = self.bce(pred, target)

        if self.weights['dice'] > 0:
            losses['dice'] = self.dice(pred, target)

        if self.weights['iou'] > 0:
            losses['iou'] = self.iou(pred, target)

        if self.weights['ssim'] > 0:
            losses['ssim'] = self.ssim(pred, target)

        if self.weights['focal'] > 0:
            losses['focal'] = self.focal(pred, target)

        # 총 손실 계산
        total_loss = sum(self.weights[k] * v for k, v in losses.items() if k in self.weights)

        # 로깅을 위한 개별 손실값 반환
        loss_dict = {k: v.item() for k, v in losses.items()}

        return total_loss, loss_dict