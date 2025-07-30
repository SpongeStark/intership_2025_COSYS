"""
File: losses_dexined.py
Author: YANG Kai
Date: 2025-06-21
Description: The class of loss for DexiNed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class WeightedLoss(nn.Module, ABC):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight if weight else [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]

    def forward(self, prediction, target):
        assert len(prediction) == len(self.weight), "The weights should have the same length as the output of the model"
        losses = self._loss_fn(prediction, target)
        return sum([loss * l_w for loss, l_w in zip(losses, self.weight)]) / sum(self.weight)
    
    @abstractmethod
    def _loss_fn(self, prediction, target):
        pass

class WeightedMSELoss(WeightedLoss):
    def _loss_fn(self, prediction, target):
        return [F.mse_loss(pred.squeeze(1), target) for pred in prediction]

class WeightedMAELoss(WeightedLoss):
    def _loss_fn(self, prediction, target):
        return [F.l1_loss(pred.squeeze(1), target) for pred in prediction]

class WeightedMSELossWithDilatation(WeightedMSELoss):
    def __init__(self, weight=None, kernel_size=3):
        super().__init__(weight)
        self.kernel_size = kernel_size

    def _loss_fn(self, prediction, target):
        # dilatation (max pooling)
        y_dilatation = F.max_pool2d(target, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        return super()._loss_fn(prediction, y_dilatation)

class WeightedMSELossWithMeanBlur(WeightedMSELoss):
    def __init__(self, weight=None, kernel_size=3):
        super().__init__(weight)
        self.kernel_size = kernel_size
    
    def mean_blur(self, img):
        k = self.kernel_size
        kernel = torch.ones((1, 1, k, k), device=img.device) / (k * k)
        blurred = F.conv2d(img.unsqueeze(1), kernel, padding=k // 2)
        return blurred.squeeze()

    def _loss_fn(self, prediction, target):
        y_blurred = self.mean_blur(target)
        return super()._loss_fn(prediction, y_blurred)
    
def test(criterion, device='cpu'):
    N, C, H, W = 8, 3, 480, 360
    x = torch.randn(N, C, H, W, requires_grad=True, device=device)
    y = torch.randn(N, H, W, requires_grad=True, device=device)
    outputs = [torch.randn(N, 1, H, W, requires_grad=True, device=device) for i in range(7)]
    loss = criterion(outputs, y)
    print(loss.detach().item())

if __name__=="__main__":
    criterion = WeightedMSELossWithMeanBlur()
    print(f"Test {criterion.__class__.__name__}")
    test(criterion, 'cuda' if torch.cuda.is_available() else "cpu")