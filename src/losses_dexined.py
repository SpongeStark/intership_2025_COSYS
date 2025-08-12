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

class WeightedMaskedBCELoss(WeightedLoss):
    def bdcn_loss2(self, yhat, y):
        n_positive = torch.where(y.long()>0, 1., 0.).sum().item()
        n_negative = torch.where(y.long()<=0, 1., 0.).sum().item()
        
        # mask = torch.zeros_like(y)
        # mask[y>0] = 1.0 * n_negative / (n_positive + n_negative)
        # mask[y<=0] = 1.1 * n_positive / (n_positive + n_negative)
        mask = torch.where(y>0., 1.0 * n_negative, 1.1 * n_positive) / (n_positive + n_negative)
        yhat= torch.sigmoid(yhat)
        cost = torch.nn.BCELoss(mask, reduction='none')(yhat, y.float())
        cost = torch.sum(cost.float().mean((1, 2)))
        return cost
    
    def _loss_fn(self, prediction, target):
        return [self.bdcn_loss2(pred.squeeze(1), target) for pred in prediction]
    
    def forward(self, prediction, target):
        assert len(prediction) == len(self.weight), "The weights should have the same length as the output of the model"
        losses = self._loss_fn(prediction, target)
        return sum([loss * l_w for loss, l_w in zip(losses, self.weight)])

class WeightedBCELoss(WeightedLoss):
    def bdcn_loss2(self, yhat, y):
        yhat= torch.sigmoid(yhat)
        cost = torch.nn.BCELoss()(yhat, y)
        # cost = torch.sum(cost.float().mean((1, 2)))
        return cost
    
    def _loss_fn(self, prediction, target):
        return [self.bdcn_loss2(pred.squeeze(1), target) for pred in prediction]
    
    def forward(self, prediction, target):
        assert len(prediction) == len(self.weight), "The weights should have the same length as the output of the model"
        losses = self._loss_fn(prediction, target)
        return sum([loss * l_w for loss, l_w in zip(losses, self.weight)])

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

class WeightedCombineLoss(WeightedLoss):
    def __init__(self, weight=None, beta=0.51):
        super().__init__(weight)
        self.beta = beta

    def _loss_fn(self, prediction, target):
        return [torch.pow((pred-target)**2, self.beta).mean() for pred in prediction]
    
def test(criterion, device='cpu'):
    from dataset import BIPEDv2
    from pathlib import Path
    import os
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dataset = BIPEDv2(
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/imgs/test/rgbr/",
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
    batch = next(iter(loader))
    x, y = batch['image_tensor'].to(device), batch['edge_tensor'].to(device)
    # x = torch.randn(N, C, H, W, requires_grad=True, device=device)
    # y = torch.rand((N, H, W), requires_grad=True, device=device)
    N, C, H, W = x.shape
    outputs = [torch.randn(N, 1, H, W, requires_grad=True, device=device) for i in range(7)]
    loss = criterion(outputs, y)
    print(loss.detach().item())

def test_bce(device):
    # loss by class
    criterion = WeightedBCELoss()
    from dataset import BIPEDv2
    from pathlib import Path
    import os
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dataset = BIPEDv2(
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/imgs/test/rgbr/",
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    batch = next(iter(loader))
    x, y = batch['image_tensor'].to(device), batch['edge_tensor'].to(device)
    # x = torch.randn(N, C, H, W, requires_grad=True, device=device)
    # y = torch.rand((N, H, W), requires_grad=True, device=device)
    N, C, H, W = x.shape
    outputs = [torch.randn(N, 1, H, W, requires_grad=True, device=device) for i in range(7)]
    loss1 = criterion(outputs, y).item()
    # loss by function
    import sys
    sys.path.append(str(PROJECT_ROOT / 'resource/DexiNed'))
    from losses import bdcn_loss2
    loss2 = sum([bdcn_loss2(preds.squeeze(), y, l_w) for preds, l_w in zip(outputs,criterion.weight)]) / len(criterion.weight)
    loss2 = loss2.item()
    print(loss1, loss2)


if __name__=="__main__":
    criterion = WeightedCombineLoss()
    print(f"Test {criterion.__class__.__name__}")
    # test(criterion)
    test(criterion, 'cuda' if torch.cuda.is_available() else "cpu")
    # test_bce("cpu")