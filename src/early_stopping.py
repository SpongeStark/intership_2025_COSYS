import torch
import os
from pathlib import Path

class EarlyStopping:
    def __init__(self, save_dir, patience=5, min_delta=1e-7, verbose=True):
        """
        参数:
            patience (int): 容忍验证指标不提升的轮数
            min_delta (float): 最小变化量，小于该值不被视为提升
            path (str): 最优模型保存路径
            verbose (bool): 是否打印提示信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir
        self.path = str(Path(save_dir) / "best_model.pth")
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        """
        参数:
            val_score: 当前验证集上的得分（比如 loss 越小越好，或 accuracy 越大越好）
            model: 当前模型（将被保存）
        """
        # 如果第一次评估
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        # 检查是否有提升
        elif val_score > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else: # loss又降了
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0  # 重置

    def save_checkpoint(self, model):
        """保存当前最优模型"""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation score improved. Saving model to {self.path}")
