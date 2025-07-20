"""
File: pl_model.py
Author: YANG Kai
Date: 2025-07-01
Description: A pytorch lightning model and its training function using pl.trainer
"""

from pathlib import Path
project_root = Path("/disk/hd/cosys/intership_2025_COSYS")

import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from dataset import BIPEDv2
import numpy as np
# from skimage.morphology import thin

import sys
model_path = str(project_root / "resource/DexiNed")
if str(model_path) not in sys.path:
    sys.path.append(str(model_path))
from model import DexiNed

# 用LightningModule包装模型和训练流程
class LitModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = DexiNed()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.train_losses = []
        self.training_step_outputs = []
        self.val_losses = []
        self.validation_step_outputs = []
        self.lr = lr

    def forward(self, x):
        outputs = self.model(x)
        return outputs # [thin(edge_tensor>0.9) for edge_tensor in outputs]
    
    def get_loss(self, outputs, y):
        return sum([self.criterion(output.squeeze(), y) for output in outputs]) / len(outputs)

    def training_step(self, batch, batch_idx):
        x, y = batch['image_tensor'], batch['visibility_map']
        outputs = self.model(x)
        loss = self.get_loss(outputs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image_tensor'], batch['visibility_map']
        outputs = self.model(x)
        loss = self.get_loss(outputs, y)
        self.log('val_loss', loss)
        self.validation_step_outputs.append(loss.item())
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
            # "monitor": "val_loss"  # 可选，用于某些 scheduler（如 ReduceLROnPlateau）
        }
    
    def on_train_epoch_end(self):
        self.train_losses.append(np.mean(self.training_step_outputs))
        self.training_step_outputs = []  # free memory

    def on_validation_epoch_end(self):
        self.val_losses.append(np.mean(self.validation_step_outputs))
        self.validation_step_outputs = []  # free memory


def main(description, save_dir, epoch=1, batch_size=4, n_work=53, learning_rate=1e-5):
    # 创建数据集和dataloader（示例数据）
    biped_dataset = BIPEDv2(
        project_root / "data/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/",
        project_root / "data/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real/"
    )
    # biped_dataset = Subset(biped_dataset, list(range(16)))
    train_loader = DataLoader(biped_dataset, batch_size=batch_size, num_workers=n_work)
    # test set | 测试集
    test_dataset = BIPEDv2(
        project_root / "data/BIPEDv2/BIPED/edges/imgs/test/rgbr/",
        project_root / "data/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
    )
    # test_dataset = Subset(test_dataset, list(range(8)))
    val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_work)

    # 初始化模型和Trainer，开始训练
    model = LitModel(learning_rate)
    trainer = pl.Trainer(
        max_epochs=epoch,
        log_every_n_steps=10,
        logger=CSVLogger(save_dir=str(save_dir), name="lightning_logs")
    )
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

    trainer.save_checkpoint(str(save_dir / "model.ckpt"))
    print(f"Succeed saving mdoel parameters in {save_dir}/model.ckpt .")

    # logging = {'metadata': {}, 'train_loss': model.train_losses, 'val_loss': model.val_losses}
    log_metadata = {
        "description": description,
        "num_epoch": epoch, 
        "batch_size": batch_size,
        # "criterion": criterion.__class__.__name__, 
        "learning_rate": learning_rate,
        "train_losses":  model.train_losses,
        "val_losses":  model.val_losses
    }

    with open(save_dir / "log_metadata.json", "w") as f:
        json.dump(log_metadata, f)
    print(f"Succeed saving log in {save_dir}/log_metadata.json")

if __name__ == "__main__":
    save_dir = project_root / "data/checkpoints"
    save_dir = save_dir / "pl_point06"
    save_dir.mkdir(parents=True, exist_ok=False)
    main("Change loss to the mean absolute error", save_dir, epoch=100)
    # save_dir = Path("/home/yangk/intership_2025_COSYS/src/checkpoints")
    # model = LitModel.load_from_checkpoint(str(save_dir / "point05" / "model.ckpt"))
