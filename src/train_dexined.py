import os
from pathlib import Path
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import json
import numpy as np
import sys

import sys
model_path = str(PROJECT_ROOT / "resource/DexiNed")
if str(model_path) not in sys.path:
    sys.path.append(str(model_path))
from model import DexiNed

from dataset import BIPEDv2, transforms
from tools import weighted_mean_for_last_layer
from nms import get_gradient_canny, nms_fully_vectorized, get_gradient
from losses_dexined import * 
from early_stopping import EarlyStopping

from dataclasses import dataclass

@dataclass
class TrainingArgs:
    output_dir: Path
    criterion: WeightedMSELoss
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    device: str = "cpu"

def training(description, args:TrainingArgs, use_nms=False):
    output_dir = args.output_dir 
    early_stopping = EarlyStopping(save_dir=output_dir)
    # preapre data | 准备数据
    batch_size = args.batch_size
    # train set | 训练集
    biped_dataset = BIPEDv2(
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/",
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real/"
    )
    train_loader = DataLoader(biped_dataset, batch_size=batch_size)
    # test set | 测试集
    test_dataset = BIPEDv2(
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/imgs/test/rgbr/",
        PROJECT_ROOT / "data/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
    )
    val_loader = DataLoader(test_dataset, batch_size=batch_size)
    # set hyper-parameters | 设置超参数
    device = args.device
    epoch = args.epochs
    learning_rate = args.learning_rate
    model = DexiNed() # load model 
    criterion = args.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logging = {'metadata': {}, 'train_loss': [], 'val_loss': []}
    # file_stem = "cpt_visibility_04"
    logging['metadata'] = {
        "description": description, 
        "num_epoch":epoch, 
        "criterion": criterion.__class__.__name__, 
        "batch_size":batch_size, 
        "learning_rate": learning_rate
    }
    # prefix = PROJECT_ROOT / "data/checkpoints/torch_point05"
    model_path = output_dir / "model.pth"
    log_path = output_dir / "log.json"
    # train or load | 训练
    if not log_path.is_file(): # First train
        output_dir.mkdir(parents=True, exist_ok=True)
        model = model.to(device)
        for e in range(epoch):
            # train step
            model.train()
            epoche_loss = []
            for batch in train_loader:
                # x, y = batch['image_tensor'].to(device), batch['edge_tensor'].to(device)
                x, y = batch['image_tensor'].to(device), batch['visibility_map'].to(device)
                optimizer.zero_grad()
                outputs = model(x)
                # # NMS
                if use_nms:
                    gx, gy = get_gradient(x.mean(dim=1))
                    outputs[-1] = nms_fully_vectorized(outputs[-1].squeeze(), gx, gy).unsqueeze(1)
                loss = criterion(outputs, y) # sum([criterion(output, y) for output in outputs]) / len(outputs)
                loss.backward()
                optimizer.step()
                print("*", end="", flush=True)
                epoche_loss.append(loss.detach().item())
            logging['train_loss'].append(np.mean(epoche_loss))
            print(f"\nIn epoch {e}, the average  training  loss is {logging['train_loss'][-1]}")
            # validation step
            model.eval()
            val_epoch_loss = []
            with torch.no_grad():
                for batch in val_loader:
                    # x, y = batch['image_tensor'].to(device), batch['edge_tensor'].to(device)
                    x, y = batch['image_tensor'].to(device), batch['visibility_map'].to(device)
                    outputs = model(x)
                    # # NMS
                    if use_nms:
                        gx, gy = get_gradient(x.mean(dim=1))
                        outputs[-1] = nms_fully_vectorized(outputs[-1].squeeze(), gx, gy).unsqueeze(1)
                    loss = criterion(outputs, y) # sum([criterion(output, y) for output in outputs]) / len(outputs)
                    val_epoch_loss.append(loss.detach().item())
            logging['val_loss'].append(np.mean(val_epoch_loss))
            print(f"In epoch {e}, the average validation loss is {logging['val_loss'][-1]}")
            # early stopping | 防止 overfitting
            early_stopping(logging['val_loss'][-1], model.cpu())
            if early_stopping.early_stop:
                print("Stop training, early stop.")
                break
            model = model.to(device)
        model = model.to('cpu')
        # save files
        torch.save(model.state_dict(), str(model_path))
        print(f"Succeed saving mdoel parameters in {model_path}.")
        with open(str(log_path), "w") as f:
            json.dump(logging, f)
        print(f"Succeed saving log in {log_path}.")
    # else: # already trained
    #     # load model
    #     model.load_state_dict(torch.load(f"./checkpoints/{file_stem}.pth", weights_only=True))
    #     # load log
    #     with open(f"./checkpoints/{file_stem}.json", 'r') as f:
    #         logging = json.load(f)
    #     # print the loss
    #     for e, (train_loss, val_loss) in enumerate(zip(logging['train_loss'], logging['val_loss'])):
    #         print("-".join(["-"]*30))
    #         print(f"\nIn epoch {e}, the average  training  loss is {train_loss}")
    #         print(f"In epoch {e}, the average validation loss is {val_loss}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    args = TrainingArgs(
        epochs=100,
        batch_size=8,
        learning_rate=1e-5,
        device=device,
        output_dir=PROJECT_ROOT/"data/checkpoints/torch_point08_bis",
        criterion=WeightedMSELoss()
    )
    training("train with NMS, Sobel for Gx and Gy", args)

def multi_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    args = TrainingArgs(
        epochs=100,
        batch_size=8,
        learning_rate=1e-5,
        device=device,
        output_dir=PROJECT_ROOT/"data/checkpoints/torch_point00",
        criterion=WeightedMSELoss()
    )
    # train with 0.99 MAE + 0.01 MSE
    args.output_dir = PROJECT_ROOT/"data/checkpoints/torch_point14"
    args.criterion = WeightedLinearCombineLoss(alpha=0.99)
    training("train with a linear combination loss of MSE and MAE, 0.99 MAE + 0.01 MSE", args, use_nms=False)
    # train with 0.9 MAE + 0.1 MSE
    args.output_dir = PROJECT_ROOT/"data/checkpoints/torch_point15"
    args.criterion = WeightedLinearCombineLoss(alpha=0.9)
    training("try with 0.9 MAE + 0.1 MSE", args)

if __name__=="__main__":
    multi_train()

