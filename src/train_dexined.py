import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import json
import numpy as np
import sys

sys.path.append("/home/yangk/intership_2025_COSYS/resource/DexiNed")
from model import DexiNed

from dataset import BIPEDv2, transforms
from nms import get_nms_edge_batch


if __name__=="__main__":
    # preapre data | 准备数据
    # train set | 训练集
    biped_dataset = BIPEDv2(
        "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/",
        "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real/"
    )
    train_loader = DataLoader(biped_dataset, batch_size=4)
    # test set | 测试集
    test_dataset = BIPEDv2(
        "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/imgs/test/rgbr/",
        "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/edge_maps/test/rgbr/"
    )
    val_loader = DataLoader(test_dataset, batch_size=4)
    # set hyper-parameters | 设置超参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    epoch = 100
    batch_size = 8
    learning_rate = 1e-4
    model = DexiNed() # load model 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logging = {'metadata': {}, 'train_loss': [], 'val_loss': []}
    file_stem = "cpt_visibility_04"
    logging['metadata'] = {
        "description": "add non-max suppression", 
        "num_epoch":epoch, 
        "batch_size":batch_size, 
        "criterion": criterion.__class__.__name__, 
        "learning_rate": learning_rate
    }
    # train or load | 训练
    if not os.path.isfile(f"./checkpoints/{file_stem}.json"): # First train
        model = model.to(device)
        for e in range(epoch):
            # train step
            model.train()
            epoche_loss = []
            for batch in train_loader:
                x, y = batch['image_tensor'].to(device), batch['visibility_map'].to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = sum([criterion(get_nms_edge_batch(output), y) for output in outputs]) / len(outputs)
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
                    x, y = batch['image_tensor'].to(device), batch['visibility_map'].to(device)
                    outputs = model(x)
                    loss = sum([criterion(get_nms_edge_batch(output), y) for output in outputs]) / len(outputs)
                    val_epoch_loss.append(loss.detach().item())
            logging['val_loss'].append(np.mean(val_epoch_loss))
            print(f"In epoch {e}, the average validation loss is {logging['val_loss'][-1]}")
        model = model.to('cpu')
        # save files
        # file_stem = "cpt_visibility_01"
        torch.save(model.state_dict(), f"./checkpoints/{file_stem}.pth")
        print(f"Succeed saving mdoel parameters in ./checkpoints/{file_stem}.pth.")
        with open(f"./checkpoints/{file_stem}.json", "w") as f:
            json.dump(logging, f)
        print(f"Succeed saving log in ./checkpoints/{file_stem}.json")
    else: # already trained
        # load model
        model.load_state_dict(torch.load(f"./checkpoints/{file_stem}.pth", weights_only=True))
        # load log
        with open(f"./checkpoints/{file_stem}.json", 'r') as f:
            logging = json.load(f)
        # print the loss
        for e, (train_loss, val_loss) in enumerate(zip(logging['train_loss'], logging['val_loss'])):
            print("-".join(["-"]*30))
            print(f"\nIn epoch {e}, the average  training  loss is {train_loss}")
            print(f"In epoch {e}, the average validation loss is {val_loss}")


