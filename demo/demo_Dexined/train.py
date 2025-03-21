import argparse
import json
from model import DexiNed

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from PIL import Image
from pathlib import Path
import numpy as np

class BIPEDv2(Dataset):
    def __init__(self, ori_path, gt_path):
        self.ori_path = Path(ori_path)
        self.gt_path = Path(gt_path)
        self.indexes = list(set([x.stem for x in self.ori_path.iterdir() if not x.name.startswith(".")]) & set([x.stem for x in self.gt_path.iterdir() if not x.name.startswith(".")]))
    
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        x = np.array(Image.open(self.ori_path.joinpath(self.indexes[i]).with_suffix(".jpg")))
        denom = 255 if x.max() > 1 else 1
        y = np.array(Image.open(self.gt_path.joinpath(self.indexes[i]).with_suffix(".png")))
        
        return torch.Tensor(x / denom).permute(2, 0, 1), torch.Tensor(y / denom)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train')
    parser.add_argument("dir_image", help="The directory of the images", type=Path)
    parser.add_argument("dir_edge", help="The directory of the edge maps", type=Path)
    parser.add_argument("-o", "--output_path", help="The path of checkpoint file", default="./checkpoint.pth", type=Path)
    parser.add_argument("-e", "--epoch", help="The number of epoch", default=1, type=int)
    parser.add_argument("-b", "--batch_size", help="The number of batch size", default=4, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate for the optimiser", default=1e-3, type=float)
    parser.add_argument("-d", "--device", default='cpu')
    parser.add_argument("-log", "--logging", help="Save the logging", action="store_true")
    args = parser.parse_args()

    device = args.device
    epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    biped_dataset = BIPEDv2(args.dir_image, args.dir_edge)

    model = DexiNed()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(biped_dataset, batch_size=4)
    model = model.to(device)
    logging = {
        'train_loss': []
    }

    for e in range(epoch):
        epoche_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)[-1].squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print("*", end="", flush=True)
            epoche_loss.append(loss.detach().item())
        
        logging['train_loss'].append(np.mean(epoche_loss))
        print(f"\nIn epoch {e}, the average loss is {logging['train_loss'][-1]}")

    # save files
    model = model.to('cpu')
    torch.save(model.state_dict(), args.output_path)
    print(f"Succeed saving mdoel parameters in {args.output_path}.")
    if args.logging:
        with open(args.output_path.with_name(args.output_path.stem + '_log.json'), "w") as f:
            json.dump(logging, f)
        print(f"Succeed saving log in {args.output_path.with_name(args.output_path.stem + '_log.json')}")

# python DexiNed//train.py DexiNed/BIPEDv2/BIPED/edges/imgs/train/rgbr/real DexiNed/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real -o DexiNed/checkpoints/cpt_e30_lr4.pth -e 30 -log -d cuda -lr 1e-4 
