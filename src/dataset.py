import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

import sys

sys.path.append("../resource/ContrastVisibilityProject_Lucas")

from ImageProcessing.ImageAnalyzer import ImageAnalyzer

from ImageProcessing.ConvolutionFilter import Filter
from Parameters import WEIGHT_LIST, SIGMA_LIST

ANALYSER = ImageAnalyzer(
    Filter(distance_from_screen=50, sigma_list=SIGMA_LIST, weight_list=WEIGHT_LIST) # sDoG filter
)


class BIPEDv2(Dataset):
    def __init__(self, ori_path, gt_path, analyer=ANALYSER):
        self.ori_path = Path(ori_path)
        self.gt_path = Path(gt_path)
        self.indexes = list(set([x.stem for x in self.ori_path.iterdir() if not x.name.startswith(".")]) & set([x.stem for x in self.gt_path.iterdir() if not x.name.startswith(".")]))
        self.indexes = np.sort(self.indexes)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) # statistic from ImageNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.analyzer = analyer

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        x = Image.open(self.ori_path.joinpath(self.indexes[i]).with_suffix(".jpg")).convert('RGB')
        y = Image.open(self.gt_path.joinpath(self.indexes[i]).with_suffix(".png"))# .convert('RGB')

        # self.analyzer.generate_visibility_map(x, edge_map=np.where(y > 250, 1, 0))

        # x, y = self.transform(x), np.array(y) / 255
        
        return x, y

if __name__=="__main__":
    image_path = "/Users/yk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/imgs/train/rgbr/real"
    edge_path = "/Users/yk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real"
    ds = BIPEDv2(image_path, edge_path)
    print(np.asarray(ds[0][0]).shape)