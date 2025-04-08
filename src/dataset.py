import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

import sys
import os
# 计算 `project_root/` 目录的路径
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ../
# 添加lucas的项目
sys.path.append(os.path.join(project_root_path, "resource/ContrastVisibilityProject_Lucas"))

from ImageProcessing.ImageAnalyzer import ImageAnalyzer
from ImageProcessing.ImageGenerator import Image as IamgeGenerator

from ImageProcessing.ConvolutionFilter import Filter
from Parameters import WEIGHT_LIST, SIGMA_LIST

ANALYSER = ImageAnalyzer(
    Filter(distance_from_screen=50, sigma_list=SIGMA_LIST, weight_list=WEIGHT_LIST) # sDoG filter
)


class BIPEDv2(Dataset):
    def __init__(self, ori_path, gt_path, analyer=ANALYSER, return_map=True):
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
        self.return_map = return_map

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i, normalize=True):
        # read x
        x = np.asarray(Image.open(self.ori_path.joinpath(self.indexes[i]).with_suffix(".jpg")).convert('RGB')) / 255
        x = self.transform(x).float() if normalize else torch.from_numpy(x).float()
        y = np.asarray(Image.open(self.gt_path.joinpath(self.indexes[i]).with_suffix(".png"))) / 255
        if self.return_map:
            img = IamgeGenerator()
            img.load_image(self.ori_path.joinpath(self.indexes[i]).with_suffix(".jpg"))
            img.convert_into_linear_space()
            self.analyzer.generate_visibility_map(img, edge_map=np.where( y > 250/255, 1, 0))
            y = self.analyzer.visibility_map
        return x, torch.from_numpy(y).float()
        

if __name__=="__main__":
    image_path = "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/imgs/train/rgbr/real/"
    edge_path = "/home/yangk/intership_2025_COSYS/resource/DexiNed/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real/"
    ds = BIPEDv2(image_path, edge_path, return_map=False)
    print(ds[0][0].shape)
    print(ds[0][1].shape)
    print(ds[0][1].max())
    print(ds[0][0].dtype)
    print(ds[0][1].dtype)