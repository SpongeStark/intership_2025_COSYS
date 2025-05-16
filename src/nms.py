import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import math


def non_max_suppression_torch(edge_mag: torch.Tensor, edge_dir: torch.Tensor) -> torch.Tensor:
    """
    对边缘图应用 PyTorch 实现的 Non-Maximum Suppression（非极大值抑制）

    参数:
        edge_mag: (1, 1, H, W) 边缘幅值图（Tensor，值范围任意）
        edge_dir: (1, 1, H, W) 边缘方向图（角度，单位为度，范围 [0, 180)）

    返回:
        Tensor: (1, 1, H, W) 经过 NMS 的边缘图
    """
    # device = edge_mag.device
    B, C, H, W = edge_mag.shape
    assert edge_mag.shape == edge_dir.shape, "edge_mag 和 edge_dir 的 shape 应该一致"

    # pad 原图，避免边界问题
    mag = F.pad(edge_mag, (1, 1, 1, 1), mode='replicate')
    dir_deg = edge_dir % 180  # 限制在 [0,180)

    # 定义采样方向
    angle_bin = torch.zeros_like(dir_deg)
    angle_bin[(0 <= dir_deg) & (dir_deg < 22.5)] = 0
    angle_bin[(157.5 <= dir_deg) & (dir_deg < 180)] = 0
    angle_bin[(22.5 <= dir_deg) & (dir_deg < 67.5)] = 1
    angle_bin[(67.5 <= dir_deg) & (dir_deg < 112.5)] = 2
    angle_bin[(112.5 <= dir_deg) & (dir_deg < 157.5)] = 3

    # 构造索引坐标（相对于中心像素）
    offsets = {
        0: [(0, -1), (0, 1)],      # 左右
        1: [(-1, 1), (1, -1)],     # ↘↖
        2: [(-1, 0), (1, 0)],      # 上下
        3: [(-1, -1), (1, 1)],     # ↙↗
    }

    output = torch.zeros_like(edge_mag)

    for direction, (offset1, offset2) in offsets.items():
        mask = (angle_bin == direction)

        # 获取方向上的两个邻居像素
        q = mag[:, :, 1 + offset1[0]:1 + offset1[0] + H, 1 + offset1[1]:1 + offset1[1] + W]
        r = mag[:, :, 1 + offset2[0]:1 + offset2[0] + H, 1 + offset2[1]:1 + offset2[1] + W]

        # 中心点大于两侧 -> 保留
        center = edge_mag
        keep = (center >= q) & (center >= r) & mask
        output = output.where(~keep, center)

    return output

def get_nms_edge(edge):
    edge = np.asarray(edge)
    # Sobel 边缘 + 方向
    gx = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.hypot(gx, gy).astype(np.float32)
    dir = np.degrees(np.arctan2(gy, gx)).astype(np.float32)

    # 转换为 PyTorch Tensor
    edge_mag_t = torch.tensor(mag).unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    edge_dir_t = torch.tensor(dir).unsqueeze(0).unsqueeze(0)

    # 应用 NMS
    return non_max_suppression_torch(edge_mag_t, edge_dir_t).squeeze()


def compute_sobel_edges(img_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用卷积方式计算 Sobel 边缘与方向（支持批量）

    输入:
        img_batch: (N, 1, H, W) 灰度图，值范围任意

    输出:
        mag: (N, 1, H, W) 边缘强度
        dir: (N, 1, H, W) 边缘方向（角度，单位：度）
    """
    sobel_kernel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=img_batch.device).view(1, 1, 3, 3)

    sobel_kernel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=img_batch.device).view(1, 1, 3, 3)

    gx = F.conv2d(img_batch, sobel_kernel_x, padding=1)
    gy = F.conv2d(img_batch, sobel_kernel_y, padding=1)

    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    dir = torch.atan2(gy, gx) * (180.0 / math.pi)

    return mag, dir


def get_nms_edge_batch(images: torch.Tensor) -> torch.Tensor:
    """
    输入 torch.Tensor 图像 (N, H, W) 或 (N, 1, H, W)，返回 NMS 后的边缘图 (N, H, W)

    参数:
        images: torch.Tensor, shape=(N, H, W) 或 (N, 1, H, W)

    返回:
        torch.Tensor: shape=(N, H, W)
    """
    if images.ndim == 3:
        images = images.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    assert images.ndim == 4 and images.shape[1] == 1, "必须为 (N, 1, H, W)"

    mag, direction = compute_sobel_edges(images)
    nms_result = non_max_suppression_torch(mag, direction)

    return nms_result.squeeze(1)  # 返回 (N, H, W)


if __name__=="__main__":
    img = torch.randn(720, 480)
    # 应用 NMS
    nms_edge = get_nms_edge(img)
    print(nms_edge.shape)
    img = torch.randn(4, 720, 480)
    # 应用 NMS
    nms_edge = get_nms_edge_batch(img)
    print(nms_edge.shape)

