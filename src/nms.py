import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np
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


def get_gradient(image):
    dim = image.dim()
    if dim == 3:
        image = image.unsqueeze(0)
    if dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    # Compute gradients with Sobel filters
    sobel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]])
    
    sobel_y = torch.tensor([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]])
    
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # shape [1, 1, 3, 3]
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_y, padding=1)

    if dim == 3:
        return gx.squeeze(0), gy.squeeze(0)
    if dim == 2:
        return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0)
    return gx, gy

def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """创建 2D 高斯核 (kernel_size x kernel_size)"""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_blur(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """使用给定高斯核对图像进行卷积"""
    C = img.shape[1]
    kernel = kernel.expand(C, 1, *kernel.shape)  # 扩展为 (C, 1, k, k)
    img_blur = F.conv2d(img, kernel, padding=kernel.shape[-1] // 2, groups=C)
    return img_blur

def get_gradient_canny(image):
    dim = image.dim()
    if dim == 3:
        image = image.unsqueeze(0)
    if dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)

    kernel = gaussian_kernel(kernel_size=7, sigma=1.5)
    img_blurred = apply_gaussian_blur(image, kernel)
    gx, gy = get_gradient(img_blurred)

    if dim == 3:
        return gx.squeeze(0), gy.squeeze(0)
    if dim == 2:
        return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0)
    return gx, gy


def nms_fully_vectorized(norm, pente_x, pente_y):
    """
    Fully vectorized NMS with directional interpolation for (N, H, W) inputs.
    Returns a tensor of the same shape as norm.
    """

    if norm.dim() == 2:
        norm = norm.unsqueeze(0)
        pente_x = pente_x.unsqueeze(0)
        pente_y = pente_y.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    N, H, W = norm.shape
    device = norm.device

    eps = 1e-10
    gy_safe = pente_y.clone()
    gy_safe[gy_safe == 0] = eps
    wd = pente_x / gy_safe  # (N, H, W)

    a = torch.zeros_like(norm)
    b = torch.zeros_like(norm)

    # -------- Shifted versions of norm (8 directions) --------
    def shift(x, dy, dx):
        # shift x by (dy, dx), pad with 0
        return F.pad(x, (1, 1, 1, 1), mode='constant')[..., 1+dy:H+1+dy, 1+dx:W+1+dx]

    n  = norm
    n_up       = shift(n, -1,  0)
    n_down     = shift(n,  1,  0)
    n_left     = shift(n,  0, -1)
    n_right    = shift(n,  0,  1)
    n_upleft   = shift(n, -1, -1)
    n_upright  = shift(n, -1,  1)
    n_downleft = shift(n,  1, -1)
    n_downright= shift(n,  1,  1)

    # ---------- Build masks ----------
    mask1 = wd >= 1
    mask2 = (wd >= 0) & (wd < 1)
    mask3 = (wd >= -1) & (wd < 0)
    mask4 = wd < -1
    gy_neg = pente_y < 0

    # ---- mask1: wd >= 1 (↘ ↖)，取 (i,j+1)->(i+1,j+1) and (i,j-1)->(i-1,j-1)
    wd1 = torch.zeros_like(wd)
    wd1[mask1] = 1 / wd[mask1]

    a1 = n_right + (n_downright - n_right) * wd1
    b1 = n_left + (n_upleft - n_left) * wd1

    # ---- mask2: 0 <= wd < 1 (→↘ ←↖)
    a2 = n_down + (n_downright - n_down) * wd
    b2 = n_up + (n_upleft - n_up) * wd

    # ---- mask3: -1 <= wd < 0 (→↙ ←↗)
    a3 = n_down + (n_downleft - n_down) * wd
    b3 = n_up + (n_upright - n_up) * wd

    # ---- mask4: wd < -1 (↙ ↗)
    wd4 = torch.zeros_like(wd)
    wd4[mask4] = 1 / wd[mask4]
    a4 = n_left + (n_downleft - n_left) * wd4
    b4 = n_right + (n_upright - n_right) * wd4

    # ---- assemble a, b according to masks ----
    a = torch.where(mask1, a1, a)
    b = torch.where(mask1, b1, b)
    a = torch.where(mask2, a2, a)
    b = torch.where(mask2, b2, b)
    a = torch.where(mask3, a3, a)
    b = torch.where(mask3, b3, b)
    a = torch.where(mask4, a4, a)
    b = torch.where(mask4, b4, b)

    # flip a, b if pente_y < 0
    a_, b_ = a.clone(), b.clone()
    a = torch.where(gy_neg, b_, a)
    b = torch.where(gy_neg, a_, b)

    # If both gradients are 0 → retain original
    zero_grad = (pente_x == 0) & (pente_y == 0)

    # Final suppression decision
    keep = (norm >= a) & (norm > b)
    keep = keep | zero_grad

    out = torch.where(keep, norm, torch.zeros_like(norm))

    # Set borders to 0
    out[:, 0, :] = 0
    out[:, -1, :] = 0
    out[:, :, 0] = 0
    out[:, :, -1] = 0

    return out.squeeze(0) if squeeze_out else out


if __name__=="__main__":
    img = torch.randn(720, 480)
    # 应用 NMS
    nms_edge = get_nms_edge(img)
    print(nms_edge.shape)
    img = torch.randn(4, 720, 480)
    # 应用 NMS
    nms_edge = get_nms_edge_batch(img)
    print(nms_edge.shape)

