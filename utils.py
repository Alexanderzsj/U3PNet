import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import torch
import logging
import os
import shutil
import time
from skimage.metrics import structural_similarity
from thop import profile
from thop import clever_format
import shutil
import copy
from PIL import Image
from SpectralUtils import *
import torchvision
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure, mean_squared_error


class Metrics:
    """量化指标计算对象
    """
    def __init__(self, logger):
        pass
    
    @staticmethod
    @torch.no_grad()
    def cal_psnr(pre_hsi, gt):
        return peak_signal_noise_ratio(pre_hsi, gt, gt.max())
        
    @staticmethod
    @torch.no_grad()
    def cal_ssim(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        return structural_similarity_index_measure(pre_hsi, gt)
    
    @staticmethod
    @torch.no_grad()
    def cal_sam(pre_hsi, gt):
        assert gt.shape == pre_hsi.shape
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        gt = gt.permute(0, 2, 3, 1)
        pre_hsi = pre_hsi.permute(0, 2, 3, 1)
        dot_product = torch.sum(gt * pre_hsi, dim=-1)
        norm_reference = torch.norm(gt, dim=-1)
        norm_target = torch.norm(pre_hsi, dim=-1)
        cos_theta = dot_product / (norm_reference * norm_target + 1e-10)
        sam_map = torch.acos(cos_theta)
        sam = torch.mean(sam_map)*180/torch.pi
        return sam


    @staticmethod
    @torch.no_grad()
    def cal_ergas(pre_hsi, gt, up_scale=4):
        assert gt.shape == pre_hsi.shape
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        rmse_bands = torch.sqrt(torch.mean((gt - pre_hsi) ** 2, dim=(2, 3)))  # (B, C)
        mean_bands = torch.mean(gt, dim=(2, 3))  # (B, C)
        ergas = 100 / up_scale * torch.sqrt(torch.mean((rmse_bands / mean_bands) ** 2, dim=1))  # (B,)
        return ergas.mean().item()
    
    @staticmethod
    @torch.no_grad()
    def cal_mse(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        return mean_squared_error(pre_hsi, gt)


def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size


def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def beijing_time():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d,%H:%M:%S'
    now_fmt = beijing_now.strftime(fmt)
    return now_fmt


def set_seed(seed=9999):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(model_name, logger_dir, log_out):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    if log_out == 1:
        ## make out.log
        log_file = f"{logger_dir}/out.log"
        if not os.path.exists(log_file):
            os.mknod(log_file)
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        ## copy model file 
        model_file_path = f'./models/{model_name}.py'
        shutil.copy(model_file_path, f"{logger_dir}/{model_name}.py")

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger


@torch.no_grad()
def test_speed(model, device, HSI_bands, MSI_bands):
    model.eval()
    HSI = torch.randn((1, HSI_bands, 16, 16)).to(device)
    RGB = torch.randn((1, MSI_bands, 64, 64)).to(device)
    flops, params = profile(model, inputs=(HSI, RGB))
    start_time = time.time()
    with torch.no_grad():
        model(HSI, RGB)
    inference_time = time.time() - start_time
    return inference_time, flops / 1000000000., params / 1000000.0


def get_cave_hsi_image(image_path):
    hsi = []
    for dir in sorted(os.listdir(image_path)):
        if dir.endswith('.png'):
            image_dir = os.path.join(image_path,dir)
            image_data = Image.open(image_dir)
            image_data = np.array(image_data)
            hsi.append(image_data)
    hsi = np.stack(hsi,axis=0) / 65535
    return hsi


def plot_tensor_image(image_data):
    """plot 格式"""
    if len(image_data.shape)==2:
        image_data = image_data.unsqueeze(0)
    return image_data.permute(1,2,0).detach().cpu().numpy()

def down_sample(hsi,scale=1/4):
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi)
    down_hsi = F.interpolate(down_hsi,scale_factor=scale)
    down_hsi = down_hsi.squeeze(0)
    return down_hsi


def get_clean_RGB(hsi):
    range = 65535
    filtersPath = "./example_D40_camera_w_gain.npz"
    filters = np.load(filtersPath)['filters']
    rgbIm = np.true_divide(projectToRGB(hsi.transpose(1,2,0), filters), range)
    return rgbIm.transpose(2,0,1)


def get_RGB(hsi):
    hsi = np.array(hsi)
    rgb = get_clean_RGB(hsi)
    return rgb


"""def restrurct_hsi(model, hsi, lr_hsi, msi, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
 
 '''   滑动重建整张高光谱图像
    Args:
        model (torch.tensor): 超分的模型
        hsi (torch.tensor): 参考的HSI图像
        lr_hsi (torch.tensor): 低分辨率的HSI图像
        msi (torch.tensor): 高分辨率的MSI图像
        device (torch.tensor, optional): _description_. Defaults to torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').

    Returns:
        torch.tensor: 预测的整个HSI图像'''

    
    c, h, w = msi.shape
    # print(c, h, w)
    scale = 4
    p = 64
    stride = 64
    hsi = hsi.to(device)
    msi = msi.to(device)
    lr_hsi = lr_hsi.to(device)
    pre_hsi = torch.zeros_like(hsi).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        i, j = 0, 0
        while True:
            if i + p > h:
                i = h - p
            ###############################################
            while True:
                if j + p > w:
                    j = w - p
                msi_patch = msi[:, i:i + p, j:j + p]
                lr_hsi_patch = lr_hsi[:, i // scale:i // scale + p // scale, j // scale:j // scale + p // scale]
                pre_hsi[:, i:i + p, j:j + p] = model(lr_hsi_patch.unsqueeze(0), msi_patch.unsqueeze(0)).squeeze(0)
                if j + p == w:
                    j = 0
                    is_last_row = True
                    break
                else:
                    j += stride
            ###############################################
            if i + p == h and is_last_row:
                break
            i += stride
            is_last_row = False

    return pre_hsi"""

def restrurct_hsi(model, hsi, lr_hsi, msi=None, upscale=4, patch_size=64, stride=64,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    滑动重建整张高光谱图像
    Args:
        model (torch.tensor): 超分的模型
        hsi (torch.tensor): 参考的HSI图像
        lr_hsi (torch.tensor): 低分辨率的HSI图像
        msi (torch.tensor): 高分辨率的MSI图像
        device (torch.tensor, optional): _description_. Defaults to torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').

    Returns:
        torch.tensor: 预测的整个HSI图像
    """
    c,h,w = lr_hsi.shape
    h,w = h*upscale,w*upscale
    scale = upscale
    
    p = patch_size
    stride = stride
    hsi = hsi.to(device)
    
    if msi is not None:
        msi = msi.to(device)
    lr_hsi = lr_hsi.to(device)
    
    pre_hsi = torch.zeros_like(hsi).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        i,j = 0,0
        while True:
            if i+p > h:
                i = h-p
            while True:
                if j+p > w:
                    j = w-p
                if msi == None:
                    lr_hsi_patch = lr_hsi[:,i//scale:i//scale+p//scale,j//scale:j//scale+p//scale]
                    pre_hsi[:,i:i+p,j:j+p] = model(lr_hsi_patch.unsqueeze(0)).squeeze(0)
                else:
                    lr_hsi_patch = lr_hsi[:,i//scale:i//scale+p//scale,j//scale:j//scale+p//scale]
                    msi_patch = msi[:,i:i+p,j:j+p]
                    pre_hsi[:,i:i+p,j:j+p] = model(lr_hsi_patch.unsqueeze(0),msi_patch.unsqueeze(0)).squeeze(0)

                if j+p == w:
                    j = 0
                    is_last_row = True
                    break
                else:
                    j += stride
            if i + p == h and is_last_row:
                break
            i += stride
            is_last_row = False
    return pre_hsi


def crop_patchs(image,strid,patch_size):
    """切割图像为patchs

    Args:
        image (np.array): 输入图像 (h,w,c)
        strid (int): 步长
        patch_size (int): patch大小

    Returns:
        list[np.array]: patchs (nums,patch_size,patch_size,c)
    """
    patchs = []
    c,h,w = image.shape
    for i in range(0,h-patch_size+1,strid):
        for j in range(0,w-patch_size+1,strid):
            patch = image[:,i:i+patch_size,j:j+patch_size]
            patchs.append(patch)
    return patchs