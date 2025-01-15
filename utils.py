import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init
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
    

