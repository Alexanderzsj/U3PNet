import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import os
import logging

from models.U3PNet import U3PNet
from models.HSRNet import HSRNet
from models.PSRT import PSRT
# from models.SGANet import SGANet
from models.Fusformer import Fusformer
from models.UnetModel import UnetModel
from models.MUCNN import MUCNN
import logging
from data import NPZDataset
import numpy as np
from utils import *
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parse = argparse.ArgumentParser()
parse.add_argument('--log_out', type=int, default=1)
parse.add_argument('--model_name', type=str, default='HSRNet')
parse.add_argument('--dataset', type=str, default='Houston')
parse.add_argument('--check_point', type=str, default=None)
parse.add_argument('--check_step', type=int, default=50)
parse.add_argument('--lr', type=int, default=4e-4)
parse.add_argument('--batch_size', type=int, default=8)
parse.add_argument('--epochs', type=int, default=1000)
parse.add_argument('--seed', type=int, default=3407)
args = parse.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model_name
model = None
HSI_bands, MSI_bands = 31, 3
test_dataset_path = None
train_dataset_path = None

if args.dataset == 'CAVE':
    test_dataset_path = '../datasets/form_datax4/CAVE_test.npz'
    train_dataset_path = '../datasets/form_datax4/CAVE_train.npz'
    HSI_bands = 31
    MSI_bands = 3
if args.dataset == "PaviaU":
    test_dataset_path = '../datasets/form_datax4/PaviaU_test.npz'
    train_dataset_path = '../datasets/form_datax4/PaviaU_train.npz'
    HSI_bands = 103
    MSI_bands = 3
if args.dataset == "Havard":
    test_dataset_path = '../datasets/Havard_test.npz'
    train_dataset_path = '../datasets/Havard_test.npz'
    HSI_bands = 31
if args.dataset == "Urban":
    test_dataset_path = '../datasets/Urban_test.npz'
    train_dataset_path = '../datasets/Urban_test.npz'
    HSI_bands = 162
if args.dataset == "Chikusei":
    test_dataset_path = '../datasets/form_datax4/chikusei_test.npz'
    train_dataset_path = '../datasets/form_datax4/chikusei_train.npz'
    HSI_bands = 128
    MSI_bands = 4

if args.dataset == "Houston":
    test_dataset_path = '../datasets/form_datax4/Houston_test.npz'
    train_dataset_path = '../datasets/form_datax4/Houston_train.npz'
    HSI_bands = 144
    MSI_bands = 4


if model_name.startswith('SGANet'):
    model = SGANet(HSI_bands, MSI_bands)
if model_name.startswith('SSRNet'):
    model = SSRNet(HSI_bands, )
if model_name.startswith('HSRNet'):
    model = HSRNet(HSI_bands, MSI_bands)
if model_name.startswith('MHFNet'):
    model = MHFNet(HSI_bands)
if model_name.startswith('CSSNet'):
    model = CSSNet(HSI_bands)
if model_name.startswith('ResTFNet'):
    model = ResTFNet(HSI_bands)
# if model_name.startswith('DCTFormer'):
#     model = DCTFormer(HSI_bands)
if model_name.startswith('SKNet'):
    model = SKNet(HSI_bands)
if model_name.startswith('Fusformer'):
    model = Fusformer(HSI_bands, MSI_bands)
if model_name.startswith('TestFormer'):
    model = TestFormer(HSI_bands)
if model_name.startswith('WDT'):
    model = WDT(HSI_bands)
if model_name.startswith('VisionTransformer'):
    model = VisionTransformer()
if model_name.startswith('UnetModel'):
    model = UnetModel(HSI_bands, MSI_bands)
if model_name.startswith('U2Net'):
    model = U2Net(HSI_bands, MSI_bands)
if model_name.startswith('PSRT'):
    model = PSRT(HSI_bands, MSI_bands)
if model_name.startswith('MUCNN'):
    model = MUCNN(HSI_bands, MSI_bands)
if model_name.startswith('U3PNet'):
    model = U3PNet(HSI_bands, MSI_bands)


model = model.to(device)
set_seed(args.seed)
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters())
scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
test_dataset = NPZDataset(test_dataset_path)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 4)
start_epoch = 0

model_size = get_model_size(model)
inference_time, flops, params = test_speed(model, device, HSI_bands, MSI_bands)


if args.check_point is not None:
    checkpoint = torch.load(args.check_point)
    model.load_state_dict(checkpoint['net'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    scheduler.load_state_dict(checkpoint['scheduler'])
    log_dir, _ = os.path.split(args.check_point)
    print(f'check_point: {args.check_point}')

if args.check_point is None:
    init_weights(model)
    log_dir = f'./trained_models/{args.dataset}/{model_name},{args.dataset},{beijing_time()}'
    if not os.path.exists(log_dir) and args.log_out == 1:
        os.makedirs(log_dir)

logger = set_logger(model_name, log_dir, args.log_out)
logger.info(
    f'[model:{args.model_name},dataset:{args.dataset}],model_size:{params:.6f} M,inference_time:{inference_time:.6f}S,FLOPs:{flops:.6f} G')


def train():
    model.train()
    loss_list = []
    for epoch in range(start_epoch, args.epochs):
        for idx, loader_data in enumerate(train_dataloader):
            GT, LRHSI, RGB = loader_data[0].to(device), loader_data[1].to(device), loader_data[2].to(device)
            optimizer.zero_grad()
            preHSI = model(LRHSI, RGB)
            loss = loss_func(GT, preHSI)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        test(epoch=epoch)


@torch.no_grad()
def test(epoch=-1):
    model.eval()
    loss_val_list, pre_hsi_list, gt_list = [], [], []
    for _,loader_data in enumerate(test_dataloader):
        gt,lrhsi,msi = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
        pre_hsi = model(lrhsi,msi)
        loss_val = loss_func(gt,pre_hsi)
        loss_val_list.append(loss_val.item())
        pre_hsi_list.append(pre_hsi)
        gt_list.append(gt)
    
    pre_hsi_list = torch.cat(pre_hsi_list,dim=0)
    gt_list = torch.cat(gt_list,dim=0)
    psnr = Metrics.cal_psnr(pre_hsi_list, gt_list)
    ssim = Metrics.cal_ssim(pre_hsi_list, gt_list)
    ergas = Metrics.cal_ergas(pre_hsi_list, gt_list)
    sam = Metrics.cal_sam(pre_hsi_list, gt_list)
    if args.log_out == 1 and (epoch + 1) % args.check_step == 0: 
        checkpoint= {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler':scheduler.state_dict()
            }
        
        torch.save(checkpoint,f'{log_dir}/epoch:{epoch},PSNR:{psnr:.4f},SSIM:{ssim:.4f}.pth')
    logger.info(f"{beijing_time()}, {args.model_name}, {args.dataset},epoch:{epoch}, loss:{np.mean(loss_val_list):.4f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, SAM:{sam:.4f}, ERGAS:{ergas:.4f}")  


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
