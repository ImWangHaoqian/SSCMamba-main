import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='/home/wanghaoqian/桌面/RGB2HS-main/train_code/exp/hybrid/net_680epoch.pth')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()

# Validate - 根据 data_root 自动判断数据集类型并获取文件列表
data_root_normalized = opt.data_root.rstrip('/')
if 'NTIRE2022' in data_root_normalized:
    # NTIRE2022 格式
    with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
        hyper_list = [line.replace('\n', '.mat') for line in fin]
    hyper_list.sort()
elif 'NTIRE2020' in data_root_normalized:
    # NTIRE2020 格式 - 从数据集获取文件名
    # 如果路径中包含 _realworld 或 _clean，需要转换为正确的数据集根目录
    if '_realworld' in data_root_normalized.lower() or '_clean' in data_root_normalized.lower():
        # 将路径转换为正确的数据集根目录
        # 例如: ../dataset/NTIRE2020_realworld/ -> ../dataset/NTIRE2020/
        dataset_root = data_root_normalized
        if '_realworld' in dataset_root.lower():
            dataset_root = dataset_root.replace('_realworld', '').replace('_RealWorld', '')
        elif '_clean' in dataset_root.lower():
            dataset_root = dataset_root.replace('_clean', '').replace('_Clean', '')
        dataset_root = dataset_root.rstrip('/') + '/'
    else:
        dataset_root = opt.data_root.rstrip('/') + '/'
    
    # 首先检查路径中是否包含关键字
    if 'RealWorld' in data_root_normalized or 'realworld' in data_root_normalized.lower():
        # 明确指定为 RealWorld
        bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
        rgb_suffix = '_RealWorld.jpg'
    elif 'Clean' in data_root_normalized or 'clean' in data_root_normalized.lower():
        # 明确指定为 Clean
        bgr_data_path = f'{dataset_root}NTIRE2020_Validation_Clean/'
        rgb_suffix = '_clean.png'
    else:
        # 路径中没有明确指定，检查目录存在性
        realworld_exists = os.path.exists(f'{dataset_root}NTIRE2020_Validation_RealWorld')
        clean_exists = os.path.exists(f'{dataset_root}NTIRE2020_Validation_Clean')
        
        if realworld_exists and clean_exists:
            # 两个都存在，默认使用 RealWorld，但给出警告
            print(f'Warning: Both Clean and RealWorld directories exist. Using RealWorld by default.')
            bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
            rgb_suffix = '_RealWorld.jpg'
        elif realworld_exists:
            # 只有 RealWorld
            bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
            rgb_suffix = '_RealWorld.jpg'
        elif clean_exists:
            # 只有 Clean
            bgr_data_path = f'{dataset_root}NTIRE2020_Validation_Clean/'
            rgb_suffix = '_clean.png'
        else:
            raise ValueError(f'Neither NTIRE2020_Validation_Clean nor NTIRE2020_Validation_RealWorld directory found in {dataset_root}')
    
    bgr_files = [f for f in os.listdir(bgr_data_path) if f.endswith(rgb_suffix)]
    bgr_files.sort()
    hyper_list = []
    for bgr_file in bgr_files:
        if rgb_suffix == '_RealWorld.jpg':
            file_id = bgr_file.replace('_RealWorld.jpg', '')
        else:
            file_id = bgr_file.replace('_clean.png', '')
        hyper_list.append(file_id + '.mat')
else:
    raise ValueError(f'Unknown dataset type. data_root should contain NTIRE2022 or NTIRE2020, got: {opt.data_root}')

var_name = 'cube'
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            if method=='awan':   # To avoid out of memory, we crop the center region as input for AWAN.
                output = model(input[:, :, 118:-118, 118:-118])
                loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            else:
                output = model(input)
                loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr = validate(val_loader, model)
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}')