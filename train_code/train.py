import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import sys
import time  # 添加time模块
import csv
import matplotlib
# 尝试使用交互式后端，如果失败则使用Agg
try:
    import tkinter
    matplotlib.use('TkAgg')  # 交互式后端，可以显示图表
except:
    matplotlib.use('Agg')  # 非交互式后端，只能保存文件
import matplotlib.pyplot as plt
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, my_summary
import datetime
import torch.nn.functional as F  # [新增] 用于插值操作

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/', help='Root directory of datasets (default: ../dataset/)')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--debug", action='store_true', help="debug mode with limited samples")
parser.add_argument("--max_samples", type=int, default=None, help="maximum number of samples to load")
parser.add_argument('--dataset', type=str, default='ntire2022', 
                    choices=['ntire2022', 'ntire2020_clean', 'ntire2020_realworld'],
                    help='Dataset type: ntire2022 (default), ntire2020_clean, or ntire2020_realworld')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
# 根据dataset类型自动设置路径（简化：直接根据dataset参数设置路径）
if opt.dataset == 'ntire2022':
    dataset_root = opt.data_root.rstrip('/') + '/NTIRE2022/'
elif opt.dataset == 'ntire2020_clean':
    dataset_root = opt.data_root.rstrip('/') + '/NTIRE2020/'
elif opt.dataset == 'ntire2020_realworld':
    dataset_root = opt.data_root.rstrip('/') + '/NTIRE2020/'
else:
    raise ValueError(f'Unknown dataset type: {opt.dataset}')

print(f'Using dataset: {opt.dataset}')
print(f'Dataset root: {dataset_root}')

train_data = TrainDataset(data_root=dataset_root, crop_size=opt.patch_size, 
                         bgr2rgb=True, arg=True, stride=opt.stride, 
                         max_samples=opt.max_samples if not opt.debug else 10,
                         dataset_type=opt.dataset)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=dataset_root, bgr2rgb=True, dataset_type=opt.dataset)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))
my_summary(model, 256, 256, 3, 1)

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
if opt.outf == './exp/mst_plus_plus/':
    opt.outf = f'./exp/{opt.method}/'
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)
# [新增] 记录训练命令到日志文件
training_command = ' '.join(sys.argv)
logger.info("=" * 80)
logger.info("Training Command:")
logger.info(training_command)
logger.info("=" * 80)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000
    start_time = time.time()  # 记录训练开始时间
    
    # 创建CSV文件记录loss
    loss_csv_path = os.path.join(opt.outf, 'loss.csv')
    loss_csv = open(loss_csv_path, 'w', newline='')
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow(['iteration', 'epoch', 'train_loss', 'val_mrae', 'val_rmse', 'val_psnr', 'lr'])
    loss_csv.flush()
    
    try:
        while iteration<total_iteration:
            model.train()
            losses = AverageMeter()
            train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=False, drop_last=True)
            val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()
                images = Variable(images)
                labels = Variable(labels)
                lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()
                output = model(images)


                # [修改部分开始] ---------------------------------------------------------
                # 方法2：检测是否为多尺度输出 (Tuple)，计算多尺度 Loss
                if isinstance(output, (tuple, list)):
                    # 此时 output 包含 (out_1x, out_0.5x, out_0.25x)
                    out1, out2, out3 = output
                    
                    # 对 Label 进行下采样以匹配输出尺寸
                    # 你的网络中使用 align_corners=True, 这里最好保持一致
                    labels_2 = F.interpolate(labels, scale_factor=0.5, mode='bilinear', align_corners=True)
                    labels_3 = F.interpolate(labels, scale_factor=0.25, mode='bilinear', align_corners=True)
                    
                    # 计算各尺度的 Loss
                    loss1 = criterion_mrae(out1, labels)
                    loss2 = criterion_mrae(out2, labels_2)
                    loss3 = criterion_mrae(out3, labels_3)
                    
                    # 加权求和 (主loss权重1，辅助loss权重递减)
                    loss = loss1 + 0.5 * loss2 + 0.25 * loss3
                    
                    # 为了后续日志记录，将 output 指向主输出
                    output = out1
                else:
                    # 正常单输出模型
                    loss = criterion_mrae(output, labels)
                # [修改部分结束] ---------------------------------------------------------


                # loss = criterion_mrae(output, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.update(loss.data)
                iteration = iteration+1
                if iteration % 20 == 0:
                    print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                          % (iteration, total_iteration, lr, losses.avg))
                if iteration % 1000 == 0:
                    # 计算剩余训练时间
                    elapsed_time = time.time() - start_time  # 已用时间（秒）
                    avg_time_per_iter = elapsed_time / iteration  # 平均每次迭代时间
                    remaining_iters = total_iteration - iteration  # 剩余迭代次数
                    remaining_time = avg_time_per_iter * remaining_iters  # 剩余时间（秒）
                    
                    # 转换为小时和分钟
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)
                    
                    mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                    print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                    # Save model
                    if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                        print(f'Saving to {opt.outf}')
                        save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                        if mrae_loss < record_mrae_loss:
                            record_mrae_loss = mrae_loss
                    # print loss
                    print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                          "Test RMSE: %.9f, Test PSNR: %.9f, Remaining Time: %d hours %d minutes" 
                          % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, 
                             remaining_hours, remaining_minutes))
                    logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                          "Test RMSE: %.9f, Test PSNR: %.9f, Remaining Time: %d hours %d minutes" 
                          % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss,
                             remaining_hours, remaining_minutes))
                    
                    # 记录到CSV
                    loss_writer.writerow([
                        iteration, 
                        iteration//1000, 
                        float(losses.avg), 
                        float(mrae_loss), 
                        float(rmse_loss), 
                        float(psnr_loss), 
                        float(lr)
                    ])
                    loss_csv.flush()
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user (Ctrl+C)")
        print("="*60)
        logger.info("Training interrupted by user")
    
    except Exception as e:
        print(f"\nTraining error: {e}")
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 无论正常结束还是被中断，都关闭CSV并绘制曲线
        if 'loss_csv' in locals():
            loss_csv.close()
        if os.path.exists(loss_csv_path):
            print("\nGenerating loss curves...")
            plot_loss_curves(loss_csv_path, opt.outf)
        else:
            print("Warning: No loss data recorded, skipping plot generation.")
    
    return 0

# Validate
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
            output = model(input)

            # [修改部分开始] ---------------------------------------------------------
            # [新增] 这里必须加判断！
            # 如果是多尺度输出，只取第一个（最高分辨率结果）进行验证
            if isinstance(output, (tuple, list)):
                output = output[0]
            # [修改部分结束] ---------------------------------------------------------    
            
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

def plot_loss_curves(csv_path, output_dir):
    """从CSV文件读取数据并绘制loss曲线"""
    try:
        iterations = []
        train_losses = []
        val_mraes = []
        val_rmses = []
        val_psnrs = []
        lrs = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row['iteration']))
                train_losses.append(float(row['train_loss']))
                val_mraes.append(float(row['val_mrae']))
                val_rmses.append(float(row['val_rmse']))
                val_psnrs.append(float(row['val_psnr']))
                lrs.append(float(row['lr']))
        
        if len(iterations) == 0:
            print("Warning: No data found in CSV file, skipping plot generation.")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Train Loss
        axes[0, 0].plot(iterations, train_losses, 'b-', linewidth=1.5, label='Train Loss')
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Train Loss (MRAE)', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation MRAE
        axes[0, 1].plot(iterations, val_mraes, 'r-', linewidth=1.5, label='Val MRAE')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Validation MRAE', fontsize=12)
        axes[0, 1].set_title('Validation MRAE', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation RMSE
        axes[0, 2].plot(iterations, val_rmses, 'g-', linewidth=1.5, label='Val RMSE')
        axes[0, 2].set_xlabel('Iteration', fontsize=12)
        axes[0, 2].set_ylabel('Validation RMSE', fontsize=12)
        axes[0, 2].set_title('Validation RMSE', fontsize=14, fontweight='bold')
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Validation PSNR
        axes[1, 0].plot(iterations, val_psnrs, 'm-', linewidth=1.5, label='Val PSNR')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Validation PSNR (dB)', fontsize=12)
        axes[1, 0].set_title('Validation PSNR', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(iterations, lrs, 'orange', linewidth=1.5, label='Learning Rate')
        axes[1, 1].set_xlabel('Iteration', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined: Train Loss and Val MRAE (对比图)
        ax1 = axes[1, 2]
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(iterations, train_losses, 'b-', linewidth=1.5, label='Train Loss')
        line2 = ax1_twin.plot(iterations, val_mraes, 'r-', linewidth=1.5, label='Val MRAE')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Train Loss', fontsize=12, color='b')
        ax1_twin.set_ylabel('Val MRAE', fontsize=12, color='r')
        ax1.set_title('Train Loss vs Val MRAE', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # 设置整个figure的标题
        fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # 为标题留出空间
        save_path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n{'='*60}")
        print(f"Loss curves saved to: {save_path}")
        print(f"{'='*60}\n")
        
        # 尝试显示图表（如果环境支持GUI）
        try:
            # 检查是否有显示环境
            if 'DISPLAY' in os.environ or sys.platform == 'win32' or sys.platform == 'darwin':
                # 使用block=True，让用户可以看到图表并手动关闭
                print("Displaying loss curves. Close the window when done.")
                plt.show(block=True)  # block=True 会阻塞直到窗口关闭
            else:
                print("No display available. Loss curves saved to file only.")
        except Exception as e:
            print(f"Could not display plot: {e}. Loss curves saved to file only.")
        
        plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to plot loss curves: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
    print(torch.__version__)