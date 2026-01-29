from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import os
import scipy.io as sio

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8, max_samples=None, dataset_type='ntire2022'):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        if dataset_type == 'ntire2022':
            # 原有逻辑完全不变
            hyper_data_path = f'{data_root}/Train_Spec/'
            bgr_data_path = f'{data_root}/Train_RGB/'

            with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
                hyper_list = [line.replace('\n','.mat') for line in fin]
                bgr_list = [line.replace('mat','jpg') for line in hyper_list]
            hyper_list.sort()
            bgr_list.sort()
            print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
            print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
            for i in range(len(hyper_list)):
                if max_samples is not None and i >= max_samples:
                    break
                hyper_path = hyper_data_path + hyper_list[i]
                if 'mat' not in hyper_path:
                    continue
                with h5py.File(hyper_path, 'r') as mat:
                    hyper =np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                bgr_path = bgr_data_path + bgr_list[i]
                assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
                bgr = cv2.imread(bgr_path)
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr)
                bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                print(f'Ntire2022 scene {i} is loaded.')
        
        elif dataset_type in ['ntire2020_clean', 'ntire2020_realworld']:
            # NTIRE2020数据集处理逻辑
            if dataset_type == 'ntire2020_clean':
                bgr_data_path = f'{data_root}/NTIRE2020_Train_Clean/'
                rgb_suffix = '_clean.png'
            else:  # ntire2020_realworld
                bgr_data_path = f'{data_root}/NTIRE2020_Train_RealWorld/'
                rgb_suffix = '_RealWorld.jpg'
            
            hyper_data_path = f'{data_root}/NTIRE2020_Train_Spectral/'
            
            # 从RGB文件夹读取文件列表
            bgr_files = [f for f in os.listdir(bgr_data_path) if f.endswith(rgb_suffix)]
            bgr_files.sort()
            
            # 提取文件ID并匹配对应的.mat文件
            for bgr_file in bgr_files:
                if max_samples is not None and len(self.hypers) >= max_samples:
                    break
                
                # 提取文件ID：从 ARAD_HS_XXXX_clean.png 或 ARAD_HS_XXXX_RealWorld.jpg 提取 ARAD_HS_XXXX
                if dataset_type == 'ntire2020_clean':
                    file_id = bgr_file.replace('_clean.png', '')
                else:  # ntire2020_realworld
                    file_id = bgr_file.replace('_RealWorld.jpg', '')
                
                hyper_file = file_id + '.mat'
                hyper_path = os.path.join(hyper_data_path, hyper_file)
                bgr_path = os.path.join(bgr_data_path, bgr_file)
                
                if not os.path.exists(hyper_path):
                    print(f'Warning: {hyper_path} not found, skipping {bgr_file}')
                    continue
                
                # 加载hyperspectral数据
                try:
                    # 尝试使用 h5py 打开（MATLAB v7.3 HDF5 格式，如 NTIRE2022）
                    try:
                        with h5py.File(hyper_path, 'r') as mat:
                            if 'cube' not in mat.keys():
                                print(f'Warning: {hyper_path} does not contain "cube" key, skipping {bgr_file}')
                                continue
                            hyper = np.float32(np.array(mat['cube']))
                            # NTIRE2022格式: (C, H, W)，转置为 (C, W, H)
                            hyper = np.transpose(hyper, [0, 2, 1])
                    except (OSError, IOError):
                        # 如果 h5py 失败，尝试使用 scipy.io.loadmat（MATLAB v5 格式，如 NTIRE2020）
                        mat_data = sio.loadmat(hyper_path)
                        if 'cube' not in mat_data:
                            print(f'Warning: {hyper_path} does not contain "cube" key, skipping {bgr_file}')
                            continue
                        hyper = np.float32(mat_data['cube'])
                        # NTIRE2020格式: (H, W, C)，转置为 (C, H, W)，保持和BGR一致的维度顺序
                        if len(hyper.shape) == 3 and hyper.shape[2] < hyper.shape[0]:
                            # 如果最后一个维度最小，说明是 (H, W, C) 格式
                            hyper = np.transpose(hyper, [2, 0, 1])  # (H, W, C) -> (C, H, W)
                        # 注意：NTIRE2020不需要再次转置，保持 (C, H, W) 格式
                except Exception as e:
                    print(f'Error loading {hyper_path}: {e}')
                    print(f'File exists: {os.path.exists(hyper_path)}, File size: {os.path.getsize(hyper_path) if os.path.exists(hyper_path) else "N/A"} bytes')
                    continue
                
                # 加载RGB数据
                bgr = cv2.imread(bgr_path)
                if bgr is None:
                    print(f'Warning: Failed to load {bgr_path}, skipping')
                    continue
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr)
                bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
                
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                print(f'NTIRE2020 {dataset_type} scene {len(self.hypers)-1} is loaded: {file_id}')
            
            print(f'len(hyper) of ntire2020 {dataset_type} dataset:{len(self.hypers)}')
            print(f'len(bgr) of ntire2020 {dataset_type} dataset:{len(self.bgrs)}')
        else:
            raise ValueError(f'Unknown dataset_type: {dataset_type}')
        
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        # 确保返回连续的数组
        return np.ascontiguousarray(img, dtype=np.float32)

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        
        # 计算patch的起始位置
        h_start = h_idx * stride
        w_start = w_idx * stride
        h_max, w_max = bgr.shape[1], bgr.shape[2]
        
        # 确保patch不超出图像边界，如果超出则调整起始位置
        if h_start + crop_size > h_max:
            h_start = max(0, h_max - crop_size)
        if w_start + crop_size > w_max:
            w_start = max(0, w_max - crop_size)
        
        # 确保索引不为负
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        
        # 计算结束位置（确保不超出边界）
        h_end = min(h_max, h_start + crop_size)
        w_end = min(w_max, w_start + crop_size)
        
        # 再次检查并调整起始位置，确保能提取到crop_size大小的patch
        if h_end - h_start < crop_size:
            h_start = max(0, h_end - crop_size)
        if w_end - w_start < crop_size:
            w_start = max(0, w_end - crop_size)
        
        # 提取patch
        bgr_patch = bgr[:, h_start:h_start+crop_size, w_start:w_start+crop_size].copy()
        hyper_patch = hyper[:, h_start:h_start+crop_size, w_start:w_start+crop_size].copy()
        
        # 调试信息（如果尺寸不对）
        if bgr_patch.shape[1] != crop_size or bgr_patch.shape[2] != crop_size:
            print(f"ERROR: BGR patch size mismatch: expected ({bgr_patch.shape[0]}, {crop_size}, {crop_size}), got {bgr_patch.shape}")
            print(f"  img_idx={img_idx}, patch_idx={patch_idx}, h_idx={h_idx}, w_idx={w_idx}")
            print(f"  h_max={h_max}, w_max={w_max}, h_start={h_start}, w_start={w_start}, h_end={h_end}, w_end={w_end}")
            print(f"  bgr.shape={bgr.shape}, hyper.shape={hyper.shape}")
        if hyper_patch.shape[1] != crop_size or hyper_patch.shape[2] != crop_size:
            print(f"ERROR: Hyper patch size mismatch: expected ({hyper_patch.shape[0]}, {crop_size}, {crop_size}), got {hyper_patch.shape}")
            print(f"  img_idx={img_idx}, patch_idx={patch_idx}, h_idx={h_idx}, w_idx={w_idx}")
            print(f"  h_max={h_max}, w_max={w_max}, h_start={h_start}, w_start={w_start}, h_end={h_end}, w_end={w_end}")
            print(f"  bgr.shape={bgr.shape}, hyper.shape={hyper.shape}")
        
        # 最终检查：确保patch尺寸正确
        assert bgr_patch.shape[1] == crop_size and bgr_patch.shape[2] == crop_size, \
            f"BGR patch size mismatch: expected ({bgr_patch.shape[0]}, {crop_size}, {crop_size}), got {bgr_patch.shape}, h_max={h_max}, w_max={w_max}, h_start={h_start}, w_start={w_start}"
        assert hyper_patch.shape[1] == crop_size and hyper_patch.shape[2] == crop_size, \
            f"Hyper patch size mismatch: expected ({hyper_patch.shape[0]}, {crop_size}, {crop_size}), got {hyper_patch.shape}, h_max={h_max}, w_max={w_max}, h_start={h_start}, w_start={w_start}"
        
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr_patch = self.arguement(bgr_patch, rotTimes, vFlip, hFlip)
            hyper_patch = self.arguement(hyper_patch, rotTimes, vFlip, hFlip)
        else:
            # 即使不使用数据增强，也要确保数组是连续的
            bgr_patch = np.ascontiguousarray(bgr_patch, dtype=np.float32)
            hyper_patch = np.ascontiguousarray(hyper_patch, dtype=np.float32)
        
        # 确保返回连续的数组
        return bgr_patch, hyper_patch

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True, dataset_type='ntire2022'):
        self.hypers = []
        self.bgrs = []
        
        if dataset_type == 'ntire2022':
            # 原有逻辑完全不变
            hyper_data_path = f'{data_root}/Valid_spectral/'
            bgr_data_path = f'{data_root}/Valid_RGB/'
            with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
                hyper_list = [line.replace('\n', '.mat') for line in fin]
                bgr_list = [line.replace('mat','jpg') for line in hyper_list]
            hyper_list.sort()
            bgr_list.sort()
            print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
            print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
            for i in range(len(hyper_list)):
                hyper_path = hyper_data_path + hyper_list[i]
                if 'mat' not in hyper_path:
                    continue
                with h5py.File(hyper_path, 'r') as mat:
                    hyper = np.float32(np.array(mat['cube']))
                hyper = np.transpose(hyper, [0, 2, 1])
                bgr_path = bgr_data_path + bgr_list[i]
                assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
                bgr = cv2.imread(bgr_path)
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr)
                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1])
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                print(f'Ntire2022 scene {i} is loaded.')
        
        elif dataset_type in ['ntire2020_clean', 'ntire2020_realworld']:
            # NTIRE2020验证集处理逻辑
            if dataset_type == 'ntire2020_clean':
                bgr_data_path = f'{data_root}/NTIRE2020_Validation_Clean/'
                rgb_suffix = '_clean.png'
            else:  # ntire2020_realworld
                bgr_data_path = f'{data_root}/NTIRE2020_Validation_RealWorld/'
                rgb_suffix = '_RealWorld.jpg'
            
            hyper_data_path = f'{data_root}/NTIRE2020_Validation_Spectral/'
            
            # 从RGB文件夹读取文件列表
            bgr_files = [f for f in os.listdir(bgr_data_path) if f.endswith(rgb_suffix)]
            bgr_files.sort()
            
            # 提取文件ID并匹配对应的.mat文件
            for bgr_file in bgr_files:
                # 提取文件ID：从 ARAD_HS_XXXX_clean.png 或 ARAD_HS_XXXX_RealWorld.jpg 提取 ARAD_HS_XXXX
                if dataset_type == 'ntire2020_clean':
                    file_id = bgr_file.replace('_clean.png', '')
                else:  # ntire2020_realworld
                    file_id = bgr_file.replace('_RealWorld.jpg', '')
                
                hyper_file = file_id + '.mat'
                hyper_path = os.path.join(hyper_data_path, hyper_file)
                bgr_path = os.path.join(bgr_data_path, bgr_file)
                
                if not os.path.exists(hyper_path):
                    print(f'Warning: {hyper_path} not found, skipping {bgr_file}')
                    continue
                
                # 加载hyperspectral数据
                try:
                    # 尝试使用 h5py 打开（MATLAB v7.3 HDF5 格式，如 NTIRE2022）
                    try:
                        with h5py.File(hyper_path, 'r') as mat:
                            hyper = np.float32(np.array(mat['cube']))
                            # NTIRE2022格式: (C, H, W)，转置为 (C, W, H)
                            hyper = np.transpose(hyper, [0, 2, 1])
                    except (OSError, IOError):
                        # 如果 h5py 失败，尝试使用 scipy.io.loadmat（MATLAB v5 格式，如 NTIRE2020）
                        mat_data = sio.loadmat(hyper_path)
                        hyper = np.float32(mat_data['cube'])
                        # NTIRE2020格式: (H, W, C)，转置为 (C, H, W)，保持和BGR一致的维度顺序
                        if len(hyper.shape) == 3 and hyper.shape[2] < hyper.shape[0]:
                            # 如果最后一个维度最小，说明是 (H, W, C) 格式
                            hyper = np.transpose(hyper, [2, 0, 1])  # (H, W, C) -> (C, H, W)
                        # 注意：NTIRE2020不需要再次转置，保持 (C, H, W) 格式
                except Exception as e:
                    print(f'Error loading {hyper_path}: {e}')
                    continue
                
                # 加载RGB数据
                bgr = cv2.imread(bgr_path)
                if bgr is None:
                    print(f'Warning: Failed to load {bgr_path}, skipping')
                    continue
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr)
                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1])
                
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                print(f'NTIRE2020 {dataset_type} validation scene {len(self.hypers)-1} is loaded: {file_id}')
            
            print(f'len(hyper_valid) of ntire2020 {dataset_type} dataset:{len(self.hypers)}')
            print(f'len(bgr_valid) of ntire2020 {dataset_type} dataset:{len(self.bgrs)}')
        else:
            raise ValueError(f'Unknown dataset_type: {dataset_type}')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)