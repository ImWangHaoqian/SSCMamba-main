from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import scipy.io as sio
import os

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

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
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
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
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        
        # 根据 data_root 自动识别数据集类型
        data_root_normalized = data_root.rstrip('/')
        if 'NTIRE2022' in data_root_normalized:
            # NTIRE2022 格式
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
                mat.close()
                print(f'Ntire2022 scene {i} is loaded.')
        
        elif 'NTIRE2020' in data_root_normalized:
            # NTIRE2020 格式 - 通过路径或目录存在性判断是 clean 还是 realworld
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
                dataset_root = data_root
            
            # 首先检查路径中是否包含关键字
            if 'RealWorld' in data_root_normalized or 'realworld' in data_root_normalized.lower():
                # 明确指定为 RealWorld
                bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
                rgb_suffix = '_RealWorld.jpg'
                dataset_type = 'realworld'
            elif 'Clean' in data_root_normalized or 'clean' in data_root_normalized.lower():
                # 明确指定为 Clean
                bgr_data_path = f'{dataset_root}NTIRE2020_Validation_Clean/'
                rgb_suffix = '_clean.png'
                dataset_type = 'clean'
            else:
                # 路径中没有明确指定，检查目录存在性
                realworld_exists = os.path.exists(f'{dataset_root}NTIRE2020_Validation_RealWorld')
                clean_exists = os.path.exists(f'{dataset_root}NTIRE2020_Validation_Clean')
                
                if realworld_exists and clean_exists:
                    # 两个都存在，默认使用 RealWorld，但给出警告
                    print(f'Warning: Both Clean and RealWorld directories exist. Using RealWorld by default.')
                    bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
                    rgb_suffix = '_RealWorld.jpg'
                    dataset_type = 'realworld'
                elif realworld_exists:
                    # 只有 RealWorld
                    bgr_data_path = f'{dataset_root}NTIRE2020_Validation_RealWorld/'
                    rgb_suffix = '_RealWorld.jpg'
                    dataset_type = 'realworld'
                elif clean_exists:
                    # 只有 Clean
                    bgr_data_path = f'{dataset_root}NTIRE2020_Validation_Clean/'
                    rgb_suffix = '_clean.png'
                    dataset_type = 'clean'
                else:
                    raise ValueError(f'Neither NTIRE2020_Validation_Clean nor NTIRE2020_Validation_RealWorld directory found in {dataset_root}')
            
            hyper_data_path = f'{dataset_root}NTIRE2020_Validation_Spectral/'
            
            # 从RGB文件夹读取文件列表
            bgr_files = [f for f in os.listdir(bgr_data_path) if f.endswith(rgb_suffix)]
            bgr_files.sort()
            
            # 提取文件ID并匹配对应的.mat文件
            for bgr_file in bgr_files:
                # 提取文件ID：从 ARAD_HS_XXXX_clean.png 或 ARAD_HS_XXXX_RealWorld.jpg 提取 ARAD_HS_XXXX
                if dataset_type == 'clean':
                    file_id = bgr_file.replace('_clean.png', '')
                else:  # realworld
                    file_id = bgr_file.replace('_RealWorld.jpg', '')
                
                hyper_file = file_id + '.mat'
                hyper_path = os.path.join(hyper_data_path, hyper_file)
                bgr_path = os.path.join(bgr_data_path, bgr_file)
                
                if not os.path.exists(hyper_path):
                    print(f'Warning: {hyper_path} not found, skipping {bgr_file}')
                    continue
                
                # 加载hyperspectral数据
                try:
                    # 尝试使用 h5py 打开（MATLAB v7.3 HDF5 格式）
                    try:
                        with h5py.File(hyper_path, 'r') as mat:
                            hyper = np.float32(np.array(mat['cube']))
                            # NTIRE2022格式: (C, H, W)，转置为 (C, W, H)
                            hyper = np.transpose(hyper, [0, 2, 1])
                    except (OSError, IOError):
                        # 如果 h5py 失败，尝试使用 scipy.io.loadmat（MATLAB v5 格式，如 NTIRE2020）
                        mat_data = sio.loadmat(hyper_path)
                        hyper = np.float32(mat_data['cube'])
                        # NTIRE2020格式: (H, W, C)，转置为 (C, H, W)
                        if len(hyper.shape) == 3 and hyper.shape[2] < hyper.shape[0]:
                            # 如果最后一个维度最小，说明是 (H, W, C) 格式
                            hyper = np.transpose(hyper, [2, 0, 1])  # (H, W, C) -> (C, H, W)
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
            raise ValueError(f'Unknown dataset type. data_root should contain NTIRE2022 or NTIRE2020, got: {data_root}')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)