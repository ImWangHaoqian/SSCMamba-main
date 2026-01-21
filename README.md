# SSCMamba: Ortho-Diagonal 3D Mamba for Hyperspectral Image Reconstruction

本仓库基于 MST++ 框架，围绕我们提出的 **SSCMamba** 方法，提供在 NTIRE2020 / NTIRE2022 数据集上的训练与测试代码。  
同时保留一批经典 SOTA 方法作为对比，包括：HSCNN+、HRNet、EDSR、HDNet、MIRNet、HINet、MPRNet、Restormer、MST、MST++、GMSR、SSTHyper 等。

---

## 1. 代码结构

当前仓库的主要结构如下：

```
SSCMamba-main/
  ├─ train_code/                 # 训练代码
  │   ├─ architecture/           # 所有可选模型结构（训练）
  │   │   ├─ __init__.py         # 模型生成器（统一入口）
  │   │   ├─ SSCMamba.py         # SSCMamba 模型实现
  │   │   ├─ edsr.py
  │   │   ├─ HDNet.py
  │   │   ├─ hinet.py
  │   │   ├─ hrnet.py
  │   │   ├─ HSCNN_Plus.py
  │   │   ├─ MIRNet.py
  │   │   ├─ MPRNet.py
  │   │   ├─ MST.py
  │   │   ├─ MST_Plus_Plus.py
  │   │   ├─ Restormer.py
  │   │   ├─ GMSR.py
  │   │   └─ SSTHyper.py
  │   ├─ train.py                # 主训练脚本
  │   ├─ hsi_dataset.py          # 数据加载
  │   └─ utils.py                # 训练/指标工具函数
  │
  ├─ test_develop_code/          # 验证集测试代码
  │   ├─ architecture/           # 所有可选模型结构（测试）
  │   │   ├─ __init__.py         # 模型生成器（统一入口）
  │   │   └─ [同上模型文件]
  │   ├─ test.py                 # 主测试脚本（验证集）
  │   ├─ hsi_dataset.py
  │   ├─ utils.py
  │   └─ model_zoo/              # 预训练模型存放目录
  │       ├─ NTIRE2022/
  │       ├─ NTIRE2020_clean/
  │       └─ NTIRE2020_realworld/
  │
  ├─ dataset/                    # 数据集根目录（需自行下载）
  │   ├─ NTIRE2022/              # NTIRE2022 RGB/HSI
  │   ├─ NTIRE2020/              # NTIRE2020 Clean / RealWorld
  │   └─ split_txt/              # train_list.txt / valid_list.txt
  │
  ├─ visualization/              # 可视化脚本（Matlab）
  ├─ requirements.txt
  └─ README.md                   # 本文件
```

**训练/测试真正会用到的模型架构都集中在：**

- 训练：`train_code/architecture/__init__.py`
- 测试：`test_develop_code/architecture/__init__.py`

这两个文件中的 `model_generator(method, ...)` 是统一入口。

---

## 2. 环境配置

- Python >= 3.8（推荐使用 Anaconda）
- NVIDIA GPU + CUDA（建议 CUDA ≥ 11.x）
- 安装依赖：

```bash
cd SSCMamba-main
pip install -r requirements.txt
```

> **说明：** `mamba-ssm` 已在代码中通过 `try/except` 做了兼容，如果没有安装，会自动回退或给出提示。但为了获得完整的 3D Mamba 能力，建议安装对应版本的 `mamba-ssm`。

---

## 3. 支持的模型（统一由 `--method` 指定）

在当前的 `train_code/architecture/__init__.py` 与 `test_develop_code/architecture/__init__.py` 中，支持的 `method` 关键字为：

### 我们的方法
- **`sscmamba`**（推荐名字）
- 兼容旧名：`mst_ss3d_orthodiagonal_repconv`（会映射到同一个 SSCMamba 实现）

### 对比方法（基于原 MST++ 工具箱）
- `edsr` - EDSR
- `gmsr` - GMSR
- `hdnet` - HDNet
- `hinet` - HINet
- `hrnet` - HRNet
- `hscnn_plus` - HSCNN+
- `mirnet` - MIRNet
- `mprnet` - MPRNet
- `mst` - MST-L
- `mst_plus_plus` - MST++
- `restormer` - Restormer
- `ssthyper` - SSTHyper

---

## 4. 数据准备

训练代码支持三种数据配置，通过 `--dataset` 参数选择（默认 `ntire2022`）：

### NTIRE2022（默认）
- 训练 RGB：`dataset/NTIRE2022/Train_RGB/`
- 训练 GT：`dataset/NTIRE2022/Train_Spec/`
- 验证 RGB：`dataset/NTIRE2022/Valid_RGB/`
- 验证 GT：`dataset/NTIRE2022/Valid_spectral/`

### NTIRE2020 Clean
- 训练 RGB：`dataset/NTIRE2020/NTIRE2020_Train_Clean/`
- 训练 GT：`dataset/NTIRE2020/NTIRE2020_Train_Spectral/`
- 验证 RGB：`dataset/NTIRE2020/NTIRE2020_Validation_Clean/`
- 验证 GT：`dataset/NTIRE2020/NTIRE2020_Validation_Spectral/`

### NTIRE2020 RealWorld
- 训练 RGB：`dataset/NTIRE2020/NTIRE2020_Train_RealWorld/`
- 训练 GT：`dataset/NTIRE2020/NTIRE2020_Train_Spectral/`
- 验证 RGB：`dataset/NTIRE2020/NTIRE2020_Validation_RealWorld/`
- 验证 GT：`dataset/NTIRE2020/NTIRE2020_Validation_Spectral/`

数据集整体结构示例：

```
dataset/
  ├─ NTIRE2022/
  │   ├─ Train_RGB/
  │   ├─ Train_Spec/
  │   ├─ Valid_RGB/
  │   └─ Valid_spectral/
  │
  ├─ NTIRE2020/
  │   ├─ NTIRE2020_Train_Clean/
  │   ├─ NTIRE2020_Train_RealWorld/
  │   ├─ NTIRE2020_Train_Spectral/
  │   ├─ NTIRE2020_Validation_Clean/
  │   ├─ NTIRE2020_Validation_RealWorld/
  │   └─ NTIRE2020_Validation_Spectral/
  │
  └─ split_txt/
      ├─ train_list.txt
      └─ valid_list.txt
```

---

## 5. 训练指令示例

进入训练目录：

```bash
cd SSCMamba-main/train_code
```

### 5.1 训练 SSCMamba

#### NTIRE2020 Clean
```bash
python train.py \
  --method sscmamba \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 4e-4 \
  --outf ./exp/sscmamba_ntire2020_clean/ \
  --dataset ntire2020_clean \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  --debug
```

#### NTIRE2020 RealWorld
```bash
python train.py \
  --method sscmamba \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 4e-4 \
  --outf ./exp/sscmamba_ntire2020_realworld/ \
  --dataset ntire2020_realworld \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  --debug
```

#### NTIRE2022
```bash
python train.py \
  --method sscmamba \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 4e-4 \
  --outf ./exp/sscmamba_ntire2022/ \
  --dataset ntire2022 \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  --debug
```

### 5.2 训练其他对比方法（示例）

#### MST++
```bash
python train.py \
  --method mst_plus_plus \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 4e-4 \
  --outf ./exp/mst_plus_plus_ntire2020_clean/ \
  --dataset ntire2020_clean \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0
```

#### GMSR
```bash
python train.py \
  --method gmsr \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 1e-4 \
  --outf ./exp/gmsr_ntire2020_clean/ \
  --dataset ntire2020_clean \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  --debug
```

#### SSTHyper
```bash
python train.py \
  --method ssthyper \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 1e-4 \
  --outf ./exp/ssthyper_ntire2020_clean/ \
  --dataset ntire2020_clean \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  --debug
```

> **提示：** 其它方法（`edsr`, `hdnet`, `hinet`, `hrnet`, `hscnn_plus`, `mirnet`, `mprnet`, `restormer`, `mst`）只需替换 `--method` 和 `--outf` 即可。不同方法的学习率可能不同，请根据实际情况调整 `--init_lr`。

---

## 6. 验证 / 测试指令（Validation）

进入测试目录：

```bash
cd SSCMamba-main/test_develop_code
```

### 6.1 NTIRE2022 数据集测试

#### SSCMamba
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method sscmamba \
  --pretrained_model_path ./model_zoo/NTIRE2022/sscmamba_net_xxxepoch.pth \
  --outf ./exp/NTIRE2022/sscmamba/ \
  --gpu_id 0
```

#### 其他方法示例

**EDSR:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method edsr \
  --pretrained_model_path ./model_zoo/NTIRE2022/edsr_net_132epoch.pth \
  --outf ./exp/NTIRE2022/edsr/ \
  --gpu_id 0
```

**MST++:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method mst_plus_plus \
  --pretrained_model_path ./model_zoo/NTIRE2022/mst_plus_plus_net_370epoch.pth \
  --outf ./exp/NTIRE2022/mst_plus_plus/ \
  --gpu_id 0
```

**GMSR:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method gmsr \
  --pretrained_model_path ./model_zoo/NTIRE2022/gmsr_net_29epoch.pth \
  --outf ./exp/NTIRE2022/gmsr/ \
  --gpu_id 0
```

**HDNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method hdnet \
  --pretrained_model_path ./model_zoo/NTIRE2022/hdnet_net_91epoch.pth \
  --outf ./exp/NTIRE2022/hdnet/ \
  --gpu_id 0
```

**HINet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method hinet \
  --pretrained_model_path ./model_zoo/NTIRE2022/hinet_net_160epoch.pth \
  --outf ./exp/NTIRE2022/hinet/ \
  --gpu_id 0
```

**HRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method hrnet \
  --pretrained_model_path ./model_zoo/NTIRE2022/hrnet_net_163epoch.pth \
  --outf ./exp/NTIRE2022/hrnet/ \
  --gpu_id 0
```

**MIRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method mirnet \
  --pretrained_model_path ./model_zoo/NTIRE2022/mirnet_net_55epoch.pth \
  --outf ./exp/NTIRE2022/mirnet/ \
  --gpu_id 0
```

**HSCNN_Plus:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method hscnn_plus \
  --pretrained_model_path ./model_zoo/NTIRE2022/hscnn_plus_net_90epoch.pth \
  --outf ./exp/NTIRE2022/hscnn_plus/ \
  --gpu_id 0
```

**MPRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method mprnet \
  --pretrained_model_path ./model_zoo/NTIRE2022/mprnet_net_110epoch.pth \
  --outf ./exp/NTIRE2022/mprnet/ \
  --gpu_id 0
```

**MST:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method mst \
  --pretrained_model_path ./model_zoo/NTIRE2022/mst_net_170epoch.pth \
  --outf ./exp/NTIRE2022/mst/ \
  --gpu_id 0
```

**Restormer:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method restormer \
  --pretrained_model_path ./model_zoo/NTIRE2022/restormer_net_125epoch.pth \
  --outf ./exp/NTIRE2022/restormer/ \
  --gpu_id 0
```

**SSTHyper:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method ssthyper \
  --pretrained_model_path ./model_zoo/NTIRE2022/ssthyper_net_195epoch.pth \
  --outf ./exp/NTIRE2022/ssthyper/ \
  --gpu_id 0
```

### 6.2 NTIRE2020 RealWorld 数据集测试

#### SSCMamba
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method sscmamba \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/sscmamba_net_xxxepoch.pth \
  --outf ./exp/NTIRE2020_realworld/sscmamba/ \
  --gpu_id 0
```

#### 其他方法示例

**EDSR:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method edsr \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/edsr_net_242epoch.pth \
  --outf ./exp/NTIRE2020_realworld/edsr/ \
  --gpu_id 0
```

**GMSR:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method gmsr \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/gmsr_net_149epoch.pth \
  --outf ./exp/NTIRE2020_realworld/gmsr/ \
  --gpu_id 0
```

**HDNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method hdnet \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/hdnet_net_239epoch.pth \
  --outf ./exp/NTIRE2020_realworld/hdnet/ \
  --gpu_id 0
```

**HINet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method hinet \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/hinet_net_333epoch.pth \
  --outf ./exp/NTIRE2020_realworld/hinet/ \
  --gpu_id 0
```

**HRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method hrnet \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/hrnet_net_142epoch.pth \
  --outf ./exp/NTIRE2020_realworld/hrnet/ \
  --gpu_id 0
```

**HSCNN_Plus:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method hscnn_plus \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/hscnn_plus_net_6epoch.pth \
  --outf ./exp/NTIRE2020_realworld/hscnn_plus/ \
  --gpu_id 0
```

**MIRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method mirnet \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/mirnet_net_187epoch.pth \
  --outf ./exp/NTIRE2020_realworld/mirnet/ \
  --gpu_id 0
```

**MPRNet:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method mprnet \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/mprnet_net_192epoch.pth \
  --outf ./exp/NTIRE2020_realworld/mprnet/ \
  --gpu_id 0
```

**MST:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method mst \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/mst_net_155epoch.pth \
  --outf ./exp/NTIRE2020_realworld/mst/ \
  --gpu_id 0
```

**MST_Plus_Plus:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method mst_plus_plus \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/mst_plus_plus_net_183epoch.pth \
  --outf ./exp/NTIRE2020_realworld/mst_plus_plus/ \
  --gpu_id 0
```

**Restormer:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method restormer \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/restormer_net_197epoch.pth \
  --outf ./exp/NTIRE2020_realworld/restormer/ \
  --gpu_id 0
```

**SSTHyper:**
```bash
python test.py \
  --data_root ../dataset/NTIRE2020_realworld/ \
  --method ssthyper \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/ssthyper_net_206epoch.pth \
  --outf ./exp/NTIRE2020_realworld/ssthyper/ \
  --gpu_id 0
```

> **说明：** 测试脚本会输出 **MRAE / RMSE / PSNR** 指标，并将重建的 HSI 保存为 `.mat` 文件到指定的输出目录。

---

## 7. 模型复杂度评估

我们提供了参数量和计算复杂度评估函数 `my_summary()`，位于 `test_develop_code/utils.py`。使用方法：

```python
from utils import my_summary
from architecture import model_generator

# 创建模型
model = model_generator('sscmamba')

# 评估参数量和 FLOPS
# 参数：模型, 图像高度, 图像宽度, 输入通道数, 批次大小
my_summary(model, 256, 256, 3, 1)
```

---

## 8. 可视化工具

可视化脚本位于 `visualization/` 目录，包含以下 Matlab 脚本：

- `NTIRE2020Clean_one_generate_comparison_figures.m` - 生成对比图
- `NTIRE2020Clean_preview_RGB.m` - RGB 预览
- `NTIRE2020Clean_two_remove_white_borders.m` - 去除白边
- `NTIRE2020Clean_three_show_line.m` - 显示光谱曲线
- `NTIRE2020RealWorld_*.m` - RealWorld 数据集对应脚本
- `NTIRE2022_*.m` - NTIRE2022 数据集对应脚本

使用方法：

```bash
cd visualization/
# 在 Matlab 中运行对应的 .m 文件
```

---

## 9. 预训练模型

预训练模型应存放在以下目录结构：

```
test_develop_code/model_zoo/
  ├─ NTIRE2022/
  │   ├─ sscmamba_net_xxxepoch.pth
  │   ├─ edsr_net_132epoch.pth
  │   ├─ gmsr_net_29epoch.pth
  │   └─ ...
  ├─ NTIRE2020_clean/
  │   └─ sscmamba.pth
  └─ NTIRE2020_realworld/
      └─ sscmamba.pth
```

> **提示：** 你可以根据实际情况提供预训练模型的下载链接（Google Drive / 百度网盘等），方便他人复现实验。

---

## 10. 论文表格方法名与代码映射

论文表格中的方法名与代码中的 `--method` 参数对应关系：

| 论文方法名 | `--method` 参数 |
|:---------:|:--------------:|
| HSCNN+ | `hscnn_plus` |
| HRNet | `hrnet` |
| EDSR | `edsr` |
| HDNet | `hdnet` |
| MIRNet | `mirnet` |
| HINet | `hinet` |
| MPRNet | `mprnet` |
| Restormer | `restormer` |
| MST | `mst` |
| MST++ | `mst_plus_plus` |
| SSTHyper | `ssthyper` |
| GMSR | `gmsr` |
| **SSCMamba (Ours)** | `sscmamba` |

---

## 11. 致谢与引用

本仓库基于 MST / MST++ 框架进行扩展，感谢原作者公开代码与模型。

如果本项目对你的研究有帮助，请引用相关论文：

```bibtex
% MST
@inproceedings{mst,
  title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
  author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={CVPR},
  year={2022}
}

% MST++
@inproceedings{mst_pp,
  title={MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction},
  author={Yuanhao Cai and Jing Lin and Zudi Lin and Haoqian Wang and Yulun Zhang and Hanspeter Pfister and Radu Timofte and Luc Van Gool},
  booktitle={CVPRW},
  year={2022}
}

% HDNet
@inproceedings{hdnet,
  title={HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging},
  author={Xiaowan Hu and Yuanhao Cai and Jing Lin and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={CVPR},
  year={2022}
}

% SSCMamba (请在此处添加你的论文引用)
@article{sscmamba,
  title={SSCMamba: Ortho-Diagonal 3D Mamba for Efficient Hyperspectral Image Reconstruction},
  author={Your Name and Co-authors},
  journal={Your Journal/Conference},
  year={2024}
}
```

---

## License

本项目遵循原 MST++ 项目的许可证。请参考原仓库的 LICENSE 文件。

---

## Contact

如有问题或建议，请通过 Issue 或邮件联系。
