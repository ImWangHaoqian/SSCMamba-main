# SSCMamba: Spatial-Spectral Combination Selective State Space Model for Spectral Reconstruction

This repository provides training and testing code for our proposed **SSCMamba** method on NTIRE2020 (Clean && Realworld) / NTIRE2022 datasets.  
We also include several classic SOTA methods for comparison, including: HSCNN+, HRNet, EDSR, HDNet, MIRNet, HINet, MPRNet, Restormer, MST, MST++, GMSR, SSTHyper, etc.

<hr />

## 📚 Table of Contents
- [Installation](#-1-installation)
- [Data Preparation](#-2-data-preparation)
- [Training](#-3-training)
- [Testing](#-4-testing--validation)
- [Model Complexity Evaluation](#-5-model-complexity-evaluation)
- [Visualization Tools](#-6-visualization-tools)
- [Pretrained Models](#-7-pretrained-models)
- [Acknowledgments](#-8-acknowledgments)
- [Contact](#-9-contact)

<br>
<hr />

## 💻 1. Installation

### System Requirements

- Python >= 3.8
- CUDA 11.8

### Install Dependencies

```bash
cd SSCMamba-main
pip install -r requirements.txt
```

<br>

<hr />

## 📦 2. Data Preparation

The training code supports three data configurations, selected via the `--dataset` parameter (default: `ntire2022`):

<br>

### NTIRE2022 (Default)

| Data Type | Path |
|-----------|------|
| Train RGB | `dataset/NTIRE2022/Train_RGB/` |
| Train GT | `dataset/NTIRE2022/Train_Spec/` |
| Valid RGB | `dataset/NTIRE2022/Valid_RGB/` |
| Valid GT | `dataset/NTIRE2022/Valid_spectral/` |

### NTIRE2020 Clean

| Data Type | Path |
|-----------|------|
| Train RGB | `dataset/NTIRE2020/NTIRE2020_Train_Clean/` |
| Train GT | `dataset/NTIRE2020/NTIRE2020_Train_Spectral/` |
| Valid RGB | `dataset/NTIRE2020/NTIRE2020_Validation_Clean/` |
| Valid GT | `dataset/NTIRE2020/NTIRE2020_Validation_Spectral/` |

### NTIRE2020 RealWorld

| Data Type | Path |
|-----------|------|
| Train RGB | `dataset/NTIRE2020/NTIRE2020_Train_RealWorld/` |
| Train GT | `dataset/NTIRE2020/NTIRE2020_Train_Spectral/` |
| Valid RGB | `dataset/NTIRE2020/NTIRE2020_Validation_RealWorld/` |
| Valid GT | `dataset/NTIRE2020/NTIRE2020_Validation_Spectral/` |

Dataset structure example:

```text
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

<hr />

## 🚂 3. Training

### 📌 Navigate to Training Directory

```bash
cd SSCMamba-main/train_code
```

### 📌 Train SSCMamba

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

> **💡 Tip:** For other methods (`edsr`, `hdnet`, `hinet`, `hrnet`, `hscnn_plus`, `mirnet`, `mprnet`, `restormer`, `mst`), simply replace `--method` and `--outf` accordingly.

<br>

<hr />

## 🧪 4. Testing / Validation

### 📌 Navigate to Testing Directory

```bash
cd SSCMamba-main/test_develop_code
```

### 📌 Test Command Format

```bash
python test.py \
  --data_root [DATA_ROOT] \
  --method [METHOD_NAME] \
  --pretrained_model_path [MODEL_PATH] \
  --outf [OUTPUT_DIR] \
  --gpu_id [GPU_ID]
```

| Parameter | Description |
|-----------|-------------|
| **`--data_root`** | Root directory path of the dataset |
| **`--method`** | Model method name (e.g., `sscmamba`) |
| **`--pretrained_model_path`** | Path to the pretrained model file |
| **`--outf`** | Output directory for results |
| **`--gpu_id`** | GPU ID to use |

<br>

### 📌 NTIRE2022 Dataset Testing

```bash
python test.py \
  --data_root ../dataset/NTIRE2022/ \
  --method sscmamba \
  --pretrained_model_path ./model_zoo/NTIRE2022/sscmamba.pth \
  --outf ./exp/NTIRE2022/sscmamba/ \
  --gpu_id 0
```

### 📌 NTIRE2020 RealWorld Dataset Testing

```bash
python test.py \
  --data_root ../dataset/NTIRE2020/ \
  --method sscmamba \
  --pretrained_model_path ./model_zoo/NTIRE2020_realworld/sscmamba.pth \
  --outf ./exp/NTIRE2020_realworld/sscmamba/ \
  --gpu_id 0
```

### 📌 NTIRE2020 Clean Dataset Testing

```bash
python test.py \
  --data_root ../dataset/NTIRE2020/ \
  --method sscmamba \
  --pretrained_model_path ./model_zoo/NTIRE2020_clean/sscmamba.pth \
  --outf ./exp/NTIRE2020_clean/sscmamba/ \
  --gpu_id 0
```

<br>

<hr />

## 📊 5. Model Complexity Evaluation

We provide a parameter count and computational complexity evaluation function `my_summary()` located in `test_develop_code/utils.py`.

### 📌 Usage

```python
from utils import my_summary
from architecture import model_generator

# Create model
model = model_generator('sscmamba')

# Evaluate parameters and FLOPS
# Parameters: model, image height, image width, input channels, batch size
my_summary(model, 256, 256, 3, 1)
```

| Parameter | Description |
|-----------|-------------|
| **Model** | Model instance to evaluate |
| **Image Height** | Height of input image (pixels) |
| **Image Width** | Width of input image (pixels) |
| **Input Channels** | Number of input channels (3 for RGB) |
| **Batch Size** | Batch size (usually 1) |

<br>

<hr />

## 🎨 6. Visualization Tools

Visualization scripts are located in the `visualization/` directory, including the following Matlab scripts:

| Script File | Function |
|------------|----------|
| `NTIRE2020Clean_one_generate_comparison_figures.m` | Generate comparison figures |
| `NTIRE2020Clean_preview_RGB.m` | RGB preview |
| `NTIRE2020Clean_two_remove_white_borders.m` | Remove white borders |
| `NTIRE2020Clean_three_show_line.m` | Display spectral curves |
| `NTIRE2020RealWorld_*.m` | RealWorld dataset corresponding scripts |
| `NTIRE2022_*.m` | NTIRE2022 dataset corresponding scripts |

<br>

<hr />

## 📥 7. Pretrained Models

Pretrained models should be stored in the following directory structure:

```text
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

<br>

<hr />

## 🙏 8. Acknowledgments

This repository is extended based on the [MST++](https://github.com/caiyuanhao1998/MST-plus-plus) framework. We thank the original authors for publicly releasing their code and models.

<br>
<hr />

## 📧 9. Contact

For questions or suggestions, please contact via email: **wanghaoqian22@nudt.edu.cn**

<br>
<hr />
