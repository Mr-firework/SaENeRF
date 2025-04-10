# SaENeRF: Suppressing Artifacts in Event-based Neural Radiance Fields
Yuanjian Wang, Yufei Deng, Rong Xiao, Jiahao Fan, Chenwei Tang, Xiong Deng, and Jiancheng Lv
IJCNN 2025
## 1. Installation
```bash
# (Optional) create a fresh conda env
conda create --name nerfstudio -y "python<3.11"
conda activate nerfstudio

# install dependencies
pip install --upgrade pip setuptools
pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install saenerf
git clone https://github.com/Mr-firework/SaENeRF
cd SaENeRF
pip install .
```

## 2. Download Dataset
https://nextcloud.mpi-klsb.mpg.de/index.php/s/xDqwRHiWKeSRyes

## 3. Training
```bash
ns-train saenerf --data data/eventnerf/nerf/lego
```
