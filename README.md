#MWAG-Net
## ⚙️ How to use
```bash
conda create -n mwagnet python=3.10 -y
conda activate mwagnet

# PyTorch 1.13 + CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116  

# dependencies
pip install opencv-python albumentations==1.4.11 albucore==0.0.12 medpy thop numpy==1.26.4 tensorboardX
```
