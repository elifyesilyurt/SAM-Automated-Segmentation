# 🚀 INSTALLATION GUIDE — Kurulum Rehberi

SAM Automated Segmentation projesini farklı ortamlarda kurmak için adım adım rehber.

## 📑 Hızlı Navigasyon

- [Google Colab (Önerilen)](#1-google-colab-önerilen)
- [Linux (Ubuntu/Debian)](#2-linux-ubuntdebian)
- [macOS](#3-macos)
- [Windows 10/11](#4-windows-1011)
- [Docker](#5-docker)
- [Sorun Giderme](#sorun-giderme)

---

## 1. Google Colab (Önerilen) ⭐

**En Hızlı & En Kolay Seçenek**
- ✅ GPU önceden yüklü (T4, V100)
- ✅ Python 3.10 hazır
- ✅ Herhangi bir kurulum gerekmez
- ⚠️ 12 saatlik session sınırı var

### Adımlar

1. **Colab'i Aç**
   ```
   https://colab.research.google.com/
   ```

2. **GPU Etkinleştir**
   ```
   Çalışma Zamanı → Çalışma zamanı türünü değiştir → 
   İşlemci: GPU → T4 seç → Kaydet
   ```

3. **İlk Hücrede (Cell 1) Kurulum Yap**
   ```python
   # ── Segment Anything'i Git'ten kur ──
   !pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
   
   # ── Ek paketleri kur ──
   !pip install -q supervision pycocotools opencv-python-headless
   ```

4. **Yapıştırma (Paste)**
   - Notebook dosyasını (`SAM_Automated_Segmentation.ipynb`) 
   - Colab'in File menüsünden "Upload notebook" yap
   - Veya GitHub link'den doğrudan aç

5. **Görüntü Yükle**
   - Sol paneldeki Files sekmesine resim sürükle
   - Veya Cell 10'a Google Drive monte kodu ekle:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     IMAGE_PATH = "/content/drive/MyDrive/deneme.jpeg"
     ```

6. **Çalıştır**
   - Cell 2'den başlayarak hücreleri sırayla çalıştır
   - Veya `Ctrl+F9` (tüm hücreleri çalıştır)

**İlk çalıştırma süresi:** ~10–15 dakika (model indirme)  
**Sonraki çalıştırmalar:** ~30–60 saniye

---

## 2. Linux (Ubuntu/Debian)

### Sistem Gereksinimleri

```bash
# Python 3.8+ kontrol et
python3 --version

# pip güncelleyin
sudo apt update && sudo apt upgrade
sudo apt install python3-pip python3-venv build-essential

# GPU Drivers (NVIDIA GPU varsa)
nvidia-smi  # NVIDIA sürücüsü yüklü mü kontrol et
```

### CUDA Kurulumu (GPU için)

GPU kullanacaksanız:

```bash
# CUDA Toolkit 11.8 indir ve kur (Ubuntu 22.04 için)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --driver

# CUDA PATH'ı ekle (~/.bashrc veya ~/.zshrc'ye)
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
source ~/.bashrc
```

### Python Environment Kurulumu

```bash
# Proje dizinine git
cd ~/projects
mkdir sam-segmentation && cd sam-segmentation

# Virtual environment oluştur
python3.10 -m venv venv

# Aktif et
source venv/bin/activate

# pip güncelleyin
pip install --upgrade pip setuptools wheel

# PyTorch'u GPU desteği ile kur
# CUDA 11.8 için:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# SAM ve bağımlılıkları kur
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install supervision pycocotools opencv-python numpy matplotlib

# Kurulumu doğrula
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Notebook Kurulumu

```bash
# Jupyter kur
pip install jupyter jupyterlab

# Notebook'ı başlat
jupyter notebook SAM_Automated_Segmentation.ipynb

# veya JupyterLab
jupyter lab SAM_Automated_Segmentation.ipynb
```

### Hızlı Komut Seti

```bash
#!/bin/bash
# setup.sh — Otomatik kurulum script'i

PYTHON_VER=3.10
VENV_NAME=venv_sam

# Virtual env oluştur
python${PYTHON_VER} -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

# PyTorch (CUDA 11.8)
pip install --upgrade pip
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# SAM ve paketler
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install supervision pycocotools opencv-python
pip install jupyter jupyterlab

echo "✅ Kurulum tamamlandı! Başlamak için:"
echo "   source ${VENV_NAME}/bin/activate"
echo "   jupyter lab"
```

Kullan:
```bash
chmod +x setup.sh
./setup.sh
```

---

## 3. macOS

### Apple Silicon (M1/M2/M3) vs Intel

**Apple Silicon (M1+)**: PyTorch Metal Acceleration otomatik olarak kullanılır ✅  
**Intel Mac**: CPU modunda çalışır (yavaş)

### Kurulum

```bash
# Homebrew'den Python 3.10+ kur
brew install python@3.10

# Virtual env oluştur
python3.10 -m venv venv_sam
source venv_sam/bin/activate

# pip güncelleyin
pip install --upgrade pip

# PyTorch (Apple Silicon otomatik algılanır)
pip install torch torchvision torchaudio

# SAM ve paketler
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install supervision pycocotools opencv-python matplotlib numpy

# Jupyter
pip install jupyter

# Başlat
jupyter notebook SAM_Automated_Segmentation.ipynb
```

### M1/M2 Optimizasyon

```python
# Notebook'ta:
import torch
print(torch.backends.mps.is_available())  # True olmalı

# GPU kullan (otomatik)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Troubleshooting for M1/M2

```bash
# Eğer PyTorch yüklenmezse:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# veya Conda kullan:
conda install -c pytorch::pytorch::pytorch-cpu
```

---

## 4. Windows 10/11

### Seçenek A: WSL2 + Linux (Önerilen)

Windows Subsystem for Linux kullanmak daha kolay:

```powershell
# PowerShell'de WSL2 kur (Administrator)
wsl --install -d Ubuntu-22.04

# WSL2'ye giriş
wsl

# Ardından Linux bölümündeki kurulum adımlarını takip et
```

### Seçenek B: Native Windows

#### Prerequisites

```powershell
# Python 3.10+ indir ve kur
# https://www.python.org/downloads/windows/

# Git kur (Git Bash için)
# https://git-scm.com/download/win

# Visual Studio Build Tools (isteğe bağlı)
# https://visualstudio.microsoft.com/downloads/
```

#### Virtual Environment

```powershell
# PowerShell'i Administrator olarak aç

# Direktöri oluştur
mkdir C:\projects\sam-segmentation
cd C:\projects\sam-segmentation

# Virtual env oluştur
python -m venv venv

# Aktif et
.\venv\Scripts\Activate.ps1

# pip güncelleyin
python -m pip install --upgrade pip setuptools wheel
```

#### PyTorch & SAM Kurulumu

```powershell
# PyTorch (CPU veya CUDA 11.8)
# GPU varsa (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Yoksa CPU:
pip install torch torchvision torchaudio

# SAM ve paketler
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install supervision pycocotools opencv-python matplotlib numpy

# Jupyter
pip install jupyter jupyterlab

# Başlat
jupyter notebook SAM_Automated_Segmentation.ipynb
```

#### Sorun: Git Komutu Bulunamadı

```powershell
# Visual Studio Code ve Git Bash kulllan veya:
# PyCharm'da terminal olarak Git Bash seç

# Veya direkt pip kullan:
pip install segment-anything  # Yerine git+ komutu
```

---

## 5. Docker

### Dockerfile

```dockerfile
# Dockerfile

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/segment-anything.git' \
    supervision \
    pycocotools \
    opencv-python \
    jupyter \
    numpy \
    matplotlib

# Jupyter Notebook portunu aç
EXPOSE 8888

# Notebook'u başlat
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

### Docker Kurulumu & Çalıştırma

```bash
# Docker image'i build et
docker build -t sam-segmentation .

# Container'ı çalıştır
docker run --gpus all -p 8888:8888 \
  -v $(pwd):/app \
  sam-segmentation

# Logs'da şu tür bir URL görünecek:
# http://127.0.0.1:8888/?token=abc123...
# Bunu browser'da aç
```

### Docker Compose (Tercihen)

```yaml
# docker-compose.yml

version: '3.8'

services:
  sam-notebook:
    build: .
    image: sam-segmentation:latest
    container_name: sam-notebook
    
    ports:
      - "8888:8888"
    
    volumes:
      - ./data:/app/data
      - ./weights:/app/weights
      - ./notebooks:/app/notebooks
    
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - CUDA_VISIBLE_DEVICES=0
    
    runtime: nvidia
    
    command: >
      jupyter lab 
      --ip=0.0.0.0 
      --no-browser 
      --allow-root
```

Çalıştırma:
```bash
docker-compose up -d

# Loglar
docker-compose logs -f

# Durdur
docker-compose down
```

---

## 🔍 Sorun Giderme

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Çözüm:**
```bash
# Virtual env'i kontrol et
which python
pip list | grep torch

# Yeniden kur
pip install --force-reinstall torch torchvision torchaudio
```

### Problem: "CUDA out of memory"

**Çözümler:**
```python
# 1) Modeli küçült
MODEL_TYPE = "vit_b"  # vit_h yerine

# 2) GPU belleğini boşalt
import torch
del model
torch.cuda.empty_cache()

# 3) İmage çözünürlüğünü azalt
image_small = cv2.resize(image, (640, 480))
```

### Problem: "No GPU found / CUDA not available"

```bash
# CUDA durumunu kontrol et
nvidia-smi

# PyTorch GPU desteğini kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# GPU driver'ı yüklü mü?
lspci | grep -i nvidia

# CUDA Toolkit'i yeniden kur (Linux)
# https://developer.nvidia.com/cuda-11-8-0-download-archive
```

### Problem: "Permission denied" (Linux)

```bash
# Setup script'e execute izni ver
chmod +x setup.sh

# Colab'de şu hata varsa:
# "You do not have permission to access this resource"
# → Colab'i yenile (F5)
```

### Problem: Jupyter kernel dead/crash

```bash
# Kernel'i yeniden başlat
# Jupyter UI: Kernel → Restart Kernel

# Veya command line'da:
jupyter kernelspec list
jupyter kernelspec uninstall python3
python -m ipykernel install --user
```

### Problem: "ImportError: libGL.so.1"

Linux'ta (headless ortam):
```bash
# Çözüm:
pip install opencv-python-headless  # opencv-python yerine

# Veya sistem libs:
apt-get install libgl1-mesa-glx libglib2.0-0
```

---

## ✅ Kurulum Doğrulama

Kurulumun başarılı olup olmadığını test et:

```python
# test_installation.py

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from segment_anything import sam_model_registry
print("✅ SAM imported successfully")

import supervision as sv
print("✅ Supervision imported successfully")

import cv2
print("✅ OpenCV imported successfully")

print("\n✅ Installation successful!")
```

Çalıştır:
```bash
python test_installation.py
```

**Beklenen Output:**
```
Python: 3.10.12 (main, ...)
PyTorch: 2.1.0
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
✅ SAM imported successfully
✅ Supervision imported successfully
✅ OpenCV imported successfully

✅ Installation successful!
```

---

## 📊 Kurulum Özeti

| Ortam | Sürü | Zorluk | Not |
|-------|------|--------|-----|
| **Colab** | ⚡ 5 dk | ★☆☆ | Önerilen |
| **Linux** | ⏱️ 15 dk | ★★☆ | Sorunsuz |
| **macOS (Silicon)** | ⏱️ 10 dk | ★★☆ | GPU otomatik |
| **Windows WSL2** | ⏱️ 20 dk | ★★★ | GPU kurulumu karmaşık |
| **Docker** | ⏱️ 30 dk | ★★★ | Prod ortamları için |

---

## 📞 Yardım & Destek

Kurulum sorunları yaşıyorsanız:

1. **İlk Adım:** Bu rehberde sorunu ara
2. **GitHub Issues:** Detaylı hata mesajı ve ortam bilgisiyle issue aç
3. **Stack Overflow:** Tag: `pytorch`, `segment-anything`, `cuda`

---

**Son Güncellenme:** Mayıs 2026  
**Versiyon:** 1.0

Happy Segmenting! 
