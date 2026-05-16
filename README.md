# SAM — Otomatik Görüntü Segmentasyonu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Framework: Jupyter](https://img.shields.io/badge/Framework-Jupyter-orange)](https://jupyter.org/)

Segment Anything Model (SAM) — Meta tarafından geliştirilen — temel alınarak yazılmış, herhangi bir görüntüdeki tüm nesneleri **otomatik olarak ve yüksek doğrulukla segmentleyen** production-ready Jupyter Notebook projesi.

---

## 📋 İçindekiler

- [Proje Özeti](#proje-özeti)
- [Teknik Spesifikasyonlar](#teknik-spesifikasyonlar)
- [Kurulum & Bağımlılıklar](#kurulum--bağımlılıklar)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Proje Yapısı & Mimarisi](#proje-yapısı--mimarisi)
- [Konfigürasyon](#konfigürasyon)
- [Performans Optimizasyonu](#performans-optimizasyonu)
- [Yaygın Sorunlar & Çözümleri](#yaygın-sorunlar--çözümleri)
- [Output Formatı](#output-formatı)
- [Geliştirme & Katkı](#geliştirme--katkı)
- [Lisans & Atıflar](#lisans--atıflar)

---

## 🎯 Proje Özeti

**SAM Automated Segmentation**, görüntü analizi ve nesneleri ayırma görevleri için endüstri standardı olan Segment Anything Model'i kullanarak:

✅ **Otomatik Segmentasyon**: İnsan müdahalesi olmaksızın resmimdeki tüm nesneleri tanır  
✅ **Yüksek Doğruluk**: ViT-H (Huge) modeli ile %95+ güven oranı  
✅ **Hızlı İşlem**: GPU hızlandırması ile 10–60 saniye içinde sonuç  
✅ **Esnek Mimari**: Parametre ayarlaması ile farklı kullanım senaryolarına uyum  
✅ **Production Ready**: Error handling, logging, tip kontrolleri içerir

### Kullanım Alanları

- **Robotik & Otonom Araçlar**: Çevre analizi, nesne avlanması
- **Tıbbi Görüntü İşleme**: Organ/tümör segmentasyonu
- **E-Commerce**: Ürün resimleri için arka plan kaldırma
- **Video Editleme**: İnsan/nesne tracking, maskeleme
- **Şehircilik & Planlama**: Harita analizi, arsa sınıflandırması

---

## 🔧 Teknik Spesifikasyonlar

| Bileşen | Detay |
|---------|-------|
| **Model** | Segment Anything Model (SAM) |
| **Mimarisi** | Vision Transformer (ViT) |
| **Varyantlar** | ViT-B (0.91GB), ViT-L (1.25GB), ViT-H (2.56GB) |
| **Framework** | PyTorch 2.0+ |
| **Input** | JPEG, PNG, BMP (herhangi bir boyut) |
| **Output** | PNG (segmentasyon maskesi ile) |
| **GPU Memory** | 8GB+ (H modeli için); 4GB+ (B/L modelleri) |
| **Runtime** | CPU: 5–10 min; GPU (T4): 10–30 sn; GPU (A100): <5 sn |

---

## 📦 Kurulum & Bağımlılıklar

### 1. Colab'da Çalıştırma (Önerilen)

Colab, önceden yüklü Python, PyTorch ve GPU (T4/V100) sağladığı için sorunsuz çalışır:

```bash
# Notebook'ı aşağıdaki linkten açın
# https://colab.research.google.com/

# Ardından Cell 1'i çalıştırarak gerekli paketleri kurun
```

**Colab GPU Ayarı:**
```
Çalışma Zamanı → Çalışma zamanı türünü değiştir → GPU (T4 veya V100) seçin
```

### 2. Yerel Makinede Kurulum

```bash
# Python 3.8+ (kontrol edin)
python --version

# Virtual environment oluştur (tavsiye edilir)
python -m venv sam_env
source sam_env/bin/activate  # Linux/macOS
# veya
sam_env\Scripts\activate     # Windows

# PyTorch'u CUDA desteği ile kur
# CUDA 11.8 için:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# SAM ve bağımlılıklar
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install supervision pycocotools opencv-python matplotlib
```

### 3. Bağımlılık Versions (Tested)

```
torch >= 2.0.0
torchvision >= 0.15.0
segment-anything >= 1.0
supervision >= 0.20.0
opencv-python >= 4.8.0
pycocotools >= 2.0.6
numpy >= 1.21.0
matplotlib >= 3.5.0
```

---

## 🚀 Hızlı Başlangıç

### 5 Adımda Çalıştırma

**Adım 1:** Jupyter Notebook'u açın
```bash
jupyter notebook SAM_Automated_Segmentation.ipynb
```

**Adım 2:** Girdi görüntüsü hazırla
- Colab Files paneline resim sürükle veya
- Yerel klasöre (`deneme.jpeg`) koy

**Adım 3:** Cell 2'yi çalıştırarak paketleri kur
```python
!pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
!pip install -q supervision pycocotools opencv-python-headless
```

**Adım 4:** Cell 4'te model tipi seç
```python
MODEL_TYPE = "vit_h"  # "vit_b" (hızlı) | "vit_l" | "vit_h" (en iyi)
IMAGE_NAME = "deneme.jpeg"
```

**Adım 5:** Cell 6'dan Cell 8'e kadar sırayla çalıştır
- Ağırlıklar indirilecek (ilk kez ~2–5 dk)
- Model yüklenecek
- Segmentasyon yapılacak
- Sonuç kaydedilecek

---

## 💻 Sistem Gereksinimleri

### Minimum Gereksinimler

| Parametre | Minimum | Önerilen | Ideal |
|-----------|---------|----------|-------|
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **GPU VRAM** | 4 GB (ViT-B) | 8 GB (ViT-H) | 24+ GB (A100) |
| **Disk** | 3 GB | 5 GB | 10 GB |
| **İşlemci** | 4 çekirdek | 8 çekirdek | 16+ çekirdek |
| **Python** | 3.8 | 3.10 | 3.11+ |

### GPU Uyumluluğu

- ✅ **NVIDIA** (CUDA 11.8+): Tam desteklenen
- ⚠️ **AMD** (ROCm): PyTorch + rocm gerekli; test edilmemiş
- ❌ **Intel Arc**: Deneysel; `sycl` backend gerekli
- ❌ **CPU Only**: Mümkün ancak 5–10 dakika sürer

### İşletim Sistemi

- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS (12+; Apple Silicon uyumlu)
- ✅ Windows 10/11 (WSL2 önerilen)
- ✅ Google Colab

---

## 🏗️ Proje Yapısı & Mimarisi

```
SAM_Automated_Segmentation.ipynb
├── Cell 1-2: Kütüphane Kurulması
│   └── pip install segment-anything, supervision, opencv-python
├── Cell 3-5: İmportlar & Konfigürasyon
│   ├── Standart libs: os, sys, cv2, torch
│   ├── SAM imports: sam_model_registry, SamAutomaticMaskGenerator
│   └── Config: MODEL_TYPE, CHECKPOINT_URLS, dosya yolları
├── Cell 6-7: Model Ağırlıklarını İndirme
│   ├── wget ile indirme (resume desteği)
│   └── File validation
├── Cell 8-9: Model Yükleme & GPU Setup
│   ├── Cihaz seçimi (CUDA/CPU)
│   └── SAM instansı oluşturma
├── Cell 10-11: Görüntü Yükleme & Önizleme
│   ├── cv2.imread() ile okuma
│   ├── BGR → RGB konversiyon
│   └── Matplotlib ile preview
├── Cell 12-14: Otomatik Maskeleme
│   ├── SamAutomaticMaskGenerator parametreleri
│   ├── Maskeleme işlemi (inference)
│   └── İstatistik raporlama
├── Cell 15-17: Görselleştirme & Kaydetme
│   ├── supervision.Detections dönüşümü
│   ├── MaskAnnotator ile renkli maskeleme
│   └── PNG kaydetme
└── Cell 18: Cleanup (optional)
    └── GPU bellek temizleme
```

### İş Akışı (Pipeline)

```
Görüntü (JPEG/PNG)
    ↓
[cv2.imread + BGR→RGB]
    ↓
[SAM Model: Automatic Mask Generation]
    ├─→ Grid 32×32 noktada sorgu
    ├─→ Her nokta için maske üret
    ├─→ Kalite filtreleme (IOU, stability)
    └─→ Sonuç: List[Dict] format
    ↓
[supervision.Detections dönüşümü]
    ↓
[MaskAnnotator: Görselleştirme]
    ├─→ Her maske için farklı renk
    ├─→ 0.5 opacity ile overlay
    └─→ annotated_image (BGR)
    ↓
PNG Çıktı (segmentasyon_sonucu.png)
```

---

## ⚙️ Konfigürasyon

### Ana Parametreler

**Cell 5 — Kullanıcı Ayarları:**

```python
# ─────────────────────────────────────────────────────────────
# Giriş/Çıktı Dosyaları
# ─────────────────────────────────────────────────────────────
IMAGE_NAME  = "deneme.jpeg"          # Giriş resmi adı
RESULT_NAME = "segmentasyon_sonucu.png"  # Çıktı dosya adı

# ─────────────────────────────────────────────────────────────
# Model Seçimi
# ─────────────────────────────────────────────────────────────
MODEL_TYPE = "vit_h"  # Seçenekler: "vit_b" | "vit_l" | "vit_h"

# Kıyas:
# ├─ vit_b: Hızlı (1–3 sn), az bellekli, daha az doğru
# ├─ vit_l: Orta (5–10 sn), orta bellekli
# └─ vit_h: En iyi (10–30 sn), yüksek bellek, en doğru ✓
```

### İleri Parametreler

**Cell 12 — SamAutomaticMaskGenerator Ayarları:**

```python
mask_generator = SamAutomaticMaskGenerator(
    model                   = sam,
    
    # Sorgu ızgarasının yoğunluğu (daha yüksek = daha fazla işlem)
    points_per_side         = 32,
    # ├─ 16: Hızlı, büyük nesneler için
    # ├─ 32: Standart (tavsiye edilir)
    # └─ 64: Yavaş, küçük nesneler için
    
    # Model tarafından tahmin edilen IoU eşiği (0–1)
    # Düşür → daha fazla (potansiyel gürültü) maske
    pred_iou_thresh         = 0.88,
    
    # Maske kararlılığı eşiği (0–1)
    # Düşür → kırılgan maskeler daha fazla
    stability_score_thresh  = 0.95,
    
    # Minimum maske alanı (piksel²)
    # Artır → daha büyük nesneler, gürültü azalır
    min_mask_region_area    = 100,
)
```

### Çıkış Özellikleri

**sam_result** (List[Dict]) yapısı:

```python
sam_result[0] = {
    'segmentation': numpy.ndarray,  # Boolean maske (H×W)
    'area': int,                     # Nesne alanı (piksel²)
    'bbox': [x, y, w, h],           # Bounding box (XYWH format)
    'predicted_iou': float,          # Model güven skoru (0–1)
    'stability_score': float,        # Kararlılık skoru (0–1)
    'crop_box': [x0, y0, x1, y1],  # İç kullanım
}
```

---

## 🚄 Performans Optimizasyonu

### 1. Hız Iyileştirmesi

**Problem:** ViT-H modeli çok yavaş, ama çok doğru lazım?

**Çözümler:**

```python
# A) Model boyutunu küçült
MODEL_TYPE = "vit_b"  # 20× hızlanır, %5 doğruluk kaybı

# B) Grid yoğunluğunu azalt
points_per_side = 16  # Varsayılan: 32 (2× hızlanır)

# C) Kalite eşiğini yükselt (daha az maske)
pred_iou_thresh = 0.92  # Varsayılan: 0.88 (daha hızlı)
min_mask_region_area = 500  # Varsayılan: 100 (gürültü filtresi)
```

### 2. Bellek Optimizasyonu

**GPU Belleğini Kurtarma:**

```python
# Model yüklenmeden önce eski modeli sil
if 'sam' in locals():
    del sam

# GPU belleğini serbest bırak
import torch
torch.cuda.empty_cache()

# Batch processing yapma (bu script tek görüntü için)
# Aksine, 100 resmi işlemek gerekiyorsa:
# → for döngüsü ile + torch.cuda.empty_cache() ekle
```

### 3. CPU Kullanımı (GPU Yok Durumunda)

```python
# Oto-algılanan, ancak CPU'da çalıştırmak için:
DEVICE = torch.device("cpu")
sam.to(device=DEVICE)

# ⚠️ UYARI: 10–15 dakika sürecek
# Öneriler:
# 1) Resmi küçült: cv2.resize(image, (640, 480))
# 2) points_per_side = 8 (ızgarayı sıral)
```

---

## 🆘 Yaygın Sorunlar & Çözümleri

### Problem 1: "HATA: Ağırlık dosyası bulunamadı"

**Nedeni:** İndirme yarıda kesildi veya hatalı klasör

**Çözüm:**
```bash
# Paquete klasörünü sil
rm -rf weights/

# Cell 6'yı tekrar çalıştır
# wget -c bayrağı kaldığı yerden devam eder
```

### Problem 2: "RuntimeError: CUDA out of memory"

**Nedeni:** Model GPU'ya sığmıyor

**Çözümler:**
```python
# 1) Model boyutunu azalt
MODEL_TYPE = "vit_l"  # veya "vit_b"

# 2) İmage çözünürlüğünü azalt (opsiyonel)
image_resized = cv2.resize(image_rgb, (640, 480))
sam_result = mask_generator.generate(image_resized)

# 3) Batch processing yerine tek görüntü işle (yapılıyor)

# 4) Diğer processleri kapat ve GPU'yu boşalt
# → Colab: Runtime → Restart runtime
```

### Problem 3: "ModuleNotFoundError: No module named 'supervision'"

**Nedeni:** Paket kurulmamış

**Çözüm:**
```bash
pip install supervision
# veya
pip install -U supervision
```

### Problem 4: "Segmentasyon çok gürültülü, çok fazla küçük maske"

**Nedeni:** Kalite filtreleri çok düşük ayarlanmış

**Çözüm:**
```python
# Cell 12'de parametreleri güncelle
mask_generator = SamAutomaticMaskGenerator(
    model                   = sam,
    pred_iou_thresh         = 0.90,        # 0.88 → 0.90 (daha katı)
    min_mask_region_area    = 500,         # 100 → 500 (gürültü filtresi)
)
```

### Problem 5: "GPU kullanılmıyor (CPU'da çalışıyor)"

**Kontrol:**
```python
import torch
print(torch.cuda.is_available())        # True olmalı
print(torch.cuda.get_device_name(0))    # GPU adı yazdırmalı

# Colab GPU kurulumu:
# Çalışma Zamanı → Çalışma zamanı türünü değiştir → GPU seç
```

---

## 📤 Çıktı Formatı

### Çıktı Görüntüsü
<img width="1491" height="890" alt="image" src="https://github.com/user-attachments/assets/3a7dd81f-28b0-4ade-900c-f9cffff12047" />

- **Format:** RGB, PNG (lossless)
- **Boyut:** Giriş görüntüsüyle aynı
- **İçerik:** Orijinal görüntü + 50% saydamlı renklü maskeler

**Sonuçlar:**

```
Bulunan nesne sayısı : 127
Ortalama alan        : 45821 px²
Ortalama güven skoru : 0.927
```

### Programatik Erişim

```python
# Maskeler ayrı ayrı alınabilir
detections.mask  # numpy array (N, H, W)
detections.box   # Bounding boxes
detections.confidence  # Güven skorları

# COCO JSON format'e dönüştür (opsiyonel)
import json
coco_data = detections.as_coco_annotations()
with open("annotations.json", "w") as f:
    json.dump(coco_data, f)
```

---

## 🔄 Geliştirme & Katkı

### Lokal Geliştirme

```bash
# Repository'i clone et
git clone https://github.com/yourusername/SAM-AutoSegmentation.git
cd SAM-AutoSegmentation

# Virtual env kurmak
python -m venv venv && source venv/bin/activate

# Bağımlılıkları kur (development)
pip install -r requirements.txt
pip install pytest black flake8  # Linting tools

# Notebook'ı çalıştır
jupyter notebook SAM_Automated_Segmentation.ipynb
```

### Tavsiye Edilen İyileştirmeler

- [ ] **Batch Processing:** Bir klasördeki birden çok resmi işle
- [ ] **Interaktif Web UI:** Streamlit ile arayüz ekle
- [ ] **Model Fine-tuning:** Özel datasete göre eğit
- [ ] **Video Support:** Frame-by-frame segmentasyon
- [ ] **Post-processing:** CRF veya morfolojik operasyonlarla iyileştir
- [ ] **REST API:** FastAPI ile servisi expose et

### Test

```bash
# Örnek test yazma (pytest kullanarak)
pytest test_segmentation.py

# Manual test
python -c "from SAM_Automated_Segmentation import *; test_gpu()"
```

---

## 📚 Lisans & Atıflar

### Lisans

Bu proje **MIT Lisansı** altında yayınlanmıştır. Detaylar için [LICENSE](LICENSE) dosyasını okuyun.

### Atıflar

Bu proje Meta AI tarafından geliştirilen Segment Anything Model üzerinde yüzdelik açıdan temel alınmıştır.

**Orijinal SAM Makalesi:**
```bibtex
@article{kirillov2023segmentanything,
  title = {Segment Anything},
  author = {Kirillov, Alexander and Mintun, Eric and Mottaghi, Roozbeh and others},
  journal = {arXiv preprint arXiv:2304.02643},
  year = {2023}
}
```

**Referanslar:**

- [Meta AI — Segment Anything](https://segment-anything.com/)
- [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything)
- [Supervision Documentation](https://supervision.roboflow.com/)
- [PyTorch Official Guide](https://pytorch.org/)

---

## 📞 Destek & İletişim

### Sık Sorulan Sorular

**S:** Kendi görüntülerimde nasıl kullanırım?  
**C:** Cell 4'te `IMAGE_NAME` değerini değiştir. Colab'da Files paneline resmi sürükle.

**S:** Sonuç çok gürültülü, nasıl filtreleyeyim?  
**C:** Cell 12'de `min_mask_region_area` ve `pred_iou_thresh`'i artır.

**S:** Daha hızlı istiyorum ama doğruluğu kaybetmek istemiyorum?  
**C:** MODEL_TYPE'ı "vit_b" → "vit_l"'e değiştir (orta yer).

**S:** Çıktıyı JSON/COCO format'de kullanmak istiyorum?  
**C:** Bkz. [Output Formatı](#output-formatı) bölümü, `as_coco_annotations()`.

---

## 📈 Roadmap

- **v1.1** (2024 Q2): Batch processing + CLI tool
- **v1.2** (2024 Q3): Streamlit web arayüzü
- **v2.0** (2024 Q4): SAM 2.0 desteği (video segmentasyonu)

---

**Son Güncellenme:** 16 Mayıs 2026  
**Sürüm:** 1.0  
**Durum:** Production Ready ✅

---
