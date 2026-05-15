# 🎯 SAM Automated Segmentation

Meta'nın **Segment Anything Model (SAM)** kullanarak herhangi bir görüntüdeki nesneleri otomatik olarak tespit edip segmente eden bir Google Colab notebook'u.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![SAM](https://img.shields.io/badge/SAM-ViT--H-green) ![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange)

---

## 📌 Proje Hakkında

Bu proje, Meta Research tarafından yayınlanan SAM modelini kullanarak görüntülerdeki her nesneyi herhangi bir etiket, eğitim ya da ön bilgi gerektirmeden otomatik olarak segmente eder. Prompt vermene gerek yok — model resmin tamamını tarar ve bulduğu her nesne için renkli bir maske üretir.

### Ne Yapıyor?

- Verdiğin herhangi bir görseldeki tüm nesneleri otomatik olarak tespit eder
- Her nesneye rastgele renk atanmış maskeler oluşturur
- Sonucu görselleştirir ve diske kaydeder

---

## 🛠️ Gereksinimler

| Bileşen | Versiyon |
|---|---|
| Python | 3.8+ |
| PyTorch | 2.0+ (CUDA destekli önerilir) |
| segment-anything | Meta GitHub'dan |
| supervision | `pip install supervision` |
| OpenCV | `pip install opencv-python` |
| pycocotools | `pip install pycocotools` |

> **Not:** SAM ViT-H modeli yaklaşık **2.5 GB** model ağırlığı indirir. GPU olmadan çalışır ama son derece yavaş olur.

---

## 🚀 Kullanım

### 1. Notebook'u Aç

Google Colab üzerinde çalıştırmak için önerilir (ücretsiz GPU için Runtime > Change runtime type > T4 GPU seç).

### 2. Kütüphaneleri Kur

```bash
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install supervision pycocotools
```

### 3. Model Ağırlıklarını İndir

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 4. Görüntünü Yükle

Colab'in Files paneline `deneme.jpeg` adında bir görsel yükle (ya da kodda `IMAGE_NAME` değişkenini değiştir).

### 5. Hücreleri Sırayla Çalıştır

Her hücreyi sırayla çalıştır. Son adımda `segmentasyon_sonucu.jpg` dosyası çalışma dizinine kaydedilir.

---

## 📁 Proje Yapısı

```
├── SAM_Automated_Segmentation.ipynb   # Ana notebook
├── weights/
│   └── sam_vit_h_4b8939.pth           # İndirilen model ağırlıkları (gitignore'da!)
├── deneme.jpeg                         # Girdi görüntüsü (örnek)
└── segmentasyon_sonucu.jpg            # Çıktı görüntüsü
```

---

## 🧠 Kullanılan Model

**SAM ViT-H (Huge)** — SAM'in en büyük ve en doğru versiyonu.

| Model | Parametre | Dosya Boyutu |
|---|---|---|
| ViT-B (Base) | 91M | ~375 MB |
| ViT-L (Large) | 308M | ~1.2 GB |
| **ViT-H (Huge)** | **636M** | **~2.5 GB** |

---

## 📚 Referanslar

- [Segment Anything — Meta AI Research](https://github.com/facebookresearch/segment-anything)
- [SAM Paper (arXiv)](https://arxiv.org/abs/2304.02643)
- [Supervision by Roboflow](https://github.com/roboflow/supervision)

---

## ⚠️ Kısıtlamalar

- Büyük ve karmaşık görsellerde çok sayıda küçük maske üretebilir (gürültü)
- GPU olmadan çalıştırılması pratik değil (~30+ dakika/görsel)
- Sabit dosya adı (`deneme.jpeg`) beklentisi kullanıcı dostu değil
