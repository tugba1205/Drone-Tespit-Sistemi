# Siber Drone Tespit Sistemi (Real-Time Drone Detection)

Bu proje; kamera ve mikrofondan gelen anlık verileri eş zamanlı olarak analiz eden, **YOLOv8x (Optik Analiz)** ve **MFCC + Makine Öğrenmesi (Akustik Analiz)** tabanlı bir drone tespit sistemidir.

Sistem, görüntü ve ses verilerini ayrı ayrı değerlendirerek elde edilen sonuçları birleştirir ve kullanıcıya gerçek zamanlı olarak sunar. Kullanıcı arayüzü Gradio v6 altyapısı kullanılarak geliştirilmiş olup modern ve kullanıcı dostu bir tasarıma sahiptir.

---

# Sistem Gereksinimleri

## Minimum Donanım Gereksinimleri
- Intel Core i5 veya AMD Ryzen 5
- 8 GB RAM
- En az 2 GB boş depolama alanı
- Dahili veya harici webcam
- Dahili veya harici mikrofon

## Yazılım Gereksinimleri
- Windows 10/11 (64-bit) / Ubuntu 20.04+ / macOS Monterey+
- Python 3.10 veya Python 3.11
- PyCharm veya Visual Studio Code

---

# Projeyi İndirme

## Yöntem 1: ZIP Olarak İndirme
1. GitHub proje sayfasını açın.
2. Sayfanın üst kısmındaki Code butonuna tıklayın.
3. Download ZIP seçeneğini seçin. 
4. İndirilen ZIP dosyasını bilgisayarınıza kaydedin.
5. ZIP dosyasını çıkartın.
6. Çıkartılan klasörü PyCharm veya Visual Studio Code ile açın.

## Yöntem 2: Git ile Klonlama

Bilgisayarınızda Git kuruluysa projeyi aşağıdaki komut ile klonlayabilirsiniz:

```bash
git clone https://github.com/tugba1205/Drone-Tespit-Sistemi
cd Drone-Tespit-Sistemi
```

---


# PyCharm ile Projeyi  Açma

## GitHub Üzerinden Klonlama
1. PyCharm uygulamasını açın.
2. Ana ekranda bulunan Get From VCS seçeneğine tıklayın.
3. GitHub depo bağlantısını ilgili alana yapıştırın.
4. Projeyi kaydetmek istediğiniz konumu seçin.
5. Clone butonuna tıklayın.
6. Proje otomatik olarak açılacaktır.

## ZIP Dosyasından Açma
1. PyCharm uygulamasını açın.
2. Open seçeneğine tıklayın.
3. ZIP dosyasından çıkarttığınız proje klasörünü seçin.
4. Proje açıldıktan sonra gerekli Python yorumlayıcısını (Interpreter) seçin.

---

# Visual Studio Code ile Açma

## ZIP Dosyasından Açma
1. Visual Studio Code'u açın.
2. File > Open Folder menüsünü seçin.
3. Proje klasörünü seçin.
4. Terminal açarak kurulum adımlarına devam edin.

## Git ile Klonlama

VS Code terminalinde aşağıdaki komutu çalıştırın:

```bash
git clone https://github.com/tugba1205/Drone-Tespit-Sistemi
cd Drone-Tespit-Sistemi
code .
```

---

# Python Kurulumu

Bu proje Python programlama dili ile geliştirilmiştir.

Bilgisayarınızda Python kurulu değilse aşağıdaki adımları uygulayın:

1. Python'u resmi web sitesinden indirin.
2. Kurulum sırasında Add Python to PATH seçeneğini işaretleyin.
3. Kurulumu tamamlayın.

Kurulumun başarılı olup olmadığını kontrol etmek için terminal açın ve aşağıdaki komutu çalıştırın:

```bash
python --version
```

veya

```bash
py --version
```
Ekranda Python sürümü görüntüleniyorsa kurulum başarılıdır.

---

# Kurulum Adımları

## 1. Sanal Ortam Oluşturma (Önerilen)

Bağımlılık çakışmalarını önlemek için sanal ortam oluşturulması önerilir.

```bash
python -m venv venv
```

Sanal ortamı etkinleştirin:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

## 2. Gerekli Kütüphaneleri Kurma

Terminali açın. Gerekli Kütüphaneleri indirmek için şu komutu girin:

```bash
pip install -r requirements.txt
```

Alternatif:

```bash
pip install gradio numpy opencv-python ultralytics librosa joblib scikit-learn
```


---

# Kullanılan Model ve Ağırlıklar

Bu projede drone tespiti için YOLOv8x modeli kullanılmaktadır.

**Model Nasıl İndirilir?**

1. Aşağıdaki bağlantıyı açın:

https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x

2. Repository içerisinden best.pt dosyasını indirin.
3. İndirdiğiniz dosyayı proje klasörüne yerleştirin.

**Örnek:**

```text
Drone-Tespit-Sistemi
│
├── main.py
├── best.pt
├── yolo_module.py
```

---

# Proje Klasör Yapısı

```text
Drone-Tespit-Sistemi
│
├── main.py
├── yolo_module.py
├── mfcc_module.py
├── audio_module.py
├── camera_module.py
├── train_audio_model.py
├── best.pt
├── yolov8n.pt
├── requirements.txt
├── drone_logo.png
└── README.md
```
---

# Ses modelinin oluşturulması

İlk kullanım öncesinde aşağıdaki komut çalıştırılmalıdır:

```bash
python train_audio_model.py
```

Bu işlem sonucunda **drone_audio_model.pkl** isimli makine öğrenmesi modeli oluşturulur ve proje klasörüne kaydedilir.

---

# Projeyi Çalıştırma

Tüm kurulumlar tamamlandıktan sonra:

```bash
python main.py
```
veya 

```bash
py main.py
```
komutunu çalıştırın.

Uygulama başlatıldıktan sonra Gradio arayüzü açılacaktır. Arayüz otomatik açılmazsa terminalde görünen localhost adresini tarayıcıya yapıştırın.

---
# Kurulum Videosu
<p align="left">
  <a href="https://www.youtube.com/watch?v=CaioE9i2KdQm">
    <img alt="Kurulum Videosu" width="70%">
  </a>
</p>

---

# Sistemin Kullanımı

1. Uygulamayı başlatın.
2. Tarayıcıda Gradio arayüzü açılacaktır.
3. Kamera erişimine izin verin.
4. Mikrofon erişimine izin verin.
5. Sistem gerçek zamanlı görüntü ve ses akışını analiz etmeye başlayacaktır.
6. Kamera modülü drone nesnelerini tespit eder.
7. Ses modülü drone seslerini analiz eder.
8. Elde edilen sonuçlar Füzyon Karar Merkezi tarafından değerlendirilir.
9. Sonuçlar kullanıcı arayüzünde anlık olarak gösterilir.
---

# Kullanılan Teknolojiler

## Yapay Zeka ve Makine Öğrenmesi
- Ultralytics YOLOv8x
- Scikit-Learn
- Joblib

## Ses ve Görüntü İşleme
- Librosa
- OpenCV
- NumPy

## Arayüz
- Gradio v6

---

# Proje Modülleri

- **main.py** → Ana kontrol merkezi
- **yolo_module.py** → Optik drone tespiti
- **mfcc_module.py** → Akustik analiz
- **audio_module.py** → Ses ön işleme
- **camera_module.py** → Görüntü ön işleme
- **train_audio_model.py** → Akustik model eğitimi

---

# Sık Karşılaşılan Hatalar

## best.pt bulunamadı

Çözüm: best.pt dosyasını proje klasörüne ekleyin.

## ModuleNotFoundError

```bash
pip install -r requirements.txt
```
komutunu çalıştırın.

## Kamera Açılmıyor
Tarayıcı izinlerini kontrol edin.

## Mikrofon Açılmıyor
İşletim sistemi ve tarayıcı izinlerini kontrol edin.

---

# Referanslar ve Teşekkür

Drone tespit modülünde kullanılan veri seti ve eğitilmiş YOLOv8x modeli için Doğu İlmak'a teşekkür ederiz.

GitHub:
https://github.com/doguilmak/Drone-Detection-YOLOv8x

Hugging Face:
https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x

---

# Önemli Not

Bu proje eğitim ve araştırma amaçlı geliştirilmiş bir prototiptir. Tespit sonuçları kullanılan modellerin doğruluğuna, görüntü kalitesine, ses kalitesine ve çevresel koşullara bağlı olarak değişiklik gösterebilir. Gerçek güvenlik sistemlerinde kullanılmadan önce kapsamlı testlerden geçirilmesi önerilir.
