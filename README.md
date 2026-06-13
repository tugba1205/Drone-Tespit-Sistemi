# Siber Drone Tespit Sistemi (Real-Time Drone Detection)

Bu proje; kamera ve mikrofondan gelen anlık akışları (streaming) eş zamanlı olarak inceleyen, **YOLOv8x (Optik)** ve **MFCC / Makine Öğrenmesi (Akustik)** tabanlı siber savunma ve drone tespit prototipidir. 

Gradio v6 mimarisi üzerine kurulu modern, karanlık siber arayüzü sayesinde tehditleri anlık olarak Füzyon Karar Merkezi'nde raporlar.

---

## Kurulum Adımları

Sistemi yerel bilgisayarınızda (localhost) çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Projeyi Klonlayın veya İndirin
Öncelikle bu depoyu bilgisayarınıza PyCharm üzerinden klonlayın ve açın.

### 3. Gerekli Kütüphaneleri ve Modeli Kurun
Projenizin ihtiyaç duyduğu yapay zeka, ses işleme ve arayüz kütüphanelerini yüklemek için terminali açın ve aşağıdaki yöntemlerden birini kullanın:

**Yöntem 1 — `requirements.txt` ile toplu kurulum (önerilen):**

```bash
pip install -r requirements.txt
```

**Yöntem 2 — Tek tek kurulum:**

```bash
pip install gradio numpy opencv-python ultralytics librosa joblib scikit-learn
```

> **Not:** `requirements.txt` dosyası projenin kök dizininde yer almaktadır ve tüm bağımlılıkları içerir.

> **ÖNEMLİ:** Projenin çalışması için önemli olan pre-trained açık kaynak `best.pt` modelini [Hugging Face Repository](https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x) bağlantısı üzerinden indirin ve projenize ekleyin. Bunu yapmaz iseniz main.py dosyasının çalıştırılması durumunda FileNotFound hatası alırsınız.

### 4. Projeyi Çalıştırın
Tüm kurulumlar tamamlandıktan sonra uygulamayı başlatmak için:

> Eğer projeyi kurduktan sonra ilk kez başlatacaksanız `train_audio_model.py` dosyasını çalıştırın ve proje dosyalarınızın içinde `drone_audio_model.psl` scriptinin oluştuğundan emin olun.

> Projenizde `drone_audio_model.psl` varlığından eminseniz, `main.py` dosyasını başlatabilirsiniz.

Uygulama başlatıldığında tarayıcınızda Gradio arayüzü otomatik olarak açılacaktır. Açılmazsa terminalde görünen `http://localhost:xxxx` adresini tarayıcınıza yapıştırın.

---

# Kurulum Videosu
<p align="left">
  <a href="https://www.youtube.com/watch?v=CaioE9i2KdQm">
    <img alt="Kurulum Videosu" width="70%">
  </a>
</p>


---

# Yazılım Gereksinimleri

## İşletim Sistemi:
Windows 10/11 (64-bit), macOS (Monterey ve üzeri) veya Linux (Ubuntu 20.04/22.04 LTS).

## Geliştirme Ortamı (IDE):
PyCharm veya Visual Studio Code

## Önerilen Python Sürümü:
Python 3.13 veya üstü

# Minimum Donanım Gereksinimleri

* **İşlemci (CPU)**: Intel Core i5 veya AMD Ryzen 5 serisi.

* **Bellek (RAM)**: 8 GB RAM.

* **Ekran Kartı (GPU)**: Paylaşımlı dahili grafik kartı (Intel HD Graphics / AMD Radeon Vega). İşlemler tamamen CPU'ya yüklenir.

* **Optik Sensör**: Bilgisayarın dahili entegre web kamerası.

* **Akustik Sensör**: Bilgisayarın dahili mikrofonu.

---

## Kullanılan Teknolojiler ve Bağımlılıklar

### Yapay Zeka ve Makine Öğrenmesi

* **Ultralytics (YOLOv8x)** : Gerçek zamanlı nesne tespiti için kullanılan derin öğrenme modeli. Kamera akışından drone tespiti yapar.

* **scikit-learn** : Akustik analiz için `RandomForestClassifier` modeli eğitimi ve tahmin işlemlerinde kullanılır.

* **Joblib** : Eğitilmiş ML modellerinin (`.pkl`) diske kaydedilmesi ve yüklenmesi için kullanılır.

### Ses ve Görüntü İşleme

* **Librosa** : Ses sinyallerinden MFCC (Mel-Frequency Cepstral Coefficients) öznitelik çıkarımı için kullanılır.

* **OpenCV (cv2)** : Kamera görüntüsü işleme, renk dönüşümleri ve parlaklık analizi için kullanılır.

* **NumPy** : Sayısal hesaplamalar, sinyal enerjisi analizi ve matris işlemleri için temel bağımlılıktır.

### Arayüz

* **Gradio (v6)** : Web tabanlı gerçek zamanlı kullanıcı arayüzü. Kamera ve mikrofon streaming desteği sağlar.

### Proje Modül Yapısı

* **main.py** : Ana kontrol merkezi — Gradio arayüzü ve füzyon karar mekanizması

* **yolo_module.py** : YOLOv8x ile optik drone tespiti

* **mfcc_module.py** : MFCC öznitelik çıkarımı ve ML tabanlı akustik analiz

* **audio_module.py** : Ham ses sinyali ön işleme (enerji hesaplama, örnekleme frekansı)

* **camera_module.py** : Kamera görüntüsü ön işleme (çözünürlük, parlaklık analizi)

* **train_audio_model.py** : Akustik ML modelini eğitip `drone_audio_model.pkl` olarak kaydeder

---

## Kullanılan Model ve Ağırlıklar (Pre-trained Weights)

Bu projede drone tespiti için YOLOv8x mimarisi üzerine eğitilmiş hazır model ağırlıkları kullanılmıştır. GitHub dosya boyutu sınırlarından dolayı modelin `.pt` dosyası bu depoya (repository) dahil edilmemiştir.

### Model Nasıl Temin Edilir?
Projeyi yerelinizde çalıştırmadan önce model ağırlıklarını indirmeniz gerekmektedir:

1. Modeli doğrudan [Hugging Face Repository](https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x) üzerinden indirin.
2. İndirdiğiniz `best.pt` model dosyasını projenin **kök dizinine** (yani `main.py`, `yolo_module.py` gibi dosyaların bulunduğu klasöre) yerleştirin.

---

## Referanslar ve Teşekkür (Acknowledgments & Citations)

Projemizde yer alan nesne algılama (drone tespiti) modülünde, **Doğu İlmak** tarafından geliştirilen veri seti ve eğitilmiş YOLOv8x modeli kullanılmıştır. Geliştiriciye açık kaynak paylaşımları için teşekkür ederiz.

* **Modelin Orijinal GitHub Deposu:** [doguilmak/Drone-Detection-YOLOv8x](https://github.com/doguilmak/Drone-Detection-YOLOv8x)
* **Modelin Hugging Face Sayfası:** [doguilmak/Drone-Detection-YOLOv8x on Hugging Face](https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x)

---
