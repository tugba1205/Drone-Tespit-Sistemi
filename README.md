# Siber Drone Tespit Sistemi (Real-Time Drone Detection)

Bu proje; kamera ve mikrofondan gelen anlık akışları (streaming) eş zamanlı olarak inceleyen, **YOLOv8x (Optik)** ve **MFCC / Makine Öğrenmesi (Akustik)** tabanlı siber savunma ve drone tespit prototipidir. 

Gradio v6 mimarisi üzerine kurulu modern, karanlık siber arayüzü sayesinde tehditleri anlık olarak Füzyon Karar Merkezi'nde raporlar.

---

## Kurulum Adımları

Sistemi yerel bilgisayarınızda (localhost) çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Projeyi Klonlayın veya İndirin
Öncelikle bu depoyu bilgisayarınıza indirin ve bir kod editöründe (tercihen PyCharm veya VS Code) açın.

### 2. Gerekli Kütüphaneleri Kurun
Projenizin ihtiyaç duyduğu yapay zeka, ses işleme ve arayüz kütüphanelerini yüklemek için terminali açın ve şu komutu çalıştırın:

```bash
pip install ultralytics gradio opencv-python librosa joblib
