import cv2
import numpy as np

def process_camera(image):
    if image is None:
        return "(Kamera) / Görüntü bekleniyor..."

    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    return f"(Optik Matris) / Çözünürlük: {width}x{height} | Parlaklık Seviyesi: {brightness:.1f}"
