import cv2
import numpy as np

def process_camera(image):  # Görüntünün boyut ve parlaklık analizini yapar.

    if image is None:
        return "(Kamera Analizi) Görüntü bekleniyor..."

    # Görüntü boyutu
    height, width, _ = image.shape

    # Gri tonlama ve ortalama parlaklık
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    result = f"""[Kamera Temel Analizi]
    Boyut: {width}x{height}
    Ortalama Parlaklık: {brightness:.2f}
    """

    return result
