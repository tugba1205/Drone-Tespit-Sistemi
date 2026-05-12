import cv2
import numpy as np


def process_camera(image):

    # Görüntü boyutu
    height, width, _ = image.shape

    # Gri tonlama
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Ortalama parlaklık
    brightness = np.mean(gray)

    result = f""" [KAMERA ANALİZİ]
Boyut: {width}x{height}
Parlaklık: {brightness:.2f}
"""

    return result