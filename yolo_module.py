import os
import cv2
import numpy as np
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f" \n MODELNOTFOUNDERROR: Özel drone modeli (best.pt) bulunamadı! \n"
        f"Lütfen indirdiğiniz dosyayı şu klasörün içine atın:\n{current_dir}"
    )

# YOLOv8x modelini yükle
model = YOLO(model_path)


# ──────────────────────────────────────────────────────────────
# Sınıf İsimlerini ve ID'lerini Emniyete Alma
# ──────────────────────────────────────────────────────────────
def get_drone_class_ids(yolo_model):
    drone_ids = []

    # 1. Aşama: Modelin içindeki tüm sınıfları terminale yazdır (Teşhis için)
    if hasattr(yolo_model, 'names') and yolo_model.names:
        print("TESPİT EDİLEN YOLO MODEL SINIFLARI:")
        for cid, cname in yolo_model.names.items():
            print(f" ID {cid}: {cname}")

        # 2. Aşama: Türkçe ve İngilizce tüm olası drone/İHA etiketlerini tara
        olasi_kelimeler = ["drone", "uav", "quadcopter", "aircraft", "iha", "target", "aircraft", "uça", "helicopter"]
        for class_id, class_name in yolo_model.names.items():
            if any(x in class_name.lower() for x in olasi_kelimeler):
                drone_ids.append(class_id)

    return drone_ids


DRONE_CLASS_IDS = get_drone_class_ids(model)


# ──────────────────────────────────────────────────────────────
# Optimize Edilmiş Görüntü İşleme Motoru
# ──────────────────────────────────────────────────────────────
def yolo_detect(image):
    if image is None:
        return False, "(YOLO) / Kamera görüntüsü alınamadı."

    # Renk Uzayı Dönüşümü: Gradio (RGB) -> OpenCV/YOLO (BGR)
    # OBS'ten gelen renklerin kaymasını ve YOLO'nun körleşmesini engeller.
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        image_bgr = image  # Olası bir hata durumunda orijinal resmi koru

    # YOLO tahmini yürüt (Terminali kirletmemesi için verbose=False)
    results = model(image_bgr, verbose=False)

    drone_detected = False
    highest_confidence = 0.0
    detected_label = "DRONE"

    for r in results:
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for idx, class_id in enumerate(classes):
                # Emniyet Kilidi: Eğer model sınıfları boşsa veya eşleşme bulunamadıysa
                # hata vermemesi için her nesneyi taramaya izin ver, aksi takdirde filtrelenmiş ID'leri kullan.
                is_drone = (class_id in DRONE_CLASS_IDS) if DRONE_CLASS_IDS else True

                if is_drone:
                    drone_detected = True
                    if confidences[idx] > highest_confidence:
                        highest_confidence = confidences[idx]
                        try:
                            detected_label = model.names[int(class_id)].upper()
                        except:
                            pass

    if drone_detected:
        text = f"(YOLO - {detected_label}) / Drone algılandı! (Güven: %{highest_confidence * 100:.1f})"
        return True, text

    return False, "(YOLO) / Drone izine rastlanmadı."
