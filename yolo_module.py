import os
from ultralytics import YOLO

# Klasör yolunu garanti altına alıyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"\n[HATA] GitHub'dan indirilen özel drone modeli (best.pt) bulunamadı!\n"
        f"Lütfen indirdiğiniz dosyayı şu klasörün içine atın:\n{current_dir}"
    )

# İndirdiğimiz özel YOLOv8x drone modelini yüklüyoruz
model = YOLO(model_path)


def get_drone_class_ids(yolo_model):
    """
    Model içindeki sınıfları tarar. Özel eğitilmiş modellerde sadece
    'drone' sınıfı olacağı için eşleşen ID'yi dinamik yakalar.
    """
    drone_ids = []
    if hasattr(yolo_model, 'names'):
        for class_id, class_name in yolo_model.names.items():
            # Model içindeki isimlendirmeleri tarar (drone, uav vb.)
            if any(x in class_name.lower() for x in ["drone", "uav", "quadcopter"]):
                drone_ids.append(class_id)
    return drone_ids


DRONE_CLASS_IDS = get_drone_class_ids(model)


def yolo_detect(image): # Görüntüyü özel eğitilmiş YOLOv8x modeli ile tarayarak drone avlar.

    if image is None:
        return False, "[YOLO] Kamera görüntüsü alınamadı."

    # Model tahmini gerçekleştiriliyor
    results = model(image)

    for r in results:
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for idx, class_id in enumerate(classes):
                # Özel modelde drone kelimesi etiketlenmemiş olsa bile (DRONE_CLASS_IDS boşsa)
                # tespiti drone kabul et, çünkü model zaten sadece drone için eğitildi.
                is_drone = (class_id in DRONE_CLASS_IDS) or (not DRONE_CLASS_IDS)

                if is_drone:
                    confidence = confidences[idx] * 100

                    # Model etiket adını dinamik alıyoruz (Örn: DRONE)
                    try:
                        detected_label = model.names[int(class_id)].upper()
                    except:
                        detected_label = "DRONE"

                    text = f"[YOLO - {detected_label}] Tehdit Algılandı! (Güven: %{confidence:.1f})"
                    return True, text

    return False, "[YOLO] Drone izine rastlanmadı."
