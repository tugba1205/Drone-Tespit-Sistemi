import os
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f" \n MODELNOTFOUNDERROR: Özel drone modeli (best.pt) bulunamadı! \n Modeli indirmek için lütfen README'yi okuyunuz! "
        f"Lütfen indirdiğiniz dosyayı şu klasörün içine atın:\n{current_dir}"
    )

model = YOLO(model_path)


def get_drone_class_ids(yolo_model):
    drone_ids = []
    if hasattr(yolo_model, 'names'):
        for class_id, class_name in yolo_model.names.items():
            if any(x in class_name.lower() for x in ["drone", "uav", "quadcopter"]):
                drone_ids.append(class_id)
    return drone_ids


DRONE_CLASS_IDS = get_drone_class_ids(model)


def yolo_detect(image):
    if image is None:
        return False, "(YOLO) / Kamera görüntüsü alınamadı."

    results = model(image)

    for r in results:
        if len(r.boxes) > 0:
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for idx, class_id in enumerate(classes):
                is_drone = (class_id in DRONE_CLASS_IDS) or (not DRONE_CLASS_IDS)

                if is_drone:
                    confidence = confidences[idx] * 100
                    try:
                        detected_label = model.names[int(class_id)].upper()
                    except:
                        detected_label = "DRONE"

                    text = f"(YOLO - {detected_label}) / Tehdit algılandı! (Güven: %{confidence:.1f})"
                    return True, text

    return False, "(YOLO) / Bölge Temiz. Drone izine rastlanmadı."
