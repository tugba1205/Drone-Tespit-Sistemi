from ultralytics import YOLO


# YOLO modeli yükleniyor
model = YOLO("best.pt")


def yolo_detect(image):

    results = model(image)

    for r in results:
        if len(r.boxes) > 0:
            confidence = r.boxes.conf.max().item() * 100
            return f"[YOLO] DRONE TESPİT EDİLDİ! (Doğruluk: %{confidence:.1f})"


    return "[YOLO] Drone tespit edilemedi."