from ultralytics import YOLO


# YOLO modeli yükleniyor
model = YOLO("yolov8n.pt")


def yolo_detect(image):

    results = model(image)

    detected_objects = []

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            detected_objects.append(label)

    if "drone" in detected_objects:
        return "[YOLO] DRONE TESPİT EDİLDİ!"
    else:
        return f"[YOLO] Tespit edilen nesneler: {detected_objects}"