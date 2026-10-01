import cv2
from ultralytics import YOLO

model = YOLO(r"F:\vehicle_detection\runs\detect\train_960-2\weights\best.pt")
cap = cv2.VideoCapture(r"F:\vehicle_detection\vehicle_detection video.mp4")

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
