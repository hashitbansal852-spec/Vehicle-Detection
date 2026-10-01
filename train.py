from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11n.pt")

    model.train(
        data="vehicle_dataset/data.yaml",
        epochs=50,
        imgsz=416,
        batch=8,
        device=0
    )
