# 🚗 Vehicle Detection System Using YOLO

A custom vehicle detection system built using **YOLO11, Python, and OpenCV**.

The system detects different types of vehicles in images and videos and was trained using the **BDD100K dataset**.

## 🚀 Features

- Real-time vehicle detection
- Video-based vehicle detection
- Detects 5 vehicle categories:
  - 🚗 Car
  - 🚚 Truck
  - 🚌 Bus
  - 🏍️ Motorcycle
  - 🚲 Bicycle
- Custom YOLO11n training
- OpenCV-based video processing
- GPU/CUDA accelerated training

## 🛠️ Technologies Used

- Python
- YOLO11 (Ultralytics)
- OpenCV
- BDD100K Dataset
- CUDA
- PyTorch

## 📊 Model Performance

The final model was trained at **960×960 image resolution**.

| Metric | Result |
|---|---:|
| mAP50 | **55.7%** |
| mAP50-95 | **35.6%** |
| Car mAP50 | **80.4%** |
| Truck mAP50 | **61.8%** |
| Bus mAP50 | **60.3%** |
| Motorcycle mAP50 | **34.5%** |
| Bicycle mAP50 | **41.5%** |

The model performs particularly well on car detection, while motorcycle and bicycle detection still have room for improvement.

## 📂 Dataset

The project uses the **BDD100K** dataset.

The original BDD100K annotations are provided in JSON format. They were converted into YOLO-compatible `.txt` annotation files.

The project uses the following classes:

```text
0 → car
1 → truck
2 → bus
3 → motor
4 → bike
