import json
import os
import shutil

os.makedirs("vehicle_dataset/labels/train", exist_ok=True)
os.makedirs("vehicle_dataset/images/train", exist_ok=True)
os.makedirs("vehicle_dataset/labels/val", exist_ok=True)
os.makedirs("vehicle_dataset/images/val", exist_ok=True)

class_map = {"car": 0, "truck": 1, "bus": 2, "motor": 3, "bike": 4}

path = "archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"

with open(path, "r") as f:
    data = json.load(f)

train_image_root = "archive/bdd100k/bdd100k/images/100k/train"
image_paths = {}

for root, folders, files in os.walk(train_image_root):

    for file in files:
        image_paths[file] = os.path.join(root, file)

for image in data:
    image_name = image["name"]
    source_path = image_paths[image_name]
    destination_path = os.path.join("vehicle_dataset/images/train", image_name)
    shutil.copy2(source_path, destination_path)
    yolo_lines = []

    for label in image["labels"]:
        category = label["category"]

        if category in class_map:
            bounding_box = label["box2d"]
            class_id = class_map[category]
            width = (bounding_box["x2"] - bounding_box["x1"]) / 1280
            height = (bounding_box["y2"] - bounding_box["y1"]) / 720
            x_centre = (bounding_box["x2"] + bounding_box["x1"]) / 2560
            y_centre = (bounding_box["y2"] + bounding_box["y1"]) / 1440
            yolo_line = (
                f"{class_id} " f"{x_centre} " f"{y_centre} " f"{width} " f"{height}"
            )
            yolo_lines.append(yolo_line)

    label_filename = image_name.replace(".jpg", ".txt")
    label_path = os.path.join("vehicle_dataset/labels/train", label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

path = "archive/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

with open(path, "r") as f:
    data = json.load(f)

val_image_root = "archive/bdd100k/bdd100k/images/100k/val"
image_paths = {}

for root, folders, files in os.walk(val_image_root):
    for file in files:
        image_paths[file] = os.path.join(root, file)

for image in data:
    image_name = image["name"]
    source_path = image_paths[image_name]
    destination_path = os.path.join("vehicle_dataset/images/val", image_name)
    shutil.copy2(source_path, destination_path)
    yolo_lines = []

    for label in image["labels"]:
        category = label["category"]
        if category in class_map:
            bounding_box = label["box2d"]
            class_id = class_map[category]
            width = (bounding_box["x2"] - bounding_box["x1"]) / 1280
            height = (bounding_box["y2"] - bounding_box["y1"]) / 720
            x_centre = (bounding_box["x2"] + bounding_box["x1"]) / 2560
            y_centre = (bounding_box["y2"] + bounding_box["y1"]) / 1440
            yolo_line = (
                f"{class_id} " f"{x_centre} " f"{y_centre} " f"{width} " f"{height}"
            )
            yolo_lines.append(yolo_line)
    label_filename = image_name.replace(".jpg", ".txt")
    label_path = os.path.join("vehicle_dataset/labels/val", label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
