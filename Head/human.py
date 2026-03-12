import os
from ultralytics import YOLO
import cv2

# Paths
car_folder = "Z:\\img"  # folder with car images
human_folder = "Z:\\person"    # folder to save cropped humans
path_Model = "Z:\\ML_Models\\yolov8n.pt"
os.makedirs(human_folder, exist_ok=True)

# Load YOLO model pretrained on COCO (detects person class)
model = YOLO(path_Model)  # you can replace with better model if you want

# COCO class id for person is 0
PERSON_CLASS_ID = 0

for img_name in os.listdir(car_folder):
    img_path = os.path.join(car_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}")
        continue

    results = model(img)

    # Collect detected persons
    persons = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes: [x1,y1,x2,y2]
        classes = r.boxes.cls.cpu().numpy().astype(int)  # class IDs
        for box, cls in zip(boxes, classes):
            if cls == PERSON_CLASS_ID:
                persons.append(box)

    if len(persons) == 0:
        print(f"No person detected in {img_name}")
        continue

    # Crop and save each detected person
    for idx, box in enumerate(persons):
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2]

        # Save with same name + _idx if multiple persons
        base_name, ext = os.path.splitext(img_name)
        if len(persons) == 1:
            save_name = f"{base_name}{ext}"
        else:
            save_name = f"{base_name}_{idx}{ext}"

        save_path = os.path.join(human_folder, save_name)
        cv2.imwrite(save_path, crop_img)
        print(f"Saved human crop {save_path}")
