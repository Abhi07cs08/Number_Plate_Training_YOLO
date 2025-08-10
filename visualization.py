import os
import cv2
from glob import glob

# FOLDERS 
image_folder = "/Users/Abhinn/Downloads/archive (1)/Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample"     # e.g., "./images"
label_folder = "/Users/Abhinn/Downloads/archive (1)/Annotations/Annotations" # e.g., "./labels"
output_folder = "/Users/Abhinn/Downloads/archive (1)/Annotations/Annotations/Visualization"
os.makedirs(output_folder, exist_ok=True)

# COLOR & CLASS NAMES
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # You can define more
class_names = ["fire", "smoke", "class_2"]  # Optional

# PROCESS EACH IMAGE
for img_path in glob(os.path.join(image_folder, "*.png")) + glob(os.path.join(image_folder, "*.jpg")):
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    filename = os.path.basename(img_path)
    label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"No label for {filename}, skipping.")
        continue

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping invalid line: {line.strip()}")
                print(label_path)
                continue

            class_id, x_center, y_center, w, h = map(float, line.strip().split())

            # Convert YOLO format to pixel coordinates
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Draw rectangle
            color = colors[int(class_id) % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw class label
            label = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save image with boxes
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, image)

print("Bounding boxes drawn and saved!")
