import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
model_path = r"/Users/Abhinn/Downloads/detect 3/train_yolov11/weights/best.pt"
input_folder_path = r"/Users/Abhinn/Downloads/fire_smoke_inference"
output_folder_path = r"/Users/Abhinn/Downloads/fire_smoke_inference_outputs"
conf_thresh = 0.001

imgsz = 640

# Load Model
model = YOLO(model_path)

for image_name in os.listdir(input_folder_path):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder_path, image_name)
        output_image_path = os.path.join(output_folder_path, image_name)

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run YOLO
        results = model.predict(source=frame, conf=conf_thresh, imgsz=imgsz, verbose=True)

        # Show results
        results[0].show()

        # Save result image
        annotated_frame = results[0].plot()
        cv2.imwrite(output_image_path, annotated_frame)
        print("image saved")