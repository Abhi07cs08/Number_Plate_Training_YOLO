# Number_Plate_Training_YOLO
YOLOv11-SimAM ANPR Inference Guide (Kaggle)
This repository demonstrates how to run inference using a customized YOLOv11 model trained for Automatic Number Plate Recognition (ANPR). The model includes architectural modifications such as GhostConv, SimAM attention, and a modified C3k2 detection head.

Folder Structure (Kaggle Environment)
Ensure the following directory structure is set up correctly in your Kaggle Notebook environment:


/kaggle
├── /input
│   ├── datasetruns/                 # Contains the trained model
│   │   └── detect/train2/weights/best.pt
│   ├── comparison-inputs/          # Folder of test images
│   │   └── comparison_inputs/*.jpg
│
├── /working
│   └── Outputs_Abhinn/             # Output directory for predicted images
Dependencies
Install the required Ultralytics YOLO package in the first notebook cell:


!pip install ultralytics --quiet
SimAM Definition and Registration
Because SimAM is a custom attention module, it must be defined and registered before loading the model:


import sys
import os
import torch
import torch.nn as nn

# Optional: Add custom repository path if using a local copy
# sys.path.insert(0, "F:/zygalnewpre/Abhinn/ultralytics_abhi")

# Define SimAM module
class SimAM(nn.Module):
    def __init__(self, channels, lambda_=1e-4):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        N, C, H, W = x.shape
        n = H * W - 1
        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean) ** 2
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4 * (v + self.lambda_)) + 0.5
        return x * torch.sigmoid(e_inv)

# Register SimAM with Ultralytics model parser
import ultralytics.nn.tasks as _tasks
_tasks.__dict__['SimAM'] = SimAM
Inference Script

from ultralytics import YOLO
import os

# Set input/output paths
input_folder = "/kaggle/input/comparison-inputs/comparison_inputs"
output_folder = "/kaggle/working/Outputs_Abhinn"
os.makedirs(output_folder, exist_ok=True)

# Load trained YOLOv11 model
model = YOLO("/kaggle/input/datasetruns/detect/train2/weights/best.pt")

# Run inference on all images
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    results = model(img_path, conf=0.15)
    results[0].save(f"{output_folder}/{filename}")
Notes
Ensure that the best.pt file being used for inference was trained with the modified YOLOv11 architecture containing SimAM.

If running the model locally, use sys.path.insert() to reference your local Ultralytics repository.

Adjust the confidence threshold (conf=0.15) as needed to reduce missed detections.

