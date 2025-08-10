import os
import random
import shutil

source_dir = "/Users/Abhinn/Downloads/archive/valid/images"
output_dir = "/Users/Abhinn/Downloads/archive/reduced_valid"

# Create output folders
os.makedirs(output_dir, exist_ok=True)

# Get all .jpg files in source directory
image_files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]

# Randomly select 700 images
selected_images = random.sample(image_files, 700)

# Copy image and label files
for img_file in selected_images:
    base_name = os.path.splitext(img_file)[0]
    label_file = f"{base_name}.txt"

    # Source paths
    img_src = os.path.join(source_dir, img_file)
    label_src = os.path.join(source_dir, label_file)

    # Destination paths
    img_dst = os.path.join(output_dir, img_file)
    label_dst = os.path.join(output_dir, label_file)

    # Copy both
    shutil.copy(img_src, img_dst)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"Warning: Missing label for {img_file}")
