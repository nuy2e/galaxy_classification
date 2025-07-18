"""
Train/Validation/Test Splitter for Galaxy Zoo Dataset

This script splits image data from pre-sorted class folders into training, validation,
and test datasets according to user-defined ratios. It supports recursive file collection,
ensures reproducible splits via a fixed random seed, and maintains class subfolder structure
within each output split.

Functionality:
- Recursively collects all image files from each class directory under the `sorted_data` folder.
- Randomly shuffles the images and splits them into train/val/test according to specified ratios.
- Copies the images to corresponding `train_data`, `val_data`, and `test_data` folders, preserving
  the original class label and any internal subfolder structure.
- Automatically clears previous contents in output directories before splitting.

Configuration:
- `USE_DEFAULT_PATHS`: Set to `False` to use custom paths.
- `train_ratio`, `val_ratio`, `test_ratio`: Must sum to 1.0.
- Default input: `sorted_data/`
- Default outputs: `dataset/train_data/`, `dataset/val_data/`, `dataset/test_data/`

Requirements:
- Directory structure under `sorted_data/` must be organised by class labels (e.g., `sorted_data/E/`).
- Supports `.jpg`, `.jpeg`, and `.png` image formats.
- Some subdirectory nesting is allowed within each class folder.

Dependencies:
- os
- shutil
- random
"""


import os
import shutil
import random

# Set source and destination paths
USE_DEFAULT_PATHS = True

if USE_DEFAULT_PATHS:
    source_dir = os.path.join(".", "sorted_data")
    train_dir = os.path.join(".", "dataset", "train_data")
    val_dir = os.path.join(".", "dataset", "val_data")
    test_dir = os.path.join(".", "dataset", "test_data")
else:
    source_dir = r"path\to\source"
    train_dir = r"path\to\train data"
    val_dir = r"path\to\val data"
    test_dir = r"path\to\test data"

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Seed
random.seed(42)

# Optional: clear previous contents
for dir_path in [train_dir, val_dir, test_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Process each class subfolder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Collect all image files (recursively)
    images = []
    for root, _, files in os.walk(class_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, f)
                images.append(full_path)

    # Shuffle and split
    random.shuffle(images)
    total = len(images)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # ensures all images are used

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    print(f"{class_name}: total={total}, train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    # Copy files with preserved subfolder structure
    for dataset_images, base_dir in [
        (train_images, train_dir),
        (val_images, val_dir),
        (test_images, test_dir)
    ]:
        for img in dataset_images:
            rel_path = os.path.relpath(img, class_path)  # e.g. SNB/image.jpg
            dest_path = os.path.join(base_dir, class_name, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(img, dest_path)

print("✅ Dataset split into train / val / test completed.")
