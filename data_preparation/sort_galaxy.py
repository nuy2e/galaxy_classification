"""
Galaxy Zoo 2 Image Sorter

This script organises raw galaxy image files into class-based folders based on the
Galaxy Zoo 2 classifications. It merges the filename mapping CSV with the classification
CSV to associate each image with a morphological class label. Images are then copied
into folders named by category (e.g., 'E', 'S', etc.), where the category is extracted
as the first alphabetical token from the 'gz2_class' string.

Configuration:
- `USE_DEFAULT_PATHS`: Set to `False` to manually specify paths.
- Paths include the raw image folder, output folder, filename mapping CSV,
  and classification CSV.

Expected Input:
- Image files in `.jpg` format named with a prefix corresponding to Galaxy Zoo object IDs.
- `gz2_filename_mapping.csv` containing 'objid' and 'asset_id'.
- `gz2_hart16.csv.gz` containing 'dr7objid' and 'gz2_class'.

Output:
- Sorted images are copied to `sorted_data/<category>/` folders.
- A live progress counter is shown during processing.

Notes:
- Unmatched images (not found in the mapping) are copied to an `unmatched` folder.
- Existing images in the target location are skipped (not overwritten).
- Special characters in category names are replaced with underscores before folder creation.

Dependencies:
- pandas
- shutil
- os
- re
"""


import os
import shutil
import pandas as pd
import re

USE_DEFAULT_PATHS = True

if USE_DEFAULT_PATHS:
    image_dir = os.path.join(".", "raw_images")
    output_dir = os.path.join(".", "sorted_data")
    mapping_csv = os.path.join(".", "initial_raw_data", "gz2_filename_mapping.csv")
    classification_csv = os.path.join(".", "initial_raw_data", "gz2_hart16.csv.gz")
else:
    image_dir = r"path\to\images"
    output_dir = r"path\to\output"
    mapping_csv = r"path\to\gz2_filename_mapping.csv"
    classification_csv = r"path\to\gz2_hart16.csv.gz"

# Load CSVs
mapping_df = pd.read_csv(mapping_csv)  # contains 'objid', 'asset_id'
classification_df = pd.read_csv(classification_csv)  # contains 'dr7objid', 'gz2_class'

# Merge mapping and classification on objid
merged_df = pd.merge(mapping_df, classification_df, left_on="objid", right_on="dr7objid", how="inner")

# Create a dictionary for quick lookup
id_to_class = dict(zip(merged_df["asset_id"].astype(str), merged_df["gz2_class"]))

# Process images
count = 0
copied = 0
unmatched = 0
exist = 0

for filename in os.listdir(image_dir):
    if filename.lower().endswith(".jpg"):
        count += 1
        image_id = filename.split(".")[0].split("_")[0]

        if image_id in id_to_class:
            raw_class = id_to_class[image_id]

            #substitute inappropriate characters to underscore
            category = re.sub(r'[<>:"/\\|?*]', '_', raw_class)
            #Edit here to change the rule of classification
            category = re.split(r'[^a-zA-Z]', category)[0]
        else:
            category = "unmatched"
            unmatched += 1

        # Create output directory if needed
        dest_dir = os.path.join(output_dir, category)
        os.makedirs(dest_dir, exist_ok=True)

        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(dest_dir, filename)

        if os.path.exists(dst_path):
            exist += 1
        else:
            shutil.copy2(src_path, dst_path)
            copied += 1

        #Live counter
        print(f"\rProcessed: {count} | Copied: {copied} | Skipped: {exist} | Unmatched: {unmatched}", end="", flush=True)
