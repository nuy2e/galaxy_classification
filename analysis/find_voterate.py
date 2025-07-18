"""
Galaxy Zoo 2 Vote Rate Extractor

This script extracts vote rates for a particular morphological class (e.g. "smooth")
from the Galaxy Zoo 2 classification dataset and matches them to image files present
in a specified directory. It outputs a CSV containing the vote fraction, weighted
fraction, and debiased vote for each matched image ID.

Expected inputs:
- `gz2_filename_mapping.csv` with Galaxy Zoo object IDs and asset IDs
- `gz2_hart16.csv.gz` with vote fraction data for each object
- Raw images whose filenames include asset IDs

Output:
- `vote_rate.csv` with selected vote metrics for matched images

Dependencies: pandas, os
"""

import os
import pandas as pd

# Paths
USE_DEFAULT_PATHS = True

if USE_DEFAULT_PATHS:
    image_dir = os.path.join("..", "data_preparation", "raw_images")
    mapping_csv = os.path.join("..", "data_preparation", "initial_raw_data", "gz2_filename_mapping.csv")
    classification_csv = os.path.join("..", "data_preparation", "initial_raw_data", "gz2_hart16.csv.gz")
    output_csv = os.path.join(".", "vote_rate.csv")
else:
    image_dir = os.path.join("path", "to", "images")
    mapping_csv = os.path.join("path", "to", "gz2_filename_mapping.csv")
    classification_csv = os.path.join("path", "to", "gz2_hart16.csv.gz")
    output_csv = os.path.join("path", "to", "output")

# Load CSVs
mapping_df = pd.read_csv(mapping_csv)  # contains 'objid', 'asset_id'
classification_df = pd.read_csv(classification_csv)  # contains vote rates

# Merge on objid = dr7objid
merged_df = pd.merge(mapping_df, classification_df, left_on="objid", right_on="dr7objid", how="inner")

# Extract relevant columns
merged_df["asset_id"] = merged_df["asset_id"].astype(str)

#Edit here to find vote rate of other categories
result_df = merged_df[
    [
        "asset_id",
        "dr7objid",
        "t01_smooth_or_features_a01_smooth_fraction",
        "t01_smooth_or_features_a01_smooth_weighted_fraction",
        "t01_smooth_or_features_a01_smooth_debiased"
    ]
].rename(columns={
    "asset_id": "image_id",
    "t01_smooth_or_features_a01_smooth_fraction": "vote_fraction_smooth",
    "t01_smooth_or_features_a01_smooth_weighted_fraction": "vote_weighted_smooth",
    "t01_smooth_or_features_a01_smooth_debiased": "vote_debiased_smooth"
})

# Optionally filter to only images present in image_dir
image_ids = {
    filename.split(".")[0].split("_")[0]
    for filename in os.listdir(image_dir)
    if filename.lower().endswith(".jpg")
}
result_df = result_df[result_df["image_id"].isin(image_ids)]

# Save to CSV
result_df.to_csv(output_csv, index=False)
print(f"Saved {len(result_df)} entries to {output_csv}")
