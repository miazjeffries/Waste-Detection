"""  
    Standardize image size and pad for detection and classification models
    Maintains train/val/test split
"""

import pandas as pd
import os
from pathlib import Path
import cv2
import numpy as np

input_csv = 'data/processed/split_data.csv'
output_dir = 'data/standardized_images'
output_csv = 'data/processed/standardized_split_data.csv'
target_size = (640, 640)

def main():
    df = pd.read_csv(input_csv)

    # Create output directories to maintain train/val/test split
    for split in ['train', 'val', 'test']:
        Path(os.path.join(output_dir, split)).mkdir(parents=True, exist_ok=True)

    failed = 0
    new_paths = []

    print(f"Resizing images...")
    for i, row in df.iterrows():
        src_path = row['image_path']
        split = row['split']

        image = cv2.imread(src_path)
        if image is None:
            failed += 1
            new_paths.append(None)
            continue

        # Resize images to 640x640 with padding
        h, w = image.shape[:2] # Original height and width
        scale = min(target_size[0] / w, target_size[1] / h) # Calculate scale to fit image into 640x640
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad with black border
        padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8) # Create 640x640 black image
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized # Place resized image center on black image

        # Save standardized iamge
        filename = os.path.basename(src_path)
        output_path = os.path.join(output_dir, split, filename)
        cv2.imwrite(output_path, padded)
        new_paths.append(output_path)

        # Print progress updates to ensure file is running
        if (i + 1) % 10000 == 0:
            print(f"{i + 1}/{len(df)} images processed")

    # Update DataFrame
    df['image_path'] = new_paths
    df = df[df['image_path'].notna()]

    df.to_csv(output_csv, index=False)
    print(f"\nCompleted: {len(df)}/{len(df) + failed} images processed")
    print(f"\nStandardized images saved to {output_csv}")


if __name__ == "__main__":
    main()
