import pandas as pd
import os
import shutil
from pathlib import Path

input_csv = "data/processed/deduplicated_data.csv"
output_dir = "data/spot_check"
samples_per_class = 25


def main():
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Group by class, pull samples
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        sampled = class_df.sample(n=min(samples_per_class, len(class_df)), random_state=42)

        # Create class folder
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        # Copy sampled images into folder
        for i, row in sampled.iterrows():
            src = row['image_path']
            if os.path.exists(src):
                filename = os.path.basename(src)
                dst = os.path.join(class_dir, filename)
                shutil.copy(src, dst)

        print(f"{label}: {len(sampled)} samples")

    print(f"Spot check samples saved to {output_dir}")

if __name__ == "__main__":
    main()