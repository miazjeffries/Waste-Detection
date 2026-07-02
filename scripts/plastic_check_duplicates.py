""" 
    Check unified dataset for duplicated images, and remove them for a cleaner dataset.
    File hashing is used as it opens and looks at each image file, taking care of cases where the same image has different filepaths

    Looks at dataset with Plastic_Recyclable and Plastic_NonRecyclable
"""

import hashlib
import pandas as pd
from collections import defaultdict

input_csv = "data/processed/plastic_unified_class_data.csv"
output_csv = "data/processed/plastic_deduplicated_data.csv"


''' Reads an image file and returns it's unique hash code'''
def hash_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return None
    

def main():
    df = pd.read_csv(input_csv)
    print(f"Total images before removing duplicates: {len(df)}")

    # Gather hash code for every image
    print(f"\nHashing images...")
    df['hash'] = df['image_path'].apply(hash_image)

    # Drop rows where a file was not found
    missing = df['hash'].isna().sum()
    if missing > 0:
        print(f"\n{missing} images not found, dropping from dataset.")
    df = df.dropna(subset=['hash'])

    # Find duplicates (images with the same unique hash code)
    duplicates = df[df.duplicated(subset='hash', keep=False)]
    print(f"\n{len(duplicates)} duplicated images found.")

    if len(duplicates) > 0:
        print(f"Sample duplicates: ")
        print(duplicates.head(10).to_string())

    # Drop second occurence of duplicated images
    df_dedup = df.drop_duplicates(subset='hash', keep='first')
    df_dedup = df_dedup.drop(columns=['hash'])

    print(f"\nTotal images after deduplicating: {len(df_dedup)}")
    print(f"Removed {len(df) - len(df_dedup)} duplicated images.\n")
    print(df_dedup['label'].value_counts())

    df_dedup.to_csv(output_csv, index=False)
    print(f"\nDe-depulicated data saved to {output_csv}")


if __name__ == "__main__":
    main()