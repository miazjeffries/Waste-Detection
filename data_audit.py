""" 
    Scans data/raw/ and summarizes image counts per class for each dataset.
    Because of the TACO dataset organization, the annotation JSON is read to display image
    supercategories instead of folder names, which are just batch id numbers.
"""

import os
import json

raw_data_dir = "data/raw"
image_types = ['.jpg', '.jpeg', '.png']


''' Goes through a dataset and counts images per class, handles nested folder structure '''
def dataset_audit(dataset_path):
    class_count = {}
    total = 0

    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_types]

        # Fix for recyclable-and-household waste dataset, which has an extra folder level to navigate
        parts = root.replace("\\", "/").split("/")
        parent = parts[-2] if len(parts) >= 2 else ""
        current = parts[-1]

        if current in ("default", "real_world"):
            class_name = parent
        else:
            class_name = current

        if images:
            class_count[class_name] = class_count.get(class_name, 0) + len(images)
            total += len(images)

    return class_count, total 


''' TACO dataset was organized by batch number instead of class. 
    This function reads the annotation JSON to get 'supercategory' counts'''
def taco_audit(dataset_path):
    annot_file = os.path.join(dataset_path, 'data/annotations.json')

    if not os.path.exists(annot_file):
        print(f"annotations.json not found in TACO dataset.")
        return {}, 0
    
    with open(annot_file) as f:
        data = json.load(f)

    # Build lookup: category_id = supercategory name
    id_to_supercategory = {cat['id']: cat['supercategory'] for cat in data['categories']}

    # Count annotations per supercategory
    class_count = {}
    for ann in data['annotations']:
        supercategory = id_to_supercategory.get(ann['category_id'], 'Unknown')
        class_count[supercategory] = class_count.get(supercategory, 0) + 1

    total = sum(class_count.values())
    return class_count, total


''' Uses above function to run data audit '''
def main():
    if not os.path.exists(raw_data_dir):
        print("data/raw/ folder not found. Run download_data.py first.")
        return
    
    datasets = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d)) and not d.startswith(".")]

    total_data = 0

    for dataset in datasets:
        dataset_path = os.path.join(raw_data_dir, dataset)
        print(f"Dataset: {dataset}")

        # Special handling for TACO dataset
        if dataset == 'taco-trash':
            class_count, total = taco_audit(dataset_path)
        else:
            class_count, total = dataset_audit(dataset_path)

        print(f"Total images: {total}")
        print(f"Total classes: {class_count}")
        total_data += total

    print(f"Total images across datasets: {total_data}")

if __name__ == "__main__":
    main()

