""" 
    This file cleans and processes the raw datasets to create a processed .CSV for modelling.
    Each image is mapped to a standardized, final material class label

    Duplicated file that creates a new dataset with two plastic classes: Plastic_Recyclable and Plastic_NonRecyclable
"""

import pandas as pd
import os
import json

# Directory containing raw downloaded datasets
raw_data_dir = "data/raw/"

# Output path for final .CSV
output_csv = "data/processed/plastic_unified_class_data.csv"

image_types = {'.jpg', '.jpeg', '.png'}

# Standardized material classes that images will map to
material_classes = ['Glass', 'Paper', 'Cardboard', 'Plastic_Recyclable', 'Plastic_NonRecyclable', 
                    'Textile', 'Metal', 'Organics', 'Ewaste', 'Other']


""" TACO SPECIFIC PROCESSING
    The taco-trash dataset had its labels stored in a JSON annotation file rather than by 
    folder name. These files load and parse that file to gather category information.
"""
def get_taco_categories(dataset_path):
    annot_file = os.path.join(dataset_path, 'data/annotations.json')
    with open(annot_file) as f:
        data = json.load(f)
    return data


""" 
    CLASS MAPPING 
    For each dataset, map the raw labels to the unified set of material labels
"""

# Map raw dataset labels to the unified material class labels using dictionaries
garbage_classification_v2_map = {'paper': 'Paper', 
                                 'clothes': 'Textile', 
                                 'metal': 'Metal', 
                                 'cardboard': 'Cardboard', 
                                 'trash': 'Other', 
                                 'glass': 'Glass', 
                                 'biological': 'Organics', 
                                 'battery': 'Ewaste', 
                                 'plastic': 'Plastic_Recyclable', # Images assumed to be mostly plastic bottles, containers, etc.
                                 'shoes': 'Textile'}

recyclable_and_household_waste_map = {'disposable_plastic_cutlery': 'Plastic_NonRecyclable',
                                      'food_waste': 'Organics', 
                                      'office_paper': 'Paper',
                                      'glass_food_jars': 'Glass',
                                      'aluminum_soda_cans': 'Metal', 
                                      'magazines': 'Paper', 
                                      'clothing': 'Textile', 
                                      'plastic_shopping_bags': 'Plastic_NonRecyclable',
                                      'plastic_soda_bottles': 'Plastic_Recyclable', 
                                      'styrofoam_food_containers': 'Other', 
                                      'aerosol_cans': 'Metal', 
                                      'aluminum_food_cans': 'Metal', 
                                      'newspaper': 'Paper', 
                                      'eggshells': 'Organics', 
                                      'glass_cosmetic_containers': 'Glass', 
                                      'paper_cups': 'Paper', 
                                      'plastic_water_bottles': 'Plastic_Recyclable', 
                                      'coffee_grounds': 'Organics', 
                                      'steel_food_cans': 'Metal', 
                                      'plastic_cup_lids': 'Plastic_NonRecyclable', 
                                      'cardboard_packaging': 'Cardboard', 
                                      'cardboard_boxes': 'Cardboard', 
                                      'plastic_straws': 'Plastic_NonRecyclable', 
                                      'styrofoam_cups': 'Other',
                                      'glass_beverage_bottles': 'Glass',
                                      'shoes': 'Textile', 
                                      'plastic_trash_bags': 'Plastic_NonRecyclable', 
                                      'tea_bags': 'Other', 
                                      'plastic_food_containers': 'Plastic_Recyclable', 
                                      'plastic_detergent_bottles': 'Plastic_Recyclable'}

trash_net_map = {'metal': 'Metal', 
                 'other': 'Other', 
                 'glass': 'Glass', 
                 'plastic': 'Plastic_Recyclable'} # Images assumed to be mostly plastic bottles, containers, etc.

real_waste_map = {'Paper': 'Paper', 
                  'Plastic': 'Plastic_Recyclable', # Images assumed to be mostly plastic bottles, containers, etc.
                  'Metal': 'Metal', 
                  'Cardboard': 'Cardboard', 
                  'Food Organics': 'Organics', 
                  'Glass': 'Glass', 
                  'Vegetation': 'Organics', 
                  'Textile Trash': 'Textile', 
                  'Miscellaneous Trash': 'Other'}

# TACO uses supercategories for broad grouping, and has more specified sub-categories
# Some ambiguous supercategories require further sub-category mapping for accuracy
taco_trash_map = {'Battery': 'Ewaste', 
                  'Paper': 'Paper', 
                  'Glass jar': 'Glass', 
                  'Shoe': 'Textile', 
                  'Aluminium foil': 'Metal', 
                  'Rope & strings': 'Other', 
                  'Scrap metal': 'Metal', 
                  'Other plastic': 'Plastic_NonRecyclable', 
                  'Plastic utensils': 'Plastic_NonRecyclable',  
                  'Broken glass': 'Glass', 
                  'Plastic bag & wrapper': 'Plastic_NonRecyclable', 
                  'Squeezable tube': 'Plastic_NonRecyclable', 
                  'Paper bag': 'Paper', 
                  'Pop tab': 'Metal', 
                  'Plastic gloves': 'Plastic_NonRecyclable',  
                  'Unlabeled litter': 'Other', 
                  'Food waste': 'Organics', 
                  'Plastic container': 'Plastic_Recyclable', 
                  'Styrofoam piece': 'Other', 
                  'Cigarette': 'Other', 
                  'Can': 'Metal',
                  # More specific sub-category names of ambiguous supercategories
                  'Paper cup': 'Paper', 
                  'Disposable plastic cup': 'Plastic_Recyclable', 
                  'Foam cup': 'Other', 
                  'Glass cup': 'Glass', 
                  'Other plastic cup': 'Plastic_Recyclable', 
                  'Clear plastic bottle': 'Plastic_Recyclable', 
                  'Glass bottle': 'Glass', 
                  'Other plastic bottle': 'Plastic_Recyclable', 
                  'Aluminium blister pack': 'Metal', 
                  'Carded blister pack': 'Paper', 
                  'Plastic straw': 'Plastic_NonRecyclable', 
                  'Paper straw': 'Paper',
                  'Plastic lid': 'Plastic_NonRecyclable', # Assumed to be loose, not on container
                  'Metal lid': 'Metal',
                  'Plastic bottle cap': 'Plastic_NonRecyclable', # Assumed to be loose, not screwed onto bottle
                  'Metal bottle cap': 'Metal', 
                  'Corrugated carton': 'Cardboard', 
                  'Drink carton': 'Other', 
                  'Egg carton': 'Paper', 
                  'Meal carton': 'Paper', 
                  'Pizza box': 'Cardboard', 
                  'Toilet tube': 'Cardboard', 
                  'Other carton': 'Cardboard'}

# Ambiguous TACO supercategories that require looking at sub-category names for accurate mapping
taco_ambiguous = ['Cup', 'Bottle', 'Blister pack', 'Straw', 'Lid', 'Bottle cap', 'Carton']

# Dataset configuration (TAOO excluded and handled separately)
datasets = [('garbage-classification-v2', garbage_classification_v2_map), 
            ('recyclable-and-household-waste', recyclable_and_household_waste_map), 
            ('trash-net', trash_net_map), 
            ('real-waste', real_waste_map)]



""" FUNCTION DEFINITIONS """

''' Goes through raw datasets and each image to its appropriate material class '''
def get_images(dataset_path, label_map):
    rows = []

    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_types]
        if not images:
            continue

        # Determine class labels
        parts = root.replace('\\', '/').split('/')
        parent = parts[-2] if len(parts) >= 2 else ""
        current = parts[-1]

        # Handls recyclable-and-household-waste's nested folder structure
        if current in ('default', 'real_world'):
            raw_label = parent
        else:
            raw_label = current

        # Map raw labels to unified class, default to "Other" if needed
        unified_label = label_map.get(raw_label, "Other")
        
        for f_name in images:
            image_path = os.path.join(root, f_name)
            rows.append((image_path, unified_label))

    return rows


''' Processes TACO dataset using its annotations.json file '''
def get_taco_images(dataset_path):
    rows = []

    # Load annotation data
    data = get_taco_categories(dataset_path)

    # Lookups for annotation file
    id_to_supercategory = {cat['id']: cat['supercategory'] for cat in data['categories']}
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']

        supercategory = id_to_supercategory.get(category_id, 'Other')

        # Specify sub-category names for accurate mapping of ambiguous supercategories 
        if supercategory in taco_ambiguous:
            lookup_key = id_to_name.get(category_id, 'Other')
        else:
            lookup_key = supercategory

        unified_label = taco_trash_map.get(lookup_key, 'Other')
        image_path = os.path.join(dataset_path, 'data', id_to_filename[image_id])

        rows.append((image_path, unified_label, 'taco-trash'))

    return rows


''' Loop over all datasets, collect image rows, returned a combined list '''
def build_dataframe(datasets):
    all_rows = []

    for folder_name,label_map in datasets:
        dataset_path = os.path.join(raw_data_dir, folder_name)

        if not os.path.exists(dataset_path):
            print(f"{folder_name} not found, moving on to next dataset.")
            continue

        rows = get_images(dataset_path, label_map)

        for image_path, unified_label in rows:
            all_rows.append((image_path, unified_label, folder_name))

        print(f"{folder_name}: {len(rows)} images")

    # Process TACO dataset separately
    taco_path = os.path.join(raw_data_dir, 'taco-trash')
    if os.path.exists(taco_path):
        taco_rows = get_taco_images(taco_path)
        all_rows.extend(taco_rows)
        print(f"TACO: {len(taco_rows)} images")
    else:
        print(f"taco-trash not found, skipping.")

    print(f"\nTotal images: {len(all_rows)}")
    return all_rows


''' Build the full DataFrame, save to .CSV, print final label break down '''
def main():
    rows = build_dataframe(datasets)
    df = pd.DataFrame(rows, columns=['image_path', 'label', 'source_dataset'])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} images to {output_csv} successfully.")

    print(df['label'].value_counts())


if __name__ == "__main__":
    main()