# Waste Detection
A computer vision project for waste detection and classification on a college campus

# Setup
### 1. Clone the repo
git clone https://github.com/miazjeffries/Waste-Detection
cd Waste-Detection

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run data scripts
Run in order from the repo root:
    
    scripts/download_data.py 
    - Downloads the five raw source datasets via kagglehub
    - Saves to data/raw/

    scripts/data_audit.py (OPTIONAL)
    - Audits the downloaded datasets: image count, class types and count, etc.

    scripts/class_mapping.py
    - Unifies the five source datasets into a single processed dataset
    - Original source labels mapped to Waste-Detection project's 9 standardized classes:
      Cardboard, Glass, Paper, Plastic, Textile, Metal, Organics, Ewaste, and Other
    - Saves to data/processed/unified_class_data.csv


    scripts/check_duplicates.py
    - Checks for and removes any duplicated images within the dataset
    - Saves to data/processed/deduplicated_data.csv 

    script/spot_check.py (OPTIONAL)
    - Manual check of data, randomly samples images per class to ensure accuracy

    script/split_data.py
    - Creates 80/10/10 train/val/test split stratified by class
    - Saves to data/processed/split_data.csv

    script/standardize_images.py
    - Resizes and pads all images to standard 640x640 image size
    - Final image dataset
    - Saves to data/processed/standardized_split_data.csv
    ** Dataset to be used in model creation