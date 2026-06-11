import kagglehub
import shutil
import os

# Download datasets
datasets = [
    ("sumn2u/garbage-classification-v2", "garbage-classification-v2"),
    ("vishwasmishra1234/trash-net", "trash-net"),
    ("alistairking/recyclable-and-household-waste-classification", "recyclable-and-household-waste"), 
    ("joebeachcapital/realwaste", "real-waste"), 
    ("kneroma/tacotrashdataset", "taco-trash")
]

for slug, folder_name in datasets:
    print(f"Downloading {folder_name}...")
    path = kagglehub.dataset_download(slug)
    dest = f"data/raw/{folder_name}"
    if not os.path.exists(dest):
        shutil.copytree(path, dest)
        print(f"Saved to {dest}")
    else:
        print(f"Dataset already exists")

print("Datasets downloaded successfully")