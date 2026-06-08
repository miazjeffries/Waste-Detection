import kagglehub
import shutil
import os

# Download Garbage Classification dataset
path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
print("Downloaded to: ", path)

# Copy into project data/raw folder
dest = "data/raw/garbage-classification-v2"
shutil.copytree(path, dest)
print("Copied to:", dest)