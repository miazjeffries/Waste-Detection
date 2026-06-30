"""
    ResNet34 Evaluation (Test Set)
    Loads best checkpoint from training, computes performance metrics
"""

import pandas as pd
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import models
from sklearn.metrics import f1_score, classification_report, confusion_matrix

''' Map class string labels to integer values '''
class_to_index = {"Cardboard": 0, "Ewaste": 1, "Glass": 2, "Metal": 3, "Organics": 4, 
                  "Other": 5, "Paper": 6, "Plastic": 7, "Textile": 8}


''' TRANSFORMS '''
# No image agumentation in testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


''' BUILD DATASET CLASS '''
class WasteDataset(Dataset):

    # Load dataset and tools
    def __init__(self, input_csv, root_dir, transform, class_to_index, split):
        self.data = pd.read_csv(input_csv)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True) # Filter dataset by split column labels
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_index = class_to_index
    
    # Returns dataset length
    def __len__(self):
        return len(self.data)
    
    # Retrieves data
    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB') # Ensures images open to standard RGB format (avoids mismatch error)
        image = self.transform(image) 
        label = self.class_to_index[row['label']]
        return image, label


''''''
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths assumed relative to repo root, resnet34_test.py expected to run from models/classification/
    root_dir = Path(__file__).resolve().parents[2]
    input_csv = os.path.join(root_dir, "data/processed/standardized_split_data.csv")
    checkpoint_path = os.path.join(Path(__file__).resolve().parent, "best_resnet34.pt")
    print(checkpoint_path) # Check file path

    # Test dataset filtered out of 'split' column
    test_dataset = WasteDataset(input_csv, root_dir, test_transform, class_to_index, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ''' MODEL DEFINITION '''
    model = models.resnet34(weights=None) # Weights loaded from checkpoint file
    model.fc = nn.Linear(model.fc.in_features, 9)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()


    ''' INFRERENCE LOOP '''
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


    ''' PERFORMANCE METRICS '''
    test_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\nTest macro F1: {test_macro_f1:.4f}\n")

    class_names = list(class_to_index.keys())
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"\nPer-class precision/recall/F1:\n{report}")

    matrix = confusion_matrix(all_labels, all_preds)
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    print(f"\nConfusion matrix (rows = true label, columns = predicted label):\n{matrix_df}")

    ''' SAVE RESULTS '''
    results_path = Path(__file__).resolve().parent / "resnet34_test_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Test macro F1: {test_macro_f1:.4f}\n\n")
        f.write("Per-class precision/recall/F1:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows = true label, columns = predicted label):\n")
        f.write(matrix_df.to_string())
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
     main()