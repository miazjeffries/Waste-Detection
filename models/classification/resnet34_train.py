""" 
    ResNet34 Image Classification Model
    Analyzes and groups images by material class
"""
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from pathlib import Path

''' Map class string labels to integer values '''
class_to_index = {"Cardboard": 0, "Ewaste": 1, "Glass": 2, "Metal": 3, "Organics": 4, 
                 "Other": 5, "Paper": 6, "Plastic": 7, "Textile": 8}


''' 
    TRANSFORMS 
    Standardize images for model type, optimize for performance
'''
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Standard image size for ResNet
    transforms.RandAugment(), # Applies random image transformations (e.g. rotation, color shifting) for increased robustness
    transforms.ToTensor(), # Prepares image data for use in neural network
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


''' BUILD DATASET CLASS '''
class WasteDataset(Dataset):
    # Loads dataset and tools
    def __init__(self, input_csv, root_dir, transform, class_to_index, split):
        self.data = pd.read_csv(input_csv)
        self.data = self.data[self.data['split'] == split].reset_index(drop=True) # Filter dataset by split column
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Selects hardware to run model (GPU or CPU)
    
    # Paths assumed relative to repo root, resnet_train.py expected to run from models/classification/
    root_dir = Path(__file__).resolve().parents[2]
    input_csv = os.path.join(root_dir, "data/processed/standardized_split_data.csv")
    print(os.path.abspath(input_csv)) # Check file location

    # Train and val sets from same .csv, filtered by split column
    train_dataset = WasteDataset(input_csv, root_dir, train_transform, class_to_index, split="train")
    val_dataset = WasteDataset(input_csv, root_dir, val_transform, class_to_index, split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_f1 = 0.0 # For checkpoints at best validation macro F1 scores
    
    ''' MODEL DEFINITION '''
    model = models.resnet34(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 9) # Replace last layer with 9 waste class outputs
    model = model.to(device)


    ''' 
        CLASS WEIGHTS
        Helps with ~5x class imbalance, assigning higher weight to 
        uncommon classes (e.g. Ewaste), lower weight to common classes (e.g. Plastic)
    '''
    class_counts = {"Cardboard": 5432, "Ewaste": 2269, "Glass": 7874, "Metal": 6121, "Organics": 4018, 
                    "Other": 5420, "Paper": 6396, "Plastic": 10032, "Textile": 11321}

    # Use inverse frequency: weight = 1 / (num samples in class)
    # Order weights to match class_to_index ordering
    weights = []

    for class_name in class_to_index:
        weight = 1.0 / class_counts[class_name] # Inverse frequency: rarer classes have higher weights
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * len(weights) # Normalize weights to average to 1.0
    class_weights = weights.to(device)


    ''' LOSS AND OPTIMIZER '''
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Small learning rate for fine-tuning


    ''' CONNECT W/ WEIGHTS & BIASES'''
    wandb.init(
        project="Waste-Detection", 
        name="resnet34-train",
        config={
        "model": "resnet34",
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 25
    })


    ''' TRAINING AND VALIDATION LOOP '''
    for epoch in range(25):
        print(f"Epoch {epoch + 1}...")
        
        # Training loop
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # Clear gradients from previous batch
            outputs = model(images) # Forward pass
            loss = loss_func(outputs, labels)
            loss.backward() # Compute gradients
            optimizer.step() # Update weights

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_dataset)
        train_macro_f1 = f1_score(train_labels, train_preds, average="macro")

        # Validation loop (weights not updated)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro")

        # Save model weights when val_macro_f1 improves
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), "best_resnet34.pt")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1
        })

        print(f"\nEpoch {epoch+1}/25 -- train_loss: {train_loss:.4f}, train_macro_f1: {train_macro_f1:.4f}, val_loss: {val_loss:.4f}, val_macro_f1: {val_macro_f1:.4f}")

if __name__ == "__main__":
    main()