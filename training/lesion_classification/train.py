import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# Original training configuration
csv_path = r"C:\Users\evere\Desktop\ham10000\data\HAM10000_metadata.csv"
image_dir = r"C:\Users\evere\Desktop\Ham10000_yeni\images_all"
save_path = r"C:\Users\evere\Desktop\Ham10000_yeni\best_model.pth"
batch_size = 32
num_epochs = 40
learning_rate = 3e-4
num_classes = 7
image_size = 260
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
class_mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
df['label'] = df['dx'].map(class_mapping)

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(df, df['label'], groups=df['lesion_id']))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_df['label']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        try: image = Image.open(row['image_path']).convert("RGB")
        except Exception: image = Image.new("RGB", (image_size, image_size))
        if self.transform: image = self.transform(image)
        return image, torch.tensor(row['label'], dtype=torch.long)

train_dataset = SkinDataset(train_df, transform=train_transforms)
val_dataset = SkinDataset(val_df, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features, num_classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

def train():
    best_val_f1 = 0.0
    patience_counter = 0
    patience_limit = 12
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        train_loss = running_loss / len(train_loader)
        model.eval(); val_loss = 0.0; all_preds = []; all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        scheduler.step()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; torch.save(model.state_dict(), save_path)
            print(classification_report(all_labels, all_preds, target_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit: break

if __name__ == "__main__": train()
