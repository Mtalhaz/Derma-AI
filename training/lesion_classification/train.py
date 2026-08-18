# ============================================================
# Derma AI - 7-Class Skin Lesion Classification Training
# PyTorch / EfficientNet-B2
# ============================================================

import os
from pathlib import Path

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

# ============================================================
# CONFIGURATION
# ============================================================

# Portable paths. Set DERMA_HAM10000_ROOT to your local dataset root.
DATA_ROOT = Path(os.environ.get("DERMA_HAM10000_ROOT", "./data/ham10000"))
CSV_PATH = DATA_ROOT / "HAM10000_metadata.csv"
IMAGE_DIR = DATA_ROOT / "images_all"
OUTPUT_DIR = Path(os.environ.get("DERMA_OUTPUT_DIR", "./outputs/lesion_classification"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUTPUT_DIR / "best_model.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 3e-4
NUM_CLASSES = 7
IMAGE_SIZE = 260
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPING = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ============================================================
# DATA LOADING AND LEAKAGE-FREE SPLIT
# ============================================================

df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image_id"].apply(
    lambda image_id: os.path.join(IMAGE_DIR, f"{image_id}.jpg")
)
df["label"] = df["dx"].map(CLASS_MAPPING)

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(
    gss.split(df, df["label"], groups=df["lesion_id"])
)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

# ============================================================
# CLASS BALANCING
# ============================================================

class_counts = train_df["label"].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_df["label"]]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True,
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(row["label"], dtype=torch.long)


train_dataset = SkinDataset(train_df, transform=train_transforms)
val_dataset = SkinDataset(val_df, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

# ============================================================
# MODEL
# ============================================================

model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(in_features, NUM_CLASSES),
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=1,
    eta_min=1e-6,
)

# ============================================================
# TRAINING
# ============================================================

def train():
    best_val_f1 = 0.0
    patience_counter = 0
    patience_limit = 12

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")

        for images, labels in train_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)

            print(
                classification_report(
                    all_labels,
                    all_preds,
                    target_names=CLASS_NAMES,
                )
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
