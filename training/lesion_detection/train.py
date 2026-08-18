# ============================================================
# PyTorch - Lezyon / Lezyon Değil Binary Model
# Progress çıktılı Kaggle eğitim kodu
# ============================================================

import os 
import io 
import json 
import time 
import random 
from pathlib import Path 

import numpy as np 
import pandas as pd 
from PIL import Image, ImageFile, ImageFilter 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 

from torchvision import transforms, models 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import ( 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report 
) 

from tqdm.auto import tqdm 

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 8
LR_HEAD = 1e-3
LR_FINE = 1e-5
NUM_WORKERS = 0
INPUT_DIR = Path("/kaggle/input/datasets")
OUTPUT_DIR = Path("/kaggle/working")
MODEL_PATH = OUTPUT_DIR / "best_lesion_binary_model.pth"
TORCHSCRIPT_PATH = OUTPUT_DIR / "lesion_binary_model_torchscript.pt"
THRESHOLD_PATH = OUTPUT_DIR / "lesion_binary_threshold.json"
LABELS_PATH = OUTPUT_DIR / "lesion_binary_labels.json"
CSV_PATH = OUTPUT_DIR / "lesion_binary_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# The remainder of this training script is the original training implementation.
# Dataset paths intentionally retain the original Kaggle environment configuration.


def list_images(root):
    paths = []
    root = Path(root)
    if not root.exists():
        return paths
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
    return paths

positive_roots = [
    INPUT_DIR / "kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1",
    INPUT_DIR / "kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2",
]
negative_roots = [
    INPUT_DIR / "eeshawn/flickr30k/flickr30k_images",
    INPUT_DIR / "shakyadissanayake/oily-dry-and-normal-skin-types-dataset/Oily-Dry-Skin-Types/train",
    INPUT_DIR / "shakyadissanayake/oily-dry-and-normal-skin-types-dataset/Oily-Dry-Skin-Types/valid",
    INPUT_DIR / "shakyadissanayake/oily-dry-and-normal-skin-types-dataset/Oily-Dry-Skin-Types/test",
    INPUT_DIR / "prasunroy/natural-images/natural_images",
    INPUT_DIR / "puneet6060/intel-image-classification/seg_train/seg_train",
    INPUT_DIR / "puneet6060/intel-image-classification/seg_test/seg_test",
    INPUT_DIR / "puneet6060/intel-image-classification/seg_pred/seg_pred",
]

positive_paths = sorted(list(set(sum((list_images(root) for root in positive_roots), []))))
negative_paths = sorted(list(set(sum((list_images(root) for root in negative_roots), []))))

if len(positive_paths) == 0:
    raise ValueError("Pozitif HAM10000 görselleri bulunamadı.")
if len(negative_paths) == 0:
    raise ValueError("Negatif görseller bulunamadı.")

random.shuffle(positive_paths)
random.shuffle(negative_paths)
NEGATIVE_RATIO = 2
max_negatives = min(len(negative_paths), len(positive_paths) * NEGATIVE_RATIO)
negative_paths = negative_paths[:max_negatives]

paths = positive_paths + negative_paths
labels = [1] * len(positive_paths) + [0] * len(negative_paths)
df = pd.DataFrame({"path": paths, "label": labels})
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)

train_df, temp_df = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

class RandomLesionLowQualityAugment:
    def __init__(self, p=0.45): self.p = p
    def __call__(self, image):
        if random.random() > self.p: return image
        if random.random() < 0.65: image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 2.2)))
        if random.random() < 0.45:
            buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=random.randint(25, 70)); buffer.seek(0); image = Image.open(buffer).convert("RGB")
        if random.random() < 0.45:
            w, h = image.size; scale = random.uniform(0.45, 0.85); nw, nh = max(32, int(w * scale)), max(32, int(h * scale)); image = image.resize((nw, nh), resample=Image.BILINEAR); image = image.resize((w, h), resample=Image.BILINEAR)
        if random.random() < 0.75: image = transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04)(image)
        return image

class AddGaussianNoise:
    def __init__(self, p=0.20, mean=0.0, std=0.020): self.p, self.mean, self.std = p, mean, std
    def __call__(self, tensor):
        if random.random() < self.p: tensor = torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0.0, 1.0)
        return tensor

class LesionBinaryDataset(Dataset):
    def __init__(self, dataframe, base_transform=None, positive_low_quality_aug=None, tensor_transform=None, train_mode=False):
        self.df = dataframe.reset_index(drop=True); self.base_transform = base_transform; self.positive_low_quality_aug = positive_low_quality_aug; self.tensor_transform = tensor_transform; self.train_mode = train_mode
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]; label_value = int(self.df.loc[idx, "label"])
        try: image = Image.open(img_path).convert("RGB")
        except Exception: image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.train_mode and label_value == 1 and self.positive_low_quality_aug is not None: image = self.positive_low_quality_aug(image)
        if self.base_transform: image = self.base_transform(image)
        if self.train_mode and self.tensor_transform: image = self.tensor_transform(image)
        return image, torch.tensor(float(label_value), dtype=torch.float32)

positive_low_quality_aug = RandomLesionLowQualityAugment(p=0.45)
train_base_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=12), transforms.RandomResizedCrop(IMG_SIZE, scale=(0.78, 1.00), ratio=(0.90, 1.10)), transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02), transforms.ToTensor()])
train_tensor_transform = transforms.Compose([AddGaussianNoise(p=0.20, std=0.020), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
eval_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = LesionBinaryDataset(train_df, train_base_transform, positive_low_quality_aug, train_tensor_transform, True)
val_dataset = LesionBinaryDataset(val_df, eval_transform, train_mode=False)
test_dataset = LesionBinaryDataset(test_df, eval_transform, train_mode=False)

class_counts = train_df["label"].value_counts().to_dict()
class_weights = {cls: len(train_df) / count for cls, count in class_counts.items()}
sample_weights = torch.DoubleTensor(train_df["label"].map(class_weights).values)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def build_model():
    try: model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    except Exception: model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model

model = build_model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

def set_trainable(model, fine_tune=False):
    for param in model.features.parameters(): param.requires_grad = False
    if fine_tune:
        for layer in model.features[-4:]:
            for param in layer.parameters(): param.requires_grad = True
    for param in model.classifier.parameters(): param.requires_grad = True

def train_one_epoch(model, loader, optimizer):
    model.train(); total_loss = 0.0; all_probs = []; all_targets = []
    for images, labels in tqdm(loader):
        images = images.to(DEVICE, non_blocking=True); labels = labels.to(DEVICE, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(); logits = model(images); loss = criterion(logits, labels); loss.backward(); optimizer.step()
        total_loss += loss.item() * images.size(0); all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel()); all_targets.extend(labels.detach().cpu().numpy().ravel())
    probs = np.array(all_probs); targets = np.array(all_targets).astype(int); preds = (probs >= 0.5).astype(int)
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), roc_auc_score(targets, probs)

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval(); total_loss = 0.0; all_probs = []; all_targets = []
    for images, labels in tqdm(loader):
        images = images.to(DEVICE, non_blocking=True); labels = labels.to(DEVICE, non_blocking=True).view(-1, 1); logits = model(images); loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0); all_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel()); all_targets.extend(labels.cpu().numpy().ravel())
    probs = np.array(all_probs); targets = np.array(all_targets).astype(int); preds = (probs >= threshold).astype(int)
    return {"loss": total_loss / len(loader.dataset), "accuracy": accuracy_score(targets, preds), "precision": precision_score(targets, preds, zero_division=0), "recall": recall_score(targets, preds, zero_division=0), "f1": f1_score(targets, preds, zero_division=0), "auc": roc_auc_score(targets, probs)}, targets, probs

def save_checkpoint(model, path):
    torch.save({"model_state_dict": model.state_dict(), "img_size": IMG_SIZE, "architecture": "mobilenet_v3_small", "labels": {"0": "lezyon_degil", "1": "lezyon"}}, path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"]); return model

set_trainable(model, fine_tune=False)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-4)
best_val_auc = -1.0
patience_counter = 0
for epoch in range(1, EPOCHS_HEAD + 1):
    train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer)
    val_metrics, _, _ = evaluate(model, val_loader)
    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]; patience_counter = 0; save_checkpoint(model, MODEL_PATH)
    else:
        patience_counter += 1
    if patience_counter >= 4: break

model = load_checkpoint(model, MODEL_PATH)
set_trainable(model, fine_tune=True)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE, weight_decay=1e-5)
patience_counter = 0
for epoch in range(1, EPOCHS_FINE + 1):
    train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer)
    val_metrics, _, _ = evaluate(model, val_loader)
    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]; patience_counter = 0; save_checkpoint(model, MODEL_PATH)
    else:
        patience_counter += 1
    if patience_counter >= 4: break

model = load_checkpoint(model, MODEL_PATH).to(DEVICE).eval()
val_metrics, val_true, val_probs = evaluate(model, val_loader)
best_threshold = 0.5; best_f1 = -1.0
for t in np.arange(0.05, 0.96, 0.01):
    score = f1_score(val_true, (val_probs >= t).astype(int), zero_division=0)
    if score > best_f1: best_f1, best_threshold = score, float(t)
with open(THRESHOLD_PATH, "w") as f: json.dump({"threshold": best_threshold}, f, indent=2)
with open(LABELS_PATH, "w") as f: json.dump({"0": "lezyon_degil", "1": "lezyon"}, f, indent=2)

test_metrics, test_true, test_probs = evaluate(model, test_loader, threshold=best_threshold)
test_preds = (test_probs >= best_threshold).astype(int)
print(test_metrics)
print(confusion_matrix(test_true, test_preds))
print(classification_report(test_true, test_preds, target_names=["lezyon_degil", "lezyon"], zero_division=0))

example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(str(TORCHSCRIPT_PATH))
