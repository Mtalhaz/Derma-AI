# Lesion Classification Model

Seven-class image classification model used as the second stage of Derma AI.

## Purpose

After the first model determines that an image contains a lesion, this model predicts one of seven HAM10000 lesion categories.

| Code | Class |
|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

## Model

- Architecture: **EfficientNet-B2**
- Initialization: ImageNet pretrained weights
- Input size: `260 × 260`
- Loss: Cross Entropy
- Optimizer: `AdamW`
- Class balancing: `WeightedRandomSampler`
- Split strategy: `GroupShuffleSplit`
- Group key: `lesion_id`
- Validation metric: Macro F1-score
- Early stopping: validation Macro F1

Using `lesion_id` for the group split helps prevent images belonging to the same lesion from being distributed between training and validation sets.

## Dataset

The model was trained using the HAM10000 metadata and image collection. The dataset is **not included in this repository**.

The local dataset and output locations are configured at the top of `train.py` and should be adapted to the environment where training is performed.

## Run

```bash
python train.py
```

A CUDA-enabled PyTorch installation is recommended for training.

## Output

The best model is saved according to validation Macro F1 as a PyTorch state dictionary.

Large model artifacts and datasets should remain outside the Git repository.
