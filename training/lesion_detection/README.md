# Lesion Detection Model

Binary image classifier used as the first stage of Derma AI.

## Purpose

Determines whether an input image contains a skin lesion.

| Label | Meaning |
|---|---|
| `0` | `lezyon_degil` |
| `1` | `lezyon` |

## Model

- Architecture: **MobileNetV3-Small**
- Initialization: ImageNet pretrained weights
- Input size: `224 × 224`
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Training strategy: classifier training followed by fine-tuning
- Validation model-selection metric: ROC-AUC
- Decision threshold: selected on the validation set by maximizing F1-score
- Export: PyTorch checkpoint and TorchScript

## Dataset sources

Positive samples are sourced from HAM10000. Negative samples are assembled from the non-lesion image datasets configured in `train.py`.

The datasets are **not included in this repository**.

## Configuration

The original training workflow was developed in Kaggle. The script defaults to:

```text
/kaggle/input/datasets
/kaggle/working
```

For a different environment, set these environment variables before running:

```bash
DERMA_DATASET_DIR=/path/to/datasets
DERMA_OUTPUT_DIR=/path/to/output
```

This keeps machine-specific absolute paths out of the repository while preserving the training workflow.

## Run

```bash
python train.py
```

A CUDA-enabled PyTorch installation is recommended for training.

## Outputs

The training script produces:

- `best_lesion_binary_model.pth`
- `lesion_binary_model_torchscript.pt`
- `lesion_binary_threshold.json`
- `lesion_binary_labels.json`
- `lesion_binary_dataset.csv`

Large model artifacts and generated datasets should remain outside the Git repository.
