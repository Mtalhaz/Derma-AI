# Derma AI

Derma AI is a two-stage deep learning application for skin lesion detection and classification.

## Project Overview

The application follows a two-stage analysis pipeline:

1. **Lesion Detection**
   - The first model determines whether the input image contains a skin lesion.

2. **Lesion Classification**
   - If a lesion is detected, the second model classifies the detected lesion into one of seven categories.

The machine learning models are developed using Python and PyTorch, while the application interface is developed with Flutter and Dart.

## Analysis Pipeline

```text
Input Image
     |
     v
Lesion Detection Model
     |
     +-- No Lesion
     |
     +-- Lesion Detected
              |
              v
      Lesion Classification Model
              |
              v
        Predicted Class
```

## Lesion Classification

The classification model performs seven-class classification using the HAM10000 dataset.

The seven classes are:

- **AKIEC** — Actinic Keratoses and Intraepithelial Carcinoma
- **BCC** — Basal Cell Carcinoma
- **BKL** — Benign Keratosis-like Lesions
- **DF** — Dermatofibroma
- **MEL** — Melanoma
- **NV** — Melanocytic Nevi
- **VASC** — Vascular Lesions

## Machine Learning

The lesion classification model is based on **EfficientNet-B2** with transfer learning.

The training process includes:

- Image preprocessing
- Data augmentation
- Group-based train/validation splitting using lesion IDs
- Class balancing
- Weighted sampling
- Cross-entropy loss
- AdamW optimizer
- Cosine Annealing Warm Restarts learning-rate scheduler
- Macro F1-score evaluation
- Early stopping
- Best model checkpointing based on validation Macro F1-score

## Technologies

- Python
- PyTorch
- Torchvision
- EfficientNet-B2
- Flutter
- Dart
- HAM10000 Dataset

## Project Structure

```text
Derma-AI/
|
+-- flutter_application_HAM10000/
|   +-- lib/
|   +-- assets/
|   +-- android/
|   +-- ios/
|   +-- pubspec.yaml
|
+-- convert_model.py
+-- inspect_model.py
+-- test_arch.py
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## Application

The Flutter application provides the user interface for image-based skin lesion analysis.

The application first performs lesion detection. When a lesion is detected, the classification model is used to determine the corresponding lesion category.

## Dataset

The lesion classification model was developed using the **HAM10000** skin lesion image dataset.

The dataset and trained model files are not included in this repository because of their large file size.

## Purpose

This project was developed as a computer engineering project involving deep learning, image classification, model training, and mobile application development.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical evaluation.