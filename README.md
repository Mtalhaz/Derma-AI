# Derma AI

> Two-stage deep learning system for skin lesion detection and classification, with a Flutter-based application interface and Grad-CAM explainability support.

## Overview

**Derma AI** is a computer vision and deep learning project designed to analyze skin lesion images through a two-stage pipeline:

1. **Lesion Detection** — determines whether the input image contains a skin lesion.
2. **Lesion Classification** — classifies the detected lesion into one of seven categories from the HAM10000 dataset.

The machine learning components are implemented with **Python and PyTorch**, while the user-facing application is built with **Flutter and Dart**. The repository also contains a Grad-CAM API component for visual model explainability.

> **Important:** This project is intended for educational and research purposes. It is not a medical diagnostic system and must not be used as a substitute for evaluation by a qualified healthcare professional.

---

## Key Features

- Two-stage skin lesion analysis pipeline
- Seven-class lesion classification
- Transfer learning with **EfficientNet-B2**
- HAM10000-based classification workflow
- Data preprocessing and augmentation
- Class balancing and weighted sampling
- Validation using **Macro F1-score**
- Early stopping and best-model checkpointing
- **Grad-CAM** support for model interpretability
- Flutter mobile application interface

---

## System Architecture

```text
                         Input Image
                              │
                              ▼
                    ┌───────────────────┐
                    │ Lesion Detection  │
                    │      Model        │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 No Lesion        Lesion Detected
                    │                   │
                    ▼                   ▼
                  Result       ┌───────────────────┐
                               │ Lesion Classification │
                               │      EfficientNet-B2  │
                               └─────────┬─────────┘
                                         │
                                         ▼
                                  Predicted Class
                                         │
                                         ▼
                                  Grad-CAM Analysis
```

---

## Classification Classes

The classification stage uses the seven diagnostic categories provided by the HAM10000 dataset:

| Code | Category |
|------|----------|
| `AKIEC` | Actinic Keratoses and Intraepithelial Carcinoma |
| `BCC` | Basal Cell Carcinoma |
| `BKL` | Benign Keratosis-like Lesions |
| `DF` | Dermatofibroma |
| `MEL` | Melanoma |
| `NV` | Melanocytic Nevi |
| `VASC` | Vascular Lesions |

---

## Machine Learning Pipeline

The classification workflow includes:

- Image preprocessing
- Data augmentation
- Group-based train/validation splitting using lesion IDs
- Class balancing
- Weighted sampling
- Cross-entropy loss
- AdamW optimization
- Cosine Annealing Warm Restarts learning-rate scheduling
- Macro F1-score evaluation
- Early stopping
- Best-model checkpointing based on validation Macro F1-score

### Model

The lesion classification model is based on **EfficientNet-B2** using transfer learning.

Model weights and large dataset files are intentionally excluded from the repository because of their size. The repository therefore contains the supporting source code and application structure without committing large binary artifacts.

---

## Explainability

Derma AI includes a **Grad-CAM** component to support visual interpretation of the classification model.

Grad-CAM can be used to highlight image regions that contribute to a model's prediction, providing an additional interpretability layer for the computer vision pipeline.

The repository contains the corresponding API implementation under:

```text
flutter_application_HAM10000/gradcam_api.py
```

---

## Technology Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- EfficientNet-B2
- Grad-CAM
- OpenCV
- NumPy
- Pillow

### Application

- Flutter
- Dart

### Dataset

- HAM10000 — Human Against Machine with 10000 training images

---

## Repository Structure

```text
Derma-AI/
│
├── flutter_application_HAM10000/
│   ├── lib/                  # Flutter application source
│   ├── assets/               # Application assets
│   ├── gradcam_api.py        # Grad-CAM API component
│   ├── android/              # Android platform files
│   ├── ios/                  # iOS platform files
│   ├── linux/                # Linux platform files
│   ├── macos/                # macOS platform files
│   ├── pubspec.yaml          # Flutter dependencies and configuration
│   └── test/                 # Flutter tests
│
├── convert_model.py          # Model conversion utility
├── inspect_model.py          # Model inspection utility
├── test_arch.py              # Model architecture testing utility
├── requirements.txt          # Python dependencies
├── .gitignore                # Repository and environment exclusions
└── README.md                 # Project documentation
```

---

## Installation

### Python Environment

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Flutter Application

Make sure Flutter is installed and available in your system PATH. Then move into the application directory:

```bash
cd flutter_application_HAM10000
flutter pub get
```

Run the application with:

```bash
flutter run
```

> The exact runtime configuration may depend on the model files, API configuration, and execution environment used with the project.

---

## Dataset

The classification pipeline was developed using the **HAM10000** skin lesion image dataset.

The dataset itself is **not included** in this repository. This keeps the repository lightweight and avoids committing a large collection of image files.

If you work with the project locally, place the dataset and model artifacts according to the configuration expected by the training/inference workflow.

---

## Project Goals

Derma AI was developed to demonstrate an end-to-end application of deep learning to medical image analysis, combining:

- Computer vision
- Transfer learning
- Image classification
- Model evaluation
- Explainable AI
- Mobile application development

The project is structured to demonstrate how a trained computer vision model can be integrated into an application-oriented workflow rather than remaining as an isolated training experiment.

---

## Limitations

- Model predictions are not medical diagnoses.
- Performance can vary depending on image quality, acquisition conditions, and lesion characteristics.
- The project is based on a specific dataset and may not generalize to every real-world population or imaging environment.
- Large model and dataset artifacts are not included in the repository.

---

## Disclaimer

**Derma AI is an educational and research project. It is not intended for clinical use, medical diagnosis, treatment decisions, or emergency decision-making. Always consult a qualified healthcare professional for medical evaluation.**

---

## Author

**Musa Talha Öz**

Computer Engineering

GitHub: [@Mtalhaz](https://github.com/Mtalhaz)
