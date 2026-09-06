# Soccer Player Recognition

A reconstructed and refactored face-recognition pipeline for identifying football players from images using:

**MTCNN → FaceNet (VGGFace2) → SVM**

> **Reconstruction Project**
>
> This repository is a reconstruction of an earlier notebook-based Soccer Player Recognition project. The original notebooks were reorganized into a modular `src/` package, while several data-leakage, label-alignment, persistence, and code-duplication issues were fixed.

## Pipeline

```text
Raw player photos
       │
       ▼
MTCNN face detection & crop
       │
       ▼
Stratified train/val split
       │
       ▼
Augment TRAIN split only
       │
       ▼
FaceNet embeddings (512-dim)
       │
       ▼
SVM classifier
       │
       ├──► Validation
       │
       └──► Held-out Test
```

## Project Structure

```text
data/
├── raw/                    # Raw images — not tracked
└── processed/              # Generated processed images — not tracked

src/
├── config.py               # Central configuration
├── pipeline.py             # End-to-end pipeline
│
├── data/
│   ├── download.py         # Kaggle dataset downloader
│   └── scrape_test_images.py
│                           # Supplementary test-image scraper
│
├── detection/
│   ├── mtcnn_detector.py   # MTCNN face detection
│   └── haar_detector.py    # Haar-cascade alternative
│
├── preprocessing/
│   ├── cleaning.py
│   ├── augmentation.py
│   └── split.py
│
├── embeddings/
│   └── facenet_embedder.py # FaceNet embeddings
│
├── models/
│   └── svm_classifier.py   # SVM training/persistence
│
└── evaluation/
    └── metrics.py          # Reports + confusion matrix

XML/                        # Haar cascade weights
models/                     # Generated model/embeddings
outputs/
├── figures/                # Generated figures
└── metrics/                # Generated metrics

requirements.txt
.gitignore
LICENSE
README.md
```

## Reconstruction & Improvements

The original project consisted of exploratory notebooks with separate Haar-cascade and MTCNN pipelines. During reconstruction, the code was reorganized and several issues were addressed.

### 1. Prevented augmentation leakage

The original notebooks augmented the dataset before splitting it into train and validation sets. This could place near-duplicate images in both splits.

The reconstructed pipeline splits the clean dataset first and augments only the training split.

### 2. Added class-alignment validation

`ImageFolder` determines class indices independently for each directory. The reconstructed embedding pipeline verifies that the test classes match the training classes before extracting embeddings, preventing silent label misalignment.

### 3. Added SVM persistence

The original implementation imported `joblib` but did not save the trained classifier.

The reconstruction adds:

```python
save_svm()
load_svm()
```

so the trained model can be reused without retraining.

### 4. Made preprocessing non-destructive

The original Haar pipeline could delete raw images when face detection failed.

The reconstructed pipeline never modifies `data/raw/`; rejected images are simply skipped.

### 5. Removed duplicated logic

Repeated augmentation and MTCNN preparation code from the notebooks was consolidated into reusable modules under `src/`.

### 6. Centralized configuration

Hardcoded relative paths were replaced with a central configuration in:

```text
src/config.py
```

This makes the pipeline independent of the current working directory.

### 7. Added confusion-matrix evaluation

The evaluation module now generates both classification reports and confusion matrices, making class-level recognition errors easier to analyze.

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset

Configure your Kaggle credentials, then run:

```bash
python -m src.data.download
```

### Run the complete pipeline

```bash
python -m src.pipeline
```

The pipeline performs:

1. Face detection and cropping
2. Stratified train/validation split
3. Training-only augmentation
4. FaceNet embedding extraction
5. SVM training
6. Model saving
7. Validation evaluation
8. Test evaluation

### Optional: collect additional test images

```bash
python -m src.data.scrape_test_images
```

These supplementary images can be used as an additional out-of-distribution evaluation set.

## Configuration

Main settings are defined in `src/config.py`.

| Setting              |      Default | Description                         |
| -------------------- | -----------: | ----------------------------------- |
| `FACE_SIZE`          | `(160, 160)` | FaceNet input size                  |
| `N_AUGMENTED_COPIES` |          `3` | Augmented copies per training image |
| `FACENET_PRETRAINED` | `"vggface2"` | FaceNet pretrained weights          |
| `VAL_SIZE`           |       `0.15` | Validation split                    |
| `SEED`               |         `42` | Random seed                         |

## Face Recognition Stack

* **MTCNN** — face detection and cropping
* **FaceNet / InceptionResnetV1** — 512-dimensional face embeddings
* **VGGFace2** — pretrained FaceNet weights
* **SVM** — player classification
* **Scikit-learn** — training and evaluation

A Haar-cascade detector is also included as a lightweight alternative to MTCNN.

## Notes

Generated datasets, embeddings, trained models, and evaluation outputs are excluded from Git through `.gitignore`.

The project is intended primarily as a learning and reconstruction exercise covering face detection, deep face embeddings, classical classification, data leakage prevention, and ML project organization.

## Attribution

This repository is a reconstruction and refactoring of an earlier notebook-based Soccer Player Recognition project.

The original implementation was used as the starting point for the reconstruction. The current repository reorganizes the implementation and introduces the fixes and structural improvements described above.

**Original Repository:**
`[Add original repository link here]`

## License

MIT License. See `LICENSE` for details.
