"""
pipeline.py
===========
End-to-end pipeline, MTCNN-based (the recommended path):

    raw images -> crop faces (MTCNN) -> stratified split -> augment TRAIN
    only -> FaceNet embeddings -> train SVM -> evaluate on val -> prepare
    test set (also MTCNN-cropped) -> evaluate on test.

Note the split happens *before* augmentation (see
src/preprocessing/split.py's docstring for why — the original notebooks
augmented first and split second, which let near-duplicate augmented
copies leak between train and val).

Usage:
    python -m src.pipeline
"""

from src.config import (
    AUGMENTED_DIR,
    CLASSES_NPY,
    CROPPED_FACES_DIR,
    TEST_EMB_PREFIX,
    TEST_READY_DIR,
    TEST_SOURCE_DIR,
    TRAIN_EMB_PREFIX,
    TRAIN_SOURCE_DIR,
    TRAIN_SPLIT_DIR,
    VAL_EMB_PREFIX,
    VAL_SPLIT_DIR,
)
from src.detection.mtcnn_detector import crop_faces_mtcnn
from src.embeddings.facenet_embedder import extract_embeddings, load_facenet_model, save_embeddings
from src.evaluation.metrics import evaluate
from src.models.svm_classifier import predict, save_svm, train_svm
from src.preprocessing.augmentation import augment_dataset
from src.preprocessing.split import split_dataset


def run_pipeline(download: bool = False, evaluate_on_test: bool = True) -> None:
    if download:
        print("\n### 0. Downloading datasets ###")
        from src.data.download import download_test_dataset, download_train_dataset
        download_train_dataset()
        download_test_dataset()

    print("\n### 1. Detecting & cropping faces (MTCNN) ###")
    crop_faces_mtcnn(TRAIN_SOURCE_DIR, CROPPED_FACES_DIR)

    print("\n### 2. Splitting BEFORE augmentation (avoids train/val leakage) ###")
    split_dataset(CROPPED_FACES_DIR, TRAIN_SPLIT_DIR, VAL_SPLIT_DIR)

    print("\n### 3. Augmenting the train split only ###")
    augmented_train_dir = AUGMENTED_DIR / "train"
    augment_dataset(TRAIN_SPLIT_DIR, augmented_train_dir)

    print("\n### 4. Extracting FaceNet embeddings ###")
    model = load_facenet_model()
    train_emb, train_labels, classes = extract_embeddings(augmented_train_dir, model)
    val_emb, val_labels, _ = extract_embeddings(VAL_SPLIT_DIR, model, expected_classes=classes)

    save_embeddings(TRAIN_EMB_PREFIX, train_emb, train_labels)
    save_embeddings(VAL_EMB_PREFIX, val_emb, val_labels)

    import numpy as np
    CLASSES_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(CLASSES_NPY, classes)

    print("\n### 5. Training the SVM ###")
    svm = train_svm(train_emb, train_labels)
    save_svm(svm)

    print("\n### 6. Evaluating on validation set ###")
    val_preds = predict(svm, val_emb)
    evaluate(val_labels, val_preds, classes, split_name="val")

    if evaluate_on_test:
        print("\n### 7. Preparing the held-out test set (MTCNN) ###")
        crop_faces_mtcnn(TEST_SOURCE_DIR, TEST_READY_DIR)

        print("\n### 8. Evaluating on the test set ###")
        test_emb, test_labels, _ = extract_embeddings(TEST_READY_DIR, model, expected_classes=classes)
        save_embeddings(TEST_EMB_PREFIX, test_emb, test_labels)

        test_preds = predict(svm, test_emb)
        evaluate(test_labels, test_preds, classes, split_name="test")

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
