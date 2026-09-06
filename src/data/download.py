"""
download.py
===========
Downloads both Kaggle datasets used in this project:
- The main training data: one folder per player (golden-foot players).
- A supplementary test set of player face images, kept separate so the
  final evaluation never sees images the model was trained/validated on.

Requires a configured Kaggle API token (~/.kaggle/kaggle.json).
"""

import os
import shutil

import kagglehub

from src.config import (
    KAGGLE_TEST_DATASET,
    KAGGLE_TRAIN_DATASET,
    TEST_SOURCE_DIR,
    TRAIN_SOURCE_DIR,
)


def _download_and_move(kaggle_dataset: str, destination) -> str:
    destination = str(destination)
    os.makedirs(destination, exist_ok=True)

    cache_path = kagglehub.dataset_download(kaggle_dataset)
    print(f"Downloaded '{kaggle_dataset}' to cache:", cache_path)

    for item in os.listdir(cache_path):
        source_path = os.path.join(cache_path, item)
        target_path = os.path.join(destination, item)
        if os.path.exists(target_path):
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)
        shutil.move(source_path, target_path)

    print("Moved to:", destination)
    return destination


def download_train_dataset() -> str:
    """Download the main player-faces training dataset into data/raw/football_golden_foot/."""
    return _download_and_move(KAGGLE_TRAIN_DATASET, TRAIN_SOURCE_DIR)


def download_test_dataset() -> str:
    """Download the supplementary test dataset into data/raw/test/."""
    return _download_and_move(KAGGLE_TEST_DATASET, TEST_SOURCE_DIR)


if __name__ == "__main__":
    download_train_dataset()
    download_test_dataset()
