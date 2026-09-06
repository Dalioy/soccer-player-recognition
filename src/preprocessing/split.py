"""
split.py
========
Stratified train/val split.

IMPORTANT FIX vs. the original notebooks: the original code augmented the
full dataset first, then split the augmented files with `train_test_split`
on a flat file list. Since an augmented copy is a near-duplicate of its
original, this let near-identical images leak across the train/val
boundary, inflating validation accuracy.

Here, splitting happens on the *cropped, pre-augmentation* images. Only
the resulting train split should be passed to
`src.preprocessing.augmentation.augment_dataset()` afterwards — validation
data stays clean and untouched, exactly like the real evaluation signal
you'll get on genuinely new photos.
"""

import os
import shutil

from sklearn.model_selection import train_test_split

from src.config import SEED, TRAIN_SPLIT_DIR, VAL_SIZE, VAL_SPLIT_DIR


def split_dataset(
    input_dir,
    train_dir=TRAIN_SPLIT_DIR,
    val_dir=VAL_SPLIT_DIR,
    test_size: float = VAL_SIZE,
    seed: int = SEED,
) -> tuple[int, int]:
    """
    Split <input_dir>/<class_name>/*.jpg into stratified train/val copies.

    Returns (n_train, n_val).
    """
    input_dir, train_dir, val_dir = str(input_dir), str(train_dir), str(val_dir)

    all_files, all_labels = [], []
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            all_files.append(os.path.join(class_path, img_name))
            all_labels.append(class_name)

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=test_size, random_state=seed, stratify=all_labels
    )

    for files, labels, split_dir in [(train_files, train_labels, train_dir), (val_files, val_labels, val_dir)]:
        for fpath, label in zip(files, labels):
            dest = os.path.join(split_dir, label)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(fpath, dest)

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")
    return len(train_files), len(val_files)


if __name__ == "__main__":
    from src.config import CROPPED_FACES_DIR

    split_dataset(CROPPED_FACES_DIR)
