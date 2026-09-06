"""
augmentation.py
===============
Light, face-appropriate augmentation: horizontal flip, small rotation,
brightness/contrast jitter, gaussian noise, and random crop+resize.

This exact set of transforms was previously duplicated verbatim across two
notebooks — it now lives here once.
"""

import os
import random

import cv2
import numpy as np

from src.config import N_AUGMENTED_COPIES, SEED


def random_flip(img):
    """Horizontal flip only — vertical flip would produce anatomically invalid faces."""
    return cv2.flip(img, 1) if random.random() > 0.5 else img


def random_rotation(img, max_angle: float = 15):
    """Small rotation (±max_angle degrees) — large angles distort facial geometry too much."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


def random_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)    # brightness
    img = img.astype(np.float32) * alpha + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, std: float = 8):
    if random.random() > 0.5:
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def random_crop_resize(img):
    if random.random() > 0.5:
        h, w = img.shape[:2]
        crop_percent = random.uniform(0.85, 0.95)
        new_h, new_w = int(h * crop_percent), int(w * crop_percent)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img = img[top:top + new_h, left:left + new_w]
        img = cv2.resize(img, (w, h))
    return img


def augment(img):
    """Apply the full augmentation chain once."""
    img = random_flip(img)
    img = random_rotation(img)
    img = random_brightness_contrast(img)
    img = add_gaussian_noise(img)
    img = random_crop_resize(img)
    return img


def augment_dataset(input_dir, output_dir, n_copies: int = N_AUGMENTED_COPIES, seed: int = SEED) -> int:
    """
    For every image in <input_dir>/<class_name>/, save the original plus
    `n_copies` augmented versions into <output_dir>/<class_name>/.

    Returns the total number of images written (originals + augmented).
    """
    random.seed(seed)
    np.random.seed(seed)

    input_dir, output_dir = str(input_dir), str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    total_written = 0

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_path = os.path.join(output_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            cv2.imwrite(os.path.join(save_class_path, img_name), img)
            total_written += 1

            for i in range(n_copies):
                aug_img = augment(img.copy())
                aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(save_class_path, aug_name), aug_img)
                total_written += 1

    print(f"Augmentation done — {total_written} images written to {output_dir}")
    return total_written


if __name__ == "__main__":
    from src.config import AUGMENTED_DIR, CROPPED_FACES_DIR

    augment_dataset(CROPPED_FACES_DIR, AUGMENTED_DIR)
