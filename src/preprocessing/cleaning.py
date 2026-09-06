"""
cleaning.py
===========
Small data-QA utilities used while exploring the raw dataset: randomly
sample a handful of images per class for visual inspection, and convert a
folder of images to grayscale (useful for quick Haar-cascade experiments,
not used by the main MTCNN + FaceNet pipeline which works in color).
"""

import math
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt


def sample_random_images(
    base_dir,
    is_flat_folder: bool = True,
    save_images: bool = True,
    sampled_dir: str = "Sampled_images",
    max_images: int = 10,
):
    """
    Sample random images from a dataset and display them in a grid — a
    quick visual sanity check before running the full pipeline.

    Parameters
    ----------
    base_dir : path to main folder containing images or per-class subfolders
    is_flat_folder : True if images sit directly in base_dir, False if
        base_dir has one subfolder per class
    save_images : whether to copy the sampled images into `sampled_dir`
    sampled_dir : destination folder for the sampled images
    max_images : how many images to sample in total
    """
    base_dir = str(base_dir)
    all_images = []

    if not is_flat_folder:
        classes = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        if not classes:
            raise FileNotFoundError(f"No class folders found inside '{base_dir}'")

        for cls in classes:
            cls_folder = os.path.join(base_dir, cls)
            files = [
                os.path.join(cls_folder, f) for f in os.listdir(cls_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            all_images.extend([(f, cls) for f in files])
    else:
        files = [
            os.path.join(base_dir, f) for f in os.listdir(base_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        all_images.extend([(f, "image") for f in files])

    if not all_images:
        raise FileNotFoundError(f"No images found in '{base_dir}'")

    if max_images > len(all_images):
        print(f"Warning: Requested {max_images} images, but only {len(all_images)} available.")
        max_images = len(all_images)

    sampled = random.sample(all_images, max_images)

    if save_images:
        sampled_dir = str(sampled_dir)
        if os.path.exists(sampled_dir):
            shutil.rmtree(sampled_dir)
        os.makedirs(sampled_dir, exist_ok=True)

    n_cols = 5
    n_rows = math.ceil(len(sampled) / n_cols)
    plt.figure(figsize=(15, 3 * n_rows))

    for i, (img_path, cls_name) in enumerate(sampled):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image {img_path}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(cls_name)
        plt.axis("off")

        if save_images:
            class_folder = os.path.join(sampled_dir, cls_name)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(class_folder, os.path.basename(img_path)))

    plt.tight_layout()
    plt.show()

    print(f"Sampled {len(sampled)} images successfully.")
    return sampled


def convert_dataset_to_gray(base_dir, output_dir, show_images: bool = False):
    """Convert every image under base_dir (recursively) to grayscale, mirroring the folder structure into output_dir."""
    base_dir, output_dir = str(base_dir), str(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    gray_images = []

    for root, _dirs, files in os.walk(base_dir):
        for file in files:
            img_path = os.path.join(root, file)
            cls_name = os.path.basename(root)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append((gray, cls_name))

            relative = os.path.relpath(root, base_dir)
            save_folder = os.path.join(output_dir, relative)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, file), gray)

    if show_images:
        n_cols = 5
        n_rows = math.ceil(len(gray_images) / n_cols)
        plt.figure(figsize=(15, 3 * n_rows))
        for i, (img, cls_name) in enumerate(gray_images):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(cls_name)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    return len(gray_images)
