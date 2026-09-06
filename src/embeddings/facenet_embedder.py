"""
facenet_embedder.py
====================
Extracts 512-dim face embeddings using FaceNet (InceptionResnetV1,
pretrained on VGGFace2) for every image in a folder-per-class directory.

CRITICAL FIX vs. the original notebooks: `torchvision.datasets.ImageFolder`
assigns label indices by alphabetically sorting whatever class folders it
finds *in that specific directory*. The original code extracted train/val
embeddings this way, then separately extracted test embeddings the same
way — but never verified the test folder had the exact same classes in the
exact same order. If even one player folder was missing from data/raw/test/
(entirely plausible for a scraped/supplementary set), every test label
after the gap would silently point to the wrong player, with no error.

`extract_embeddings()` below takes an explicit `expected_classes` list
(the training set's class order) and raises immediately if a given
directory's classes don't match exactly — no silent mislabeling.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import FACENET_PRETRAINED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_facenet_model(pretrained: str = FACENET_PRETRAINED, device: str = DEVICE) -> InceptionResnetV1:
    return InceptionResnetV1(pretrained=pretrained).eval().to(device)


def extract_embeddings(
    data_dir,
    model: InceptionResnetV1,
    expected_classes: list[str] | None = None,
    batch_size: int = 32,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract embeddings + integer labels for every image under
    <data_dir>/<class_name>/*.jpg.

    If `expected_classes` is provided (e.g. the training set's class list),
    this raises a ValueError when `data_dir`'s classes don't match exactly
    — preventing the train/val/test label-misalignment bug described above.
    """
    data_dir = str(data_dir)
    dataset = datasets.ImageFolder(data_dir, transform=_TRANSFORM)

    if expected_classes is not None and list(dataset.classes) != list(expected_classes):
        raise ValueError(
            f"Class mismatch in '{data_dir}'.\n"
            f"  Expected (from training set): {expected_classes}\n"
            f"  Found:                        {dataset.classes}\n"
            "Every split must contain the exact same classes, in the same "
            "order, or label indices won't line up across train/val/test."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            emb = model(imgs).cpu().numpy()
            embeddings.append(emb)
            labels.append(lbls.numpy())

    return np.concatenate(embeddings), np.concatenate(labels), dataset.classes


def save_embeddings(path_prefix, embeddings: np.ndarray, labels: np.ndarray) -> None:
    """Save <prefix>_emb.npy and <prefix>_labels.npy."""
    path_prefix = str(path_prefix)
    Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{path_prefix}_emb.npy", embeddings)
    np.save(f"{path_prefix}_labels.npy", labels)


def load_embeddings(path_prefix) -> tuple[np.ndarray, np.ndarray]:
    path_prefix = str(path_prefix)
    return np.load(f"{path_prefix}_emb.npy"), np.load(f"{path_prefix}_labels.npy")
