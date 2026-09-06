"""
metrics.py
==========
Evaluation utilities. The original notebooks only ever printed
`classification_report` — this adds a confusion matrix (essential for
seeing *which* players get confused with each other, not just an
aggregate score) and saves both to outputs/.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import OUTPUT_FIGURES_DIR, OUTPUT_METRICS_DIR


def evaluate(y_true, y_pred, classes, split_name: str = "val", save: bool = True) -> str:
    """Print + save a classification report, and plot + save a confusion matrix."""
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    if save:
        OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_METRICS_DIR / f"classification_report_{split_name}.txt", "w") as f:
            f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(classes) * 0.6), max(6, len(classes) * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")

    if save:
        OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_DIR / f"confusion_matrix_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    return report
