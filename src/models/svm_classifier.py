"""
svm_classifier.py
=================
Trains an SVM on top of normalized FaceNet embeddings, and — unlike the
original notebooks, which `import joblib` but never actually call it —
properly saves/loads the trained model so it doesn't need to be retrained
from scratch in every session.
"""

import joblib
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from src.config import SVM_MODEL_PATH


def train_svm(train_emb, train_labels, kernel: str = "linear", C: float = 1.0) -> SVC:
    """Normalize embeddings (important for SVM) and fit the classifier."""
    train_emb = normalize(train_emb)
    svm = SVC(kernel=kernel, probability=True, C=C)
    svm.fit(train_emb, train_labels)
    return svm


def save_svm(svm: SVC, path=SVM_MODEL_PATH) -> None:
    path = str(path)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(svm, path)
    print("Saved SVM model to:", path)


def load_svm(path=SVM_MODEL_PATH) -> SVC:
    return joblib.load(str(path))


def predict(svm: SVC, embeddings):
    """Normalize embeddings the same way as training, then predict."""
    return svm.predict(normalize(embeddings))
