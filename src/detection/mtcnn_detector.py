"""
mtcnn_detector.py
=================
Primary face-detection pipeline: MTCNN (deep-learning based) detects and
crops a single face per image, resized to the size FaceNet expects.

Preferred over the Haar-cascade detector (src/detection/haar_detector.py)
because it's far more robust to pose/lighting variation and rarely
misses a face outright — the trade-off is that it's slower per image.
"""

import os

import cv2
from mtcnn import MTCNN

from src.config import FACE_SIZE


def crop_faces_mtcnn(input_dir, output_dir, face_size=FACE_SIZE, verbose: bool = True) -> tuple[int, int]:
    """
    Walk every <input_dir>/<class_name>/*.jpg, detect exactly one face with
    MTCNN, crop + resize it, and save to <output_dir>/<class_name>/.

    Images with zero or multiple detected faces are skipped (ambiguous —
    better to drop them than risk a mislabeled or wrong-crop training example).

    Returns (saved_count, failed_count).
    """
    input_dir, output_dir = str(input_dir), str(output_dir)
    detector = MTCNN()

    saved, failed = 0, 0

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_path = os.path.join(output_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)

            if len(results) != 1:
                failed += 1
                continue

            x, y, w, h = results[0]["box"]
            x, y = max(0, x), max(0, y)
            face = img[y:y + h, x:x + w]
            if face.size == 0:
                failed += 1
                continue

            face = cv2.resize(face, face_size)
            cv2.imwrite(os.path.join(save_class_path, img_name), face)
            saved += 1

            if verbose:
                print(f"Saved: {saved} | Failed: {failed}", end="\r")

    if verbose:
        print(f"\nDone! Saved: {saved} | Failed: {failed}")

    return saved, failed


if __name__ == "__main__":
    from src.config import CROPPED_FACES_DIR, TRAIN_SOURCE_DIR

    crop_faces_mtcnn(TRAIN_SOURCE_DIR, CROPPED_FACES_DIR)
