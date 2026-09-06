"""
haar_detector.py
================
Alternative face-detection pipeline using OpenCV's Haar-cascade classifier.
Much faster than MTCNN and needs no extra model download (ships with
OpenCV's XML weights, bundled in XML/), but is noticeably less robust to
non-frontal poses, occlusion, or unusual lighting — expect more skipped
images than the MTCNN pipeline.

Use this when speed matters more than recall (e.g. very large datasets,
quick experiments) — otherwise prefer src/detection/mtcnn_detector.py.
"""

import os

import cv2

from src.config import FACE_SIZE, HAAR_EYE_CASCADE, HAAR_FACE_CASCADE


def crop_faces_haar(
    input_dir,
    output_dir,
    face_size=FACE_SIZE,
    cascade_path=HAAR_FACE_CASCADE,
    require_eyes: bool = False,
    eye_cascade_path=HAAR_EYE_CASCADE,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Walk every <input_dir>/<class_name>/*.jpg, detect exactly one face with
    a Haar cascade, crop + resize it, and save to <output_dir>/<class_name>/.

    Set `require_eyes=True` for an extra quality filter that also requires
    at least 2 eyes to be detected inside the face region — cuts down on
    false-positive face detections, at the cost of skipping more images.

    Unlike the original implementation this is based on, source images are
    NEVER deleted — rejected images are simply skipped, leaving data/raw/
    untouched no matter how this function is configured.

    Returns (saved_count, failed_count).
    """
    input_dir, output_dir = str(input_dir), str(output_dir)
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path)) if require_eyes else None

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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) != 1:
                failed += 1
                continue

            x, y, w, h = faces[0]

            if require_eyes:
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) < 2:
                    failed += 1
                    continue

            face = img[y:y + h, x:x + w]
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

    crop_faces_haar(TRAIN_SOURCE_DIR, CROPPED_FACES_DIR)
