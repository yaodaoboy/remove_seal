from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_seal_boxes(img: np.ndarray, min_area: int = 120) -> list[tuple[int, int, int, int]]:
    """Detect red seal regions and return bounding boxes as (x1, y1, x2, y2)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 43, 46]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([156, 43, 46]), np.array([180, 255, 255])),
    )

    # Connect nearby red regions to make seal candidates more complete.
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def remove_seal_from_image(img: np.ndarray, padding: int = 8) -> np.ndarray:
    """Remove red seal pixels while protecting dark text in detected seal regions."""
    h_img, w_img = img.shape[:2]
    final_result = img.copy()
    detected_boxes = detect_seal_boxes(img)

    for x1, y1, x2, y2 in detected_boxes:
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 43, 46]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([156, 43, 46]), np.array([180, 255, 255])),
        )

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_mask = cv2.inRange(gray_roi, 0, 70)

        # Only whiten red pixels that are not dark text.
        final_mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_mask))

        dilated_mask = cv2.dilate(final_mask, np.ones((3, 3), np.uint8), iterations=1)
        roi[dilated_mask == 255] = [255, 255, 255]

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.createCLAHE(clipLimit=2.0).apply(roi_gray)
        final_result[y1:y2, x1:x2] = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    return final_result


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for: {path}")
    encoded.tofile(str(path))


def process_images(input_path: Path, output_path: Path, padding: int) -> int:
    if input_path.is_file():
        img = read_image(input_path)
        result = remove_seal_from_image(img, padding=padding)
        target = output_path if output_path.suffix else output_path / input_path.name
        write_image(target, result)
        print(f"Processed: {input_path} -> {target}")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)
    image_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    for src in image_files:
        img = read_image(src)
        result = remove_seal_from_image(img, padding=padding)
        dst = output_path / src.name
        write_image(dst, result)
        print(f"Processed: {src} -> {dst}")
    return len(image_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove red seals from document images using OpenCV.")
    parser.add_argument("--input", default="input", help="Input image file or folder path.")
    parser.add_argument("--output", default="output", help="Output image file or folder path.")
    parser.add_argument("--padding", type=int, default=8, help="Expansion pixels for detected seal boxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    count = process_images(input_path, output_path, padding=args.padding)
    print(f"Done. Total processed image(s): {count}")


if __name__ == "__main__":
    main()
