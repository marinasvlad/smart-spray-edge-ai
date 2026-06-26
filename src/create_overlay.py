"""
Create an overlay image from an original image and a color-coded segmentation mask.

Expected mask format:
- black: background
- green: maize
- magenta/pink: weeds

Output:
- original image with maize highlighted blue and weeds highlighted red

Example:
    python src/create_overlay.py \
        --image assets/demo-original.jpg \
        --mask assets/demo-mask.png \
        --output assets/demo-overlay.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# OpenCV uses BGR, not RGB.
# CSS colors used in the web demo:
# weeds: #d32f2f
# maize: #1976d2
WEED_RED_BGR = np.array([47, 47, 211], dtype=np.uint8)
MAIZE_BLUE_BGR = np.array([210, 118, 25], dtype=np.uint8)


def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.80) -> np.ndarray:
    """
    Create a visual overlay from an original image and a color-coded mask.

    Args:
        original: Original BGR image loaded with OpenCV.
        mask: Color-coded BGR segmentation mask.
        alpha: Overlay opacity. Higher values make prediction colors stronger.

    Returns:
        BGR image with maize and weed overlay.
    """
    if original is None:
        raise ValueError("Original image is None.")

    if mask is None:
        raise ValueError("Mask image is None.")

    if original.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Original and mask must have the same size. "
            f"Original: {original.shape[:2]}, Mask: {mask.shape[:2]}"
        )

    b = mask[:, :, 0]
    g = mask[:, :, 1]
    r = mask[:, :, 2]

    maize_region = (g > 120) & (r < 100) & (b < 100)
    weed_region = (r > 120) & (b > 120) & (g < 100)

    result = original.copy()

    result[weed_region] = (
        original[weed_region].astype(np.float32) * (1 - alpha)
        + WEED_RED_BGR.astype(np.float32) * alpha
    ).astype(np.uint8)

    result[maize_region] = (
        original[maize_region].astype(np.float32) * (1 - alpha)
        + MAIZE_BLUE_BGR.astype(np.float32) * alpha
    ).astype(np.uint8)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create semantic segmentation overlay.")
    parser.add_argument("--image", required=True, help="Path to the original image.")
    parser.add_argument("--mask", required=True, help="Path to the color-coded mask image.")
    parser.add_argument("--output", required=True, help="Path where the overlay image will be saved.")
    parser.add_argument("--alpha", type=float, default=0.80, help="Overlay opacity between 0 and 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_path = Path(args.output)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

    if original is None:
        raise FileNotFoundError(f"Could not read original image: {image_path}")

    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    overlay = create_overlay(original, mask, alpha=args.alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), overlay)

    if not success:
        raise RuntimeError(f"Could not save overlay image to: {output_path}")

    print(f"Overlay saved to: {output_path}")


if __name__ == "__main__":
    main()