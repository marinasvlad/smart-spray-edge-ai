"""
Simplified smart spraying decision logic.

The real prototype runs semantic segmentation on an NVIDIA Jetson device and
activates a relay/pump when weed pixel coverage exceeds a configured threshold.

Class convention:
- 0: background
- 1: maize
- 2: weeds
"""

import numpy as np


def compute_class_ratio(prediction_mask: np.ndarray, class_id: int) -> float:
    """
    Compute the ratio of pixels belonging to a given class.

    Args:
        prediction_mask: 2D semantic segmentation mask.
        class_id: Class ID to measure.

    Returns:
        Pixel ratio for the selected class.
    """
    if prediction_mask.size == 0:
        return 0.0

    class_pixels = np.sum(prediction_mask == class_id)
    return float(class_pixels) / float(prediction_mask.size)


def compute_weed_ratio(prediction_mask: np.ndarray, weed_class_id: int = 2) -> float:
    """
    Compute the ratio of pixels classified as weeds.
    """
    return compute_class_ratio(prediction_mask, weed_class_id)


def should_spray(
    prediction_mask: np.ndarray,
    threshold: float = 0.03,
    weed_class_id: int = 2
) -> tuple[bool, float]:
    """
    Decide whether spraying should be triggered.

    Args:
        prediction_mask: 2D semantic segmentation mask.
        threshold: Minimum weed pixel ratio required for spraying.
        weed_class_id: Class ID used for weeds.

    Returns:
        Tuple containing:
        - spray decision
        - computed weed ratio
    """
    weed_ratio = compute_weed_ratio(prediction_mask, weed_class_id)
    return weed_ratio >= threshold, weed_ratio


def summarize_prediction(prediction_mask: np.ndarray) -> dict[str, float]:
    """
    Return class coverage summary for a 3-class segmentation mask.
    """
    return {
        "background_ratio": compute_class_ratio(prediction_mask, 0),
        "maize_ratio": compute_class_ratio(prediction_mask, 1),
        "weed_ratio": compute_class_ratio(prediction_mask, 2),
    }


if __name__ == "__main__":
    fake_mask = np.zeros((1024, 1024), dtype=np.uint8)

    # Simulate maize rows
    fake_mask[:, 250:320] = 1
    fake_mask[:, 700:770] = 1

    # Simulate weed patch
    fake_mask[300:500, 420:650] = 2

    spray, weed_ratio = should_spray(fake_mask, threshold=0.03)
    summary = summarize_prediction(fake_mask)

    print("Prediction summary:")
    for class_name, ratio in summary.items():
        print(f"- {class_name}: {ratio:.4f}")

    print(f"\nSpray decision: {spray}")
    print(f"Weed ratio: {weed_ratio:.4f}")