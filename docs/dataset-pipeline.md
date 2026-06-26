# Dataset Pipeline

This document describes the dataset creation process used for the Smart Spray Edge AI project.

## Motivation

Public top-down maize/weed pixel-wise segmentation datasets are limited. Many available datasets are either small, captured in different conditions, or do not contain realistic maize and weed mixtures in the same image.

Because of this limitation, I built a larger custom dataset by combining multiple public sources and converting different annotation formats into a unified semantic segmentation format.

## Final Dataset Format

The final dataset contains **16,000+ top-down field images** with pixel-wise segmentation masks.

The masks use three semantic classes:

| Class ID | Class Name |
|---:|---|
| 0 | Background |
| 1 | Maize |
| 2 | Weeds |

All images and masks were prepared for semantic segmentation training at `1024x1024` resolution.

## Source Annotation Types

The dataset was created from multiple sources with different annotation formats:

- images with existing segmentation masks;
- images with bounding-box annotations;
- maize/weed field images requiring additional mask refinement.

The main challenge was converting inconsistent source annotations into clean pixel-wise masks.

## Bounding Box to Mask Conversion

For datasets with bounding-box annotations, I generated semantic masks using a classical computer vision pipeline.

The process included:

1. Cropping regions from the bounding boxes.
2. Applying HSV-based green filtering.
3. Creating binary masks using thresholding.
4. Cleaning masks with morphological opening and closing.
5. Extracting plant regions with connected components.
6. Calibrating thresholds per dataset.
7. Refining difficult cases with SAM-assisted mask generation.

This approach made it possible to transform object-level annotations into segmentation-ready masks.

## Mask Classes

The final masks follow this convention:

- black / class `0` = background;
- maize / class `1` = crop;
- weeds / class `2` = unwanted vegetation.

For visualization, maize is displayed in blue and weeds are displayed in red in the demo overlays.

## Training Split

The dataset was split into training and validation subsets.

The model was trained with:

- image size: `1024x1024`;
- classes: background, maize, weeds;
- augmentations: Albumentations;
- losses: Dice Loss, Focal Loss and Cross Entropy;
- metrics: IoU and foreground mIoU.

## External Evaluation

In addition to same-domain validation, the model was evaluated on an external dataset to measure generalization under domain shift.

This helped identify how well the model performs when image style, camera position, lighting and field conditions change.

## Limitations

The dataset creation process has several limitations:

- source datasets have different camera angles and lighting conditions;
- bounding-box to mask conversion can introduce imperfect masks;
- small weeds are harder to separate from background;
- domain shift remains a challenge across fields and sensors;
- the full dataset is not redistributed due to licensing and file size constraints.

## Notes

This repository does not include the full dataset. It contains documentation and representative scripts that describe the end-to-end process.