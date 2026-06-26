# Deployment Notes

This document summarizes the deployment paths used in the Smart Spray Edge AI project.

The same semantic segmentation model was used in three deployment scenarios:

1. Web inference demo.
2. Drone field mapping.
3. Jetson-based smart spraying prototype.

## 1. Web Inference Deployment

The web demo uses a distributed architecture:

```text
Angular UI → .NET Backend → Flask ONNX API → Segmentation Overlay → Comparison Slider