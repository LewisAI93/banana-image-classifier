# 🍌 Banana Object Detection

A computer-vision project that uses **Faster R-CNN** to detect bananas in annotated images.

Built as part of my Data Engineering (Artificial Intelligence) studies at SAMK University of Applied Sciences.

## Overview

This project implements an end-to-end object-detection workflow:

- Load and prepare annotated image data
- Apply image augmentations
- Train a Faster R-CNN model
- Monitor training and validation loss
- Evaluate model predictions

Unlike image classification, object detection identifies both the **banana** and its location using a bounding box.

## Tech Stack

- Python
- PyTorch
- Faster R-CNN with ResNet-50 FPN
- Computer Vision
- Git and GitHub

## Project Structure

```text
object_detection/
├── args.py            # Training configuration
├── augmentations.py   # Image transforms
├── dataset.py         # Dataset loading
├── evaluate.py        # Evaluation utilities
├── main.py            # Project entry point
├── model.py           # Model configuration
├── trainer.py         # Training loop
└── utils.py           # Helper functions
```

## Results

| Item | Result |
|---|---|
| Dataset | 100 annotated banana images |
| Model | Faster R-CNN ResNet-50 FPN |
| Best validation loss | 0.6746 |
| Best epoch | 26 |

## Future Improvements

- Add prediction examples with bounding boxes
- Measure IoU, precision, recall, and mAP
- Add single-image inference
- Create a small web interface or API
