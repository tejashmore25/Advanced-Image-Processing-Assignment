# Assignment 4

This assignment covers three independent signal processing and computer vision tasks:

1. **Scalar quantization of a Laplacian source** using both uniform and Lloyd-Max quantizers.
2. **Objective image quality assessment** by correlating full-reference metrics with blur DMOS scores.
3. **Vehicle detection with YOLOv8**, including evaluation on a held-out test set and a campus domain-shift image set.

## Problem Solved

### 1. Uniform and Lloyd-Max Quantization

The first problem implements scalar quantization for a Laplacian random variable. It compares:

- a **uniform quantizer** optimized for the source,
- a **Lloyd-Max quantizer** obtained by iterative reconstruction/boundary updates.

The goal is to minimize mean squared error (MSE) and compare the final quantizer design.

### 2. Blur Quality Assessment

The second problem evaluates how well common image-quality measures track blur distortion. For each distorted image, the code computes:

- **PSNR**
- **SSIM luminance**
- **SSIM contrast**
- **SSIM structure**
- **overall SSIM**

These scores are then compared against the ground-truth blur DMOS values using Spearman rank correlation.

### 3. YOLO-Based Vehicle Detection

The third problem trains a **YOLOv8n** object detector on a vehicle dataset with five classes:

- bus
- car
- pickup
- truck
- van

The detector is then evaluated on:

- the held-out test split,
- a separate campus image set to study domain shift.

The code also saves qualitative inference outputs, confusion matrices, and training curves.

## Repository Structure

```text
Assignment 4/
|-- Problem1.ipynb                  # Scalar quantization experiments for Laplacian source
|-- Problem2.ipynb                  # Blur metric correlation analysis
|-- Problem3_1.ipynb                # YOLOv8 training and held-out test evaluation
|-- Problem3_2.ipynb                # Campus image inference / domain shift analysis
|-- Problem Statement.pdf           # Original assignment statement
|-- Report_TejashMore_27077.pdf     # Final report with observations and results
|-- README.md                       # Project overview
|-- campus_ouput_classified.png     # Example campus inference output
|-- data/
|   |-- hw5/
|   |   |-- gblur/                  # Gaussian-blurred images used in Problem 2
|   |   |-- refimgs/                # Reference images used in Problem 2
|   |   |-- hw5.mat                 # Blur DMOS and metadata file
|   |   `-- readme.txt              # Dataset notes
|   |-- vehicles/                   # Vehicle detection dataset for Problems 3.1 and 3.2
|   `-- campus_images/              # Campus images used for domain-shift inference
|-- runs/
|   `-- detect/
|       |-- outputs/yolo_training_run/        # YOLO training logs, curves, confusion matrices, and weights
|       |-- val/                              # Validation visuals from evaluation
|       |-- val2/                             # Additional validation visuals
|       `-- assignment_outputs/              # Saved inference images for test/campus sets
|-- yolo26n.pt                      # Saved YOLO checkpoint
`-- yolov8n.pt                      # Pretrained YOLOv8n model used as the training base
```

## Results and Observations

### 1. Quantization Results

The quantizer comparison shows that Lloyd-Max improves reconstruction quality over uniform quantization:

| Quantizer | Step Size / Boundaries | Reconstruction Points | MSE |
|---|---|---|---:|
| Uniform | `Delta = 4.6134` | `[-6.9201, -2.3067, 2.3067, 6.9201]` | `3.5334` |
| Lloyd-Max | Learned iteratively | `[-7.7793, -1.7805, 1.7805, 7.7793]` | `3.1715` |

Observation:

- Lloyd-Max converged in **11 iterations**.
- It reduced the distortion by **0.3619 MSE** compared with the uniform quantizer.
- The optimized reconstruction levels are more spread out near the tails, which matches the Laplacian source shape better.

### 2. Blur Metric Correlation

The blur-analysis notebook reports the following Spearman rank correlations with DMOS:

| Metric | Spearman Correlation |
|---|---:|
| PSNR | 0.7823 |
| SSIM luminance | 0.9335 |
| SSIM contrast | 0.9068 |
| SSIM structure | 0.9055 |
| Overall SSIM | 0.9036 |

Observation:

- SSIM-based measures correlate better with human blur scores than PSNR.
- Among the evaluated metrics, **SSIM luminance** has the strongest correlation with DMOS.
- This indicates that perceptual structure-aware metrics are more reliable than pure error-based measures for blur assessment.

### 3. Vehicle Detection Results

The final YOLOv8n model was trained for **15 epochs** on the vehicle dataset. The held-out evaluation on the test split reports:

| Metric | Value |
|---|---:|
| Precision | 0.856 |
| Recall | 0.822 |
| mAP@0.5 | 0.892 |
| mAP@0.5:0.95 | 0.699 |

Per-class performance from the final evaluation:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| bus | 0.866 | 0.792 | 0.885 | 0.628 |
| car | 0.854 | 0.790 | 0.875 | 0.589 |
| pickup | 0.854 | 0.855 | 0.906 | 0.697 |
| truck | 0.883 | 0.844 | 0.909 | 0.748 |
| van | 0.839 | 0.824 | 0.885 | 0.811 |

Observation:

- Training steadily improved the detector, with the final validation curve reaching strong vehicle-class performance.
- The best class-wise results were observed for **truck** and **van**.
- The campus inference outputs show a clear **domain shift**: some images are detected well, while others have missed detections or no detections at all.
- This suggests the model generalizes reasonably to real campus scenes, but performance degrades when the appearance differs from the training distribution.

## How to Run

The notebooks are meant to be executed from inside the `Assignment 4` directory.

Recommended packages:

```bash
pip install numpy scipy matplotlib opencv-python scikit-image scikit-learn torch torchvision ultralytics pillow
```

Suggested notebook order:

```bash
jupyter notebook Problem1.ipynb
jupyter notebook Problem2.ipynb
jupyter notebook Problem3_1.ipynb
jupyter notebook Problem3_2.ipynb
```

## Key Takeaways

- Lloyd-Max quantization outperforms uniform quantization for the Laplacian source used here.
- SSIM is a better blur-quality indicator than PSNR when compared with DMOS.
- YOLOv8n achieves strong performance on the vehicle dataset, but campus images reveal the effect of domain shift.
- The saved plots in `runs/detect/` provide the training, validation, and qualitative inference evidence for the detector.
