# Hands-on CV Project 2 – Multimodal Fusion (RGB + LiDAR)

## Overview

This project investigates **multimodal classification** using paired **RGB images and LiDAR depth maps**. The goal is to classify simple geometric objects (**cubes vs. spheres**) and compare different **fusion strategies** and **downsampling methods**.

The work is organized according to the assessment tasks: dataset handling + visualization (Tasks 1–2), fusion architectures (Task 3), and pooling comparison (Task 4).

---

## Dataset Structure

Expected dataset layout:

```text
datasets/
├── cubes/
│   ├── rgb/
│   │   └── 0000.png
│   └── lidar/
│       └── 0000.npy
└── spheres/
    ├── rgb/
    │   └── 0000.png
    └── lidar/
        └── 0000.npy
```

Labels:

0: Cube

1: Sphere

## Project Tasks
Task 1 & 2: Dataset Loading & Visualization
Data Loader: A custom PyTorch implementation that matches RGB and LiDAR files and assigns correct labels.

Visualization: Tools to sanity-check spatial alignment between sensors and verify sample integrity before training.

Task 3: Multimodal Fusion Architectures
We implemented and compared several fusion strategies to combine the two data streams:

Late Fusion: Features are extracted independently and combined at the final decision layer.

Intermediate Fusion: Features are combined within the network backbone using:

Concatenation: Stacking feature channels.

Element-wise Addition: Summing feature maps.

Hadamard Product: Element-wise multiplication to emphasize overlapping features.

Task 4: Pooling Strategy Comparison
Each fusion strategy was evaluated against two downsampling techniques:

Max Pooling: Non-linear downsampling.

Strided Convolution: Learnable downsampling.

## Results & Analysis
Across all experiments, models were evaluated based on final validation F1-score, parameter count, training time, and peak GPU memory usage.

Key Findings
Performance: Every Max-Pooling variant outperformed its Strided Convolution counterpart.

Best Model: The Intermediate Hadamard Fusion + Max-Pooling architecture achieved the highest overall accuracy.

Resource Usage: Parameter counts and training times are comparable, but Max-Pooling variants require significantly more GPU memory.

## Getting Started
Installation
Bash```

pip install -r requirements.txt
Dependencies: PyTorch, NumPy, scikit-learn, and Weights & Biases (wandb).
```

Project Structure
.
├── src/
│   ├── datasets/      # Data pipeline
│   ├── models/        # Fusion model definitions
│   ├── train.py       # Training entry point
│   ├── eval.py        # Metrics & validation
│   └── visualize.py   # Alignment & sanity checks
├── notebooks/         # Experiment analysis
├── checkpoints/       # Saved .pt models
├── results/           # metrics.csv logs
└── requirements.txt

##  Usage
To train a specific configuration, you must simply 
1. adjust the dataset path 
2. Install the requirements into your venv
3. and run the according notebook (1, 2, 3 or 4)




## Conclusion
Experimental results demonstrate that Max-Pooling provides a superior inductive bias for this specific RGB–LiDAR classification task. The Intermediate Hadamard approach is the preferred fusion method, effectively capturing the relationship between RGB textures and LiDAR geometry.