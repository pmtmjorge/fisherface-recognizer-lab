# Fisher & Eigenface Face Recognition

A Python implementation of a face recognition system using Eigenfaces (PCA) and Fisherfaces (LDA / MDC). This project provides a custom `FisherEigenFaceRecognizer` class designed to mimic the OpenCV FaceRecognizer API while being transparent and modular.

## 🚀 Overview

This repository contains tools for training and evaluating a face recognition model based on subspace projection techniques:
1.  **PCA (Principal Component Analysis)**: Used to reduce dimensionality and extract "Eigenfaces".
2.  **LDA (Linear Discriminant Analysis)**: Used to maximize class separability by finding "Fisherfaces".

## 📂 Key Files

-   **`recognizer.py`**: Contains the `FisherEigenFaceRecognizer` class. It implements the mathematical logic for training (SVD/Eigenvalue decomposition) and prediction (Nearest Neighbor).
-   **`test_accuracy.py`**: An evaluation script to test the model on a dataset. It uses `scikit-learn` for metrics and `matplotlib` for visualization.

## 📂 Project Structure

```text
.
├── experiments/       # Legacy scripts and experimental code
├── local_tmp/         # Local models, images, and data (git-ignored)
├── .gitignore         # Rules to keep the repository clean
├── README.md          # Documentation
├── recognizer.py      # Core Fisherface / Eigenface logic
├── requirements.txt   # Python dependencies
└── test_accuracy.py   # Main evaluation script
```

## 🛠️ Installation

Ensure you have the required dependencies installed:

```bash
pip install numpy opencv-python scikit-learn matplotlib
```

## 📷 Dataset Preparation

For the Fisherface and Eigenface algorithms to work effectively, the input dataset must follow these guidelines:

1.  **Uniform Size**: All images in the dataset should be resized to the same dimensions (e.g., 100x100 pixels). The `test_accuracy.py` script includes a basic auto-resize to the reference size of the first image, but pre-resized datasets are more reliable.
2.  **Geometric Alignment**: For optimal accuracy, images should be normalized such that the **eyes are at the same pixel positions** in every frame. This ensures that the PCA/LDA focus on facial features rather than background noise or tilt.
3.  **Intensity Normalization**: Applying techniques like **Histogram Equalization** can improve robustness to varying lighting conditions.
4.  **Grayscale**: The recognizer expects grayscale images.

## 🧪 Evaluation (`test_accuracy.py`)

The evaluation script allows you to split a dataset into training and testing sets, train the model, and visualize the performance.

### Features
-   **Stratified Splitting**: Ensures every person has representative images in both training and test sets.
-   **Automated Metrics**: Computes Accuracy, Precision, Recall, and F1-score using `scikit-learn`.
-   **Confusion Matrix**: Generates and saves a visual heatmap (`confusion_matrix.png`) showing which classes are being confused.

### Example: Running with Split and Seed

To run the evaluation with a specific training percentage (e.g., 75%) and a random seed for reproducibility:

```bash
python test_accuracy.py --dataset "path/to/your/dataset" --split 75.0 --seed 42
```

-   `--split 75.0`: Uses 75% of images for training and 25% for testing.
-   `--seed 42`: Fixes the random state so you get the same results every time you run the experiment.

## 💻 Programmatic Usage

You can use the recognizer in your own Python scripts as follows:

```python
from recognizer import FisherEigenFaceRecognizer
import cv2

# 1. Initialize
recognizer = FisherEigenFaceRecognizer()

# 2. Train
# images: list of grayscale numpy arrays (all same size)
# labels: list of integers
recognizer.train(train_images, train_labels)

# 3. Predict
label, confidence = recognizer.predict(test_image)
print(f"Predicted Label: {label} with distance {confidence:.2f}")

# 4. Save/Load Model
recognizer.save("my_model.npz")
recognizer.load("my_model.npz")
```

## 📊 Results Summary
After running `test_accuracy.py`, the script prints a **Classification Report** and saves a **Confusion Matrix plot**. This helps identify if the model struggles with specific individuals or requires more training data/better alignment.

## 🛠️ Context

This project was developed in collaboration with **Antigravity**, an agentic AI coding assistant. The AI assisted in refactoring the core algorithms for better readability, making the code more robust, and organizing the repository structure.
