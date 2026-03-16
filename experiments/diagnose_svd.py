import os
import cv2
import numpy as np
from recognizer import FisherEigenFaceRecognizer
from normalizer import FaceNormalizer

def diagnose():
    train_path = r'C:\geral\work\isel\disciplinas\varm\Aulas\labs\Project1A\faceDataBase\Celebrity Faces Dataset\train'
    
    if not os.path.exists(train_path):
        print(f"Error: Path not found {train_path}")
        return

    normalizer = FaceNormalizer(num_rows=128, num_cols=128)
    recognizer = FisherEigenFaceRecognizer(normalizer=normalizer)

    person_names = sorted(os.listdir(train_path))
    image_paths = []
    labels = []
    
    print("Loading data...")
    for idx, name in enumerate(person_names[:5]): # Only first 5 people for speed
        full_path = os.path.join(train_path, name)
        if not os.path.isdir(full_path): continue
        for img_name in os.listdir(full_path)[:10]: # 10 images each
            image_paths.append(os.path.join(full_path, img_name))
            labels.append(idx)

    print(f"Checking for NaNs in normalized data...")
    X, filtered_labels = normalizer.normalize(image_paths, labels, require_eyes=True)
    
    if np.any(np.isnan(X)):
        print("CRITICAL: NaNs found in X!")
    if np.any(np.isinf(X)):
        print("CRITICAL: Infs found in X!")
        
    print(f"Data shape: {X.shape}")
    X_flat = X.reshape(X.shape[0], -1)
    print(f"Flattened shape: {X_flat.shape}")

    mean_face = np.mean(X_flat, axis=0)
    X_centered = X_flat - mean_face
    
    print("Testing SVD directly...")
    try:
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        print("SVD successful.")
    except Exception as e:
        print(f"SVD failed: {e}")

    print("Testing train()...")
    try:
        recognizer.train(image_paths, labels)
        print("Training successful.")
        print(f"Mean face min/max: {recognizer.mean_face.min()}, {recognizer.mean_face.max()}")
        print(f"W projection matrix min/max: {recognizer.W.min()}, {recognizer.W.max()}")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    diagnose()
