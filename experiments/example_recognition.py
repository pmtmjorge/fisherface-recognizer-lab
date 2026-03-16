import cv2
import numpy as np
import os
from normalizer import FaceNormalizer
from recognizer import FisherEigenFaceRecognizer

def main():
    # 1. Initialize Normalizer and Recognizer
    # We use 128x128 resolution for alignment
    normalizer = FaceNormalizer(num_rows=128, num_cols=128)
    recognizer = FisherEigenFaceRecognizer(normalizer=normalizer)

    # 2. Define training data
    # Note: Fisherfaces (LDA) requires at least TWO different classes/people to work.
    image_paths = [
        'pedroMjorge1.jpg', 
        'pedroMjorge2.jpg', 
        'pedroMjorge6.jpg'
    ]
    
    # Check if files exist before proceeding
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if len(valid_paths) < 2:
        print(f"Error: Need at least 2 existing images for Fisherfaces. Found: {len(valid_paths)}")
        return

    # Assign labels (e.g., Label 1 for first two, Label 2 for the third)
    # This is just for demonstration purposes.
    labels = [1, 1, 2] 
    
    # Ensure labels match the count of valid paths
    labels = labels[:len(valid_paths)]

    print(f"Training with {len(valid_paths)} images across {len(np.unique(labels))} classes...")
    
    try:
        # 3. Train the model
        # The recognizer will use the normalizer internally to align and crop faces
        recognizer.train(valid_paths, labels)
        print("Training successful!")

        # 4. Perform a prediction
        test_img_path = valid_paths[0]  # Testing with the first image
        print(f"\nPredicting for: {test_img_path}")
        
        label, distance = recognizer.predict(test_img_path)
        
        print(f"Result -> Predicted Label: {label}, Distance: {distance:.2f}")
        
        # 5. Save the trained model
        model_name = "face_model.npz"
        recognizer.save(model_name)
        print(f"\nModel saved to {model_name}")

        # 6. Optional: Load and verify
        new_recognizer = FisherEigenFaceRecognizer(normalizer=normalizer)
        new_recognizer.load(model_name)
        print("Model reloaded from file. Testing reload...")
        label_re, dist_re = new_recognizer.predict(test_img_path)
        print(f"Reloaded Result -> Label: {label_re}, Distance: {dist_re:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
