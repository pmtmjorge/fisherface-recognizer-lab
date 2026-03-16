# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
from normalizer import FaceNormalizer
from recognizer import FisherEigenFaceRecognizer

def main():
    # 1. Configuration
    train_path = r'C:\geral\work\isel\disciplinas\varm\Aulas\labs\Project1A\faceDataBase\Celebrity Faces Dataset\train'
    model_file = 'fisher_celebrity_model.npz'
    training_flag = True # Set to False to load existing model
    
    # 2. Initialize Normalizer and Recognizer
    # Standard resolution for Fisherfaces (128x128)
    normalizer = FaceNormalizer(num_rows=128, num_cols=128)
    recognizer = FisherEigenFaceRecognizer(normalizer=normalizer, threshold=3000.0) # Adjust threshold as needed

    if not os.path.exists(train_path):
        print(f"Error: Dataset path not found: {train_path}")
        return

    person_names = sorted(os.listdir(train_path))

    if training_flag:
        print("Preparing training data...")
        image_paths = []
        labels = []
        
        # Configuration: Limit the number of images per person to manage memory and speed.
        # Set to None or a very large number to use all available images.
        max_images_per_person = 100
        
        for idx, name in enumerate(person_names):
            full_path = os.path.join(train_path, name)
            if not os.path.isdir(full_path):
                continue
                
            print(f"Loading images for: {name}")
            person_imgs = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            # Select images based on the limit
            selected_imgs = person_imgs[:max_images_per_person] if max_images_per_person else person_imgs
            
            for img_name in selected_imgs:
                img_full_path = os.path.join(full_path, img_name)
                image_paths.append(img_full_path)
                labels.append(idx)
        
        print(f"Aligning and filtering {len(image_paths)} images. Keeping only those with clear eyes...")
        try:
            # 1. Normalize and filter
            X, filtered_labels = normalizer.normalize(image_paths, labels, require_eyes=True)
            
            if len(filtered_labels) == 0:
                print("Error: No images were successfully normalized with eyes detected. Training aborted.")
                return
            
            print(f"Training on {len(filtered_labels)} high-quality images...")
            
            # 2. Train the recognizer. 
            # We pass the pre-normalized matrix X and filtered_labels.
            # To avoid the recognizer trying to re-normalize, we can temporarily disable its normalizer 
            # or simply rely on the fact that it handles matrices.
            # However, the recognizer's predict() still needs the normalizer for live frames.
            recognizer.train(X, filtered_labels)
            
            recognizer.save(model_file)
            print(f"Model trained and saved to {model_file}")
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        if os.path.exists(model_file):
            print(f"Loading model from {model_file}... (This may take a moment)")
            try:
                recognizer.load(model_file)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        else:
            print(f"Model file '{model_file}' not found. You must run with training_flag = True first.")
            return

    # 3. Live Recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device (Camera).")
        return
    print("Starting live Fisherfaces recognition (Normalized). Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # The recognizer's predict method uses the normalizer internally.
        # It will handle eye detection, alignment, and grayscale conversion.
        try:
            # require_eyes=True ensures we skip recognition if alignment is not perfect
            label_id, distance = recognizer.predict(frame, require_eyes=True)
            
            # If label_id is -2, it means eyes were not detected/aligned
            if label_id != -2:
                # We show detection for the first face (the one being predicted)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = normalizer.mp_face_detection.process(image_rgb)
                
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # 1. Draw Bounding Box
                    x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
                    x2, y2 = x1 + int(bbox.width * iw), y1 + int(bbox.height * ih)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 2. Draw Eye Landmarks (Red for Right Eye, Green for Left Eye)
                    keypoints = detection.location_data.relative_keypoints
                    if len(keypoints) >= 2:
                        re_x, re_y = int(keypoints[0].x * iw), int(keypoints[0].y * ih)
                        le_x, le_y = int(keypoints[1].x * iw), int(keypoints[1].y * ih)
                        cv2.circle(frame, (re_x, re_y), 4, (0, 0, 255), -1) 
                        cv2.circle(frame, (le_x, le_y), 4, (0, 255, 0), -1)
                    
                    # 3. Draw Label
                    if label_id != -1 and label_id < len(person_names):
                        name = person_names[label_id]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        
                    text = f"{name} ({distance:.0f})"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            pass

        cv2.imshow("Fisherfaces Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
