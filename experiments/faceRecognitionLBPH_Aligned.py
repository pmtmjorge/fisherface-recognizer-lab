# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from normalizer import FaceNormalizer

def main():
    # 1. Configuration
    train_path = r'C:\geral\work\isel\disciplinas\varm\Aulas\labs\Project1A\faceDataBase\Celebrity Faces Dataset\train'
    model_file = 'lbph_normalized_model.yml'
    training_flag = True # Set to False to load existing model
    
    # 2. Initialize Normalizer and LBPH Recognizer
    # LBPH can handle various sizes, but 128x128 with eye alignment is very effective
    normalizer = FaceNormalizer(num_rows=128, num_cols=128, apply_clahe=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.exists(train_path):
        print(f"Error: Dataset path not found: {train_path}")
        return

    person_names = sorted(os.listdir(train_path))

    if training_flag:
        print("Preparing training data with strict eye-alignment...")
        image_paths = []
        labels = []
        
        # 1. Collect all paths and labels first
        for idx, name in enumerate(person_names):
            name_path = os.path.join(train_path, name)
            if not os.path.isdir(name_path):
                continue
                
            print(f"Collecting paths for: {name}")
            for img_name in os.listdir(name_path):
                img_path = os.path.join(name_path, img_name)
                image_paths.append(img_path)
                labels.append(idx)
        
        # 2. Use the new synchronized batch normalization with strict eye requirement
        print(f"Aligning {len(image_paths)} images. Skipping those without clear eyes...")
        X, filtered_labels = normalizer.normalize(image_paths, labels, require_eyes=True)
        
        # Convert 3D array (N, H, W) to list of 2D uint8 images for OpenCV
        face_list = [face.astype(np.uint8) for face in X]
        
        print(f"Training LBPH on {len(face_list)} high-quality samples...")
        recognizer.train(face_list, filtered_labels)
        recognizer.save(model_file)
        print(f"Model saved to {model_file}")
    else:
        if os.path.exists(model_file):
            print(f"Loading model from {model_file}... (This may take a moment)")
            try:
                recognizer.read(model_file)
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
    print("Starting live LBPH recognition (Normalized). Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 1. Automatically detect eyes for high-quality alignment
            # require_eyes=True means it returns an empty array if eyes aren't found
            aligned_faces = normalizer.normalize(frame, require_eyes=True)
            
            if aligned_faces.shape[0] > 0:
                face_to_predict = aligned_faces[0].astype(np.uint8)
                label_id, confidence = recognizer.predict(face_to_predict)
                
                # To display the bounding box, we use the detection results from the normalizer
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = normalizer.mp_face_detection.process(image_rgb)
                
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # 1. Draw Bounding Box
                    x1 = int(bbox.xmin * iw)
                    y1 = int(bbox.ymin * ih)
                    x2 = x1 + int(bbox.width * iw)
                    y2 = y1 + int(bbox.height * ih)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 2. Draw Eye Landmarks (Red for Right Eye, Green for Left Eye)
                    keypoints = detection.location_data.relative_keypoints
                    if len(keypoints) >= 2:
                        re_x, re_y = int(keypoints[0].x * iw), int(keypoints[0].y * ih)
                        le_x, le_y = int(keypoints[1].x * iw), int(keypoints[1].y * ih)
                        cv2.circle(frame, (re_x, re_y), 4, (0, 0, 255), -1) # Right Eye (Red)
                        cv2.circle(frame, (le_x, le_y), 4, (0, 255, 0), -1) # Left Eye (Green)
                    
                    # 3. Draw Recognition Label
                    if label_id != -1 and label_id < len(person_names):
                        name = person_names[label_id]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        
                    text = f"{name} ({confidence:.0f})"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception:
            pass

        cv2.imshow("Normalized LBPH Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
