# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:36:26 2026

@author: pmjor
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

# 1. Criar o reconhecedor
recognizer = cv2.face.LBPHFaceRecognizer_create()

training_flag = True

if training_flag:
    #prepere database
    face_list = []
    class_list = []
    
    train_path = 'C:\\geral\\work\\isel\\disciplinas\\varm\\Aulas\\labs\\Project1A\\faceDataBase\\Celebrity Faces Dataset\\train'
    person_name = os.listdir(train_path)
    
    for idx, name in enumerate(person_name):
        full_path = train_path + '/' + name
        for img_name in os.listdir(full_path):
            img_full_path = full_path + '/' + img_name
            img = cv2.imread(img_full_path, 0)
            if img is None:
                continue
            nlin, ncol = img.shape
    
            # MediaPipe expects RGB
            img_rgb = cv2.cvtColor(cv2.imread(img_full_path), cv2.COLOR_BGR2RGB)
            results = mp_face_detection.process(img_rgb)
    
            if not results.detections:
                continue
    
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * ncol)
                y1 = int(bbox.ymin * nlin)
                w = int(bbox.width * ncol)
                h = int(bbox.height * nlin)
                x2, y2 = x1 + w, y1 + h
                
                if x1 >= 0 and y1 >= 0 and x2 <= ncol and y2 <= nlin:
                    face_img = img[y1:y2, x1:x2]
                    face_list.append(face_img)
                    class_list.append(idx)
    
    # 2. Treino
    recognizer.train(face_list, np.array(class_list))
    recognizer.save('trainingData.yml')
else:
    if os.path.exists('trainingData - with me.yml'):
        recognizer.read('trainingData - with me.yml')
    elif os.path.exists('trainingData.yml'):
        recognizer.read('trainingData.yml')
    
    train_path = 'C:\\geral\\work\\isel\\disciplinas\\varm\\Aulas\\labs\\Project1A\\faceDataBase\\Celebrity Faces Dataset\\train'
    person_name = os.listdir(train_path)

# 3. Predição
cap = cv2.VideoCapture(0)
cont_flag = True

while cont_flag:
    _, image = cap.read()
    if image is None:
        continue
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nlin, ncol = img_gray.shape
    
    results = mp_face_detection.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            # Get Bounding Box
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * ncol)
            y1 = int(bbox.ymin * nlin)
            w = int(bbox.width * ncol)
            h = int(bbox.height * nlin)
            x2, y2 = x1 + w, y1 + h
            
            if x1 >= 0 and y1 >= 0 and x2 <= ncol and y2 <= nlin:
                # Keypoints: 0=Right Eye, 1=Left Eye (relative to the person)
                # In MediaPipe: index 0 is "right eye", index 1 is "left eye"
                keypoints = detection.location_data.relative_keypoints
                
                # Draw landmarks
                # Right Eye
                right_eye_x = int(keypoints[0].x * ncol)
                right_eye_y = int(keypoints[0].y * nlin)
                cv2.circle(image, (right_eye_x, right_eye_y), 2, (0, 0, 255), -1)
                
                # Left Eye
                left_eye_x = int(keypoints[1].x * ncol)
                left_eye_y = int(keypoints[1].y * nlin)
                cv2.circle(image, (left_eye_x, left_eye_y), 2, (0, 255, 0), -1)
                
                face_img_test = img_gray[y1:y2, x1:x2]
                if face_img_test.size > 0:
                    label_id, confidence = recognizer.predict(face_img_test)
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0 , 0), 1)
                    if label_id < len(person_name):
                        text = person_name[label_id] + " : " + str(round(confidence, 2))
                    else:
                        text = "Unknown"
                    cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
