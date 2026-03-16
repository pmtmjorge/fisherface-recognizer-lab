import numpy as np
import cv2
import mediapipe as mp

class FaceNormalizer:
    """
    Normalizes face images using a similarity transform to align eye positions.
    Automatically detects eyes using MediaPipe Face Detection (Short-range model).
    """
    def __init__(self, num_rows=128, num_cols=128, apply_clahe=True):
        """
        :param num_rows: Lines in the normalized image.
        :param num_cols: Columns in the normalized image.
        :param apply_clahe: Whether to apply CLAHE.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.apply_clahe = apply_clahe
        
        # Target eye positions
        self.target_eye_y = 0.428 * num_rows
        self.target_right_eye_x = 0.348 * num_cols
        self.target_left_eye_x = 0.674 * num_cols
        
        # Initialize MediaPipe (model_selection=0 is for short-range faces)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def get_eyes_mediapipe(self, image):
        """
        Detects face landmarks using MediaPipe and extracts eye coordinates.
        Note: MP detects Right Eye at index 0 and Left Eye at index 1.
        """
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        # We take the first face detected
        detection = results.detections[0]
        keypoints = detection.location_data.relative_keypoints
        
        ih, iw, _ = image.shape
        
        # MediaPipe keypoints: 0=Right Eye, 1=Left Eye
        right_eye = (keypoints[0].x * iw, keypoints[0].y * ih)
        left_eye = (keypoints[1].x * iw, keypoints[1].y * ih)
        
        # Also return the face bounding box as fallback
        bbox = detection.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * iw))
        y1 = int(max(0, bbox.ymin * ih))
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        face_bbox = (x1, y1, w, h)
        
        return (right_eye, left_eye), face_bbox

    def align_and_crop(self, image, right_eye, left_eye):
        """
        Performs similarity transform to align the eyes.
        """
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        dist_source = np.sqrt(dx**2 + dy**2)
        if dist_source < 1e-6:
            # Avoid division by zero, return center-cropped and resized image
            ih, iw = image.shape[:2]
            center_x, center_y = iw // 2, ih // 2
            size = min(iw, ih)
            x1, y1 = max(0, center_x - size // 2), max(0, center_y - size // 2)
            face_crop = image[y1:y1+size, x1:x1+size]
            return cv2.resize(face_crop, (self.num_cols, self.num_rows), interpolation=cv2.INTER_AREA)

        dist_target = self.target_left_eye_x - self.target_right_eye_x
        scale = dist_target / dist_source
        
        eye_center = (
            int((right_eye[0] + left_eye[0]) // 2),
            int((right_eye[1] + left_eye[1]) // 2)
        )
        
        target_center = (
            (self.target_right_eye_x + self.target_left_eye_x) / 2,
            self.target_eye_y
        )
        
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        M[0, 2] += target_center[0] - eye_center[0]
        M[1, 2] += target_center[1] - eye_center[1]
        
        face_aligned = cv2.warpAffine(image, M, (self.num_cols, self.num_rows), flags=cv2.INTER_CUBIC)
        return face_aligned

    def _normalize_single(self, image_input, require_eyes=False):
        """
        Normalize a single image with automatic eye detection.
        :param image_input: Either a numpy array (BGR/Gray) or a string (file path).
        :param require_eyes: If True, returns None if eyes are not detected.
        """
        # 1. Load image if a path is provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                if require_eyes:
                    return None
                raise ValueError(f"Could not load image from path: {image_input}")
        else:
            image = image_input

        # 2. Ensure BGR for detection/conversion logic
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
            
        # 3. Detect Eyes and Bounding Box
        detection_result = self.get_eyes_mediapipe(image_bgr)
        
        # 4. Convert to grayscale for the final normalized output
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if detection_result is not None:
            eyes, bbox = detection_result
            # Check if eyes are valid (not just same point)
            if eyes is not None and abs(eyes[0][0] - eyes[1][0]) > 1e-3:
                gray = self.align_and_crop(gray, eyes[0], eyes[1])
            elif not require_eyes:
                # Fallback 1: Crop to the face bounding box first
                x, y, w, h = bbox
                face_crop = gray[y:y+h, x:x+w]
                if face_crop.size > 0:
                    gray = cv2.resize(face_crop, (self.num_cols, self.num_rows), interpolation=cv2.INTER_AREA)
                else:
                    gray = cv2.resize(gray, (self.num_cols, self.num_rows), interpolation=cv2.INTER_AREA)
            else:
                return None
        elif not require_eyes:
            # Fallback 2: No face detected at all
            gray = cv2.resize(gray, (self.num_cols, self.num_rows), interpolation=cv2.INTER_AREA)
        else:
            return None

        # 5. Contrast Enhancement
        if self.apply_clahe:
            normalized = self.clahe.apply(gray)
        else:
            normalized = cv2.equalizeHist(gray)

        # 6. Return normalized face as 2D array (not flattened)
        return normalized.astype(np.float32)

    def normalize(self, images, labels=None, require_eyes=False):
        """
        Normalizes a collection of images.
        :param images: List of images or a single image.
        :param labels: Optional list of labels to be filtered in sync with images.
        :param require_eyes: If True, filters out images where eyes are not detected.
        :return: If labels is None, a 3D numpy array. If labels is provided, (images, labels) tuple.
        """
        is_single = not isinstance(images, (list, tuple))
        if is_single:
            images = [images]

        res_images = []
        res_labels = []

        for i, img in enumerate(images):
            norm = self._normalize_single(img, require_eyes=require_eyes)
            if norm is not None:
                res_images.append(norm)
                if labels is not None:
                    res_labels.append(labels[i])

        X = np.array(res_images)
        
        if labels is not None:
            return X, np.array(res_labels)
        
        # If input was a single image and we are just returning X
        # we still return a 3D array for consistency, or we could handle it differently.
        return X

    def reconstruct(self, flat_vector):
        return flat_vector.reshape((self.num_rows, self.num_cols))
