import os
import cv2
import numpy as np
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        print(f"{desc}: Processing images...")
        return iterable
from normalizer import FaceNormalizer

def batch_normalize(input_dir, output_dir, img_size=(128, 128), require_eyes=False):
    """
    Walks through the input directory, normalizes all images found, 
    and saves them in the output directory preserving the structure.
    """
    # 1. Initialize normalizer
    normalizer = FaceNormalizer(num_rows=img_size[0], num_cols=img_size[1])
    
    # 2. Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # 3. Gather all files to process for progress bar
    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files_to_process)} images. Starting normalization...")
    
    success_count = 0
    fail_count = 0
    
    # 4. Process each image
    for file_path in tqdm(files_to_process, desc="Normalizing"):
        # Determine relative path to maintain structure
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Ensure output directory exists (including subfolders)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                print(f"\n[!] Could not load: {file_path}")
                fail_count += 1
                continue
            
            # Normalize
            # normalize() returns a 3D array (Samples, Rows, Cols)
            # Since we pass a single image, it will be (1, Rows, Cols)
            normalized_batch = normalizer.normalize(img, require_eyes=require_eyes)
            
            if len(normalized_batch) > 0:
                # Take the first (and only) normalized face
                normalized_img = normalized_batch[0]
                
                # Convert to uint8 for saving (OpenCV imwrite expects 0-255)
                # FaceNormalizer returns float32
                normalized_img_uint8 = normalized_img.astype(np.uint8)
                
                # Save normalized image (keeping same extension or forcing .jpg)
                # It's usually better to force .jpg if we want consistency, 
                # but users might want to keep the same. 
                # Let's keep the same for now.
                cv2.imwrite(output_path, normalized_img_uint8)
                success_count += 1
            else:
                # If require_eyes=True and no eyes found, it returns empty
                # If require_eyes=False, it should always return something (fallback)
                # Unless something else went wrong
                fail_count += 1
                
        except Exception as e:
            print(f"\n[!] Error processing {file_path}: {e}")
            fail_count += 1

    print("\nNormalization Complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Failed/Skipped: {fail_count}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch normalize face images while preserving directory structure.")
    parser.add_argument("input", help="Source directory containing images")
    parser.add_argument("output", help="Destination directory for normalized images")
    parser.add_argument("--size", type=int, default=128, help="Output image size (default: 128)")
    parser.add_argument("--require_eyes", action="store_true", help="Only save images where eyes were detected")
    
    args = parser.parse_args()
    
    batch_normalize(args.input, args.output, img_size=(args.size, args.size), require_eyes=args.require_eyes)
