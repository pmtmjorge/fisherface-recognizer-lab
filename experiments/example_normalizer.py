import cv2
import numpy as np
from normalizer import FaceNormalizer

def main():
    # 1. Initialize the normalizer with desired output resolution
    # Standard dimensions: 112x112 or 128x128
    rows, cols = 128, 128
    normalizer = FaceNormalizer(num_rows=rows, num_cols=cols)
    
    print(f"Normalizer initialized: {cols}x{rows}")
    print("Alignment targets: Y={:.1f}, RightX={:.1f}, LeftX={:.1f}".format(
        normalizer.target_eye_y, normalizer.target_right_eye_x, normalizer.target_left_eye_x
    ))

    # --- Example 1: Normalize a single image ---
    # Replace 'my_face.jpg' with a real path to a face image
    image_path = 'pedroMjorge2.jpg' 
    img = cv2.imread(image_path)
    '''
    if img is not None:
        print(f"\nProcessing single image: {image_path}")
        # normalize() returns a 2D matrix (Samples x Pixels)
        # Even for one image, it returns (1, Rows*Cols)
        flat_matrix = normalizer.normalize(img)
        print(f"Output shape: {flat_matrix.shape}") # Should be (1, 16384) for 128x128
        
        # To visualize the result:
        # Use reconstruct() to turn the flat vector back into a 2D image
        normalized_img = normalizer.reconstruct(flat_matrix[0])
        
        # Save or display
        cv2.imwrite('normalized_single.jpg', normalized_img)
        print("Saved normalized result to: normalized_single.jpg")
        
        # Show normalized image
        cv2.imshow('Normalized Single', normalized_img.astype(np.uint8))
        print("Press any key to continue...")
        cv2.waitKey(0)
    else:
        print(f"\n[!] Could not load {image_path}. Please provide a valid face image.")
    '''
    # --- Example 2: Normalize a set of images ---
    # Simulating a list of images based on paths you provided
    image_list = ['pedroMjorge1.jpg', 'pedroMjorge2.jpg', 'pedroMjorge6.jpg']
    
    print("\nProcessing a set of images...")
    try:
        # normalize() handles the whole batch at once
        batch_matrix = normalizer.normalize(image_list)
        print(f"Batch output shape: {batch_matrix.shape}")
        
        # Reconstruct and stack images horizontally for visualization
        reconstructed_images = [normalizer.reconstruct(row).astype(np.uint8) for row in batch_matrix]
        tiled_images = np.hstack(reconstructed_images)
        
        cv2.imshow('Normalized Batch', tiled_images)
        print("Showing batch (tiled). Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        print("Make sure files exist: ", image_list)

if __name__ == "__main__":
    main()
