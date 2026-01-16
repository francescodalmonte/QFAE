import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def get_union_bbox(image):
    """
    Get the union bounding box of all non-zero regions in the image.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        tuple: (x, y, w, h) bounding box coordinates, or None if no non-zero pixels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find all non-zero pixels
    coords = np.column_stack(np.where(gray > 0))
    
    if len(coords) == 0:
        return None
    
    # Get bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Convert to (x, y, width, height) format
    x, y = x_min, y_min
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return (x, y, w, h)

def make_image_natural(image):
    """
    Apply naturalization techniques to make medical images more natural using bilateral filtering.
    
    Args:
        image: Input grayscale image (numpy array)
    
    Returns:
        numpy array: Naturalized image
    """
    if image is None or image.size == 0:
        return image
    
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Skip processing if image is all zeros
    if np.max(image) == 0:
        return image
    
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        if image.max() > image.min():
            image_norm = (image - image.min()) / (image.max() - image.min())
            image = (image_norm * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Apply bilateral filtering to smooth discretization artifacts while preserving edges
    enhanced = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Clip to valid range and convert back to uint8
    enhanced = np.clip(enhanced, 0, 255)
    enhanced = enhanced.astype(np.uint8)
    
    return enhanced

def create_debug_image(original_img, processed_img, output_path):
    """
    Create a 2x2 debug image showing original image, its histogram, 
    processed image, and its histogram.
    
    Args:
        original_img: Original image (grayscale)
        processed_img: Processed image (grayscale)
        output_path: Path to save the debug image
    """
    # Ensure images are grayscale
    if len(original_img.shape) == 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    if len(processed_img.shape) == 3:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 2. Original image histogram
    ax2 = fig.add_subplot(gs[0, 1])
    # Only plot histogram for non-zero pixels
    non_zero_pixels = original_img[original_img > 0]
    if len(non_zero_pixels) > 0:
        ax2.hist(non_zero_pixels, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_title(f'Original Histogram\n(Non-zero pixels: {len(non_zero_pixels)})')
    else:
        ax2.set_title('Original Histogram\n(No non-zero pixels)')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')
    
    # 3. Processed image
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(processed_img, cmap='gray')
    ax3.set_title('Processed Image (Bilateral Filter)')
    ax3.axis('off')
    
    # 4. Processed image histogram
    ax4 = fig.add_subplot(gs[1, 1])
    # Only plot histogram for non-zero pixels
    non_zero_pixels_proc = processed_img[processed_img > 0]
    if len(non_zero_pixels_proc) > 0:
        ax4.hist(non_zero_pixels_proc, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_title(f'Processed Histogram\n(Non-zero pixels: {len(non_zero_pixels_proc)})')
    else:
        ax4.set_title('Processed Histogram\n(No non-zero pixels)')
    ax4.set_xlabel('Pixel Intensity')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the debug image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_image(input_path, output_path, debug_dir=None, debug_filename=None):
    """
    Process a single image: find union bbox of non-zero regions, crop, paste to 224x224, 
    and apply naturalization techniques.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        debug_dir: Directory to save debug images (optional)
        debug_filename: Filename for debug image (optional)
    """
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not read image {input_path}")
        return False
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    original_gray = gray_image.copy()  # Keep copy for debug
    
    # Get union bounding box of all non-zero regions
    bbox = get_union_bbox(gray_image)
    
    if bbox is None:
        print(f"Warning: No non-zero pixels found in {input_path}")
        # Create empty 224x224 image
        final_image = np.zeros((224, 224), dtype=np.uint8)
    else:
        x, y, w, h = bbox
        
        # Crop the image to the bounding box
        cropped = gray_image[y:y+h, x:x+w]
        
        # Paste into 224x224 empty image (centered) - resize only if ROI is too large
        target_size = 224
        empty_image = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Check if cropped region fits in target size
        if h <= target_size and w <= target_size:
            # ROI fits - paste directly without resizing
            start_y = (target_size - h) // 2
            start_x = (target_size - w) // 2
            empty_image[start_y:start_y+h, start_x:start_x+w] = cropped
        else:
            # ROI is too large - resize to fit while maintaining aspect ratio
            print(f"Info: ROI size ({h}x{w}) larger than target size, resizing...")
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h > 0 and new_w > 0:
                resized_crop = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Center the resized crop in the empty image
                start_y = (target_size - new_h) // 2
                start_x = (target_size - new_w) // 2
                empty_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_crop
        
        # Apply naturalization techniques
        final_image = make_image_natural(empty_image)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed image
    cv2.imwrite(output_path, final_image)
    
    # Create debug image if requested
    if debug_dir is not None and debug_filename is not None:
        debug_path = os.path.join(debug_dir, debug_filename)
        create_debug_image(original_gray, final_image, debug_path)
    
    return True

def process_dataset(input_dir, output_dir, debug_dir=None):
    """
    Process the entire dataset.
    
    Args:
        input_dir: Root directory of input dataset
        output_dir: Root directory for processed dataset
        debug_dir: Directory to save debug images (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    train_input = input_path / "train"
    train_output = output_path / "train"
    
    if train_input.exists():
        print("Processing training data...")
        train_debug_dir = debug_path / "train" if debug_dir else None
        process_split(train_input, train_output, train_debug_dir)
    
    # Process test data
    test_input = input_path / "test"
    test_output = output_path / "test"

    if test_input.exists():
        print("Processing test data...")
        test_debug_dir = debug_path / "test" if debug_dir else None
        process_split(test_input, test_output, test_debug_dir)

    # Process val data
    val_input = input_path / "valid"
    val_output = output_path / "val"

    if val_input.exists():
        print("Processing val data...")
        val_debug_dir = debug_path / "val" if debug_dir else None
        process_split(val_input, val_output, val_debug_dir)

def process_split(input_split_dir, output_split_dir, debug_split_dir=None):
    """
    Process a single split (train or test).
    
    Args:
        input_split_dir: Input split directory
        output_split_dir: Output split directory
        debug_split_dir: Debug split directory (optional)
    """
    for class_dir in input_split_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"  Processing class: {class_name}")
            
            # Create debug directory for this class if needed
            class_debug_dir = debug_split_dir / class_name if debug_split_dir else None
            if class_debug_dir:
                class_debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for img subdirectory
            img_dir = class_dir / "img"
            if img_dir.exists():
                output_class_dir = output_split_dir / class_name / "img"
                process_images_in_directory(img_dir, output_class_dir, class_debug_dir)
            else:
                # If no img subdirectory, process images directly in class directory
                output_class_dir = output_split_dir / class_name
                process_images_in_directory(class_dir, output_class_dir, class_debug_dir)

def process_images_in_directory(input_img_dir, output_img_dir, debug_class_dir=None):
    """
    Process all images in a directory.
    
    Args:
        input_img_dir: Input image directory
        output_img_dir: Output image directory
        debug_class_dir: Debug directory for this class (optional)
    """
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    processed_count = 0
    failed_count = 0
    debug_count = 0
    
    for img_file in input_img_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            input_path = str(img_file)
            output_path = str(output_img_dir / img_file.name)
            
            # Create debug image for first 5 images of each class
            debug_filename = None
            if debug_class_dir and debug_count < 10:
                debug_filename = f"debug_{img_file.stem}.png"
                debug_count += 1
            
            if process_image(input_path, output_path, debug_class_dir, debug_filename):
                processed_count += 1
            else:
                failed_count += 1
            
            if (processed_count + failed_count) % 100 == 0:
                print(f"    Processed {processed_count + failed_count} images...")
    
    print(f"    Completed: {processed_count} processed, {failed_count} failed")
    if debug_count > 0:
        print(f"    Debug images created: {debug_count}")

def main():
    """Main function to run the dataset processing."""
    # Set input and output directories
    input_dataset_dir = "./data/Liver"  # Current directory with train/test splits
    output_dataset_dir = "./data/Liver_processed"  # Output directory
    debug_dataset_dir = "./data/Liver_processed_debug"  # Debug images directory
    
    print(f"Processing dataset from {input_dataset_dir} to {output_dataset_dir}")
    print(f"Debug images will be saved to {debug_dataset_dir}")
    
    # Create output directories if they don't exist
    os.makedirs(output_dataset_dir, exist_ok=True)
    os.makedirs(debug_dataset_dir, exist_ok=True)
    
    # Process the dataset
    process_dataset(input_dataset_dir, output_dataset_dir, debug_dataset_dir)
    
    print("Dataset processing completed!")
    print(f"Check debug images in {debug_dataset_dir} to verify processing quality")

if __name__ == "__main__":
    main()