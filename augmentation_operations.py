import cv2
import numpy as np
import os
import shutil
from collections import defaultdict

# ==== Configuration ====
BLEND_RATIOS = [0.25, 0.5, 0.75, 1.0]
HUE_SHIFT_LIST = [30, 60, 90, 120, 150]
BRIGHTNESS_FACTORS = [0.4, 1.1, 5.5]
SATURATION_FACTORS = [-0.5, 0.5, 1.5]

# ==== Image Augmentation Functions ====
def apply_bgr_blend(image, ratio):
    bgr_swapped = image[..., ::-1]
    return cv2.addWeighted(image, 1 - ratio, bgr_swapped, ratio, 0)

def apply_hue_shift(image, hue_shift_deg):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = (h + hue_shift_deg) % 180
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_hsv_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def adjust_saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    if factor < 0:
        hsv[:, :, 1] = np.clip((1 - hsv[:, :, 1] / 255) * abs(factor) * 255, 0, 255)
    else:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ==== File Operations ====
def save_augmented_image(image, output_dir, base_name, suffix):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_name}_{suffix}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    return filename

def save_augmented_label(label_content, output_dir, base_name, suffix):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_name}_{suffix}.txt"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(label_content)
    return filename

def get_file_base_names(folder_path, extensions):
    """Get base names (without extension) of files with given extensions"""
    base_names = set()
    if not os.path.exists(folder_path):
        return base_names
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                base_name = os.path.splitext(file)[0]
                base_names.add(base_name)
    return base_names

def find_file_by_basename(folder_path, base_name, extensions):
    """Find file path by base name and extensions"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[0] == base_name and any(file.lower().endswith(ext) for ext in extensions):
                return os.path.join(root, file)
    return None

def validate_folders(images_folder, labels_folder):
    """Validate that image and label folders have matching files"""
    print(f"[INFO] Validating folders...")
    print(f"  Images folder: {images_folder}")
    print(f"  Labels folder: {labels_folder}")
    
    if not os.path.exists(images_folder):
        print(f"❌ Images folder does not exist: {images_folder}")
        return False, {}
    
    if not os.path.exists(labels_folder):
        print(f"❌ Labels folder does not exist: {labels_folder}")
        return False, {}
    
    # Get all image and label base names
    image_extensions = ['.jpg', '.jpeg', '.png']
    label_extensions = ['.txt']
    
    image_base_names = get_file_base_names(images_folder, image_extensions)
    label_base_names = get_file_base_names(labels_folder, label_extensions)
    
    print(f"  Found {len(image_base_names)} image files")
    print(f"  Found {len(label_base_names)} label files")
    
    # Find matching files
    matching_base_names = image_base_names.intersection(label_base_names)
    
    if not matching_base_names:
        print("❌ No matching files found between images and labels folders")
        return False, {}
    
    print(f"  ✅ Found {len(matching_base_names)} matching file pairs")
    
    # Build file pairs dictionary
    file_pairs = {}
    for base_name in matching_base_names:
        image_path = find_file_by_basename(images_folder, base_name, image_extensions)
        label_path = find_file_by_basename(labels_folder, base_name, label_extensions)
        
        if image_path and label_path:
            file_pairs[base_name] = {
                'image_path': image_path,
                'label_path': label_path
            }
    
    # Report mismatched files
    image_only = image_base_names - label_base_names
    label_only = label_base_names - image_base_names
    
    if image_only:
        print(f"  ⚠️  Images without labels: {len(image_only)}")
        for name in sorted(list(image_only)[:5]):  # Show first 5
            print(f"    - {name}")
        if len(image_only) > 5:
            print(f"    ... and {len(image_only) - 5} more")
    
    if label_only:
        print(f"  ⚠️  Labels without images: {len(label_only)}")
        for name in sorted(list(label_only)[:5]):  # Show first 5
            print(f"    - {name}")
        if len(label_only) > 5:
            print(f"    ... and {len(label_only) - 5} more")
    
    return True, file_pairs

def perform_augmentation(image_path, label_path, base_name, processed_img_dir, processed_txt_dir):
    """Perform all augmentations on a single image-label pair"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error reading image: {image_path}")
        return
    
    # Read label content
    try:
        with open(label_path, 'r') as f:
            label_content = f.read()
    except Exception as e:
        print(f"❌ Error reading label: {label_path}, Error: {e}")
        return
    
    # Resize image to 640x640
    image = cv2.resize(image, (640, 640))
    
    print(f"  🔁 Augmenting: {base_name}")
    
    augmentation_count = 0
    
    # BGR Blend augmentations
    for ratio in BLEND_RATIOS:
        aug_image = apply_bgr_blend(image, ratio)
        suffix = f"bgr_blend_{ratio:.2f}"
        save_augmented_image(aug_image, processed_img_dir, base_name, suffix)
        save_augmented_label(label_content, processed_txt_dir, base_name, suffix)
        augmentation_count += 1
    
    # Hue Shift augmentations
    for hue in HUE_SHIFT_LIST:
        aug_image = apply_hue_shift(image, hue)
        suffix = f"hue_shift_{hue}"
        save_augmented_image(aug_image, processed_img_dir, base_name, suffix)
        save_augmented_label(label_content, processed_txt_dir, base_name, suffix)
        augmentation_count += 1
    
    # Brightness augmentations
    for factor in BRIGHTNESS_FACTORS:
        aug_image = apply_hsv_brightness(image, factor)
        suffix = f"brightness_{factor:.1f}"
        save_augmented_image(aug_image, processed_img_dir, base_name, suffix)
        save_augmented_label(label_content, processed_txt_dir, base_name, suffix)
        augmentation_count += 1
    
    # Saturation augmentations
    for factor in SATURATION_FACTORS:
        sign = '-' if factor < 0 else '+'
        aug_image = adjust_saturation(image, factor)
        suffix = f"saturation_{sign}{abs(factor):.1f}"
        save_augmented_image(aug_image, processed_img_dir, base_name, suffix)
        save_augmented_label(label_content, processed_txt_dir, base_name, suffix)
        augmentation_count += 1
    
    print(f"    ✅ Generated {augmentation_count} augmented pairs")

def process_matched_files(file_pairs, output_base_dir="Dataset"):
    """Process all matched image-label pairs"""
    
    if not file_pairs:
        print("❌ No file pairs to process")
        return
    
    # Create output directories
    processed_img_dir = os.path.join(output_base_dir, "processed_img")
    processed_txt_dir = os.path.join(output_base_dir, "processed_txt")
    
    os.makedirs(processed_img_dir, exist_ok=True)
    os.makedirs(processed_txt_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting augmentation process...")
    print(f"  Output directories:")
    print(f"    Images: {processed_img_dir}")
    print(f"    Labels: {processed_txt_dir}")
    
    total_processed = 0
    
    for base_name, paths in file_pairs.items():
        perform_augmentation(
            paths['image_path'], 
            paths['label_path'], 
            base_name, 
            processed_img_dir, 
            processed_txt_dir
        )
        total_processed += 1
    
    print(f"\n✅ Augmentation completed!")
    print(f"  Processed {total_processed} original file pairs")
    
    # Count generated files
    img_count = len([f for f in os.listdir(processed_img_dir) if f.lower().endswith('.jpg')])
    txt_count = len([f for f in os.listdir(processed_txt_dir) if f.lower().endswith('.txt')])
    
    print(f"  Generated {img_count} augmented images")
    print(f"  Generated {txt_count} augmented labels")
    
    if img_count == txt_count:
        print("  ✅ Image and label counts match!")
    else:
        print("  ⚠️  Image and label counts don't match!")

def main():
    """Main function to run the validation and augmentation process"""
    
    # Input folders
    images_folder = "images"  # Change this to your images folder path
    labels_folder = "labels"  # Change this to your labels folder path
    
    print("="*60)
    print("  ENHANCED AUGMENTATION WITH VALIDATION")
    print("="*60)
    
    # Step 1: Validate folders and find matching files
    is_valid, file_pairs = validate_folders(images_folder, labels_folder)
    
    if not is_valid:
        print("\n❌ Validation failed. Please check your input folders.")
        return
    
    if not file_pairs:
        print("\n❌ No matching image-label pairs found.")
        return
    
    # Step 2: Process matched files
    process_matched_files(file_pairs)
    
    print("\n" + "="*60)
    print("  PROCESS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()