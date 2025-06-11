import cv2
import numpy as np
import os
import shutil

# ==== Configuration ====
BLEND_RATIOS = [0.25, 0.5, 0.75, 1.0]
HUE_SHIFT_LIST = [30, 60, 90, 120, 150]
BRIGHTNESS_FACTORS = [0.4, 1.1, 5.5]
SATURATION_FACTORS = [-0.5, 0.5, 1.5]

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

def save_augmented(image, output_dir, base_name, suffix):
    filename = f"{base_name}_{suffix}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

def apply_all_augmentations(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Skipping unreadable image: {image_path}")
        return

    image = cv2.resize(image, (640, 640))
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. BGR Blend
    for ratio in BLEND_RATIOS:
        aug = apply_bgr_blend(image, ratio)
        save_augmented(aug, output_dir, base_name, f"bgr_blend_{ratio:.2f}")

    # 2. Hue Shift
    for hue in HUE_SHIFT_LIST:
        aug = apply_hue_shift(image, hue)
        save_augmented(aug, output_dir, base_name, f"hue_shift_{hue}")

    # 3. Brightness (HSV V channel)
    for factor in BRIGHTNESS_FACTORS:
        aug = apply_hsv_brightness(image, factor)
        save_augmented(aug, output_dir, base_name, f"brightness_{factor:.1f}")

    # 4. Saturation Adjustments
    for factor in SATURATION_FACTORS:
        aug = adjust_saturation(image, factor)
        sign = '-' if factor < 0 else '+'
        save_augmented(aug, output_dir, base_name, f"saturation_{sign}{abs(factor):.1f}")

def process_images(images_root, labels_root):
    for task_folder in os.listdir(images_root):
        task_path = os.path.join(images_root, task_folder)
        if not os.path.isdir(task_path):
            continue

        print(f"\n🔁 Processing task folder: {task_folder}")
        output_dir = os.path.join("augmentation_processed", task_folder)
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(task_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = os.path.splitext(file)[0]
                label_path = os.path.join(labels_root, f"{img_name}.txt")
                image_path = os.path.join(task_path, file)

                if os.path.exists(label_path):
                    apply_all_augmentations(image_path, output_dir)
                else:
                    print(f"⚠️ Skipping unmatched image: {file}")

if __name__ == "__main__":
    images_folder = "images"
    labels_folder = "labels"
    process_images(images_folder, labels_folder)
