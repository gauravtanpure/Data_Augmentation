import zipfile
import os
import tempfile
import shutil

def sort_image_files(zip_path):
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[INFO] Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Collect image files (jpg, jpeg, png)
        image_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))

        # Sort by filename
        sorted_images = sorted(image_files, key=lambda x: os.path.basename(x).lower())

        # Output folder
        output_dir = os.path.join(os.getcwd(), zip_name)
        os.makedirs(output_dir, exist_ok=True)

        for img_path in sorted_images:
            shutil.copy(img_path, output_dir)

        print(f"\n[INFO] Total image files: {len(sorted_images)}")
        print(f"[INFO] Sorted and copied to: {output_dir}")

        # Print filenames
        for img in sorted_images:
            print(os.path.basename(img))

if __name__ == "__main__":
    zip_file = r"job_60_dataset_2025_06_10_13_29_05_cvat for images 1.1.zip"
    sort_image_files(zip_file)
