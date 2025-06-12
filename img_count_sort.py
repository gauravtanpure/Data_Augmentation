import zipfile
import os
import tempfile
import shutil

def extract_and_sort_inner_zips(parent_zip_path):
    base_output_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(base_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[INFO] Extracting parent ZIP: {parent_zip_path}")
        with zipfile.ZipFile(parent_zip_path, 'r') as parent_zip:
            parent_zip.extractall(temp_dir)

        # Find all inner zip files
        inner_zips = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.zip'):
                    inner_zips.append(os.path.join(root, file))

        print(f"[INFO] Found {len(inner_zips)} inner ZIP files.\n")

        for inner_zip_path in inner_zips:
            inner_zip_name = os.path.splitext(os.path.basename(inner_zip_path))[0]
            print(f"[INFO] Processing inner ZIP: {inner_zip_name}")

            with tempfile.TemporaryDirectory() as inner_temp_dir:
                with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(inner_temp_dir)

                # Collect and sort images
                image_files = []
                for root, _, files in os.walk(inner_temp_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(os.path.join(root, file))

                sorted_images = sorted(image_files, key=lambda x: os.path.basename(x).lower())

                
                output_dir = os.path.join(base_output_dir, inner_zip_name)
                os.makedirs(output_dir, exist_ok=True)

                for img_path in sorted_images:
                    shutil.copy(img_path, output_dir)

                print(f"    [✓] {len(sorted_images)} images copied to {output_dir}")

if __name__ == "__main__":
    parent_zip = r"sort_count_img.zip"
    extract_and_sort_inner_zips(parent_zip)