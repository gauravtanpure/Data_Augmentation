import zipfile
import os
import tempfile
import shutil

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_inner_zip(zip_file_path):
    with tempfile.TemporaryDirectory() as temp_inner:
        extract_zip(zip_file_path, temp_inner)

        # Collect .txt files excluding train.txt
        txt_files = []
        for root, _, files in os.walk(temp_inner):
            for file in files:
                if file.endswith('.txt') and file.lower() != 'train.txt':
                    txt_files.append(os.path.join(root, file))

        # Sort the files
        sorted_txt = sorted(txt_files, key=lambda x: os.path.basename(x).lower())

        # Output path: ./labels/<inner_zip_name>/
        inner_zip_name = os.path.splitext(os.path.basename(zip_file_path))[0]
        output_dir = os.path.join("labels", inner_zip_name)
        os.makedirs(output_dir, exist_ok=True)

        for txt_file in sorted_txt:
            shutil.copy(txt_file, output_dir)

        print(f"[INFO] {len(sorted_txt)} .txt files copied to: {output_dir}")

def sort_txt_from_each_inner_zip(main_zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[INFO] Extracting outer ZIP: {main_zip_path}")
        extract_zip(main_zip_path, temp_dir)

        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.zip'):
                    inner_zip_path = os.path.join(root, file)
                    print(f"\n[INFO] Processing inner ZIP: {inner_zip_path}")
                    process_inner_zip(inner_zip_path)

if __name__ == "__main__":
    main_zip_path = "Data_Augmentation_txt.zip"
    sort_txt_from_each_inner_zip(main_zip_path)
