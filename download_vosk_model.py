import os
import sys
import zipfile
import shutil
import urllib.request

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
ZIP_PATH = "vosk-model-en-us-0.22.zip"
EXTRACT_DIR = "model_full_temp"
TARGET_DIR = "model"

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading full Vosk model: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
    sys.stdout.flush()

def main():
    print(f"Starting download from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH, reporthook=report_progress)
    print("\nDownload completed! Extracting ZIP archive...")

    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print("Extraction complete. Updating 'model' folder...")

    # Inside EXTRACT_DIR there is a folder named 'vosk-model-en-us-0.22'
    extracted_folder = os.path.join(EXTRACT_DIR, "vosk-model-en-us-0.22")
    if not os.path.exists(extracted_folder):
        # Fallback if folder name differs
        items = os.listdir(EXTRACT_DIR)
        extracted_folder = os.path.join(EXTRACT_DIR, items[0])

    # Backup existing model directory if present
    if os.path.exists(TARGET_DIR):
        backup_dir = "model_small_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(TARGET_DIR, backup_dir)
        print(f"Moved existing small model to '{backup_dir}'")

    # Move full model into 'model'
    shutil.move(extracted_folder, TARGET_DIR)
    print(f"Successfully installed full Vosk model into '{TARGET_DIR}'!")

    # Cleanup temporary files
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

    print("Cleanup completed.")

if __name__ == "__main__":
    main()
