import os
import random
import shutil
import math

# === CONFIGURATION ===
# Folder with original images
source_folder = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\training images\wholeTrainingData\real"
# Folder for 70% of images
train_folder = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\ThesisModel\CNNDetection\dataset\train\real"
# Folder for 30% of images
test_folder = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\ThesisModel\CNNDetection\dataset\val\real"

# Supported image extensions
valid_extensions = {".png", ".jpg", ".jpeg"}

# Create output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# === MAIN SCRIPT ===
# Step 1: Collect all images
files = [f for f in os.listdir(source_folder)
         if os.path.splitext(f)[1].lower() in valid_extensions]

if not files:
    print("No image files found in the source folder.")
    exit()

# Step 2: Shuffle images
random.shuffle(files)

# Step 3: Split into 70% and 30%
total_count = len(files)
split_index = math.ceil(total_count * 0.7)  # 70% for training
train_files = files[:split_index]
test_files = files[split_index:]

print(f"Total images: {total_count}")
print(f"Training set: {len(train_files)} images")
print(f"Testing set: {len(test_files)} images")

# Step 4: Copy and rename files


def copy_and_rename(file_list, target_folder):
    for idx, filename in enumerate(file_list):
        # Preserve original extension
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{idx}{ext}"                     # Rename sequentially
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, new_name)
        # Copy while keeping metadata
        shutil.copy2(src_path, dst_path)


# Copy train and test sets
copy_and_rename(train_files, train_folder)
copy_and_rename(test_files, test_folder)

print("Images successfully split, shuffled, and renamed!")
