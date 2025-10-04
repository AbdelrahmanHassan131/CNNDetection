import os
import shutil

# Source directory containing subfolders with images
source_dir = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\third task\fake av dataset\fakeavceleb\outputOnlyFakeFaces"

# Destination directory where all images will be copied and renamed
destination_dir = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\training images\wholeTrainingData\fake"
os.makedirs(destination_dir, exist_ok=True)

# Supported image extensions
image_extensions = {".png", ".jpeg", ".jpg"}

# Counter for renaming images sequentially
counter = 287187

# Loop through each subfolder and collect images
for root, dirs, files in os.walk(source_dir):
    for file in sorted(files, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else float('inf')):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            source_path = os.path.join(root, file)
            new_name = f"{counter}{ext}"
            destination_path = os.path.join(destination_dir, new_name)
            shutil.copy(source_path, destination_path)
            counter += 1


# def rename_images(folder_path):
#     # Supported image extensions
#     image_extensions = ('.jpg', '.jpeg', '.png', '.gif',
#                         '.bmp', '.tiff', '.webp')

#     # List all files and filter only images
#     images = [f for f in os.listdir(
#         folder_path) if f.lower().endswith(image_extensions)]
#     images.sort()  # Sort alphabetically to maintain order

#     # Rename files
#     for index, filename in enumerate(images):
#         extension = os.path.splitext(filename)[1]  # Get file extension
#         new_name = f"{index}{extension}"
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_name)

#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} → {new_name}")

#     print("\nRenaming complete!")


# # Example usage
# # Change this to your folder path
# folder = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\training images\wholeTrainingData\real"
# rename_images(folder)
print(f"All images copied to {destination_dir} with new numbering!")
