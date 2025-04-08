#!/usr/bin/env python3

import os
import glob
import shutil
import zipfile

# script to help get images from folder
# take every X images and put them in a new folder

root_dir = '/home/dtsai/Data/cslics_datasets/cslics_2022_fert_dataset'
target_zip_dir = '/home/dtsai/Data/cslics_datasets/cslics_2022_fert_dataset/zipped_manual_images'

def list_folders(directory):
    """List all folders in the given directory."""
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders

def find_folders_with_images(directory, folders):
    """Find folders that contain a subfolder named 'images'."""
    cslics_image_folders = []
    for folder in folders:
        images_path = os.path.join(directory, folder, 'images')
        if os.path.isdir(images_path):
            cslics_image_folders.append(folder)
    return cslics_image_folders

def get_image_names(directory, skip):
    """Get all .jpg file names in the directory, skipping based on the given value."""
    image_paths = sorted(glob.glob(os.path.join(directory, '*.jpg'), recursive=True))
    # Skip images based on the manual_image_skip value
    return image_paths[::skip]

def prepare_output_directory(output_dir):
    """Create the output directory and clean up any existing files."""
    os.makedirs(output_dir, exist_ok=True)
    # Delete all files in output_dir if it contains any files
    if os.listdir(output_dir):
        print(f"Cleaning up existing files in {output_dir}...")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def zip_output_directories(root_dir, target_zip_dir):
    """Zip all output directories into a single archive."""
    os.makedirs(target_zip_dir, exist_ok=True)
    zip_file_path = os.path.join(target_zip_dir, 'manual_images.zip')
    
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in os.listdir(root_dir):
            output_dir = os.path.join(root_dir, folder, 'for_manual_labelling')
            if os.path.isdir(output_dir):
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, root_dir)  # Relative path for the zip
                        zipf.write(file_path, arcname)
    print(f"Zipped all output directories into {zip_file_path}")

if __name__ == "__main__":
    folders = list_folders(root_dir)
    print("Folders in root_dir:")
    for folder in folders:
        print(folder)
    
    cslics_image_folders = find_folders_with_images(root_dir, folders)
    print("\nFolders containing 'images' subfolder:")
    for folder in cslics_image_folders:
        print(folder)
    
    manual_image_skip = 100
    total_images = 0
    print("\nNumber of images (skipping every {} images):".format(manual_image_skip))
    for folder in cslics_image_folders:
        images_path = os.path.join(root_dir, folder, 'images')
        image_names = get_image_names(images_path, manual_image_skip)
        print(f"Folder: {folder}, number of images after skip: {len(image_names)}")
        total_images += len(image_names)
        
        # Prepare the output directory
        output_dir = os.path.join(root_dir, folder, 'for_manual_labelling')
        prepare_output_directory(output_dir)
        
        # Copy images to the output directory
        for image_path in image_names:
            shutil.copy(image_path, output_dir)
    
    print("\nTotal number of images after skip: {}".format(total_images))
    
    # Zip all output directories
    zip_output_directories(root_dir, target_zip_dir)