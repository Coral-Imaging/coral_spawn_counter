#!/usr/bin/env python3

"""
Run fertilisation counter on all subfolders in a directory - simplified for debugging
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from FertilisationCounter import FertilisationCounter

def find_image_folders(root_dir):
    """Find all folders containing an 'images' subfolder"""
    image_folders = []
    for path in Path(root_dir).iterdir():
        if path.is_dir():
            images_path = path / 'images'
            if images_path.is_dir() and list(images_path.glob('*.jpg')):
                image_folders.append(path)
    
    return image_folders

def main():
    # Set default values
    default_root_dir = '/home/dtsai/Data/cslics_datasets/cslics_2023_fert_dataset'
    default_model_path = '/home/dtsai/Data/cslics_datasets/models/fertilisation/cslics_20230905_yolov8m_640p_amtenuis1000.pt'
    default_manual_count_path = '/home/dtsai/Data/cslics_datasets/cslics_2022_fert_dataset_manual_counts/cslics_2022_fert_dataset_manual_counts.xlsx'
    
    # Parse command line arguments (optional)
    parser = argparse.ArgumentParser(description='Run fertilisation counter on all subfolders in a directory')
    parser.add_argument('--root-dir', default=default_root_dir, help='Root directory containing experiment folders')
    parser.add_argument('--model-path', default=default_model_path, help='Path to the YOLO model file')
    parser.add_argument('--manual-count-path', default=default_manual_count_path, help='Path to the Excel file with manual counts')
    parser.add_argument('--use-cache', action='store_true', default=False, help='Use cached detections if available')
    parser.add_argument('--no-manual', action='store_true', help='Exclude manual counts from analysis')
    parser.add_argument('--time-limit', type=int, default=120, help='Time limit in minutes for plotting (default: 120)')
    args = parser.parse_args()
    
    # Print configuration
    print(f"Running with the following configuration:")
    print(f"  Root directory: {args.root_dir}")
    print(f"  Model path: {args.model_path}")
    print(f"  Manual count path: {args.manual_count_path}")
    print(f"  Use cache: {args.use_cache}")
    print(f"  Include manual counts: {not args.no_manual}")
    print(f"  Time limit: {args.time_limit} minutes")
    
    # Find folders containing images
    image_folders = find_image_folders(args.root_dir)
    print(f"\nFound {len(image_folders)} folders with images:")
    for folder in image_folders:
        print(f"  - {folder.name}")
    
    # Process each folder sequentially
    print("\nProcessing folders sequentially:")
    
    successes = []
    failures = []
    
    # Process each folder one by one
    for i, folder in enumerate(tqdm(image_folders, desc="Processing folders")):
        print(f"\n[{i+1}/{len(image_folders)}] Processing {folder.name}...")
        
        try:
            # Images folder path
            images_folder = folder / 'images'
            
            # Output directory
            output_dir = folder / 'predictions'
            
            # Extract sheet name from folder name
            sheet_name = folder.name
            
            # Create and run counter
            counter = FertilisationCounter(
                root_dir=str(images_folder),
                model_path=args.model_path,
                output_dir=str(output_dir),
                manual_count_path=args.manual_count_path,
                sheet_name=sheet_name,
                use_cached_detections=args.use_cache,
                include_manual=not args.no_manual
            )
            
            # Run the counter and get results
            results = counter.run()
            
            # Plot with the specified time limit
            counter.plot_fertilisation(results, 
                                      manual_times=None if args.no_manual else counter.manual_times, 
                                      manual_ratios=None if args.no_manual else counter.manual_ratios,
                                      time_limit_minutes=args.time_limit)
                                      
            print(f"✅ Completed processing {folder.name}")
            successes.append(folder.name)
            
        except Exception as e:
            print(f"❌ Error processing {folder.name}:")
            print(f"   {str(e)}")
            import traceback
            print(traceback.format_exc())
            failures.append((folder.name, str(e)))
    
    # Final report
    print("\n=== Processing Summary ===")
    print(f"Total folders: {len(image_folders)}")
    print(f"Successfully processed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    
    if failures:
        print("\nFailed folders:")
        for folder_name, error in failures:
            print(f"  - {folder_name}: {error}")
    
    print("\n🎉 All folders processed")

if __name__ == "__main__":
    main()