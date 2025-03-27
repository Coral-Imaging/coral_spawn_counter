#!/usr/bin/env python3

from PIL import Image
import os

def create_gif_from_pngs(folder_path, output_path, duration=500, loop=0):
    """
    Create a GIF from all PNG files in a folder.

    Args:
        folder_path (str): Path to the folder containing PNG files.
        output_path (str): Path to save the generated GIF.
        duration (int): Duration of each frame in milliseconds. Default is 500ms.
        loop (int): Number of times the GIF should loop. 0 means infinite loop. Default is 0.
    """
    # Get all PNG files in the folder and sort them by filename
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])[::2]
    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    # Open all PNG files as PIL Image objects
    print(f'opening {len(png_files)} images')
    images = [Image.open(os.path.join(folder_path, file)) for file in png_files]

    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF created and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    folder_path = "/media/dtsai/CSLICSNov24/cslics_november_2024/detections/10000000f620da42/cslics_subsurface_20250205_640p_yolov8n/histograms"  # Replace with the path to your folder containing PNGs
    output_path = "/media/dtsai/CSLICSNov24/cslics_november_2024/detections/10000000f620da42/cslics_subsurface_20250205_640p_yolov8n/histogram_evolution.gif"  # Replace with the desired output path for the GIF
    create_gif_from_pngs(folder_path, output_path, duration=500, loop=0)