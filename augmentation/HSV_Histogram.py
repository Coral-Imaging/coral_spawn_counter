#!/usr/bin/env python3++
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from RGB_HSV_converter import HSVInfo
from concurrent.futures import ThreadPoolExecutor
import time
import csv

def load_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv_img, axis=(0, 1))
        return avg_hsv
    return None

def load_data(data_dirs, save_dir, max_images=10000000):
    end_dir = os.path.basename(os.path.normpath(data_dir))
    avg_hsv_data = {}
    image_counter = 0
    file_paths = []
    output_csv = os.path.join(save_dir, f'{end_dir}_HSV_data.csv')
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["File Path", "Average HSV"])  # Write header row

        for directory in data_dirs.split(', '):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('clean.jpg', )):  # Add more extensions if needed
                        file_paths.append(os.path.join(root, file))
        start_time = time.time()  # Start the stopwatch    
        with ThreadPoolExecutor() as executor:
            for file_path, avg_hsv in zip(file_paths, executor.map(load_image, file_paths)):
                if avg_hsv is not None:
                    avg_hsv_data[file_path] = avg_hsv
                    image_counter += 1
                    csv_writer.writerow([file_path, avg_hsv])
                    elapsed_time = time.time() - start_time  # Calculate elapsed time
                    print(f'Images counted: {image_counter}, Time elapsed: {elapsed_time:.2f} seconds', end='\r')
                    if image_counter >= max_images:
                        print("\nReached the maximum limit of images.")
                        break

    print()  # Move to the next line after the loop completes
    return avg_hsv_data


if __name__ == '__main__':
    #data_dirs = '/media/java/CSLICSNov24/cslics_data/2024_november_spawning, /media/java/CSLICSOct24/cslics_october_2024/20241023_spawning'
    data_dir = '/media/java/CSLICSNov24/cslics_data/2024_november_spawning/10000000f620da42'
    avg_HSV_data = load_data(data_dir,'/home/java/Java/cslics/tank_data')
    
    # Create separate histograms for hue, saturation, and value
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist([value[0] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='r')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Hue Values')

    plt.subplot(1, 3, 2)
    plt.hist([value[1] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='b')
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Saturation Values')

    plt.subplot(1, 3, 3)
    plt.hist([value[2] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='g')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Value Values')

    plt.tight_layout()
    plt.show()

