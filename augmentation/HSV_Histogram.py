#!/usr/bin/env python3++
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from RGB_HSV_converter import HSVInfo
from concurrent.futures import ThreadPoolExecutor
import time
import json

def load_data(json_dirs):
    avg_HSV_data = {}
    for json_dir in json_dirs:
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        avg_HSV_data[file] = data['avg_hsv']
    return avg_HSV_data

def load_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv_img, axis=(0, 1))
        return avg_hsv.tolist() 
    return None

def save_data(data_dir, save_dir, max_images=10000000):
    end_dir = os.path.basename(os.path.normpath(data_dir))
    save_json_dir = os.path.join(save_dir, end_dir)
    os.makedirs(save_json_dir, exist_ok=True)
    
    #avg_hsv_data = {}
    image_counter = 0
    file_paths = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('clean.jpg'):
                file_paths.append(os.path.join(root, file))

    start_time = time.time()  # Start the stopwatch    
    with ThreadPoolExecutor() as executor:
        for file_path, avg_hsv in zip(file_paths, executor.map(load_image, file_paths)):
            if avg_hsv is not None:
                file_name = os.path.splitext(os.path.basename(file_path))[0] + '.json'
                json_path = os.path.join(save_json_dir, file_name)
                
                with open(json_path, 'w') as json_file:
                    json.dump({"avg_hsv": avg_hsv}, json_file, indent=4)
                image_counter += 1
                
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                #print(f'Images processed: {image_counter}, Time elapsed: {elapsed_time:.2f} seconds', end='\r')
                
                if image_counter >= max_images:
                    print("\nReached the maximum limit of images.")
                    break
    
    print()  # Move to the next line after the loop completes
    #return avg_hsv_data

if __name__ == '__main__':
    #data_dirs = '/media/java/CSLICSNov24/cslics_data/2024_november_spawning, /media/java/CSLICSOct24/cslics_october_2024/20241023_spawning'
    data_dir = '/mnt/hpccs01/home/wardlewo/Data/10000000570f9d9c/'
    avg_HSV_data = save_data(data_dir,'/mnt/hpccs01/home/wardlewo/Data/cslics/tank_data')
    
    ## Create separate histograms for hue, saturation, and value
    #plt.figure(figsize=(15, 5))

    #plt.subplot(1, 3, 1)
    #plt.hist([value[0] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='r')
    #plt.xlabel('Hue Value')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Hue Values')

    #plt.subplot(1, 3, 2)
    #plt.hist([value[1] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='b')
    #plt.xlabel('Saturation Value')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Saturation Values')

    #plt.subplot(1, 3, 3)
    #plt.hist([value[2] for value in avg_HSV_data.values()], bins=30, alpha=0.5, color='g')
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Value Values')

    #plt.tight_layout()
    #plt.show()

