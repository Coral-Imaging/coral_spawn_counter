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
    data_dir = '/mnt/hpccs01/home/wardlewo/Data/100000009c23b5af/'
    #save_data(data_dir,'/mnt/hpccs01/home/wardlewo/Data/cslics/tank_data')
    Tank_da42 = load_data(['/home/java/Java/hpc-home/Data/cslics/tank_data/10000000f620da42'])
    Tank_438d = load_data(['/home/java/Java/hpc-home/Data/cslics/tank_data/100000001ab0438d'])
    Tank_b5af = load_data(['/home/java/Java/hpc-home/Data/cslics/tank_data/100000009c23b5af'])
    print("Loading data complete")
    for tank_name, tank_data in [("Tank_da42", Tank_da42), ("Tank_438d", Tank_438d), ("Tank_b5af", Tank_b5af)]:
        print(f"{tank_name} - Lowest Hue")
        print(min(tank_data, key=lambda x: tank_data[x][0]))
        print(f"{tank_name} - Highest Hue")
        print(max(tank_data, key=lambda x: tank_data[x][0]))
        
        print(f"{tank_name} - Lowest Saturation")
        print(min(tank_data, key=lambda x: tank_data[x][1]))
        print(f"{tank_name} - Highest Saturation")
        print(max(tank_data, key=lambda x: tank_data[x][1]))
        
        print(f"{tank_name} - Lowest Value")
        print(min(tank_data, key=lambda x: tank_data[x][2]))
        print(f"{tank_name} - Highest Value")
        print(max(tank_data, key=lambda x: tank_data[x][2]))


    legend = ['Tank_3_da42_Amil', 'Tank_4_438d_Pdae', 'Tank_2_b5af_Amil']
    ## Create separate histograms for hue, saturation, and value
    # Plot for Hue Values
    all_values = (
    [value[0] for value in Tank_da42.values()] +
    [value[0] for value in Tank_438d.values()] +
    [value[0] for value in Tank_b5af.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.hist([value[0] for value in Tank_da42.values()], bins=bin_edges, alpha=0.5, color='r')
    plt.hist([value[0] for value in Tank_438d.values()], bins=bin_edges, alpha=0.5, color='b')
    plt.hist([value[0] for value in Tank_b5af.values()], bins=bin_edges, alpha=0.5, color='g')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Hue Values')
    plt.legend()
    plt.subplot(1, 3, 2)
    
    all_values = (
    [value[1] for value in Tank_da42.values()] +
    [value[1] for value in Tank_438d.values()] +
    [value[1] for value in Tank_b5af.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.hist([value[1] for value in Tank_da42.values()], bins=bin_edges, alpha=0.5, color='r')
    plt.hist([value[1] for value in Tank_438d.values()], bins=bin_edges, alpha=0.5, color='b')
    plt.hist([value[1] for value in Tank_b5af.values()], bins=bin_edges, alpha=0.5, color='g')
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Saturation Values')
    plt.legend()

    plt.subplot(1, 3, 3)

    all_values = (
    [value[2] for value in Tank_da42.values()] +
    [value[2] for value in Tank_438d.values()] +
    [value[2] for value in Tank_b5af.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.hist([value[2] for value in Tank_da42.values()], bins=bin_edges, alpha=0.5, color='r', label=legend[0])
    plt.hist([value[2] for value in Tank_438d.values()], bins=bin_edges, alpha=0.5, color='b', label=legend[1])
    plt.hist([value[2] for value in Tank_b5af.values()], bins=bin_edges, alpha=0.5, color='g', label=legend[2])    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Value Values')
    plt.legend()
    #Place a legend to the right of this smaller subplot.
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    plt.tight_layout()
    plt.show()

