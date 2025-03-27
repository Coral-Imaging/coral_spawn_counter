#! /usr/bin/env python3

"""
read detections from ML model, saved as individual .txt files
plot them into time history
re-vamped to read in the json files from nested directories and then generate a plot
"""

import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np

# specify sampling/aggregate frequency
# 
# read json in for each image
# detectiosn are a list of pred = [x1 y1 x2 y2 conf class]

def convert_to_decimal_days(dates_list, time_zero=None):
    """from a list of datetime objects, convert to decimal days since time_zero"""
    if time_zero is None:
        time_zero = dates_list[0]  # Time zero is the first element date in the list
    else:
        time_zero = time_zero
        
    decimal_days_list = []

    for date in dates_list:
        time_difference = date - time_zero
        decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
        decimal_days_list.append(decimal_days)

    return decimal_days_list


# read in classes
root_dir = '/media/dorian/CSLICSNov24/cslics_november_2024/detections/100000000846a7ff/cslics_subsurface_20250205_640p_yolov8n'

# TODO: link with read_manuala_counts.py to automatically take in the tank_sheet_name
# 10000000f620da42 --> T3 Amil NOV24

tank_sheet_name = 'NOV24 T5 Pdae'

# set output directory
# save_plot_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024/detections/10000000f620da42/cslics_subsurface_20250205_640p_yolov8n'
save_plot_dir = root_dir
os.makedirs(save_plot_dir, exist_ok=True)


# read in all the json files
# it is assumed that because of the file naming structure, sorting the files by their filename sorts them chronologically
print(f'Gathering list of detection files (json) in all sub-directories of source directory: {root_dir}')
sample_list = sorted(Path(root_dir).rglob('*_det.json'))
print(f'Number of detection files: {len(sample_list)}')

# skipping frequency - mostly for time in development
skipping_frequency = 2

# aggregate into samples - 1 hr chunks, configurable
aggregate_size = 30
# so every 100 images, average each image-based detection into a sample

# edge cases:
# 1) length of json_list is less than skipping frequency
# 2) length of json_list is less than aggegate size
# 3) how to handle last few files that don't fit witihn aggregate? just ignore

if len(sample_list) < skipping_frequency:
    print(f'ERROR: json_list {len(sample_list)} is less than skipping frequency {skipping_frequency}. Choose a smaller skipping frequency.')
    exit
if len(sample_list) < aggregate_size:
    print(f'ERROR: json_list {len(sample_list)} is less than aggregate size {aggregate_size}. Choose a smaller aggregate size.')
    exit

# skip every X images
downsampled_list = sample_list[::skipping_frequency]

# rather than using a big for loop, we can just use list comprehension to aggreate into samples
batched_samples = [downsampled_list[i:i+aggregate_size] for i in range(0, len(downsampled_list), aggregate_size)]

# for each batch, compute relevant data:
# read in jsons
# simple approach:sum the counts, taking note of confidences 
# more nuanced: every detection has a confidence associated with it. How do the counts change wrt confidence value?
# for now, I just have to re-run the code for a diff value
confidence_threshold = 0.5

batched_image_count = []
batched_std = []
batched_time = []
MAX_SAMPLE = 1000
# iterate over all the batched samples
for i, sample in enumerate(batched_samples):
    print(f'batched sample: {i}/{len(batched_samples)}')

    if i >= MAX_SAMPLE:
        print(f'Hit MAX SAMPLE limit for debugging purposes')
        break
    sample_count = []
    # iterate for each sample, there are multiple detection files
    for detection_file in sample:
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)           

            # GET COUNTS
            detections = data['detections [xn1, yn1, xn2, yn2, conf, cls]'] # TODO it's actually conf then class - fix key

            # only take those detections that are greater than confidence threshold:
            select_detections = [d for d in detections if d[4] >= confidence_threshold]

            # if there was class-based filtering/selection, here would be the place to use it
            # though current CSLICS is single-class object detector 
            sample_count.append(len(select_detections))

        except json.JSONDecodeError as e:
            print("Error loading JSON:", e)
        except FileNotFoundError:
            print("Error: File not found.")

    # TODO there is some suggestion on alternate methods instead of taking the mean - e.g. median to smooth out the data
    sample_mean = np.mean(sample_count)
    sample_std = np.std(sample_count)

    # GET CAPTURE TIME OF MIDDLE CAPTURE
    # obtain via filename, or take min/max and then assume middle, can turn this into function later on
    capture_time_str = Path(sample[int(len(sample)/2)]).stem[:-10] # capture time minus '_clean_det' characters
    capture_time = datetime.strptime(capture_time_str, "%Y-%m-%d_%H-%M-%S")

    batched_image_count.append(sample_mean)
    batched_std.append(sample_std)
    batched_time.append(capture_time)

# convert batched_time to decimal days and 0 the time since spawning
decimal_capture_times = convert_to_decimal_days(batched_time)
# plot sample points over time

# convert from list to np array because matplotlib's fill_between cannot handle list input
decimal_capture_times = np.array(decimal_capture_times)
batched_image_count = np.array(batched_image_count)
batched_std = np.array(batched_std)

n = 1
fig0, ax = plt.subplots()
plt.plot(decimal_capture_times, batched_image_count)
plt.fill_between(decimal_capture_times, batched_image_count-n*batched_std, batched_image_count+n*batched_std,
            alpha=0.2)
plt.grid(True)
plt.xlabel('Days since spawning')
plt.ylabel(f'Image count (batched {aggregate_size} images)')
plt.title(f'CSLICS AI Count: {tank_sheet_name}')
plt.savefig(os.path.join(save_plot_dir, 'Image counts' + tank_sheet_name + '.png'))

print('done')
        
import code
code.interact(local=dict(globals(), **locals()))