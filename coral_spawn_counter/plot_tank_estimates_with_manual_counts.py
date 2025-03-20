#! /usr/bin/env python3

"""
General pipeline so far:
- organise all relevant CSLICS images into appropriate folder structure
- have coral spawn object detection model ready
- run predict_2025.py to perform coral detection on all cslics images for given cslics run
- run read_manual_counts.py to confirm you can get manual counts from the xlsx datasheets
- run read_detections.py to confirm that you can get per-image samples over time
- finally, run this script: plot_tank_estimates_with_manual_counts.py

- in a nutshell, this is a script to combine read_detections.py and read_manual_counts.py
- adds tank-scaling for the predictions and option for calibration wrt a specified manual count

# TODO shift the manual counts to start at the appropriate time - first time point?
# TODO this should all take the form of an object-based approach for code-resuability and readability
# TODO config files for the different runs/tanks
# TODO script-able for running overnight? (related to config file)

"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# THE PLAN:
# specify directories
# root directory of detections
# manual count file(s) directory
# save directory for resultant plots
# any calibration/configuration files

# read in manual counts
# plot manual counts
# read in detections
# plot image-based counts
# use default calibration concept (focus volume calculations)
# select manual calibration point (ala cslics desktop)
# output tank-based counts
# plot manual counts and tank-based counts
# calculate error with some window of the manual counts

#######################################
# HELPER FUNCTIONS

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

def read_cslics_uuid_tank_association(file, spawning_sheet_name, tank_sheet_name):
    """read the manual counts from the excel file and return the datetime objects and counts"""
    try:
        df = pd.read_excel(os.path.join(file), sheet_name=spawning_sheet_name, engine='openpyxl', header=2) 
        tank_sheet_column = df['manual count sheet name']
        camera_uuid_column = df['camera uuid']
        species_column = df['species']
        # TODO ensure each column is of the same length
        
        idx = []
        try:
            idx = list(tank_sheet_column).index(tank_sheet_name)
            print(f'Index of {tank_sheet_name}: {idx}')
        except ValueError:
            print(f'Item {tank_sheet_name} not found in listof tank_sheet_column.')
        
        cslics_uuid = camera_uuid_column[idx]
        species = species_column[idx]
        return cslics_uuid, species
    except:
        print('ERROR: reading the read_cslics_uuid_tank_association file, check the sheet name is correct and headings are as expected')


#######################################
# CONFIGURATION

# TODO this should take the form of a config file input
manual_counts_file = '/home/dorian/Data/cslics_datasets/manual_counts/cslics_2024_manual_counts.xlsx'

spawning_sheet_name = '2024 oct'
# tank_sheet_name = 'OCT24 T1 Amag'
# tank_sheet_name = 'OCT24 T2 Amag'
# tank_sheet_name = 'OCT24 T3 Amag'
# tank_sheet_name = 'OCT24 T4 Maeq'
# tank_sheet_name = 'OCT24 T5 Maeq'
tank_sheet_name = 'OCT24 T6 Aant'

# spawning_sheet_name = '2024 nov'
# tank_sheet_name = 'NOV24 T1 Amil'
# tank_sheet_name = 'NOV24 T2 Amil'
# tank_sheet_name = 'NOV24 T3 Amil'
# tank_sheet_name = 'NOV24 T4 Pdae'
# tank_sheet_name = 'NOV24 T5 Pdae'
# tank_sheet_name = 'NOV24 T6 Lcor'
cslics_associations_file = '/home/dorian/Data/cslics_datasets/manual_counts/cslics_2024_spawning_setup.xlsx'

# specify tank_sheet_name, which determines which tank/installation and thus cslics uuid we want to run
# given the tank sheet name, determine the corresponding uuid & therefore folder:
cslics_uuid, coral_species = read_cslics_uuid_tank_association(cslics_associations_file, spawning_sheet_name, tank_sheet_name)

# saving manual counts output
save_manual_plot_dir = '/home/dorian/Data/cslics_datasets/manual_counts/plots'
os.makedirs(save_manual_plot_dir, exist_ok=True)

# specify directory of detections after running inference on all the images, such that the detection json files are available
search_str = "nov"
model_name = 'cslics_subsurface_20250205_640p_yolov8n'
if search_str.lower() in tank_sheet_name.lower():
    det_dir = '/media/dorian/CSLICSNov24/cslics_november_2024/detections/' + str(cslics_uuid) + '/' + model_name
else:
    # OCT
    det_dir = '/media/dorian/CSLICSOct24/cslics_october_2024/detections/' + str(cslics_uuid) + '/' + model_name


# saving detection plots
save_det_dir = det_dir
os.makedirs(save_det_dir, exist_ok=True)

# skipping frequency - mostly for time in development
skipping_frequency = 1

# aggregate into samples - 1 hr chunks, configurable
aggregate_size = 100
# so every 100 images, average each image-based detection into a sample

# for each batch, only select detections above certain confidence threshold 
confidence_threshold = 0.5

# for dev purposes, max images to prevent run-away
MAX_SAMPLE = 1000

# manual calibration window sample size - how many aggregated samples should be used to average the calibration
calibration_window_size = 1 # choose one for now, just to confirm scaling is working

# index of manual count list that should be used for scaling ai counts
# typically first (0'th) is the initial stocking date
# second (1st) is the first date for counting cslics
# 2nd might also be valid, if suspension is not yet achieved
calibration_idx = 1
calibration_window_shift = 0 # time shift for selecting calibration samples

# boolean to plot focus volume in final combined figure
PLOT_FOCUS_VOLUME = False 

##############################################
# READ MANUAL COUNTS

def read_manual_counts(cslics_associations_file, manual_counts_file, tank_sheet_name=None):
    # read manual counts from spreadsheet file

    # extract the relevant columns from the spreadsheet
    df = pd.read_excel(manual_counts_file, sheet_name = tank_sheet_name, engine='openpyxl',header=5)
    date_column = df['Date']
    time_collected = df['Time']
    count_column = df['Count (500L)']
    std_column = df['Std Dev']

    # combine date and time into single datetime object
    # convert to decimal-based days format for easier plotting
    counts_time = pd.to_datetime(date_column.astype(str) + " " + time_collected.astype(str), dayfirst=True)
    # TEST: find nearest new day to counts_time[0]
    nearest_day = counts_time[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    # decimal_days = convert_to_decimal_days(counts_time, counts_time[0])
    decimal_days = convert_to_decimal_days(counts_time, nearest_day)

    return count_column, std_column, decimal_days, counts_time


def plot_manual_counts(counts, std, days):
    # plot manual counts

    scaled_counts = counts/1000 # for readability
    scaled_std = std/1000
    n = 1.0

    fig, ax = plt.subplots()
    ax.plot(days, scaled_counts, marker='o', color='blue')
    ax.errorbar(days, scaled_counts, yerr= n*scaled_std, fmt='o', color='blue', alpha=0.5)
    plt.grid(True)
    plt.xlabel('Days since spawning')
    plt.ylabel('Tank count (in thousands for 500L)')
    plt.title(f'CSLICS Manual Count: {tank_sheet_name}')
    plt.savefig(os.path.join(save_manual_plot_dir, 'Manual counts' + tank_sheet_name + '.png')) 

# read manual counts from spreadsheet
manual_counts, manual_std, manual_times, manual_times_dt = read_manual_counts(cslics_associations_file, manual_counts_file, tank_sheet_name)

# optionally: generate manual count plots
plot_manual_counts(manual_counts, manual_std, manual_times)


####################################
# READ DETECTIONS

def read_detections(det_dir):
    # read in all the json files
    # it is assumed that because of the file naming structure, sorting the files by their filename sorts them chronologically
    print(f'Gathering list of detection files (json) in all sub-directories of source directory: {det_dir}')
    sample_list = sorted(Path(det_dir).rglob('*_det.json'))
    print(f'Number of detection files: {len(sample_list)}')

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

    batched_image_count = []
    batched_std = []
    batched_time = []

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
                # should be: 'detections [xn1, yn1, xn2, yn2, cls, conf]'
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
        sample_avg = np.mean(sample_count)
        # sample_avg = np.median(sample_count)
        sample_std = np.std(sample_count)

        # GET CAPTURE TIME OF MIDDLE CAPTURE
        # obtain via filename, or take min/max and then assume middle, can turn this into function later on
        idx_time = int(len(sample)/2)
        capture_time_str = Path(sample[idx_time]).stem[:-10] # capture time minus '_clean_det' characters
        capture_time = datetime.strptime(capture_time_str, "%Y-%m-%d_%H-%M-%S")

        batched_image_count.append(sample_avg)
        batched_std.append(sample_std)
        batched_time.append(capture_time)

    # convert batched_time to decimal days and 0 the time since spawning
    nearest_day = manual_times_dt[0].replace(hour=0, minute=0, second=0, microsecond=0)  + timedelta(days=1) # NOTE replicated code
    decimal_capture_times = convert_to_decimal_days(batched_time, nearest_day)
    # plot sample points over time

    # convert from list to np array because matplotlib's fill_between cannot handle list input
    decimal_capture_times = np.array(decimal_capture_times)
    batched_image_count = np.array(batched_image_count)
    batched_std = np.array(batched_std)

    return batched_image_count, batched_std, decimal_capture_times


def plot_image_detections(counts, std, times):
    n = 1
    fig0, ax = plt.subplots()
    plt.plot(times, counts)
    plt.fill_between(times, counts-n*std, counts+n*std,
                alpha=0.2)
    plt.grid(True)
    plt.xlabel('Days since spawning')
    plt.ylabel(f'Image count (batched {aggregate_size} images)')
    plt.title(f'CSLICS AI Count: {tank_sheet_name}')
    plt.savefig(os.path.join(save_det_dir, 'Image counts' + tank_sheet_name + '.png'))


# read detections from detection.json files
image_counts, image_std, image_times = read_detections(det_dir)

# optionally, plot the image-based detections time series
plot_image_detections(image_counts, image_std, image_times)

###############################################
# SCALE DETECTION RESULTS

def get_hyper_focal_dist(f, c, n):
    return f + f**2 / (c * n)

def scale_by_focus_volume():
    # a physics-based approach to solving the calibration problem    
    # issue with this approach is that it requires very careful calibration, which can/may vary with different cslics
    # this approach puts significant onus on careful calibration for every single unit
    # and still relies on a somewhat nebulous variable "c", the circle of confusion

    width_pix = 4056 # pixels
    height_pix = 3040 # pixels
    pix_size = 1.55 / 1000 # um -> mm, pixel size
    sensor_width = width_pix * pix_size # mm
    sensor_height = height_pix * pix_size # mm
    f = 12 # mm, focal length
    aperture = 2.8 # f-stop number of the lens
    c = 0.1 # mm, circle of confusion, def 0.1, increase to 0.2 to double (linear) the sample volume
    hyp_dist = get_hyper_focal_dist(f, c, aperture) # hyper-focal distance = max depth of field of camera
    focus_dist = 75 #mm focusing distance, practically the working distance of the camera
    # NOTE: focus distance was kept to ~ the same in the CSLICS 2023, but may differ between CSLICS (see CSLICS tank setup notes)
    dof_far = (hyp_dist * focus_dist) / (hyp_dist - (focus_dist - f))
    dof_near = (hyp_dist * focus_dist) / (hyp_dist + (focus_dist - f))
    dof_diff = abs(dof_far - dof_near) # mm
    print(f'DoF diff = {dof_diff} mm')

    work_dist = focus_dist # mm, working distance
    # 1.33 for refraction through water, lensing effect
    hfov = work_dist * sensor_height / (1.33 * f) # mm, horizontal field-of-view
    vfov = work_dist * sensor_width / (1.33 * f) # mm, vertical field-of-view
    print(f'horizontal FOV = {hfov}')
    print(f'vertical FOV = {vfov}')

    area_cslics = hfov * vfov # mm, area of cslics
    print(f'area_cslics = {area_cslics} mm^2')

    # we can approximate the frustum as a rectangular prism, since the angular FOV is not that wide
    focus_volume = area_cslics * dof_diff # mm^3
    print(f'focus volume = {focus_volume} mm^3')
    print(f'focus volume = {focus_volume/1000} mL')

    volume_image = focus_volume / 1000 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
    volume_tank = 475 * 1000 # 500 L = 500000 ml
    
    scale_factor = volume_tank / volume_image # thus, how many cslics images will fill the whole volume of the tank
    print(f'default scale factor = {scale_factor}')

    return scale_factor

scale_factor_def = scale_by_focus_volume()

# apply scale factor to image counts
tank_counts_def = image_counts * scale_factor_def
tank_std_def = image_std * scale_factor_def

# manual scaling based on calibration index (see config)
def find_closest_time(image_time, manual_time, manual_idx):
    """ assuming both image_time, and manual_time are datetime objects"""
    t_diff = abs(image_time - manual_time[manual_idx])
    return np.argmin(t_diff), np.min(t_diff)

closest_idx, __ = find_closest_time(image_times, manual_times, calibration_idx)

# manual_counts, manual_std, manual_times
def scale_by_manual_calibration_idx(manual_count, image_counts, closest_idx, calibration_window_size=1, shift=0):
    """ determine scale factor for image_counts based on manual_counts and calibration_idx """
    scale_factor = []
    
    # added due to some potential calibration times lining up with "night" conditions
    idx_select = closest_idx + shift
    
    # find the idx for the nearest time to the specified calibration manual time
    # accounting for min/max sizes of image_counts
    idx_min = []
    idx_max = []
    idx_min = int(idx_select - calibration_window_size/2)
    if idx_min < 0:
        idx_min = int(0)
        if len(image_counts) <= calibration_window_size:
            idx_max = int(len(image_counts)-1)
        else:
            idx_max = int(calibration_window_size)
    else:
        idx_max = int(idx_min + calibration_window_size)
    if idx_max >= len(image_counts):
        idx_max = int(len(image_counts)-1)
    image_count_window = image_counts[idx_min:idx_max]
    # compute the scale factor over specified average (based on aggregate size?)
    image_sample_average = np.mean(image_count_window)
    # TODO handle the divide by zero case 
    if image_sample_average == 0:
        scale_factor = 1
    else:
        scale_factor = manual_count / image_sample_average
    print(f'calibration scale factor = {scale_factor}')
    return scale_factor, idx_select

scale_factor_manual, scaling_idx = scale_by_manual_calibration_idx(manual_counts[calibration_idx], 
                                                                   image_counts, 
                                                                   closest_idx, 
                                                                   calibration_window_size,
                                                                   calibration_window_shift)

# apply scale factor to image counts
tank_counts_cal = image_counts * scale_factor_manual
tank_std_cal = image_std * scale_factor_manual


#############################################################
# PLOT RESULTS 

n = 0.5
fig, ax = plt.subplots()

# AI counts
if PLOT_FOCUS_VOLUME:
    ax.plot(image_times, tank_counts_def, label='focus-volume scaled')
    ax.fill_between(image_times, tank_counts_def-n*tank_std_def, tank_counts_def+n*tank_std_def, alpha=0.2)

ax.plot(image_times, tank_counts_cal, label='manually scaled')
ax.fill_between(image_times, tank_counts_cal-n*tank_std_cal, tank_counts_cal+n*tank_std_cal, alpha=0.2)


# manual counts
ax.plot(manual_times, manual_counts, marker='o', color='green', label='manual count')
ax.errorbar(manual_times, manual_counts, yerr= n*manual_std, fmt='o', color='orange', alpha=0.5)

ax.plot(manual_times[calibration_idx], manual_counts[calibration_idx], marker='*', markersize=10, color='red', label='calibration')
ax.plot(image_times[scaling_idx-1], tank_counts_cal[scaling_idx-1], marker='*', markersize=10, color='black', label='shifted calibration')

plt.legend()
plt.grid(True)
plt.xlabel('Days since spawning')
plt.ylabel(f'Tank count (batched {aggregate_size} images)')
plt.title(f'CSLICS AI Count: {tank_sheet_name}')
plt.savefig(os.path.join(save_det_dir, 'Combined tank counts' + tank_sheet_name + '.png'), dpi=600)
plt.show()


print('done')
print(f'cslics uuid: {cslics_uuid}')


import code
code.interact(local=dict(globals(), **locals()))