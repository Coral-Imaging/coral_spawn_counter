#! /usr/bin/env python3

"""
use the results from SubSurface_detector and Surface detector pixkle files and plot them
assume that you have run the Surface and SubSurface Detectors on the relevant data already...
"""
#test
t = 1

import os
import cv2 as cv
import numpy as np
import glob
import random as rng
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import seaborn.objects as so
import pickle
import pandas as pd
from datetime import datetime
import time
import sys
import bisect
from sklearn.metrics import mean_squared_error
import math
sys.path.insert(0, '')

from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.read_manual_counts import read_manual_counts

from coral_spawn_counter.Surface_Detector import Surface_Detector
from coral_spawn_counter.SubSurface_Detector import SubSurface_Detector

# Consts (more below as well)
window_size = 60 # for rolling means, etc
n = 1 # how many std deviations to show
mpercent = 0.1 # range for manual counts
image_volume = 0.10 # mL
tank_volume = 500 * 1000 # 500 L * 1000 mL/L

Counts_avalible = True #if True, means pkl files avalible, else run the surface detectors
scale_detection_results = True #if True, scale the detection results to the tank size
idx_subsuface_manual_count = 4
idx_surface_manual_count = 1

# estimated tank specs area
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2 
area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04
volume_image = 35 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
volume_tank = 500 * 1000 # 500 L = 500000 ml
if scale_detection_results==False:
    nimage_to_tank_surface = area_tank / area_cslics ### replaced latter with modifier based on manual counts
    nimage_to_tank_volume = volume_tank / volume_image # thus, how many cslics images will fill the whole volume of the tank
capture_time = []

# File locations
save_plot_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06'
manual_counts_file = '/home/java/Java/data/cslics_ManualCounts/2023-12/C-SLIC culture density data sheet.xlsx'
sheet_name = 'Dec-A.lor Tank 3'
img_dir = '/home/java/Java/data/20231204_alor_tank3_cslics06/images'
object_names_file = '/home/java/Java/cslics/metadata/obj.names'
result_plot_name = 'Tankcouts_with_scalingS.png'
plot_title = 'Cslics06 '+sheet_name+' alor_aten_2000'
if Counts_avalible==True:
    subsurface_det_path = '/home/java/Java/data/20231204_alor_tank3_cslics06/detections_subsurface/subsurface_detections.pkl'
    surface_det_path = '/home/java/Java/data/20231204_alor_tank3_cslics06/detect_surface/surface_detections.pkl'
else:
    MAX_IMG = 10e10
    skip_img = 50
    subsurface_pkl_name = 'subsurface_detections2.pkl'
    surface_pkl_name = 'surface_detections2.pkl'
    save_dir_subsurface = '/home/java/Java/data/20231204_alor_tank3_cslics06/detections_subsurface_2'
    save_dir_surface = '/home/java/Java/data/20231204_alor_tank3_cslics06/detect_surface_2'
    meta_dir = '/home/java/Java/cslics' 
    weights = '/home/java/Java/ultralytics/runs/detect/train - alor_1000/weights/best.pt'
    object_names_file = '/home/java/Java/cslics/metadata/obj.names'
    result_plot_name = 'subsubface_tankcounts_without_scaling_2.png'
    Coral_Detector = Surface_Detector(weights_file=weights, meta_dir = meta_dir, img_dir=img_dir, save_dir=save_dir_surface, 
                                      output_file=surface_pkl_name, max_img=MAX_IMG, skip_img=skip_img)
    Coral_Detector.run()
    Coral_Detector = SubSurface_Detector(meta_dir = meta_dir, img_dir = img_dir, save_dir=save_dir_subsurface, 
                                         detection_file=subsurface_pkl_name, max_img = MAX_IMG, skip_img=skip_img)
    Coral_Detector.run() 
    subsurface_det_path = os.path.join(save_dir_subsurface, subsurface_pkl_name) 
    surface_det_path = os.path.join(save_dir_surface, surface_pkl_name)


# File setup
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
os.makedirs(save_plot_dir, exist_ok=True)

# load classes
#with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
with open(object_names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def convert_to_decimal_days(dates_list, time_zero=None):
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

def new_read_manual_counts(file, sheet_name):
    df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
    count_coloum = df['Calcs'].iloc[:]
    date_column = df['Date'].iloc[:]
    time_column = df['Time Collected'].iloc[:]

    # get all the dates and times
    date_objects = []
    time_objects = []
    combined_datetime_objects = []
   
    # get all the manual counts
    man_counts = []
    for i, value in enumerate(count_coloum):
        if i >= 1 and i % 6 == 0: #would normally just be i>0
            combined_datetime_obj = datetime.combine(date_column[i], time_column[i])
        if i>6 and (i - 4) % 6 == 0 and not np.isnan(value): #would normally be i>4
            man_counts.append(value)
            combined_datetime_objects.append(combined_datetime_obj)
    
    return combined_datetime_objects, man_counts

########################################################
# read manual counts file
########################################################

#dt, mc, tw = read_manual_counts(manual_counts_file)
dt, mc = new_read_manual_counts(manual_counts_file, sheet_name)
zero_time = dt[0]
plot_title = plot_title + ' ' + dt[0].strftime("%Y-%m-%d %H:%M:%S")
manual_decimal_days = convert_to_decimal_days(dt, zero_time)

#######################################################################
# Subsurface load pixle data
with open(subsurface_det_path, 'rb') as f:
    save_data = pickle.load(f)
    
blobs_list = save_data['blobs_list']
blobs_count = save_data['blobs_count']
image_index = save_data['image_index']
capture_time = save_data['capture_time']

# convert blobs_count into actual count, not interior list of indices
subsurface_imge_count = [len(blobs_index) for blobs_index in blobs_count]
subsurface_imge_count = np.array(subsurface_imge_count)

#capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time]
capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S') for d in capture_time]
decimal_days = convert_to_decimal_days(capture_time_dt)

#subsurface counts given manual count
if scale_detection_results==True:
    manual_count = mc[idx_subsuface_manual_count]
    manual_scale_factor = manual_count / np.mean(subsurface_imge_count)
    volume_image = volume_tank / manual_scale_factor
    nimage_to_tank_volume = volume_tank / volume_image
    print(f'cslics spawn subsurface scale factor: {nimage_to_tank_volume}')
subsurface_image_count_total = subsurface_imge_count * nimage_to_tank_volume 
density_count = subsurface_imge_count * image_volume
############################################
# interpolate manual counts to frequency of subsurface counts, calcualte RMSE and correlation coefficient
idx_subsurface_manual_count_time = bisect.bisect_right(capture_time_dt, dt[-1]) - 1
mc_interpolated = np.interp(decimal_days[:idx_subsurface_manual_count_time], manual_decimal_days, mc)
rmse_not_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_time]))
correlation_coefficient_not_scaled = np.corrcoef(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_time])[0, 1]
print(f'Before scaling subsurface: RMSE {rmse_not_scaled}, correlation coefficient {correlation_coefficient_not_scaled}')
rmse_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_time]))
correlation_coefficient_scaled = np.corrcoef(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_time])[0, 1]
print(f'After scaling subsurface: RMSE {rmse_scaled}, correlation coefficient {correlation_coefficient_scaled}')
##################################### surface counts ########################################
def load_surface_counts(surface_det_path):
    #with open(os.path.join(root_dir, surface_pkl_file), 'rb') as f:
    with open(surface_det_path, 'rb') as f:
        results = pickle.load(f)
    # get counts as arrays:
    count_eggs = []
    count_first = []
    count_two = []
    count_four = []
    count_adv = []
    count_dmg = []
    capture_time_str = []

    for res in results:
        # create a list of strings of all the detections for the given image
        counted_classes = [det[6] for det in res.detections]

        # do list comprehensions on counted_classes  # TODO consider putting this as a function into CoralImage/detections
        # could I replace this with iterating over the classes dictionary?
        count_eggs.append(float(counted_classes.count('Egg')))
        count_first.append(counted_classes.count('FirstCleavage')) # TODO fix class names to be all one continuous string (no spaces)
        count_two.append(counted_classes.count('TwoCell'))
        count_four.append(counted_classes.count('FourEightCell'))
        count_adv.append(counted_classes.count('Advanced'))
        count_dmg.append(counted_classes.count('Damaged'))
        capture_time_str.append(res.metadata['capture_time'])

    # parse capture_time into datetime objects so we can sort them
    surface_capture_times = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time_str]

    surface_counts = [count_eggs[i] + count_first[i] + count_two[i] + count_four[i] + count_adv[i] for i in range(len(count_eggs))]
    # apply rolling means
    return surface_capture_times, surface_counts, count_eggs, count_first, count_two, count_four, count_adv, count_dmg

#Get the surface idx_surface_manual_count
def read_scale_times(dt, file, sheet_name):
    df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
    date_column = df['Date'].iloc[:]
    notes_column = df['Notes'].iloc[:]
    time_column = df['Time Collected'].iloc[:]

    combined_datetime = np.nan
    closest_index = None
    min_difference = None

    for index, note in enumerate(notes_column):
        if note == 'Time placed into water' and not pd.isnull(combined_datetime):
            break
        elif not pd.isnull(date_column[index]):
            combined_datetime = datetime.combine(date_column[index], time_column[index])

    for index, dt_item in enumerate(dt):
        difference = abs(combined_datetime - dt_item)
        if min_difference is None or difference < min_difference:
            min_difference = difference
            closest_index = index

    return closest_index

def get_surface_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts):
    plotdatadict = {'capture times': surface_capture_times,
                    'eggs': count_eggs,
                    'first': count_first,
                    'two': count_two,
                    'four': count_four,
                    'adv': count_adv,
                    'dmg': count_dmg,
                    'total': surface_counts}

    df = pd.DataFrame(plotdatadict)

    count_eggs_mean = df['eggs'].rolling(window_size).mean()
    count_eggs_std = df['eggs'].rolling(window_size).std()

    count_first_mean = df['first'].rolling(window_size).mean()
    count_first_std = df['first'].rolling(window_size).std()

    count_two_mean = df['two'].rolling(window_size).mean()
    count_two_std = df['two'].rolling(window_size).std()

    count_four_mean = df['four'].rolling(window_size).mean()
    count_four_std = df['four'].rolling(window_size).std()

    count_adv_mean = df['adv'].rolling(window_size).mean()
    count_adv_std = df['adv'].rolling(window_size).std()

    count_dmg_mean = df['dmg'].rolling(window_size).mean()
    count_dmg_std = df['dmg'].rolling(window_size).std()

    count_total_mean = df['total'].rolling(window_size).mean()
    count_total_std = df['total'].rolling(window_size).std()

    return count_total_mean, count_total_std

surface_capture_times, surface_counts, count_eggs, count_first, count_two, count_four, count_adv, count_dmg = load_surface_counts(surface_det_path)

surface_count_total_mean, surface_count_total_std = get_surface_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts)

#Surface Count given manual count
if scale_detection_results==True:
    idx_surface_manual_count = read_scale_times(dt, manual_counts_file, 'CSLICS_'+sheet_name)
    manual_count = mc[idx_surface_manual_count]
    first_non_nan_index = surface_count_total_mean.index[surface_count_total_mean.notna() & (surface_count_total_mean != 0)][0]
    cslics_fov_est = (area_tank / manual_count)*surface_count_total_mean[first_non_nan_index] 
    nimage_to_tank_surface = area_tank / (cslics_fov_est * 50)
    print(f'cslics surface count using FOV from manual count = {nimage_to_tank_surface}')

# countperimage_total = count_eggs_mean + count_first_mean + count_two_mean + count_four_mean + count_adv # not counting damaged
surface_decimal_days = convert_to_decimal_days(surface_capture_times)
surface_counttank_total = surface_count_total_mean * nimage_to_tank_surface 

##############################################################################
## Plots
##############################################################################

# show averages to apply rolling means
plotdatadict = {
    'index': image_index,
    'capture_time_days': decimal_days,
    'image_count': subsurface_imge_count,
    'density_count': density_count,
    'tank_count': subsurface_image_count_total
}
df = pd.DataFrame(plotdatadict)


image_count_mean = df['image_count'].rolling(window_size).mean()
image_count_std = df['image_count'].rolling(window_size).std()

density_count_mean = df['density_count'].rolling(window_size).mean()
density_count_std = df['density_count'].rolling(window_size).std()

tank_count_mean = df['tank_count'].rolling(window_size).mean()
tank_count_std = df['tank_count'].rolling(window_size).std()


# TODO showcase counts over time?
sns.set_theme(style='whitegrid')


fig1, ax1 = plt.subplots()
plt.plot(decimal_days, image_count_mean, label='count')
plt.fill_between(decimal_days, 
                 image_count_mean - n*image_count_std,
                 image_count_mean + n*image_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('image count')
plt.title('Subsurface Counts vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubfacecounts.png'))


fig2, ax2 = plt.subplots()
plt.plot(decimal_days, density_count_mean, label='density [count/mL]')
plt.fill_between(decimal_days, 
                 density_count_mean - n*density_count_std,
                 density_count_mean + n*density_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('density')
plt.title('Subsurface Density count/mL vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubface_densitycounts.png'))


def wholistic_tank_count_n_plot(surface_decimal_days, counttank_total, nimage_to_tank_surface, count_total_std, n,
                            decimal_days, tank_count_mean, tank_count_std,
                            manual_decimal_days, mc, mpercent,
                            save_plot_dir):
    # wholistic tank count
    fig3, ax3 = plt.subplots()

    # surface counts
    plt.plot(surface_decimal_days, counttank_total, label='surface count', color='orange')
    plt.fill_between(surface_decimal_days,
                    counttank_total - n * nimage_to_tank_surface * count_total_std,
                    counttank_total + n * nimage_to_tank_surface * count_total_std,
                    alpha=0.2,
                    color='orange')

    # subsurface counts
    plt.plot(decimal_days, tank_count_mean, label='subsurface count')
    plt.fill_between(decimal_days, 
                    tank_count_mean - n * tank_count_std,
                    tank_count_mean + n * tank_count_std,
                    alpha=0.2)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o')
    mc_under = [mc - mpercent * mc for mc in mc]
    mc_over = [mc + mpercent * mc for mc in mc]
    plt.fill_between(manual_decimal_days, 
                    mc_under,
                    mc_over,
                    alpha=0.1,
                    color='green')
    highlight_indices = [idx_surface_manual_count, idx_subsuface_manual_count]  #idx for scale factor
    highlight_colors = ['orange', 'blue']  # colour of surface and subsurface manual counts
    for idx, color in zip(highlight_indices, highlight_colors): # Plot the highlighted points
        plt.scatter(manual_decimal_days[idx], mc[idx], color=color, s=100)  # Change s for point size if needed

    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Overall tank count vs Time')
    plt.suptitle(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, result_plot_name))

wholistic_tank_count_n_plot(surface_decimal_days, surface_counttank_total, nimage_to_tank_surface, surface_count_total_std, n,
                            decimal_days, tank_count_mean, tank_count_std,
                            manual_decimal_days, mc, mpercent,
                            save_plot_dir)

print('results ploted')


# TODO need to ascertain the validity of the blobs - not all edges are ideal - chat with Andrew
# TODO save blob counts to txt file for reading/later usage?
# TODO write to xml file for uploading blob annotations    
import code
code.interact(local=dict(globals(), **locals()))