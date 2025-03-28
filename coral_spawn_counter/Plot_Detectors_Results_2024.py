#! /usr/bin/env python3

"""
Uses the results from SubSurface_detector and Surface detector pickle files and plot them
Can be  run with or without the Surface and SubSurface Detectors on the relevant data already 
with subsurface detector being a machine learning yolo detector
"""
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn.objects as so
import pickle
import pandas as pd
from datetime import datetime
import sys
import bisect
from sklearn.metrics import mean_squared_error
import yaml
sys.path.insert(0, '')

from coral_spawn_counter.Surface_Detector import Surface_Detector

def get_hyper_focal_dist(f, c, n):
    return f + f**2 / (c * n)


## Varibles for running the script
Fert_Rate = True #if True, will calculate the fertalisation rate
Counts_avalible = True #if True, means pkl files avalible, else run the surface detectors
scale_detection_results = True #if True, scale the detection results to the tank size
scale_subsurface_at_2 = False #if True, scale the subsurface counts at the second last manual count
before_2023 = False #if True, use the old manual counts file format and std of maual coutns = n (see constant definitions)
                    # currently will need to manually set the submersion_idx
submersion_idx = 1 #will be overwittien if before_2023 is False

with open("/home/wardlewo/Reggie/corals/coral_spawn_counter/coral_spawn_counter/config_ssd.yaml", "r") as f: ##NOTE this is path to the config file
    config = yaml.safe_load(f)

dataset = config["dataset"]
data_dir = config["data_dir"]
manual_counts_file = config["manual_counts_file"]
manual_counts_sheet = config.get("manual_counts_sheet")
object_names_file = config["object_names_file"]
assesor_id = dataset[-1]

# File locations
save_plot_dir = data_dir+'Plot_Counts' ##NOTE this is save location
sheet_name = manual_counts_sheet
if scale_subsurface_at_2==True:
    result_plot_name = 'tankcounts_with_late_submersion_'
else:
    result_plot_name = 'tankcounts_'
plot_title = dataset.split('_')[-1]+ ' ' + sheet_name
if Counts_avalible==True: #if false will have to set up the paths for the detectors
    subsurface_det_path = data_dir+dataset+'/210_subsurface_detections/subsurface_detections.pkl'  # path to subsurface detections
    surface_det_path = data_dir+dataset+'/alor_atem_2000_surface_detections/surface_detections.pkl' # path to surface detections
    fert_det_path = data_dir+dataset+'/fertalisation_detections/fertalisation_detections.pkl' # path to fertalisation detections

## Constant Definitions
window_size = 100 # for rolling means, etc # TODO ascertain the effect of window_size on results, including stddev
n = 1 # how many std deviations to show
mpercent = 0.1 # range for manual counts
# image_volume = 0.10 # mL
tank_volume = 500 * 1000 # 500 L * 1000 mL/L # NOTE in practice, tank volume is less than 500
image_span = 100 # how many images to averge out for the time history comparison
idx_surface_manual_count = 0
# estimated tank specs area
rad_tank = (100.0 * 10)/2 # mm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2 
# area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04
# TODO: have done - see dof_calcs.py in focus_experiment, Dorian to integrate

## Camera & Lens characteristics
# TODO should be a packaged function self-contained that outputs Diff_DOF
width_pix = 4056 # pixels
height_pix = 3040 # pixels
pix_size = 1.55 / 1000 # um -> mm, pixel size
sensor_width = width_pix * pix_size # mm
sensor_height = height_pix * pix_size # mm
f = 12 # mm, focal length
aperture = 2.8 # f-stop number of the lens
c = 0.2 # mm, circle of confusion, def 0.1, increase to 0.2 to double (linear) the sample volume
hyp_dist = get_hyper_focal_dist(f, c, aperture) # hyper-focal distance = max depth of field of camera
focus_dist = 50 #mm focusing distance, practically the working distance of the camera
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
focus_volume = hfov * vfov * dof_diff # mm^3
print(f'focus volume = {focus_volume} mm^3')

volume_image = focus_volume / 1000 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
volume_tank = 475 * 1000 # 500 L = 500000 ml
fertilisation_time = 180 # mintues, how long the fertilisation detector runs for
if scale_detection_results==False:
    nimage_to_tank_surface = area_tank / area_cslics ### replaced latter with modifier based on manual counts
    nimage_to_tank_volume = volume_tank / volume_image # thus, how many cslics images will fill the whole volume of the tank
capture_time = []




if Counts_avalible==False:
    # if before_2023:
    # img_pattern = '*.jpg'
    # img_dir = data_dir+dataset+'/images_jpg'
    # else:
    img_pattern = '*/*.jpg'
    img_dir = data_dir+dataset+'/images'
    MAX_IMG = 100000
    skip_img = 1 # we skip every X images for speed
    subsurface_pkl_name = 'subsurface_detections.pkl'
    surface_pkl_name = 'surface_detections.pkl'
    fert_pkl_name = 'fertalisation_detections.pkl'
    save_dir_subsurface = data_dir+dataset+'/100000_subsurface_detections'
    save_dir_surface = data_dir+dataset+'/alor_atem_100000_surface_detections'
    save_dir_fert = data_dir+dataset+'/fertalisation_detections'
    meta_dir = config["meta_dir"]
    object_names_file = meta_dir+'/metadata/obj.names'
    subsurface_weights = config["subsurface_weights"]
    surface_weights = config["surface_weights"]
    Surface_Detection = Surface_Detector(weights_file=surface_weights, 
                                         meta_dir = meta_dir, 
                                         img_dir=img_dir, 
                                         save_dir=save_dir_surface, 
                                         output_file=surface_pkl_name, 
                                         max_img=MAX_IMG, 
                                         skip_img=skip_img, 
                                         img_pattern=img_pattern,
                                         conf_thresh=0.5,
                                         iou=0.5)
    Surface_Detection.run()
    Subsurface_Detection = Surface_Detector(weights_file=subsurface_weights, 
                                            meta_dir = meta_dir, 
                                            img_dir=img_dir, 
                                            save_dir=save_dir_subsurface,
                                            output_file=subsurface_pkl_name, 
                                            max_img=MAX_IMG, 
                                            skip_img=skip_img, 
                                            img_pattern=img_pattern,
                                            conf_thresh=0.5,
                                            iou=0.5)
    Subsurface_Detection.run()
    if Fert_Rate:
        Fertilisation_Detection = Surface_Detector(weights_file=surface_weights, 
                                                   meta_dir = meta_dir, 
                                                   img_dir=img_dir, 
                                                   save_dir=save_dir_fert, 
                                                   output_file=fert_pkl_name, 
                                                   max_img=MAX_IMG, 
                                                   skip_img=1, 
                                                   img_pattern=img_pattern, 
                                                   time_lim=fertilisation_time,
                                                   conf_thresh=0.5,
                                                   iou=0.5)
        Fertilisation_Detection.run()
    subsurface_det_path = os.path.join(save_dir_subsurface, subsurface_pkl_name) 
    surface_det_path = os.path.join(save_dir_surface, surface_pkl_name)
    fert_det_path = os.path.join(save_dir_fert, fert_pkl_name)

# File setup
os.makedirs(save_plot_dir, exist_ok=True)

# load classes
with open(object_names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

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

def read_scale_times(dt, file, sheet_name, assesor_id):
    """read the scale times from the excel file and return the index of the closest datetime object"""
    df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
    date_column = df['Date'].iloc[:]
    notes_column = df['Notes'].iloc[:]
    time_column = df['Time Collected'].iloc[:]
    combined_datetime = np.nan
    closest_index = None
    min_difference = None

    for index, note in enumerate(notes_column):
        if note == int(assesor_id):
            break
    combined_datetime = datetime.combine(date_column[index-1], time_column[index-1])

    for index, dt_item in enumerate(dt):
        difference = abs(combined_datetime - dt_item)
        if min_difference is None or difference < min_difference:
            min_difference = difference
            closest_index = index
    return closest_index, combined_datetime

def old_read_manual_counts(file):
    try: #sheet_name='DataExport'
        df = pd.read_excel(os.path.join(file), sheet_name='Nov 14.11 Tank3 Export', engine='openpyxl') 
    except:
        print('error reading the manual counts file, check the sheet name is correct')
        import code
        code.interact(local=dict(globals(), **locals()))

    date_column = df['Date'].iloc[:]
    time_column = df['Time'].iloc[:]
    count_column = df['Manual Count'].iloc[:]

    date_time_objects = []
    mc = []
    for i, value in enumerate(count_column):
        if pd.notna(value):
            date_time_obj = datetime.combine(date_column[i], time_column[i])
            date_time_objects.append(date_time_obj)
            mc.append(value)

    return date_time_objects, mc

def new_read_manual_counts(file, sheet_name):
    """read the manual counts from the excel file and return the datetime objects and counts"""
    try:
        df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
        count_coloum = df['Mean, last 2 digits rounded\nSD, last 2 digits rounded'].iloc[:]
        date_column = df['Date'].iloc[:]
        time_column = df['Time Collected'].iloc[:]
        time_2 = df['Unnamed: 2'].iloc[:]
    except:
        print('error reading the manual counts file, check the sheet name is correct and headings are as expected')
        import code
        code.interact(local=dict(globals(), **locals()))

    combined_datetime_objects = []
    man_counts = []
    std = []
    for i, value in enumerate(count_coloum):
        if i % 6 == 0 and pd.notna(date_column[i]) and pd.notna(time_column[i]): 
            combined_datetime_obj = datetime.combine(date_column[i], time_column[i])
            combined_datetime_objects.append(combined_datetime_obj)
        if i>3 and (i - 4) % 6 == 0 and not np.isnan(value): 
            man_counts.append(value)
        if i>4 and (i - 5) % 6 == 0 and not np.isnan(value):
            std.append(value)    
    if (len(man_counts) < 1) or len(combined_datetime_objects) < 1:
        print('error reading the manual counts, check colum headings')
        import code
        code.interact(local=dict(globals(), **locals()))
    return combined_datetime_objects, man_counts, std

    """create a time history got coral counts"""
    averaged_counts = []
    ts_of_averaged_counts = []
    stds_of_averaged_counts = []
    for i, manual_count in enumerate(manual_counts):
        time_of_man = manual_decimal_days[i]
        idx = bisect.bisect_right(coral_counts_decimal, time_of_man)
        lower_bound = int(max(0, idx - 0.5*image_span))
        upper_bound = int(min(len(scaled_coral_counts), idx + 0.5*image_span))
        range_of_cont = scaled_coral_counts[lower_bound:upper_bound]
        valid_values = range_of_cont[~np.isnan(range_of_cont)]
        
        if len(valid_values) > 0:
            averge_count = np.mean(valid_values)
            averaged_counts.append(averge_count)
            ts_of_averaged_counts.append(coral_counts_decimal[idx])
    stds_of_averaged_counts = np.std(averaged_counts)
    return averaged_counts, ts_of_averaged_counts, stds_of_averaged_counts

def load_counts(surface_det_path):
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

def get_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts, window_size):
    plotdatadict = {'capture times': surface_capture_times,
                    'eggs': count_eggs,
                    'first': count_first,
                    'two': count_two,
                    'four': count_four,
                    'adv': count_adv,
                    'dmg': count_dmg,
                    'total': surface_counts}

    df = pd.DataFrame(plotdatadict)

    count_eggs_mean = df['eggs'].rolling(window_size, center=True, min_periods=1).mean()
    count_eggs_std = df['eggs'].rolling(window_size, center=True, min_periods=1).std()

    count_first_mean = df['first'].rolling(window_size, center=True, min_periods=1).mean()
    count_first_std = df['first'].rolling(window_size, center=True, min_periods=1).std()

    count_two_mean = df['two'].rolling(window_size, center=True, min_periods=1).mean()
    count_two_std = df['two'].rolling(window_size, center=True, min_periods=1).std()

    count_four_mean = df['four'].rolling(window_size, center=True, min_periods=1).mean()
    count_four_std = df['four'].rolling(window_size, center=True, min_periods=1).std()

    count_adv_mean = df['adv'].rolling(window_size, center=True, min_periods=1).mean()
    count_adv_std = df['adv'].rolling(window_size, center=True, min_periods=1).std()

    count_dmg_mean = df['dmg'].rolling(window_size, center=True, min_periods=1).mean()
    count_dmg_std = df['dmg'].rolling(window_size, center=True, min_periods=1).std()

    count_total_mean = df['total'].rolling(window_size, center=True, min_periods=1).mean()
    count_total_std = df['total'].rolling(window_size, center=True, min_periods=1).std()

    return count_total_mean, count_total_std, count_eggs_mean, count_first_mean, count_two_mean, count_four_mean, count_adv_mean, count_dmg_mean

########################################################
# read manual counts file
########################################################
if before_2023:
    dt, mc = old_read_manual_counts(manual_counts_file)
else:
    dt, mc, manual_std = new_read_manual_counts(manual_counts_file, sheet_name)

zero_time = dt[0]
plot_title = plot_title + ' ' + dt[0].strftime("%Y-%m-%d %H:%M:%S")
manual_decimal_days = convert_to_decimal_days(dt, zero_time)

# #######################################################################
# # Subsurface load pixle data
# #######################################################################
capture_time_dt, subsurface_imge_count, count_eggs, count_first, count_two, count_four, count_adv, count_dmg = load_counts(subsurface_det_path)
subsurface_mean, subsurface_std, _, _, _, _, _, _ = get_mean_n_std(capture_time_dt, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, subsurface_imge_count, window_size)
decimal_days = convert_to_decimal_days(capture_time_dt)

if scale_detection_results==True:
    submersion_idx, submersion_time = read_scale_times(dt, manual_counts_file, 'CSLICS_'+sheet_name, assesor_id)
    if scale_subsurface_at_2:
        submersion_idx = len(mc)-2
    submersion_time = convert_to_decimal_days([submersion_time], dt[0])[0]
    manual_count = mc[submersion_idx]
    subsurface_start_idx = bisect.bisect_right(decimal_days,  manual_decimal_days[submersion_idx]) - 1
    manual_scale_factor = manual_count / (subsurface_mean[subsurface_start_idx])
    volume_image = volume_tank / manual_scale_factor
    nimage_to_tank_volume = volume_tank / volume_image
    print(f'cslics spawn subsurface scale factor: {nimage_to_tank_volume}')

submersion_idx, submersion_time = read_scale_times(dt, manual_counts_file, 'CSLICS_'+sheet_name, assesor_id)

subsurface_image_count_total = subsurface_mean * nimage_to_tank_volume
subsurface_image_count_std = subsurface_std * nimage_to_tank_volume

#######################################################################
# interpolate manual counts to frequency of subsurface counts, calcualte RMSE and correlation coefficient
idx_subsurface_manual_count_stop_time = bisect.bisect_right(capture_time_dt, dt[-1]) - 1
mc_interpolated = np.interp(decimal_days[:idx_subsurface_manual_count_stop_time], manual_decimal_days, mc)

rmse_not_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_stop_time]))
correlation_coefficient_not_scaled = np.corrcoef(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_stop_time])[0, 1]
rmse_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_stop_time]))
correlation_coefficient_scaled = np.corrcoef(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_stop_time])[0, 1]

print(f'After scaling subsurface: RMSE {rmse_scaled / nimage_to_tank_volume}, correlation coefficient {correlation_coefficient_scaled}')
print(f'Before scaling subsurface: RMSE {rmse_not_scaled}, correlation coefficient {correlation_coefficient_not_scaled}')

##################################### surface counts ########################################

surface_capture_times, surface_counts, count_eggs, count_first, count_two, count_four, count_adv, count_dmg = load_counts(surface_det_path)

surface_count_total_mean, surface_count_total_std, _, _, _, _, _, _ = get_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts, window_size)

#Surface Count given manual count
if scale_detection_results==True:
    manual_count = mc[idx_surface_manual_count]
    first_non_nan_index = surface_count_total_mean.index[surface_count_total_mean.notna() & (surface_count_total_mean != 0)][0]
    cslics_fov_est = (area_tank / manual_count)*surface_count_total_mean[first_non_nan_index] 
    nimage_to_tank_surface = area_tank / (cslics_fov_est)
    print(f'cslics surface count using FOV from manual count = {nimage_to_tank_surface}')

surface_decimal_days = convert_to_decimal_days(surface_capture_times)
surface_counttank_total = surface_count_total_mean * nimage_to_tank_surface 
surface_counttank_total_std = surface_count_total_std * nimage_to_tank_surface

##############################################################################
## Plots
##############################################################################


#### plot image count
# and std deviation of the counts

subsurface_start_idx = bisect.bisect_right(decimal_days,  manual_decimal_days[submersion_idx]) - 1
subsurface_stop_idx = bisect.bisect_right(decimal_days, manual_decimal_days[-1])

ss_days = decimal_days[subsurface_start_idx:subsurface_stop_idx]
ss_img_actual = subsurface_imge_count[subsurface_start_idx:subsurface_stop_idx]
ss_img_count = subsurface_mean[subsurface_start_idx:subsurface_stop_idx]
ss_img_count_std = subsurface_std[subsurface_start_idx:subsurface_stop_idx]
fig0, ax0 = plt.subplots()
plt.plot(ss_days, ss_img_actual, label='ss image count')
plt.plot(ss_days, ss_img_count, label='ss image count mean')
plt.fill_between(ss_days, ss_img_count - ss_img_count_std, ss_img_count + ss_img_count_std, alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('image count')
plt.title('Subsurface: Image Counts')
plt.suptitle(plot_title)
plt.legend()
plt.savefig(os.path.join(save_plot_dir, 'subsurface_image_counts.png'))

#### subsurface plot
def subsurface_plot(plot_subsurface_days, plot_subsurface_mean, plot_subsurface_std,
                    manual_decimal_days, mc, plot_manual_std,
                    idx_subsuface_manual, dt_idx_subsurface_manual,
                    result_plot_name, scale_subsurface_at_2):
    fig1, ax1 = plt.subplots()
    # subsurface counts
    plt.plot(plot_subsurface_days, plot_subsurface_mean, label='subsurface count')
    plt.fill_between(plot_subsurface_days, plot_subsurface_mean - plot_subsurface_std, 
                    plot_subsurface_mean + plot_subsurface_std, alpha=0.2)
    plt.errorbar(dt_idx_subsurface_manual, plot_subsurface_mean[idx_subsuface_manual], 
                 yerr=plot_subsurface_std[idx_subsuface_manual], fmt='o', color='blue', alpha=0.5)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='orange', marker='o', linestyle='--')
    plt.errorbar(manual_decimal_days, mc, yerr=plot_manual_std, fmt='o', color='orange', alpha=0.5)
    
    #highlight Scale point
    if scale_subsurface_at_2:
        plt.plot(manual_decimal_days[-2], mc[-2], 'ro', label='callibration point', markersize=10)
    else:
        plt.plot(manual_decimal_days[0], mc[0], 'ro', label='callibration point', markersize=10)

    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Subsurface counts')
    plt.suptitle(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, result_plot_name+"subsurface_counts.png"), dpi=300)

#get location to start and stop the plot
subsurface_start_idx = bisect.bisect_right(decimal_days,  manual_decimal_days[submersion_idx]) - 1
subsurface_stop_idx = bisect.bisect_right(decimal_days, manual_decimal_days[-1])

plot_subsurface_days = decimal_days[subsurface_start_idx:subsurface_stop_idx]
plot_subsurface_mean = subsurface_image_count_total[subsurface_start_idx:subsurface_stop_idx]
plot_subsurface_std = subsurface_image_count_std[subsurface_start_idx:subsurface_stop_idx]

#get the error for the subsurface counts at the same time of manual counts
idx_subsuface_manual = []
dt_idx_subsurface_manual = []
for i in manual_decimal_days[submersion_idx:]:
    idx = bisect.bisect_right(plot_subsurface_days, i)
    idx_subsuface_manual.append(idx+subsurface_start_idx-1)
    dt_idx_subsurface_manual.append(plot_subsurface_days[idx-1])

if before_2023:
    manual_std = np.full(len(mc), np.std(mc))


subsurface_plot(plot_subsurface_days, plot_subsurface_mean, plot_subsurface_std,
                manual_decimal_days[submersion_idx:], mc[submersion_idx:], 
                manual_std[submersion_idx:],
                idx_subsuface_manual, dt_idx_subsurface_manual, result_plot_name, scale_subsurface_at_2)


#### surface plot
def surface_plot(plot_surface_days, plot_surface_mean, plot_surface_std,
                 manual_decimal_days, mc, plot_manual_std,
                 idx_suface_manual, dt_idx_surface_manual,
                 result_plot_name, idx_surface_manual_count):
    fig2, ax2 = plt.subplots()
    # surface counts
    plt.plot(plot_surface_days, plot_surface_mean, label='surface count', color='orange')
    plt.errorbar(dt_idx_surface_manual, plot_surface_mean[idx_suface_manual], 
                 yerr=plot_surface_std[idx_suface_manual], fmt='o', color='orange', alpha=0.5)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o', linestyle='--')
    plt.errorbar(manual_decimal_days, mc, yerr=plot_manual_std, fmt='o', color='green', alpha=0.5)
    
    #highlight scale poin
    plt.plot(manual_decimal_days[idx_surface_manual_count], mc[idx_surface_manual_count], 'ro', label='callibration point', markersize=10)
    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Surface counts')
    plt.suptitle(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, result_plot_name+"surface_counts.png"))

#get location to start and stop the plot
surface_start_idx = bisect.bisect_right(surface_decimal_days,  manual_decimal_days[0]) - 1
surface_stop_idx = bisect.bisect_right(surface_decimal_days, manual_decimal_days[submersion_idx])+1

plot_surface_days = surface_decimal_days[surface_start_idx:surface_stop_idx]
plot_surface_mean = surface_counttank_total[surface_start_idx:surface_stop_idx]
plot_surface_std = surface_counttank_total_std[surface_start_idx:surface_stop_idx]

#get the error for the surface counts at the same time of manual counts
idx_suface_manual = []
dt_idx_surface_manual = []
for i in manual_decimal_days[:submersion_idx+1]:
    idx = bisect.bisect_right(plot_surface_days, i)
    idx_suface_manual.append(idx)
    dt_idx_surface_manual.append(plot_surface_days[idx-1])

if not scale_subsurface_at_2:
    surface_plot(plot_surface_days, plot_surface_mean, plot_surface_std,
                manual_decimal_days[:submersion_idx+1], 
                mc[:submersion_idx+1], manual_std[:submersion_idx+1],
                idx_suface_manual, dt_idx_surface_manual, result_plot_name, idx_surface_manual_count)

print('results ploted')

def whole_plot(manual_decimal_days, mc, manual_std,
              decimal_days, subsurface_image_count_total, subsurface_image_count_std,
              surface_decimal_days, surface_counttank_total, surface_counttank_total_std, 
              result_plot_name, submersion_idx, idx_surface_manual_count, scale_subsurface_at_2):
    fig3, ax3 = plt.subplots()
    # subsurface counts
    plt.plot(decimal_days, subsurface_image_count_total, label='subsurface count')
    plt.fill_between(decimal_days, subsurface_image_count_total - subsurface_image_count_std, 
                    subsurface_image_count_total + subsurface_image_count_std, alpha=0.2)

    # surface counts
    plt.plot(surface_decimal_days, surface_counttank_total, label='surface count', color='orange')
    plt.fill_between(surface_decimal_days, surface_counttank_total - surface_counttank_total_std,
                        surface_counttank_total + surface_counttank_total_std, alpha=0.2, color='orange')

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o', linestyle='--')
    plt.errorbar(manual_decimal_days, mc, yerr=manual_std, fmt='o', color='green', alpha=0.5)
    
    #highlight Scale point
    plt.plot(manual_decimal_days[idx_surface_manual_count], mc[idx_surface_manual_count], 'ro', label='surface callibration point', markersize=10)
    if scale_subsurface_at_2:
        plt.plot(manual_decimal_days[-2], mc[-2], 'ro', label='subsurface callibration point', markersize=10)
    else:
        plt.plot(manual_decimal_days[submersion_idx], mc[submersion_idx], 'bo', label='subsurface callibration point', markersize=10)

    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Whole counts')
    plt.suptitle(plot_title)
    plt.legend()
    ax3.set_ylim(0, 800000)
    plt.savefig(os.path.join(save_plot_dir, result_plot_name+"whole_counts.png"))
 
whole_plot(manual_decimal_days, mc, manual_std,
              decimal_days[:subsurface_stop_idx], subsurface_image_count_total[:subsurface_stop_idx], subsurface_image_count_std[:subsurface_stop_idx],
              surface_decimal_days[:subsurface_stop_idx], surface_counttank_total[:subsurface_stop_idx], surface_counttank_total_std[:subsurface_stop_idx], 
              result_plot_name, submersion_idx, idx_surface_manual_count, scale_subsurface_at_2)


################################### Fertilistion Rates ########################################
if Fert_Rate:
    fert_dt, fert_total_count, fert_count_eggs, fer_count_first, fert_count_two, fert_count_four, fert_count_adv, fert_count_dmg = load_counts(fert_det_path)
    _, _, fert_count_eggs_mean, fert_count_first_mean, fert_count_two_mean, fert_count_four_mean, fert_count_adv_mean, fert_count_dmg_mean = get_mean_n_std(fert_dt, fert_count_eggs, fer_count_first, 
                fert_count_two, fert_count_four, fert_count_adv, fert_count_dmg, fert_total_count, 20)
    fert_decimal_days = convert_to_decimal_days(fert_dt)
    fert_decimal_mins = [x *24*60 for x in fert_decimal_days] #fixing the time for the data​
    # TODO fert ratio is just first cleavage to eggs, or everything else to eggs?
    countperimage_total = fert_count_eggs_mean + fert_count_first_mean + fert_count_two_mean + fert_count_four_mean + fert_count_adv_mean # not counting damaged
    fert_ratio = (fert_count_first_mean + fert_count_two_mean + fert_count_four_mean + fert_count_adv_mean)/ countperimage_total
    fert_mean = fert_ratio.rolling(40).mean()


def fert_plot(fert_decimal_mins, fert_ratio, fert_mean, plot_title, result_plot_name):
    fig4, ax4 = plt.subplots()
    plt.plot(fert_decimal_mins, fert_ratio, 
             color='blue', label='fertilisation success')
    plt.plot(fert_decimal_mins, fert_mean, 
             color='red', label='mean fertilisation success')
    plt.xlabel('minutes since stocking')
    plt.ylabel('fertalisation ratio')
    plt.title('fertalisation success')
    plt.suptitle(plot_title)
    plt.legend()
    ax4.set_ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(save_plot_dir, result_plot_name+"fertalisation.png"))

if Fert_Rate and not scale_subsurface_at_2:
    fert_plot(fert_decimal_mins, fert_ratio, fert_mean, plot_title, result_plot_name)

print('done')
import code
code.interact(local=dict(globals(), **locals()))