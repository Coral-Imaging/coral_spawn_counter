#!/usr/bin/env python3


# read spawn_table.csv
# plot running average?
# read/output average


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from pprint import *

import PIL.Image as PIL_Image
from coral_spawn_counter.CoralImage import CoralImage


def seconds_to_date(time):
    # input seconds, convert to day, hour, minutes, seconds:

    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    min = time // 60
    time %= 60
    sec = time

    return day, hour, min, sec 

# localtion of spawn table
# root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
# root_dir = '/home/cslics/Pictures/cslics_data_Nov14_test'
# root_dir = '/media/agkelpie/cslics_ssd/2022_NovSpawning/20221112_AMaggieTenuis/'
# root_dir = '/home/cslics/Dropbox/QUT/GreatBarrierReefRestoration_Automation/Transition2Deployment/CSLICS/cslics_sample_images/time_series_example/20221114_AMaggieTenuis'
# root_dir = '/media/cslics/cslics_ssd/2022_NovSpawning/20221113_AMaggieTenuis'
root_dir = '/media/cslics/cslics_ssd/AIMS_2022_Dec_Spawning/20221213_datagrab'
host = 'cslics01_sample'
spawn_table_file = os.path.join(root_dir, host, 'metadata', 'spawn_counts.csv')

df = pd.read_csv(spawn_table_file)

img_names = df['image_name'].tolist()
capture_times = df['capture_time'].tolist()
counts = df['count'].tolist()
window_size = 100 # TODO needs careful decision making onto window size for meaningful numbers
counts_mean = df['count'].rolling(window=window_size).mean()

x = np.arange(0,len(counts), 1)
y = np.array(counts)
x_rolling_mean = x[window_size:len(x)-window_size]

# convert capture_times (string) to list of datetime objects
dt_format = "%Y%m%d_%H%M%S_%f"
dt_capture_times = [datetime.datetime.strptime(cap, dt_format) for cap in capture_times]

dt_times0 = [dt - dt_capture_times[0] for dt in dt_capture_times]
dt_sec0 = [dt.total_seconds() for dt in dt_times0]

# plot the seconds - should only be increasing
# fig, ax =  plt.subplots(1)
# plt.plot(x, dt_sec0)
# plt.ylabel('time(sec)')
# plt.xlabel('index')
# plt.show()

# TODO convert seconds into days/hours/minutes/seconds
# for i in range(len(dt_sec0)):
#     d, h, m, s = seconds_to_date(dt_sec0[i])

fig1, ax1 = plt.subplots(1)
fig1.autofmt_xdate()
plt.plot(dt_capture_times, y, label='image count')
plt.plot(dt_capture_times, counts_mean, label='image rolling mean {}'.format(window_size))
# plt.plot()

plt.title([host + 'date of spawning' ])
plt.ylabel('image count')
plt.xlabel('time')
ax1.grid(True)
plt_date_format = '%d %H:%M'
xfmt = mdates.DateFormatter(plt_date_format)
ax1.xaxis.set_major_formatter(xfmt)


# manual counts
# TODO read in .csv file from Google Sheets from the manual times

# manual counts from 13 Dec 2022 spawning, Tank 3 mycedium elephantatos
manual_counts = [540000, 540000]
manual_times = ['2022-12-13 22:35', '2022-12-14 00:30']

# manual_counts = [568200, 468400, 416500, 277700, 273000, 227500]
# manual_times = ['2022-11-14 19:30',
#                 '2022-11-15 20:51',
#                 '2022-11-16 14:35',
#                 '2022-11-16 16:03',
#                 '2022-11-17 16:39',
#                 '2022-11-18 19:22']
mtime_format = '%Y-%m-%d %H:%M'
manual_times = [datetime.datetime.strptime(mt, mtime_format) for mt in manual_times]

pprint(manual_times)


# note: cslics surface area counts differ for different cslics!!
# area_cslics = 2.3**2*(3/4) # cm^2 for cslics03 @ 15cm distance - had micro1 lens
area_cslics = 2.35**2*(3/4) # cm2 for cslics01 @ 15.5 cm distance with micro2 lens

# area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04

print(f'Estimated surface area: {area_cslics} cm^2')
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2
# manual_count = 687790 # spawn
print(f'Number of images: {len(counts)}')

nimage_to_tank_surface = area_tank / area_cslics
print(f'nimage multiplier {nimage_to_tank_surface}')

print('PHASE 1')
print('Surface area analysis')
manual_count = manual_counts[0]
expected_count_perimage = manual_count / nimage_to_tank_surface
print(f'Expected count per image = {expected_count_perimage}')

cslics_estimate_tank = counts_mean * nimage_to_tank_surface 
print(f'Manual count for entire tank = {manual_count}')
print(f'Expected range (+- 10%): min {manual_count*0.9}, max {manual_count * 1.1}')
print(f'Cslics spawn count for entire tank = {cslics_estimate_tank}')


# set manual counts to range +- 10%:
manual_count_upper = np.array(manual_counts) * 1.10
manual_count_lower = np.array(manual_counts) * 0.9



# import code
# code.interact(local=dict(globals(), **locals()))

fig2, ax2 = plt.subplots(1)
fig2.autofmt_xdate()
plt.plot(dt_capture_times, cslics_estimate_tank, label='cslics estimate')
plt.plot(manual_times, manual_counts, label='manual counts', color='green', marker='o')
plt.fill_between(manual_times, manual_count_upper, manual_count_lower, alpha=0.15, color='green', label='10% of manual counts')
plt.title(host + ' 2022 Dec 13 spawning - Myc. El. - Tank 3')
plt.ylabel('count')
plt.xlabel('time')
ax2.grid(True)
plt_date_format = '%d %H:%M'
xfmt = mdates.DateFormatter(plt_date_format)
ax2.xaxis.set_major_formatter(xfmt)
ax2.legend()

plt.show()

avg_count = np.mean(counts)
std_count = np.std(counts)
print(f'For {host}:, mean count: {avg_count}, std_dev: {std_count}')


# surface area calculations





# print('PHASE 2')
# print('Volume and density analysis')
# # volume/density calculations:
# volume_image = 35 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
# volume_tank = 500 * 1000 # 500 L = 500000 ml
# # thus, how many cslics images will fill the whole volume of the tank
# nimage_to_tank_volume = volume_tank / volume_image

# expected_count_per_imagevolume = manual_count / nimage_to_tank_volume

# cslics_estimate_tank_volume = avg_count * nimage_to_tank_volume

# print(f'nimage to tank volume: {nimage_to_tank_volume}')
# print(f'cslics spawn count density for entire tank volume: {cslics_estimate_tank_volume}')

import code
code.interact(local=dict(globals(), **locals()))