#!/usr/bin/env python3


# read spawn_table.csv
# plot running average?
# read/output average


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PIL.Image as PIL_Image
from coral_spawn_counter.CoralImage import CoralImage

# localtion of spawn table
# root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
root_dir = '/home/cslics/Pictures/cslics_data_Nov14_test'
host = 'cslics01'
spawn_table_file = os.path.join(root_dir, host, 'metadata', 'spawn_counts.csv')

df = pd.read_csv(spawn_table_file)

img_names = df['image_name'].tolist()
capture_times = df['capture_time'].tolist()
counts = df['count'].tolist()

# since we know that there's 30 images/sample, we take a moving average of 30?
# TODO organise according to time/samples?

avg_count = np.mean(counts)
std_count = np.std(counts)
print(f'For {host}:, mean count: {avg_count}, std_dev: {std_count}')

x = np.arange(0,len(counts), 1)
y = np.array(counts)
plt.plot(x, y)
plt.title(host)
plt.ylabel('count')
plt.xlabel('sample number (not yet time)')
plt.show()

# surface area calculations
area_cslics = 2.3**2*(3/4) # cm^2
print(f'Estimated surface area: {area_cslics} cm^2')
rad_tank = 105.0/2 # cm^2
area_tank = np.pi * rad_tank**2
manual_count = 687790 # spawn
print(f'Number of images: {len(counts)}')

nimage_to_tank_surface = area_tank / area_cslics
print(f'nimage multiplier {nimage_to_tank_surface}')

print('PHASE 1')
print('Surface area analysis')
expected_count_perimage = manual_count / nimage_to_tank_surface
print(f'Expected count per image = {expected_count_perimage}')

cslics_estimate_tank = avg_count * nimage_to_tank_surface 
print(f'Manual count for entire tank = {manual_count}')
print(f'Expected range (+- 10%): min {manual_count*0.9}, max {manual_count * 1.1}')
print(f'Cslics spawn count for entire tank = {cslics_estimate_tank}')


print('PHASE 2')
print('Volume and density analysis')
# volume/density calculations:
volume_image = 35 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
volume_tank = 500 * 1000 # 500 L = 500000 ml
# thus, how many cslics images will fill the whole volume of the tank
nimage_to_tank_volume = volume_tank / volume_image

expected_count_per_imagevolume = manual_count / nimage_to_tank_volume

cslics_estimate_tank_volume = avg_count * nimage_to_tank_volume

print(f'nimage to tank volume: {nimage_to_tank_volume}')
print(f'cslics spawn count density for entire tank volume: {cslics_estimate_tank_volume}')

# import code
# code.interact(local=dict(globals(), **locals()))