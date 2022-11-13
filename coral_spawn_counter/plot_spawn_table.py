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
root_dir = '/home/cslics/cslics_ws/src/rrap-downloader/cslics_data'
host = 'cslics04'
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


area_cslics = 3.0 # cm^2
rad_tank = 105.0/2 # cm^2
area_tank = np.pi * rad_tank**2
manual_count = 420000 # spawn

nimage_to_tank = area_tank / area_cslics

expected_count_perimage = manual_count / nimage_to_tank
print(f'Expected count per image = {expected_count_perimage}')

cslics_estimate_tank = avg_count * nimage_to_tank 
print(f'Manual count for entire tank = {manual_count}')
print(f'Expected range (+- 10%): min {manual_count*0.9}, max {manual_count * 1.1}')
print(f'Cslics spawn count for entire tank = {cslics_estimate_tank}')
# import code
# code.interact(local=dict(globals(), **locals()))