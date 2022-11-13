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

# import code
# code.interact(local=dict(globals(), **locals()))