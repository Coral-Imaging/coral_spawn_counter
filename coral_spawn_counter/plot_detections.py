#! /usr/bin/env python3
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib as mpl
import seaborn.objects as so

from coral_spawn_counter.CoralImage import CoralImage

# load variables from file
with open('detection_results.pkl', 'rb') as f:
    results = pickle.load(f)

root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20221113_amtenuis_cslics03'

# get counts as arrays:
print('getting counts from detections')
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
capture_times = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time_str]

# apply rolling means
plotdatadict = {'capture times': capture_times,
                'eggs': count_eggs,
                'first': count_first,
                'two': count_two,
                'four': count_four,
                'adv': count_adv,
                'dmg': count_dmg}
df = pd.DataFrame(plotdatadict)
# plotdata = pd.Series(count_eggs)

window_size = 10
count_eggs_mean = df['eggs'].rolling(window_size).mean()
count_eggs_std = df['eggs'].rolling(window_size).std()

count_first_mean = df['first'].rolling(window_size).mean()
count_first_std = df['first'].rolling(window_size).std()

count_two_mean = df['two'].rolling(window_size).mean()
count_two_std = df['two'].rolling(window_size).std()

count_four_mean = df['four'].rolling(window_size).mean()
count_four_std = df['four'].rolling(window_size).std()

count_adv = df['adv'].rolling(window_size).mean()
count_adv = df['adv'].rolling(window_size).std()

count_dmg = df['dmg'].rolling(window_size).mean()
count_adv = df['dmg'].rolling(window_size).std()

# TODO fert ratio is just first cleavage to eggs, or everything else to eggs?
count_total = count_eggs_mean + count_first_mean + count_two_mean + count_four_mean + count_adv # not counting damaged

fert_ratio = (count_first_mean + count_two_mean + count_four_mean + count_adv)/ count_total



# sns.lineplot(data=df,
#              x="capture times",
#              y="eggs",
#              err_style='band',
#              errorbar=('sd', 2))

# plot those detections into a matplotlib graph
# plt.plot(capture_times, label='capture times', marker='o')

sns.set_theme(style='whitegrid')

# plt.plot(capture_times, count_eggs, label='Egg')
n = 1
fig1, ax1 = plt.subplots()

plt.plot(capture_times, count_eggs_mean, label='Egg', color='b')
plt.fill_between(capture_times, count_eggs_mean - n*count_eggs_std, count_eggs_mean + n*count_eggs_std, color='b', alpha=0.2)

plt.plot(capture_times, count_first_mean, label='First Cleavage', color='orange')
plt.fill_between(capture_times, count_first_mean - count_first_std*n, count_first_mean + n*count_first_std, color='orange', alpha=0.2)

# plt.plot(capture_times, count_first, label='First Cleavage')
# plt.plot(capture_times, count_two, label='Two Cell Stage')
# plt.plot(capture_times, count_four, label='Four-Eight Cell Stage')
# plt.plot(capture_times, count_adv, label='Advanced')
# plt.plot(capture_times, count_dmg, label='Damaged')

# TODO calculate the standard deviation
# then up/low bounds by adding dev to mean
# plot this both

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Cell Counts over Time')
plt.legend()
plt.savefig(os.path.join(root_dir,'detections','CellCounts.png'))



fig2, ax2 = plt.subplots()
sns.set_theme(style='whitegrid')
plt.plot(capture_times, fert_ratio, label='Fertilisation Ratio')
plt.xlabel('Date')
plt.ylabel('fert ratio')
plt.title('Fertilisation Ratio over Time')
plt.legend()
plt.savefig(os.path.join(root_dir, 'detections','FertRatio.png'))


fig3, ax3 = plt.subplots()
sns.set_theme(style='darkgrid')
plt.plot(capture_times, count_total, label='Total Count')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Total Coral Spawn Count over Time')
plt.legend()
plt.savefig(os.path.join(root_dir, 'detections', 'TotalCoralSpawnCount.png'))

# # plot time deltas:
# from datetime import timedelta
# # time_delta = [cap[i+1] - cap[i] for i, cap in enumerate(capture_times) if i < len(capture_times)]
# time_delta = []
# for i in range(len(capture_times) - 1):
#     time_delta.append(capture_times[i+1] - capture_times[i])

# time_delta_sec = [td.seconds for td in time_delta]

# plt.plot(time_delta_sec, label='time between image capture times', marker='o')
# plt.title('Capture Time Deltas (seconds)')
# plt.xlabel('index')
# plt.ylabel('seconds')
# plt.savefig(os.path.join(root_dir, 'detections','CaptureTimeDeltas100.png'))

plt.show()

import code
code.interact(local=dict(globals(), **locals()))