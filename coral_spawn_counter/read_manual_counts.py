#! /usr/bin/env python3

"""
script to read in read in  manual counts, save them into a json file
"""

# valid only for 2024 coral spawning data
# specify location of xlsx file
# read in file from specific locations in tables
# setup relevant data into dictionary
# save as a json file

import os
import pandas as pd
# from datetime import datetime

import matplotlib.pyplot as plt



manual_counts_file = '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_manual_counts.xlsx'
cslics_associations_file = '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_spawning_setup.xlsx'
save_plot_dir = '/home/dtsai/Data/cslics_datasets/manual_counts/plots'
os.makedirs(save_plot_dir, exist_ok=True)

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

def read_manual_count_sheet_correspondence(file, sheet_name):
    """read the manual counts from the excel file and return the datetime objects and counts"""
    try:
        df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
        tank_sheet_column = df['manual count sheet name']
        camera_uuid_column = df['camera uuid']
        species_column = df['species']

        # count_coloum = df['Mean, last 2 digits rounded\nSD, last 2 digits rounded'].iloc[:]
        # date_column = df['Date'].iloc[:]
        # time_column = df['Time Collected'].iloc[:]
        # time_2 = df['Unnamed: 2'].iloc[:]
        return tank_sheet_column, camera_uuid_column, species_column
    except:
        print('ERROR: reading the manual counts file, check the sheet name is correct and headings are as expected')
        # import code
        # code.interact(local=dict(globals(), **locals()))
    

    # combined_datetime_objects = []
    # man_counts = []
    # std = []
    # for i, value in enumerate(count_coloum):
    #     if i % 6 == 0 and pd.notna(date_column[i]) and pd.notna(time_column[i]): 
    #         combined_datetime_obj = datetime.combine(date_column[i], time_column[i])
    #         combined_datetime_objects.append(combined_datetime_obj)
    #     if i>3 and (i - 4) % 6 == 0 and not np.isnan(value): 
    #         man_counts.append(value)
    #     if i>4 and (i - 5) % 6 == 0 and not np.isnan(value):
    #         std.append(value)    
    # if (len(man_counts) < 1) or len(combined_datetime_objects) < 1:
    #     print('error reading the manual counts, check colum headings')
    #     import code
    #     code.interact(local=dict(globals(), **locals()))
    # return combined_datetime_objects, man_counts, std


# this sheet tells us which sheet name in the manual counts data sheet to look for
# corresponds uuid to species and manual count sheet
# spawning_sheet_name = '2024 oct'
spawning_sheet_name = '2024 nov'
df = pd.read_excel(cslics_associations_file, sheet_name=spawning_sheet_name, engine='openpyxl', header=2) 

tank_sheet_column, camera_uuid_column, species_column = read_manual_count_sheet_correspondence(cslics_associations_file, spawning_sheet_name)

# specify which sheet to look at:
# TODO later specify this through a config file
tank_sheet_name = tank_sheet_column[4]

# next, we look at the cslics manual counts datasheet for 2024
# TODO need to create sheet for each manual count session

df = pd.read_excel(manual_counts_file, sheet_name = tank_sheet_name, engine='openpyxl',header=5)
# date_column = [datetime.strptime(date ,"%d/%m/%Y") for date in df['Date']]
date_column = df['Date']
time_collected = df['Time']
count_column = df['Count (500L)']
std_column = df['Std Dev']

# combine date and time into single datetime object
# TODO probably a more efficient way of combining these datetime objects than to str and then back
counts_time = pd.to_datetime(date_column.astype(str) + " " + time_collected.astype(str), dayfirst=True)
decimal_days = convert_to_decimal_days(counts_time)

# plot the manual counts over time for the given series
fig, ax = plt.subplots()
scaled_counts = count_column/1000 # for readability
scaled_std = std_column/1000
n = 1

ax.plot(decimal_days, scaled_counts, marker='o', color='blue')
ax.errorbar(decimal_days, scaled_counts, yerr= n*scaled_std, fmt='o', color='blue', alpha=0.5)
plt.grid(True)
plt.xlabel('Days since spawning')
plt.ylabel('Tank count (in thousands for 500L)')
plt.title(f'CSLICS Manual Count: {tank_sheet_name}')
plt.savefig(os.path.join(save_plot_dir, 'Manual counts' + tank_sheet_name + '.png')) 

print('done')

import code
code.interact(local=dict(globals(), **locals()))

