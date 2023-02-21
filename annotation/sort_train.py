#! /usr/bin/env python3

"""
sort train.txt into alphabetical order
"""

# full path, name of text file:
txtfile = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/train.txt'

# output text file
outtxtfile = '/home/agkelpie/Code/cslics_ws/src/datasets/202211_amtenuis_1000/metadata/train_sorted.txt'

with open(txtfile, 'r') as f:
    lines = f.readlines()

# sort lines
lines.sort()

with open(outtxtfile, 'w') as f:
        f.writelines(lines)


print('done')
