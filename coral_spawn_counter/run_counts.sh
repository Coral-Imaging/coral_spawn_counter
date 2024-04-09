#!/bin/bash

# run counting python code for each line/folder in data/csllics_202211 file

data_file="/home/dorian/Code/cslics_ws/src/coral_spawn_counter/data/cslics_202311"
python_script="/home/dorian/Code/cslics_ws/src/coral_spawn_counter/coral_spawn_counter/counts_202311.py"
DIR="/home/dorian/Data/cslics_2023_datasets/2023_Nov_Spawning"
echo 'Running counts'
for remote in $(cat $data_file); do
    echo "Processing line: $DIR/$remote"
    python3 "$python_script" "$DIR/$remote"
done