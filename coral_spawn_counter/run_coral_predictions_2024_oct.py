#!/usr/bin/env python3

import os
from CoralSpawnPredictor import CoralSpawnPredictor

# script to run all the coral predictions on the HPC for October 2024 spawning

# root directory of all the cslics datasets for oct
root_dir = '/home/tsaid/data/cslics_datasets/cslics_october_2024'

# list of each cslics uuids to run:
cslics_uuids = ['100000000029da9b',
                '100000000846a7ff',
                '100000001ab0438d',
                '10000000570f9d9c',
                '100000009c23b5af',
                '10000000f620da42']

weights_path = '/home/tsaid/data/cslics_datasets/models/cslics_subsurface_20250205_640p_yolov8n.pt'
classes = ['coral']
class_colours = {'coral': [0, 0, 255]}


def run_predictions(cslics):
    """
    Function to run predictions for a single cslics UUID.
    """
    img_dir = os.path.join(root_dir, cslics)
    save_dir = os.path.join(root_dir, 'detections', cslics)
    print(f'Running predictions for {cslics}')
    print(f'Image directory: {img_dir}')
    print(f'Save directory: {save_dir}')
    print('')

    # Run the coral predictions
    predictor = CoralSpawnPredictor(weights_path, img_dir, save_dir, classes, class_colours)
    predictor.run()

    print(f'Finished predictions for {cslics}')
    print('')

# Main sequential execution
if __name__ == "__main__":
    for cslics in cslics_uuids:
        run_predictions(cslics)
        
    print('Completed all predictions')