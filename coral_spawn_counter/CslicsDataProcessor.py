#!/usr/bin/env python3

"""
CSLICS Data Processor
- Encapsulates functionality for processing and plotting tank estimates with manual counts.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CSLICSDataProcessor:
    def __init__(self, config):
        """
        Initialize the CSLICS Data Processor.
        """
        self.manual_counts_file = config['manual_counts_file']
        self.spawning_sheet_name = config['spawning_sheet_name']
        self.tank_sheet_name = config['tank_sheet_name']
        self.cslics_associations_file = config['cslics_associations_file']
        self.cslics_uuid, self.coral_species = self.read_cslics_uuid_tank_association()
        print(f'CSLICS UUID: {self.cslics_uuid}, Coral Species: {self.coral_species}')
        
        self.model_name = config['model_name']
        self.base_det_dir = config['base_detection_dir']
        self.model_det_dir = self._determine_detection_directory()
        self.save_manual_plot_dir = config['save_manual_plot_dir']
        self.save_det_dir = self.model_det_dir

        # Additional configuration parameters
        self.skipping_frequency = config.get('skipping_frequency', 1)
        self.aggregate_size = config.get('aggregate_size', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.MAX_SAMPLE = config.get('MAX_SAMPLE', 1000)
        self.calibration_window_size = config.get('calibration_window_size', 1)
        self.calibration_idx = config.get('calibration_idx', 1)
        self.calibration_window_shift = config.get('calibration_window_shift', 0)
        self.PLOT_FOCUS_VOLUME = config.get('PLOT_FOCUS_VOLUME', False)
        
        os.makedirs(self.save_manual_plot_dir, exist_ok=True)
        os.makedirs(self.save_det_dir, exist_ok=True)
    
    
    def _determine_detection_directory(self):
        return f'{self.base_det_dir}/{self.cslics_uuid}/{self.model_name}'    
        
        
    def read_manual_counts(self):
        """
        Reads manual count data from an Excel file and processes it.

        This function reads data from an Excel file specified by `self.manual_counts_file` 
        and extracts manual count information from the sheet specified by `self.tank_sheet_name`. 
        It calculates the corresponding decimal days and timestamps for the counts.

        Returns:
            tuple: A tuple containing:
                - pd.Series: The manual counts from the 'Count (500L)' column.
                - pd.Series: The standard deviations from the 'Std Dev' column.
                - pd.Series: The decimal days calculated relative to the nearest day.
                - pd.Series: The datetime objects representing the counts' timestamps.
        """
        df = pd.read_excel(self.manual_counts_file, sheet_name=self.tank_sheet_name, engine='openpyxl', header=5)
        counts_time = pd.to_datetime(df['Date'].astype(str) + " " + df['Time'].astype(str), dayfirst=True)
        nearest_day = counts_time[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        decimal_days = self.convert_to_decimal_days(counts_time, nearest_day)
        return df['Count (500L)'], df['Std Dev'], decimal_days, counts_time
    
    
    def convert_to_decimal_days(self, dates_list, time_zero=None):
        if time_zero is None:
            time_zero = dates_list[0]
        return [(date - time_zero).total_seconds() / (60 * 60 * 24) for date in dates_list]


    def read_cslics_uuid_tank_association(self):
        """
        Read the manual counts from the Excel file and return the camera UUID and species.

        Uses the class attributes for the associations file, spawning sheet name, and tank sheet name.
        
        Returns:
            tuple: A tuple containing:
                - str: The camera UUID.
                - str: The coral species.
        """
        try:
            # Read the Excel file using the class attributes
            df = pd.read_excel(self.cslics_associations_file, sheet_name=self.spawning_sheet_name, engine='openpyxl', header=2)
            
            # Find the index of the tank sheet name
            idx = df['manual count sheet name'].tolist().index(self.tank_sheet_name)
            print(f'Index of {self.tank_sheet_name}: {idx}')
            
            # Return the camera UUID and species
            return df.at[idx, 'camera uuid'], df.at[idx, 'species']
        except ValueError:
            print(f'Error: {self.tank_sheet_name} not found in the manual count sheet.')
        except Exception as e:
            print(f'Error reading {self.cslics_associations_file}: {e}')
        
        # Return None values if an error occurs
        return None, None
    
    def plot_manual_counts(self, counts, std, days):
        scaled_counts = counts/1000 # for readability
        scaled_std = std/1000
        n = 1.0

        __, ax = plt.subplots()
        ax.plot(days, scaled_counts, marker='o', color='blue')
        ax.errorbar(days, scaled_counts, yerr= n*scaled_std, fmt='o', color='blue', alpha=0.5)
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel('Tank count (in thousands for 500L)')
        plt.title(f'CSLICS Manual Count: {self.tank_sheet_name}')
        plt.savefig(os.path.join(self.save_manual_plot_dir, f'Manual_counts_{self.tank_sheet_name}.png'))
        plt.show()
        
        
    def read_detections(self, nearest_day):
        # read in all the json files
        # it is assumed that because of the file naming structure, sorting the files by their filename sorts them chronologically
        print(f'Gathering detection files from: {self.model_det_dir}')
        sample_list = sorted(Path(self.model_det_dir).rglob('*_det.json'))
        print(f'Found {len(sample_list)} detection files.')

        if len(sample_list) < self.skipping_frequency:
            raise ValueError(f"Not enough detection files ({len(sample_list)}) for skipping frequency ({self.skipping_frequency}).")
        if len(sample_list) < self.aggregate_size:
            raise ValueError(f"Not enough detection files ({len(sample_list)}) for aggregate size ({self.aggregate_size}).")

        # skip every X images
        downsampled_list = sample_list[::self.skipping_frequency]
        batched_samples = [downsampled_list[i:i+self.aggregate_size] for i in range(0, len(downsampled_list), self.aggregate_size)]
        
        batched_image_count, batched_std, batched_time = [], [], []

        # iterate over all the batched samples
        for i, sample in enumerate(batched_samples[:self.MAX_SAMPLE]):
            print(f'Processing batch {i+1}/{len(batched_samples)}')
            sample_count = []
            for detection_file in sample:
                try:
                    with open(detection_file, 'r') as f:
                        data = json.load(f)
                    detections = data['detections [xn1, yn1, xn2, yn2, conf, cls]']
                    sample_count.append(sum(1 for d in detections if d[4] >= self.confidence_threshold))
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f'Error reading {detection_file}: {e}')
            
            # average stats over the batch
            batched_image_count.append(np.mean(sample_count))
            batched_std.append(np.std(sample_count))
            capture_time_str = Path(sample[len(sample)//2]).stem[:-10]
            batched_time.append(datetime.strptime(capture_time_str, "%Y-%m-%d_%H-%M-%S"))
        
        # convert batched_time to decimal days and 0 the time since spawning
        decimal_capture_times = self.convert_to_decimal_days(batched_time, nearest_day)
        return np.array(batched_image_count), np.array(batched_std), np.array(decimal_capture_times)


    def plot_image_detections(self, counts, std, times):
        """
        Plot image-based detections with error bands.

        Args:
            counts (array-like): Array of detection counts.
            std (array-like): Array of standard deviations for the counts.
            times (array-like): Array of times (in decimal days) since spawning.
        """
        n = 1  # Multiplier for the error band
        __, ax = plt.subplots()
        ax.plot(times, counts, label='Detections')
        ax.fill_between(times, counts - n * std, counts + n * std, alpha=0.2, label='Error Band')
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel(f'Image count (batched {self.aggregate_size} images)')
        plt.title(f'CSLICS AI Count: {self.tank_sheet_name}')
        plt.legend()
        output_path = os.path.join(self.save_det_dir, f'Image_counts_{self.tank_sheet_name}.png')
        plt.savefig(output_path)
        print(f"Image detections plot saved to {output_path}")
        plt.show()
        

    def run(self):
        """
        Run the full processing and plotting pipeline.

        Args:
            file_path (str): Path to the input data file.
            output_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        
        # read manual counts
        manual_counts, manual_std, manual_times, manual_times_dt = self.read_manual_counts()
        # plot manual counts
        self.plot_manual_counts(manual_counts, manual_std, manual_times)
        
        # read model detections
        nearest_day = manual_times_dt[0].replace(hour=0, minute=0, second=0, microsecond=0)  + timedelta(days=1) # NOTE replicated code
        image_counts, image_std, image_times = self.read_detections(nearest_day)
        
        # plot model detections
        self.plot_image_detections(image_counts, image_std, image_times)
        
        pass


# Example usage:
if __name__ == "__main__":
    
    # spawning_sheet_name = '2024 oct'
    # tank_sheet_name = 'OCT24 T1 Amag'
    # tank_sheet_name = 'OCT24 T2 Amag'
    # tank_sheet_name = 'OCT24 T3 Amag'
    # tank_sheet_name = 'OCT24 T4 Maeq'
    # tank_sheet_name = 'OCT24 T5 Maeq'
    # tank_sheet_name = 'OCT24 T6 Aant'

    # spawning_sheet_name = '2024 nov'
    # tank_sheet_name = 'NOV24 T1 Amil'
    # tank_sheet_name = 'NOV24 T2 Amil'
    # tank_sheet_name = 'NOV24 T3 Amil'
    # tank_sheet_name = 'NOV24 T4 Pdae'
    # tank_sheet_name = 'NOV24 T5 Pdae'
    # tank_sheet_name = 'NOV24 T6 Lcor'
    config = {
        'manual_counts_file': '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_manual_counts.xlsx',
        'spawning_sheet_name': '2024 oct',
        'tank_sheet_name': 'OCT24 T6 Aant',
        'cslics_associations_file': '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_spawning_setup.xlsx',
        'model_name': 'cslics_subsurface_20250205_640p_yolov8n',
        'base_detection_dir': '/media/dtsai/CSLICSOct24/cslics_october_2024/detections',
        'save_manual_plot_dir': '/home/dtsai/Data/cslics_datasets/manual_counts/plots',
        'skipping_frequency': 1,
        'aggregate_size': 100,
        'confidence_threshold': 0.5,
        'MAX_SAMPLE': 1000,
        'calibration_window_size': 1,
        'calibration_idx': 1,
        'calibration_window_shift': 0,
        'PLOT_FOCUS_VOLUME': False
    }
    
    processor = CSLICSDataProcessor(config)
    processor.run()
    
    # manual_counts, manual_std, manual_times, manual_times_dt = processor.read_manual_counts()
    # processor.plot_manual_counts(manual_counts, manual_std, manual_times)

    # processor = CSLICSDataProcessor()
    # processor.run("path/to/data.csv", "path/to/output_plot.png")