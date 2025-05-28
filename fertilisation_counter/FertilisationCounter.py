#!/usr/bin/env python3

"""
count fertilisation for cslics runs, first 2ish hours
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2 as cv
import re
import torch
from ultralytics.engine.results import Results, Boxes
import torchvision
from tqdm import tqdm 
from datetime import datetime, timedelta
import json
from scipy.interpolate import interp1d
import pandas as pd

class FertilisationCounter:
    def __init__(self, 
                 root_dir, 
                 model_path, 
                 output_dir=None, 
                 use_cached_detections=False,
                 manual_count_path=None,
                 sheet_name=None,
                 include_manual=True):
        """Initialize the FertilisationCounter class with paths and model"""
        self.root_dir = root_dir
        self.model_path = model_path
        self.output_dir = output_dir if output_dir else os.path.join(root_dir, '../predictions')
        self.json_dir = os.path.join(self.output_dir, 'predictions_json')
        self.use_cached_detections = use_cached_detections
        self.manual_count_path = manual_count_path
        self.sheet_name = sheet_name
        self.include_manual = include_manual
        
        # Create JSON directory if it doesn't exist
        if self.use_cached_detections or not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir, exist_ok=True)
        
        # Rest of initialization...
        # Define classes and colors
        self.classes = {
            0: 'four-eight cell stage',
            1: 'first cleavage',
            2: 'two-cell stage',
            3: 'advanced',
            4: 'damaged',
            5: 'egg'
        }
        self.class_colours = self.set_class_colours()
        
        # Detection properties
        self.img_size = 640
        self.conf = 0.3
        self.iou = 0.5
        self.max_det = 2000
        
        # Drawing properties
        self.line_width = 4
        self.font_size = 2
        
        # Check for GPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda:0":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Load model only if not using cached detections
        self.model = None
        if not self.use_cached_detections:
            self.load_model()
    
    def set_class_colours(self):
        """Set classes to specific colours using a dictionary"""
        orange = [255, 128, 0]    # four-eight cell stage
        blue = [0, 212, 255]      # first cleavage
        purple = [170, 0, 255]    # two-cell stage
        yellow = [255, 255, 0]    # advanced
        brown = [144, 65, 2]      # damaged
        green = [0, 255, 0]       # egg
        colors = [orange, blue, purple, yellow, brown, green]
        
        class_colours = {}
        for i, c in enumerate(self.classes):
            class_colours.update({c: colors[i]})
        
        return class_colours
    
    def load_model(self):
        """Load the YOLO model"""
        self.model = YOLO(self.model_path)
        # Ensure model is using GPU if available
        if self.device != "cpu":
            self.model.to(self.device)
        
    def get_date_str(self, relative_filename, pattern=None):
        """Get date string from filename following cslics image naming convention"""
        if pattern is None:
            pattern = r'cslics\d+_(\d{8})'
            # regular expression to extract the date from the filename of the format:
            # 'cslics01_20231205_234344_838548_img.jpg'
        match = re.search(pattern, relative_filename)
        if match:
            date_str = match.group(1)
        else:
            date_str = None
            print('get_date_str: no date found in filename')
        return date_str
    
    def nms(self, pred, conf_thresh, iou_thresh):
        """Perform class-agnostic non-maxima suppression on predictions"""
        # Checks
        assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thresh}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thresh}, valid values are between 0.0 and 1.0'
        
        # Filter by confidence
        pred = pred[pred[:, 4] > conf_thresh]
        if len(pred) == 0:
            return pred
            
        boxes = pred[:, :4]
        scores = pred[:, 4]
        
        # Sort scores into descending order
        _, indices = torch.sort(scores, descending=True)

        # Class-agnostic NMS
        keep = []
        while indices.numel() > 0:
            # Get the highest scoring detection
            i = indices[0]
            
            # Add the detection to the list of final detections
            keep.append(i.item())

            # Calculate the IoU highest scoring detection within all other detections
            if indices.numel() == 1:
                break
            else:
                overlaps = torchvision.ops.box_iou(boxes[indices[1:]], boxes[i].unsqueeze(0)).squeeze()

            # Keep only detections with IOU below certain threshold
            indices = indices[1:]
            indices = indices[overlaps <= iou_thresh]

        return pred[keep, :]

    def detect(self, image):
        """Return detections from a single RGB image"""
        
        # Ensure we're using the device we specified
        pred = self.model.predict(source=image,
                                save=False,
                                save_txt=False,
                                save_conf=True,
                                verbose=False,
                                imgsz=self.img_size,
                                conf=self.conf,
                                iou=self.iou,
                                max_det=self.max_det,
                                device=self.device)
                                  
        boxes: Boxes = pred[0].boxes 
        pred = []
        for b in boxes:
            # Keep all variables on cuda/GPU for speed if available
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
            else:
                cls = int(b.cls)
                conf = float(b.conf)
                xyxyn = b.xyxyn.cpu().numpy()[0]
                x1n = xyxyn[0]
                y1n = xyxyn[1]
                x2n = xyxyn[2]
                y2n = xyxyn[3]  
                pred.append([x1n, y1n, x2n, y2n, conf, cls])
        
        # After iterating over boxes, make sure pred is on GPU if available (and a single tensor)
        if self.device != "cpu" and pred:
            pred = torch.tensor(pred, device=self.device)
        elif pred:
            pred = torch.tensor(pred)
        else:
            return []  # Empty case
        
            
        # Apply NMS if we have predictions
        if len(pred) > 0:
            predictions = self.nms(pred, self.conf, self.iou)
        else:
            predictions = []  # Empty case
            
        return predictions
    
    
    def compute_fertilisation(self, predictions):
        """Compute fertilisation rate from predictions per image"""
        # NOTE: previously, this was a mean across 20+ images
        
        # Count the number of detections for each class
        counts = {cls_name: 0 for cls_name in self.classes.values()}
        for p in predictions:
            cls = int(p[5]) # the 5th index is the class    
            # Check if the class is in the classes dictionary
            if cls in self.classes.keys():
                # Increment the count for the class
                counts[self.classes[cls]] += 1
            else:
                print(f"Warning: Class {cls} not in classes dictionary.")
                
        # self.classes = {
        #     0: 'four-eight cell stage',
        #     1: 'first cleavage',
        #     2: 'two-cell stage',
        #     3: 'advanced',
        #     4: 'damaged',
        #     5: 'egg'
        # }
        
        # Calculate fertilisation rate
        #fertilisation_rate = (counts[0] + counts[1] + counts[2]) / (counts[0] + counts[1] + counts[2] + counts[3] + counts[4] + counts[5])
        fertilised_counts = counts['first cleavage'] + counts['two-cell stage'] + counts['four-eight cell stage'] + counts['advanced'] + counts['damaged']
        total_counts = counts['egg'] + fertilised_counts # opt: included counts['damaged'] as well?
        if total_counts > 0:
            fertilisation_rate = fertilised_counts / total_counts
        else:
            fertilisation_rate = 0
        return counts, fertilisation_rate
    
    
    def save_image_predictions(self, predictions, img, imgname, BGR=False, quality=50):
        """Save predictions/detections on image"""
        imgw, imgh = img.shape[1], img.shape[0]
        
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            #extract back into cv lengths
            x1 = x1*imgw
            x2 = x2*imgw
            y1 = y1*imgh
            y2 = y2*imgh        
            cv.rectangle(img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         self.class_colours[cls], 
                         self.line_width)
            cv.putText(img, f"{self.classes[cls]}: {conf:.2f}", (int(x1), int(y1 - 5)), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       self.font_size, 
                       self.class_colours[cls], 
                       self.line_width)

        imgsavename = os.path.basename(imgname)
        # Add day into save directory to prevent an untenable number of images in a single folder
        date_str = self.get_date_str(imgsavename)
        os.makedirs(os.path.join(self.output_dir, date_str), exist_ok=True)
        
        imgsave_path = os.path.join(self.output_dir, date_str, imgsavename.rsplit('.',1)[0] + '.jpg')
        
        if BGR:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)        
        # To save on memory, reduce quality of saved image
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)
        return True
    
    # NOTE: duplicated code from select_manual_images.py, consider creating a commons/utilities.py class if this gets more
    def extract_datetime_from_filename(self, filename):
        """Extract the datetime object from the filename."""
        try:
            # Extract the timestamp from the filename (e.g., "cslics08_20231103_205449_514797_img.jpg")
            # Extract date (20231103) and time (205449) parts
            filename = os.path.basename(filename)
            parts = filename.split('_')
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2][:6]  # Take first 6 chars for HHMMSS
                timestamp_str = date_str + '_' + time_str
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return None
        except (IndexError, ValueError) as e:
            print(f"Error extracting datetime from {filename}: {e}")
            return None
        
    def plot_fertilisation(self, results, manual_times=None, manual_ratios=None, time_limit_minutes=120):
        """
        Plot fertilisation rate vs time since first capture
        
        Args:
            results: List of dictionaries containing capture_time and fertilisation_rate
            manual_times: List of datetime objects for manual counts
            manual_ratios: List of manual fertilisation ratios
            time_limit_minutes: Maximum time in minutes to display on the plot
        """
        # Filter out results with None capture time
        valid_results = [r for r in results if r['capture_time'] is not None]
        
        if not valid_results:
            print("No valid timestamps found for plotting")
            return
        
        # Get the earliest capture time as reference point (time zero)
        first_capture = min(result['capture_time'] for result in valid_results)
        
        # Extract times (in minutes since first capture) and fertilisation rates
        times_minutes = [(result['capture_time'] - first_capture).total_seconds() / 60.0 
                        for result in valid_results]
        fert_rates = [result['fertilisation_rate'] for result in valid_results]
        
        # Apply time limit: filter data to only include times <= time_limit_minutes
        time_limited_indices = [i for i, t in enumerate(times_minutes) if t <= time_limit_minutes]
        times_minutes = [times_minutes[i] for i in time_limited_indices]
        fert_rates = [fert_rates[i] for i in time_limited_indices]
        
        # Check if we have any data within the time limit
        if not times_minutes:
            print(f"No data points within the {time_limit_minutes} minute time limit")
            return
        
        # Compare with manual counts if available
        pearson_corr = None
        rmse = None
        if manual_times and manual_ratios and len(manual_times) > 0:
            pearson_corr, rmse, matched_pairs = self.compare_with_manual_counts(
                results, manual_times, manual_ratios
            )
            if matched_pairs:
                print(f"Pearson correlation: {pearson_corr:.3f}")
                print(f"RMSE: {rmse:.3f}")
                for pair in matched_pairs:
                    time_diff, manual_time, manual_ratio, auto_ratio = pair
                    diff = manual_ratio - auto_ratio
                    print(f"Manual time: {manual_time}, Time diff: {time_diff:.1f} min, "
                        f"Manual ratio: {manual_ratio:.3f}, Auto ratio: {auto_ratio:.3f}, "
                        f"Difference: {diff:.3f}")
        
        # Create the plot
        label_font_size = 10
        title_font_size = 12
        
        plt.figure(figsize=(8, 6))
        
        # Plot the per-image data with transparency to give a light background
        plt.plot(times_minutes, fert_rates, '.', alpha=0.3, color='blue', markersize=3, label='CSLICS Count')
        
        # create running average and standard deviation of fertilisation rates
        window_size = min(20, len(times_minutes))  # Ensure window isn't larger than our data
        if window_size > 1:  # Only compute average if we have enough data points
            # Calculate running average
            fert_rate_avg = np.convolve(fert_rates, np.ones(window_size)/window_size, mode='valid')
            
            # Calculate running standard deviation
            # First, calculate squared differences for each window
            squared_diffs = []
            fert_rates_np = np.array(fert_rates)
            for i in range(len(fert_rates) - window_size + 1):
                window = fert_rates_np[i:i+window_size]
                window_mean = np.mean(window)
                squared_diff = np.sum((window - window_mean)**2) / window_size
                squared_diffs.append(squared_diff)
            
            # Then take the square root to get standard deviation
            fert_rate_std = np.sqrt(squared_diffs)
            
            # Adjust times for running average and std
            time_avg = times_minutes[window_size-1:]
        else:
            time_avg = []
            fert_rate_avg = []
            fert_rate_std = []
        
        # Plot a semi-transparent area showing variance from running average
        if len(time_avg) > 1:  # Need at least 2 points for interpolation
            f_interp = interp1d(time_avg, fert_rate_avg, bounds_error=False, fill_value=(fert_rate_avg[0], fert_rate_avg[-1]))
            interp_avg = f_interp(times_minutes)
            
            # Plot the variance as a semi-transparent fill
            # plt.fill_between(times_minutes, fert_rates, interp_avg, alpha=0.15, color='blue', label='Variance')
        
            # Plot running average as a solid line on top
            plt.plot(time_avg, fert_rate_avg, '-', linewidth=2, color='red', label=f'CSLICS Mean ({window_size} images)')
            
            # Plot standard deviation around the running average
            upper_bound = fert_rate_avg + fert_rate_std
            lower_bound = fert_rate_avg - fert_rate_std
            
            # Clip bounds to ensure they stay within 0-1 range for fertilization rates
            upper_bound = np.clip(upper_bound, 0, 1)
            lower_bound = np.clip(lower_bound, 0, 1)
            
            # Plot the standard deviation band
            plt.fill_between(time_avg, lower_bound, upper_bound, 
                             color='orange', alpha=0.3, 
                             label=f'±1 Std Dev ({window_size} images)')
        
        # Add manual counts if available
        if manual_times and manual_ratios and len(manual_times) > 0:
            # Convert manual times to minutes since first capture (same zero point)
            manual_minutes = [(t - first_capture).total_seconds() / 60.0 for t in manual_times]
            
            # Filter manual times by time limit
            limited_manual_data = [(t, r) for t, r in zip(manual_minutes, manual_ratios) if t <= time_limit_minutes]
            if limited_manual_data:
                limited_manual_minutes, limited_manual_ratios = zip(*limited_manual_data)
                
                # Add manual counts as green squares with black edges
                plt.scatter(limited_manual_minutes, limited_manual_ratios, marker='s', color='green', s=40, 
                        edgecolor='black', linewidth=1.5, alpha=0.8, label='Manual Count')
                
                # Add annotations for the manual count values
                for i, (x, y) in enumerate(zip(limited_manual_minutes, limited_manual_ratios)):
                    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                                xytext=(0, 10), ha='center', fontsize=9)
        
        plt.legend(loc='lower right', fontsize=label_font_size)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time since stocking [minutes]', fontsize=label_font_size)
        plt.ylabel('Fertilisation success', fontsize=label_font_size)
        
        # Add correlation and RMSE to title if available
        title_text = f'Fertilisation Rate Over Time\n{Path(self.root_dir).parts[-2]}'
        if pearson_corr is not None and rmse is not None:
            title_text += f'\nPearson r: {pearson_corr:.3f}, RMSE: {rmse:.3f}'
        plt.title(title_text, fontsize=title_font_size)
        
        # Add 0-1 range for y-axis and set reasonable limits
        plt.ylim(-0.05, 1.05)
        
        # Set x-axis limits to show from -5 minutes to time_limit_minutes
        plt.xlim(-5, time_limit_minutes)
        
        # Save the plot
        plot_filename = f"fertilisation_plot_{Path(self.root_dir).parts[-2]}.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"Fertilisation plot saved to {plot_path}")
        
        return plot_path
    
    def compare_with_manual_counts(self, results, manual_times, manual_ratios):
        """
        Compare automated detection results with manual counts using Pearson correlation and RMSE
        
        Args:
            results: List of dictionaries containing automated detection results
            manual_times: List of datetime objects for manual counts
            manual_ratios: List of manual fertilisation ratios
            
        Returns:
            Tuple of (pearson_corr, rmse, matched_pairs)
        """
        if not manual_times or not manual_ratios:
            return None, None, []
        
        # Extract valid results with timestamps
        valid_results = [r for r in results if r['capture_time'] is not None]
        if not valid_results:
            return None, None, []
        
        # Find closest auto detection time point for each manual count
        matched_pairs = []
        for manual_time, manual_ratio in zip(manual_times, manual_ratios):
            if manual_time is None or np.isnan(manual_ratio):
                continue
                
            # Calculate time difference in seconds for each result
            time_diffs = [(abs((r['capture_time'] - manual_time).total_seconds()), r) 
                        for r in valid_results]
            # Find the closest result
            closest = min(time_diffs, key=lambda x: x[0])
            time_diff_minutes = closest[0] / 60.0
            closest_result = closest[1]
            
            # Store the pair for comparison (time diff in minutes, manual time, manual ratio, auto ratio)
            matched_pairs.append((
                time_diff_minutes,
                manual_time,
                manual_ratio,
                closest_result['fertilisation_rate']
            ))
        
        # Skip if no valid pairs were found
        if not matched_pairs:
            return None, None, []
        
        # Extract manual and auto ratios from matched pairs
        manual_ratios_matched = [p[2] for p in matched_pairs]
        auto_ratios_matched = [p[3] for p in matched_pairs]
        
        # Calculate Pearson correlation coefficient
        pearson_corr = np.corrcoef(manual_ratios_matched, auto_ratios_matched)[0, 1]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(manual_ratios_matched) - np.array(auto_ratios_matched))**2))
        
        return pearson_corr, rmse, matched_pairs
    
    def get_json_path_for_image(self, img_path):
        """Get the JSON file path for an image"""
        img_name = os.path.basename(img_path)
        json_name = f"{os.path.splitext(img_name)[0]}.json"
        # Store JSONs in subdirectories matching the date to avoid too many files in one dir
        date_str = self.get_date_str(img_name)
        json_subdir = os.path.join(self.json_dir, date_str if date_str else "unknown")
        os.makedirs(json_subdir, exist_ok=True)
        return os.path.join(json_subdir, json_name)
    
    def save_predictions_to_json(self, img_path, predictions, counts, fertilisation_rate):
        """Save predictions to a JSON file"""
        json_path = self.get_json_path_for_image(img_path)
        
        # Convert tensor predictions to list for JSON serialization
        if isinstance(predictions, torch.Tensor):
            predictions_list = predictions.cpu().numpy().tolist()
        else:
            predictions_list = [p.tolist() if isinstance(p, torch.Tensor) else p for p in predictions]
        
        # Create the data to save
        data = {
            'image_path': str(img_path),
            'predictions': predictions_list,
            'counts': counts,
            'fertilisation_rate': fertilisation_rate
        }
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(data, f)
    
    def load_predictions_from_json(self, img_path):
        """Load predictions from a JSON file"""
        json_path = self.get_json_path_for_image(img_path)
        
        if not os.path.exists(json_path):
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert predictions back to tensor if needed
        if self.device != "cpu" and data['predictions']:
            predictions = torch.tensor(data['predictions'], device=self.device)
        else:
            predictions = torch.tensor(data['predictions']) if data['predictions'] else []
        
        return {
            'predictions': predictions,
            'counts': data['counts'],
            'fertilisation_rate': data['fertilisation_rate']
        }
    
    def process_image(self, img_path):
        """Process a single image, either using detection or cached results"""
        img_path_str = str(img_path)
        capture_time = self.extract_datetime_from_filename(img_path_str)
        
        # Try to load from cache if using cached detections
        if self.use_cached_detections:
            cached_data = self.load_predictions_from_json(img_path_str)
            if cached_data:
                return {
                    'image_path': img_path_str,
                    'capture_time': capture_time,
                    'counts': cached_data['counts'],
                    'fertilisation_rate': cached_data['fertilisation_rate'],
                    'predictions': cached_data['predictions']
                }
        
        # If we get here, we need to run detection
        # Read image
        img_bgr = cv.imread(img_path_str)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)  # RGB
        
        # Run model
        predictions = self.detect(img_rgb)
        
        # Compute fertilisation
        counts, fertilisation_rate = self.compute_fertilisation(predictions)
        
        # Save predictions to JSON for future use
        self.save_predictions_to_json(img_path_str, predictions, counts, fertilisation_rate)
        
        # Save predictions on the image
        self.save_image_predictions(predictions, img_rgb, img_path, BGR=True)
        
        return {
            'image_path': img_path_str,
            'capture_time': capture_time,
            'counts': counts,
            'fertilisation_rate': fertilisation_rate,
            'predictions': predictions
        }
    
    def read_manual_counts(self):
        """
        Read manual counts from an Excel file
        
        Args:
            manual_count_path: Path to the Excel file
            sheet_name: Name of the sheet to read
            
        Returns:
            Tuple of manual count times and fertilisation ratios
        """
        
        
        try:
            # Read the Excel file, skipping the first 7 rows, using row 8 as header
            print(f"Reading manual counts from {self.manual_count_path}, sheet '{self.sheet_name}'")
            df = pd.read_excel(
                self.manual_count_path,
                sheet_name=self.sheet_name,
                header=7,  # 0-indexed, so row 8 becomes header
                nrows=10   # Read only 10 rows of data
            )
            
            # Check if data was found
            if df.empty:
                print(f"Warning: No data found in sheet '{self.sheet_name}'")
                return None, None
                    
            # Basic validation and cleaning
            print(f"Found {len(df)} manual count records")
            
            # Extract datetime objects from filenames
            manual_times = []
            image_names = df['Image Name'].tolist()
            for filename in image_names:
                manual_times.append(self.extract_datetime_from_filename(filename))
            
            # Get fertilization ratios
            fert_ratio_manual = df['fert ratio'].tolist()
            
            # Filter out None values
            valid_data = [(t, r) for t, r in zip(manual_times, fert_ratio_manual) if t is not None]
            if valid_data:
                valid_times, valid_ratios = zip(*valid_data)
                return valid_times, valid_ratios
            return None, None
                
        except Exception as e:
            print(f"Error reading manual counts: {e}")
            # Check if sheet exists in the Excel file
            try:
                all_sheets = pd.ExcelFile(self.manual_count_path).sheet_names
                print(f"Available sheets: {all_sheets}")
                if self.sheet_name not in all_sheets:
                    print(f"Sheet '{self.sheet_name}' not found. Please check the sheet name.")
            except:
                pass
            return None, None
        
    def run(self):
        """Process all images in the root directory"""
        # Get all images
        img_list = sorted(Path(self.root_dir).rglob('*.jpg'))
        print(f"Processing {len(img_list)} images...")
        
        # Determine mode message
        mode = "Reading from cached detections" if self.use_cached_detections else "Running detections"
        print(f"{mode} for {len(img_list)} images...")
        
        # Run model on images with progress bar
        results = []
        for img_path in tqdm(img_list, desc="Processing images", total=len(img_list), unit="img"):
            result = self.process_image(img_path)
            results.append(result)
            
        print('Processing complete')
        
        # read manual counts only if include_manual is True
        manual_times, manual_ratios = None, None
        if self.include_manual and self.manual_count_path and self.sheet_name:
            print("Including manual counts in analysis")
            manual_times, manual_ratios = self.read_manual_counts()
        elif self.include_manual and (not self.manual_count_path or not self.sheet_name):
            print("Manual counts requested but manual_count_path or sheet_name is missing")
        elif not self.include_manual:
            print("Manual counts excluded from analysis")

        # Plot fertilisation results
        self.plot_fertilisation(results, manual_times, manual_ratios)
        
        return results

def main():
    # Configuration
    root_dir = '/home/dtsai/Data/cslics_datasets/cslics_2022_fert_dataset/20221214_alor_tank4_cslics02/images'
    model_path = '/home/dtsai/Data/cslics_datasets/models/fertilisation/cslics_20230905_yolov8m_640p_amtenuis1000.pt'
    
    
    sheet_name = Path(root_dir).parts[-2]
    print(f'sheet_name: {sheet_name}')
    manual_count_path = '/home/dtsai/Data/cslics_datasets/cslics_2022_fert_dataset_manual_counts/cslics_2022_fert_dataset_manual_counts.xlsx'
    output_dir = os.path.join(os.path.dirname(root_dir), 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create counter and run
    counter = FertilisationCounter(
        root_dir=root_dir, 
        model_path=model_path, 
        output_dir=output_dir,
        manual_count_path=manual_count_path,
        sheet_name=sheet_name,
        use_cached_detections=True,
        include_manual=True)
    
    
    
    counter.run()
    
if __name__ == "__main__":
    main()

