#!/usr/bin/env python3

# Annotation pipeline - Class-based implementation

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import yaml
import time

# Import filters
from FilterSift import FilterSift   
from FilterHue import FilterHue
from FilterSaturation import FilterSaturation
from FilterLaplacian import FilterLaplacian
from FilterValue import FilterValue


class AnnotationPipeline:
    def __init__(self, config_path):
        """
        Initialize the annotation pipeline with configuration from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Set class information
        self.class_name = self.config['class']['name']
        self.class_label = self.config['class']['label']
        self.class_color = tuple(self.config['class']['color_bgr'])
        
        # Initialize filters
        self.initialize_filters()
        
        # Processing parameters
        self.max_img = 1000  # Maximum number of images to process
        
    def initialize_filters(self):
        """Initialize all filter objects based on configuration."""
        self.filters = {
            'sift': FilterSift(config=self.config['sift']) if self.config['sift']['do'] else None,
            'hue': FilterHue(config=self.config['hue']) if self.config['hue']['do'] else None,
            'saturation': FilterSaturation(config=self.config['saturation']) if self.config['saturation']['do'] else None,
            'value': FilterValue(config=self.config['value']) if self.config['value']['do'] else None,
            'laplacian': FilterLaplacian(config=self.config['laplacian']) if self.config['laplacian']['do'] else None
        }
        
    def setup_directories(self, img_dir, output_base_dir):
        """
        Set up all necessary directories for processing and output.
        
        Args:
            img_dir (str): Directory containing input images.
            output_base_dir (str): Base directory for all outputs.
            
        Returns:
            tuple: Paths to various output directories.
        """
        # Input directory
        self.img_dir = img_dir
        
        # Output directories
        self.save_dir = os.path.join(output_base_dir, 'output')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Export directories
        self.save_export_dir = os.path.join(output_base_dir, 'export')
        os.makedirs(self.save_export_dir, exist_ok=True)
        
        self.txt_save_dir = os.path.join(self.save_export_dir, 'labels/train')
        os.makedirs(self.txt_save_dir, exist_ok=True)
        
        # Files
        self.train_txt_path = os.path.join(self.save_export_dir, 'train.txt')
        # self.obj_names_path = os.path.join(self.save_export_dir, 'obj.names')
        # self.obj_data_path = os.path.join(self.save_export_dir, 'obj.data')
        self.data_yaml = os.path.join(self.save_export_dir, 'data.yaml')
        
        # Annotated images directory
        self.save_annotated_dir = os.path.join(self.save_dir, 'annotated')
        os.makedirs(self.save_annotated_dir, exist_ok=True)
        
        # Initialize train.txt file
        with open(self.train_txt_path, 'w') as train_file:
            pass
        
        # Write data.yaml file
        with open(self.data_yaml, 'w') as file:
            file.write('path: ./ # dataset root dir\n')
            file.write('train: train.txt # train images (relative to path)\n')
            file.write('# Classes\n')
            file.write('names:\n')
            file.write(f'  0: {self.class_name}\n')  # Single class name
            
        # # Write obj.names file
        # with open(self.obj_names_path, 'w') as file:
        #     file.write(self.class_name)
            
        # # Write obj.data file
        # with open(self.obj_data_path, 'w') as file:
        #     file.write('classes = 1\n')
        #     file.write('train = labels/train.txt\n')
        #     file.write('names = data/obj.names\n')
        #     file.write('backup = backup/')
    
    def get_image_list(self, pattern='*.jpg'):
        """
        Get a list of image paths matching the pattern.
        
        Args:
            pattern (str): Glob pattern for image files.
            
        Returns:
            list: Sorted list of image paths.
        """
        return sorted(glob.glob(os.path.join(self.img_dir, pattern)))
    
    def create_masks(self, img_bgr):
        """
        Create masks using all enabled filters.
        
        Args:
            img_bgr (ndarray): Input image in BGR format.
            
        Returns:
            list: List of binary masks from all enabled filters.
        """
        mask_list = []
        
        # Process with SIFT filter if enabled
        if self.filters['sift'] is not None:
            kp = self.filters['sift'].get_best_sift_features(img_bgr)
            img_ftr = self.filters['sift'].draw_keypoints(img_bgr, kp)
            self.filters['sift'].save_image(img_ftr, self.current_img_name, self.save_dir, '_sift.jpg')
            
            mask_sift = self.filters['sift'].create_sift_mask(img_bgr, kp)
            mask_sift_overlay = self.filters['sift'].display_mask_overlay(img_bgr, mask_sift)
            self.filters['sift'].save_image(mask_sift_overlay, self.current_img_name, self.save_dir, '_siftoverlay.jpg')
            mask_list.append(mask_sift)
        
        # Process with Hue filter if enabled
        if self.filters['hue'] is not None:
            mask_hue = self.filters['hue'].create_hue_mask(img_bgr)
            mask_hue_overlay = self.filters['hue'].display_mask_overlay(img_bgr, mask_hue)
            self.filters['hue'].save_image(mask_hue, self.current_img_name, self.save_dir, '_hue.jpg')
            self.filters['hue'].save_image(mask_hue_overlay, self.current_img_name, self.save_dir, '_hueoverlay.jpg')
            mask_list.append(mask_hue)
        
        # Process with Saturation filter if enabled
        if self.filters['saturation'] is not None:
            mask_sat = self.filters['saturation'].create_saturation_mask(img_bgr)
            mask_sat_overlay = self.filters['saturation'].display_mask_overlay(img_bgr, mask_sat)
            self.filters['saturation'].save_image(mask_sat, self.current_img_name, self.save_dir, '_sat.jpg')
            self.filters['saturation'].save_image(mask_sat_overlay, self.current_img_name, self.save_dir, '_satoverlay.jpg')
            mask_list.append(mask_sat)
        
        # Process with Value filter if enabled
        if self.filters['value'] is not None:
            mask_val = self.filters['value'].create_value_mask(img_bgr)
            mask_val_overlay = self.filters['value'].display_mask_overlay(img_bgr, mask_val)
            self.filters['value'].save_image(mask_val, self.current_img_name, self.save_dir, '_val.jpg')
            self.filters['value'].save_image(mask_val_overlay, self.current_img_name, self.save_dir, '_valoverlay.jpg')
            mask_list.append(mask_val)
        
        # Process with Laplacian filter if enabled
        if self.filters['laplacian'] is not None:
            mask_lapl = self.filters['laplacian'].create_laplacian_mask(img_bgr)
            mask_lapl_overlay = self.filters['laplacian'].display_mask_overlay(img_bgr, mask_lapl)
            self.filters['laplacian'].save_image(mask_lapl, self.current_img_name, self.save_dir, '_lapl.jpg')
            self.filters['laplacian'].save_image(mask_lapl_overlay, self.current_img_name, self.save_dir, '_laploverlay.jpg')
            mask_list.append(mask_lapl)
            
        return mask_list
    
    def combine_masks(self, mask_list):
        """
        Combine all masks using bitwise AND operation.
        
        Args:
            mask_list (list): List of binary masks.
            
        Returns:
            ndarray: Combined binary mask.
        """
        # Combine masks with AND operation
        mask_combined = mask_list[0]
        for m in mask_list:
            mask_combined = mask_combined & m
            
        # Post-process the combined mask
        value_filter = self.filters['value'] or next((f for f in self.filters.values() if f is not None), None)
        if value_filter:
            mask_combined = value_filter.fill_holes(mask_combined)
            mask_combined, _ = value_filter.filter_components(mask_combined)
            
        return mask_combined
    
    def extract_bounding_boxes(self, mask_combined):
        """
        Extract bounding boxes from the combined mask.
        
        Args:
            mask_combined (ndarray): Combined binary mask.
            
        Returns:
            list: List of bounding boxes in YOLO format [class, x_center, y_center, width, height].
        """
        img_width, img_height = mask_combined.shape[1], mask_combined.shape[0]
        contours, _ = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for c in contours:
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            xcen = (x + w/2.0)/img_width
            ycen = (y + h/2.0)/img_height
            
            # Append in YOLO format: class x_center y_center width height
            bounding_boxes.append([
                self.class_label, 
                xcen, 
                ycen, 
                w/img_width, 
                h/img_height
            ])
            
        return bounding_boxes
            
    def save_image_predictions(self, predictions, img, quality=50, imgformat='.jpg'):
        """
        Save predictions/detections on image.
        
        Args:
            predictions (list): List of predictions in YOLO format.
            img (ndarray): Image to draw predictions on.
            quality (int): JPEG quality (0-100).
            imgformat (str): Image format extension.
            
        Returns:
            bool: True if successful.
        """
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            cls = int(p[0])
            xcen, ycen, w, h = p[1], p[2], p[3], p[4]
            
            # Extract back into CV lengths
            x1 = (xcen - w/2) * imgw
            x2 = (xcen + w/2) * imgw
            y1 = (ycen - h/2) * imgh
            y2 = (ycen + h/2) * imgh    
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_color, 3)
        
        imgsavename = os.path.basename(self.current_img_name)
        imgsave_path = os.path.join(self.save_annotated_dir, 
                                  imgsavename.rsplit('.', 1)[0] + '_annotated' + imgformat)
        
        # To save on memory, reduce quality of saved image
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)
        return True
    
    def save_text_predictions(self, annotations, txtformat='.txt'):
        """
        Save annotations/predictions/detections into text file.
        
        Args:
            annotations (list): List of annotations in YOLO format.
            txtformat (str): Text file format extension.
            
        Returns:
            bool: True if successful.
        """
        txtsavename = os.path.basename(self.current_img_name).rsplit('.', 1)[0]
        txtsavepath = os.path.join(self.txt_save_dir, txtsavename + txtformat)
        
        with open(txtsavepath, 'w') as f:
            for a in annotations:
                class_label = int(a[0])
                x, y, w, h = a[1], a[2], a[3], a[4]
                f.write(f'{class_label:g} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
        return True
    
    def update_train_file(self):
        """
        Append current image to the train.txt file.
        
        Returns:
            bool: True if successful.
        """
        write_line = os.path.join('images/train/', 
                                 os.path.basename(self.current_img_name))
        with open(self.train_txt_path, 'a') as train_file:
            train_file.write(write_line + '\n')
        return True
    
    def process_image(self, img_path):
        """
        Process a single image.
        
        Args:
            img_path (str): Path to the input image.
            
        Returns:
            list: List of bounding boxes in YOLO format.
        """
        self.current_img_name = img_path
        img_bgr = cv.imread(img_path)
        
        # Create masks from different filters
        mask_list = self.create_masks(img_bgr)
        
        # Combine masks
        if not mask_list:
            print(f"Warning: No masks generated for {img_path}")
            return []
        
        mask_combined = self.combine_masks(mask_list)
        
        # Display and save combined mask
        display_filter = next((f for f in self.filters.values() if f is not None), None)
        if display_filter:
            mask_combined_overlay = display_filter.display_mask_overlay(img_bgr, mask_combined)
            display_filter.save_image(mask_combined, img_path, self.save_dir, '_combined.jpg')
            display_filter.save_image(mask_combined_overlay, img_path, self.save_dir, '_combinedoverlay.jpg')
        
        # Extract bounding boxes
        bounding_boxes = self.extract_bounding_boxes(mask_combined)
        
        # Save annotations and update train file
        self.save_text_predictions(bounding_boxes)
        self.save_image_predictions(bounding_boxes, img_bgr.copy())
        self.update_train_file()
        
        return bounding_boxes
    
    def run(self, img_dir, output_base_dir, img_pattern='*.jpg'):
        """
        Run the annotation pipeline on all images.
        
        Args:
            img_dir (str): Directory containing input images.
            output_base_dir (str): Base directory for all outputs.
            img_pattern (str): Glob pattern for image files.
            
        Returns:
            float: Processing duration in seconds.
        """
        # Setup directories
        self.setup_directories(img_dir, output_base_dir)
        
        # Get image list
        img_list = self.get_image_list(img_pattern)
        
        start_time = time.time()
        
        # Process each image
        for i, img_path in enumerate(img_list):
            if i >= self.max_img:
                print('Reached max image limit')
                break
                
            print(f'\n{i}: {img_path}')
            self.process_image(img_path)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print statistics
        print('Run time: {} sec'.format(duration))
        print('Run time: {} min'.format(duration / 60.0))
        print('Run time: {} hrs'.format(duration / 3600.0))
        print(f'Time[s]/image = {duration / min(len(img_list), self.max_img)}')
        print('Done!')
        
        return duration


# Example usage
if __name__ == "__main__":
    # Configuration
    # config_path = '../data_yml_files/annotation_cslics_2024_oct_amag_tank3_10000000f620da42.yaml'
    # img_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/10000000f620da42/images'
    # output_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/10000000f620da42'
    
    # November 2024 spawning
    config_path = '/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/annotation_cslics_2024_nov_pdae_tank4_100000001ab0438d.yaml'
    img_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_november_subsurface_dataset/100000001ab0438d/images'
    output_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_november_subsurface_dataset/100000001ab0438d'
    
    # Create and run pipeline
    pipeline = AnnotationPipeline(config_path)
    pipeline.run(img_dir, output_dir)
