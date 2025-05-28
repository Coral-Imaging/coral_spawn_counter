#!/usr/bin/env python3

"""
Script to plot error data from multiple JSON files
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Error files paths
error_files = [
    "/media/dtsai/CSLICSNov24/cslics_november_2024/detections/100000000029da9b/cslics_subsurface_20250205_640p_yolov8n/error_data/error_data_NOV24 T1 Amil_100000000029da9b.json",
    "/media/dtsai/CSLICSNov24/cslics_november_2024/detections/100000009c23b5af/cslics_subsurface_20250205_640p_yolov8n/error_data/error_data_NOV24 T2 Amil_100000009c23b5af.json",
    "/media/dtsai/CSLICSNov24/cslics_november_2024/detections/10000000f620da42/cslics_subsurface_20250205_640p_yolov8n/error_data/error_data_NOV24 T3 Amil_10000000f620da42.json"
]

def plot_error_data(error_files, output_path=None):
    """
    Read and plot error data from multiple JSON files.
    
    Args:
        error_files (list): List of paths to error data JSON files
        output_path (str, optional): Path to save the output plot
    """
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Track data for combined statistics
    all_errors = []
    tank_names = []
    
    # Loop through each error file
    for file_path in error_files:
        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract relevant data
        tank_name = data["tank_sheet_name"]
        tank_names.append(tank_name)
        cslics_uuid = data["cslics_uuid"]
        species = data.get("species", "Unknown")  # Use .get() with default value
        manual_times = data["manual_times"]
        errors = data["errors"]
        mae = data["statistics"]["mae"]
        rmse = data["statistics"]["rmse"]
        
        # zero the manual times
        manual_time0 = manual_times[0]
        manual_times = [t - manual_time0 for t in manual_times]
        
        # Collect errors for combined statistics
        all_errors.extend(errors)
        
        # Plot the error data
        ax.plot(manual_times, errors, 'o-', label=f'{tank_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}')
        
        # Print summary information
        print(f"\n{tank_name} (UUID: {cslics_uuid}, Species: {species})")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"Number of data points: {len(manual_times)}")
    
    # Calculate combined statistics
    combined_mae = np.mean(np.abs(all_errors))
    combined_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
    print(f"\nCombined Statistics:")
    print(f"Mean Absolute Error: {combined_mae:.2f}")
    print(f"Root Mean Square Error: {combined_rmse:.2f}")
    
    # Add a zero reference line
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    
    # Add grey transparent patch overlay from times 0 to 1
    ax.axvspan(-0.1, 1.3, alpha=0.3, color='grey', label='Water Filtration Off Period')
    
    # Configure plot aesthetics
    ax.set_xlabel('Days Since Spawning', fontsize=12)
    ax.set_ylabel('Error (AI Count - Manual Count)', fontsize=12)
    ax.set_title('CSLICS AI vs Manual Counting Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    # Force y-axis to include zero
    ax.spines['left'].set_position('zero')
    
    # Use integer ticks on x-axis if appropriate
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add text with combined statistics
    # combined_stats = f"Combined MAE: {combined_mae:.2f}\nCombined RMSE: {combined_rmse:.2f}"
    # plt.figtext(0.2, 0.02, combined_stats, fontsize=10, 
    #             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Generate output path in same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "cslics_error_comparison.png")
    
    # Create and save the plot
    plot_error_data(error_files, output_file)
    
    print("\nAnalysis complete!")