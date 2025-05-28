import json
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_multiple_error_data(error_data_files, output_path=None):
    """
    Read multiple error data files and plot them together.
    
    Args:
        error_data_files (list): List of paths to error data JSON files.
        output_path (str, optional): Path to save the combined plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(12, 8))
    
    # Track min/max values for axis scaling
    all_times = []
    all_errors = []
    
    for file_path in error_data_files:
        # Read the error data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the data
        tank_name = data["tank_sheet_name"]
        species = data["species"]
        manual_times = data["manual_times"]
        errors = data["errors"]
        mae = data["statistics"]["mae"]
        rmse = data["statistics"]["rmse"]
        
        # Plot the data
        plt.plot(manual_times, errors, marker='o', label=f'{tank_name} ({species}) - MAE: {mae:.2f}, RMSE: {rmse:.2f}')
        
        # Collect all times and errors for axis scaling
        all_times.extend(manual_times)
        all_errors.extend(errors)
    
    # Add a zero line
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Set plot properties
    plt.grid(True)
    plt.xlabel('Days since spawning')
    plt.ylabel('Error (AI - Manual Counts)')
    plt.title('Error Comparison Across Multiple Tanks')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=600)
        print(f"Combined error plot saved to {output_path}")
    else:
        plt.show()
        
if __name__ == "__main__":
    # Example usage
    error_files = [
        "/path/to/error_data_OCT24_T1_Amag_100000000029da9b.json",
        "/path/to/error_data_OCT24_T2_Amag_100000009c23b5af.json",
        "/path/to/error_data_OCT24_T3_Amag_10000000f620da42.json"
    ]
    
    output_path = 'combined_error_plot.png'  # Set to None to display the plot instead of saving
    plot_multiple_error_data(error_files, output_path)