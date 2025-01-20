#!/usr/bin/env python3++
import os
import cv2
import numpy as np

directories = ['/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_november_spawning/100000001ab0438d', 
               '/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_november_spawning/100000000029da9b', 
               '/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_november_spawning/100000009c23b5af', 
               '/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/10000000570f9d9c', 
               '/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/100000009c23b5af', 
               '/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/10000000f620da42'
               ]

# Initialize the dictionaries
banding = {}
no_banding = {}

def channel_shift_stitch(image_name: str, channel: str, intensity: int):
    """
    Create a stitched image with the same image shifted by negative intensity, original, and positive intensity 
    for a specified channel (H, S, or V) in HSV color space.
    
    Parameters:
        image_name (str): Path to the input image.
        channel (str): The channel to modify ('H', 'S', or 'V').
        intensity (int): The intensity of the shift.
    
    Returns:
        stitched_image (np.ndarray): The stitched image with applied channel shifts.
    """
    # Map the channel to index (0: H, 1: S, 2: V)
    channel_map = {'H': 0, 'S': 1, 'V': 2}
    if channel not in channel_map:
        raise ValueError("Channel must be 'H', 'S', or 'V'.")
    
    channel_index = channel_map[channel]
    
    # Read the image and convert to HSV
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Image '{image_name}' not found.")
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Function to apply channel shift
    def shift_channel(hsv_img, channel_idx, shift_val):
        shifted_img = hsv_img.copy()
        shifted_img[:, :, channel_idx] = np.clip(shifted_img[:, :, channel_idx] + shift_val, 0, 255)
        return shifted_img
    
    # Generate the three versions of the image
    left_image = shift_channel(hsv_image, channel_index, -intensity)
    right_image = shift_channel(hsv_image, channel_index, intensity)
    
    # Convert the shifted images back to BGR
    left_image_bgr = cv2.cvtColor(left_image, cv2.COLOR_HSV2BGR)
    center_image_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_HSV2BGR)
    
    # Stitch the images together
    stitched_image = np.hstack((left_image_bgr, center_image_bgr, right_image_bgr))
    
    return stitched_image



class HSVInfo:
    """
    Class to store the name of an image and the value of a channel.
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def find_min_values(dictionary):
        """
        Find the image and the value with the lowest value in the dictionary for each channel.
        Returns:
            hue_min (HSVInfo): lowest hue value.
            saturation_min (HSVInfo): lowest saturation value.
            value_min (HSVInfo):  lowest value value.
        """
        hue_min        = HSVInfo(min(dictionary, key=lambda x: dictionary[x][0]), dictionary[min(dictionary, key=lambda x: dictionary[x][0])][0])
        saturation_min = HSVInfo(min(dictionary, key=lambda x: dictionary[x][1]), dictionary[min(dictionary, key=lambda x: dictionary[x][1])][1])
        value_min      = HSVInfo(min(dictionary, key=lambda x: dictionary[x][2]), dictionary[min(dictionary, key=lambda x: dictionary[x][2])][2])
    
        return hue_min, saturation_min, value_min
    
    def find_max_values(dictionary):
        """
        Find the image and the value with the peak value in the dictionary for each channel.
        Returns:
            hue_max (HSVInfo): peak hue value.
            saturation_max (HSVInfo): peak saturation value.
            value_max (HSVInfo):  peak value value.
        """
        hue_max        = HSVInfo(max(dictionary, key=lambda x: dictionary[x][0]), dictionary[(max(dictionary, key=lambda x: dictionary[x][0]))][0])
        saturation_max = HSVInfo(max(dictionary, key=lambda x: dictionary[x][1]), dictionary[(max(dictionary, key=lambda x: dictionary[x][1]))][1])
        value_max      = HSVInfo(max(dictionary, key=lambda x: dictionary[x][2]), dictionary[(max(dictionary, key=lambda x: dictionary[x][2]))][2])

        return hue_max, saturation_max, value_max
    
    def calculate_average_values(dictionary):
        """
        Calculate the average of the average values of the channels
        Rounded to 3 decimal places
        Returns:
            hue_avg (float): The average of the hue values.
            saturation_avg (float): The average of the saturation values.
            value_avg (float): The average of the value values.
        """
        hue_avg = round(sum(value[0] for value in dictionary.values()) / len(dictionary),3)
        saturation_avg = round(sum(value[1] for value in dictionary.values()) / len(dictionary),3)
        value_avg = round(sum(value[2] for value in dictionary.values()) / len(dictionary),3)
        return hue_avg, saturation_avg, value_avg
    def calculate_std_deviation(dictionary):
        """
        Calculate the standard deviation of the values in the dictionary for each channel.
        
        Returns:
            hue_std (float): The standard deviation of the hue values.
            saturation_std (float): The standard deviation of the saturation values.
            value_std (float): The standard deviation of the value values.
        """
        hue_std = np.std([value[0] for value in dictionary.values()])
        saturation_std = np.std([value[1] for value in dictionary.values()])
        value_std = np.std([value[2] for value in dictionary.values()])
        
        return hue_std, saturation_std, value_std


if __name__ == '__main__':
    # Iterate over the directories
    for directory in directories:
        # Get the subdirectories
        subdirectories = ['banding', 'no_banding']
        
        # Iterate over the subdirectories
        for subdirectory in subdirectories:
            # Get the path to the subdirectory
            subdirectory_path = os.path.join(directory, subdirectory)
            
            # Get the list of image files in the subdirectory
            image_files = os.listdir(subdirectory_path)
            
            # Iterate over the image files
            for image_file in image_files:
                # Read the image
                image_path = os.path.join(subdirectory_path, image_file)
                image = cv2.imread(image_path)
                
                # Assign the image to the appropriate dictionary
                if subdirectory == 'banding':
                    banding[image_file] = image
                elif subdirectory == 'no_banding':
                    no_banding[image_file] = image
                else:
                    print("Invalid subdirectory.")
                    raise ValueError("Invalid subdirectory.")

    # Convert the images to HSV
    banding_hsv = {key: cv2.cvtColor(value, cv2.COLOR_RGB2HSV) for key, value in banding.items()}
    no_banding_hsv = {key: cv2.cvtColor(value, cv2.COLOR_RGB2HSV) for key, value in no_banding.items()}

    #average the values of the images in the banding and no_banding dictionaries
    banding_hsv_avg = {key: cv2.mean(value) for key, value in banding_hsv.items()}
    no_banding_hsv_avg = {key: cv2.mean(value) for key, value in no_banding_hsv.items()}

    # Find the image and the value with the peak and lowest value in the banding dictionary for each channel
    banding_hue_max, banding_saturation_max, banding_value_max          =   HSVInfo.find_max_values(banding_hsv_avg)
    no_banding_hue_max, no_banding_saturation_max, no_banding_value_max =   HSVInfo.find_max_values(no_banding_hsv_avg)
    banding_hue_min, banding_saturation_min, banding_value_min          =   HSVInfo.find_min_values(banding_hsv_avg)
    no_banding_hue_min, no_banding_saturation_min, no_banding_value_min =   HSVInfo.find_min_values(no_banding_hsv_avg)

    print()
    # Print the peak and lowest values for each channel with flipped bolding
    print("Channel     | Lowest Value | Peak Value    | % Difference | Absolute HSV Difference")
    print("------------|--------------|---------------|--------------|-------------------------")
    print(f"\033[1mBanding   H\033[0m | \033[1m{round(banding_hue_min.value, 3):<12}\033[0m | \033[1m{round(banding_hue_max.value, 3):<13}\033[0m | \033[1m{round(((banding_hue_max.value - banding_hue_min.value) / banding_hue_max.value) * 100, 3):<12}\033[0m | \033[1m{round(banding_hue_max.value - banding_hue_min.value, 3):<12}\033[0m")
    print(f"NoBanding H | {round(no_banding_hue_min.value, 3):<12} | {round(no_banding_hue_max.value, 3):<13} | {round(((no_banding_hue_max.value - no_banding_hue_min.value) / no_banding_hue_max.value) * 100, 3):<12} | {round(no_banding_hue_max.value - no_banding_hue_min.value, 3):<12}")
    print(f"\033[1mBanding   S\033[0m | \033[1m{round(banding_saturation_min.value, 3):<12}\033[0m | \033[1m{round(banding_saturation_max.value, 3):<13}\033[0m | \033[1m{round(((banding_saturation_max.value - banding_saturation_min.value) / banding_saturation_max.value) * 100, 3):<12}\033[0m | \033[1m{round(banding_saturation_max.value - banding_saturation_min.value, 3):<12}\033[0m")
    print(f"NoBanding S | {round(no_banding_saturation_min.value, 3):<12} | {round(no_banding_saturation_max.value, 3):<13} | {round(((no_banding_saturation_max.value - no_banding_saturation_min.value) / no_banding_saturation_max.value) * 100, 3):<12} | {round(no_banding_saturation_max.value - no_banding_saturation_min.value, 3):<12}")
    print(f"\033[1mBanding   V\033[0m | \033[1m{round(banding_value_min.value, 3):<12}\033[0m | \033[1m{round(banding_value_max.value, 3):<13}\033[0m | \033[1m{round(((banding_value_max.value - banding_value_min.value) / banding_value_max.value) * 100, 3):<12}\033[0m | \033[1m{round(banding_value_max.value - banding_value_min.value, 3):<12}\033[0m")
    print(f"NoBanding V | {round(no_banding_value_min.value, 3):<12} | {round(no_banding_value_max.value, 3):<13} | {round(((no_banding_value_max.value - no_banding_value_min.value) / no_banding_value_max.value) * 100, 3):<12} | {round(no_banding_value_max.value - no_banding_value_min.value, 3):<12}")
    print()
    print(banding_hue_max.name)
    #print(banding_value_min.name,banding_value_max.name)


    # Calculate the average values for banding and no banding dictionaries
    banding_hue_avg, banding_saturation_avg, banding_value_avg          = HSVInfo.calculate_average_values(banding_hsv_avg)
    no_banding_hue_avg, no_banding_saturation_avg, no_banding_value_avg = HSVInfo.calculate_average_values(no_banding_hsv_avg)
    print()

    # Calculate the standard deviation values for banding and no banding dictionaries
    banding_hue_std, banding_saturation_std, banding_value_std = HSVInfo.calculate_std_deviation(banding_hsv_avg)
    no_banding_hue_std, no_banding_saturation_std, no_banding_value_std = HSVInfo.calculate_std_deviation(no_banding_hsv_avg)

    # Print the standard deviation values
    print("Channel     | No Banding Avg | No Banding Std | Banding Avg   | Banding Std   | % Difference | Absolute HSV Difference")
    print("------------|----------------|----------------|---------------|---------------|--------------|------------------------")
    print(f"    H       | {no_banding_hue_avg:<14} | {round(no_banding_hue_std,3):<14} | {banding_hue_avg:<13} | {round(banding_hue_std,3):<13} | {round(abs(banding_hue_avg - no_banding_hue_avg) / no_banding_hue_avg * 100, 2):<12} | {round(abs(banding_hue_avg - no_banding_hue_avg), 2):<12}")
    print(f"    S       | {no_banding_saturation_avg:<14} | {round(no_banding_saturation_std,3):<14} | {(banding_saturation_avg):<13} | {round(banding_saturation_std,3):<13} | {round(abs(banding_saturation_avg - no_banding_saturation_avg) / no_banding_saturation_avg * 100, 2):<12} | {round(abs(banding_saturation_avg - no_banding_saturation_avg), 2):<12}")
    print(f"    V       | {no_banding_value_avg:<14} | {round(no_banding_value_std,3):<14} | {banding_value_avg:<13} | {round(banding_value_std,3):<13} | {round(abs(banding_value_avg - no_banding_value_avg) / no_banding_value_avg * 100, 2):<12} | {round(abs(banding_value_avg - no_banding_value_avg), 2):<12}")
    print()

    #testImage = "/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/100000009c23b5af/banding/2024-10-24_16-02-38_clean.jpg"
    #testImage = "/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_november_spawning/100000000029da9b/banding/2024-11-23_17-01-05_clean.jpg"
    testImage = "/media/wardlewo/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/100000009c23b5af/no_banding/2024-10-26_03-00-18_clean.jpg"
    result = channel_shift_stitch(testImage, 'H', banding_hue_std)
    cv2.namedWindow("HSV_Adjusted_Image", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("HSV_Adjusted_Image", 1920, 1080) 
    cv2.imshow("HSV_Adjusted_Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

