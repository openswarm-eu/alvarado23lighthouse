# import the necessary packages
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from functions.data_processing import   read_calibration_file, \
                                        camera_to_world_homography, \
                                        reorganize_data, \
                                        interpolate_camera_to_lh2, \
                                        find_closest_point

from functions.plotting import plot_trajectory_and_error

####################################################################################
###                               Options                                        ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment = 3
# Define the start and end time for the plotted trajectory. Useful for plotting smaller sections of large experiments
start_time = 2 # seconds
end_time = 20 # seconds

####################################################################################
###                            Read Dataset                                      ###
####################################################################################

# Define the input data files
lh2_file = "dataset/lighthouse/" + f'{experiment}_lh2.csv'
camera_file = "dataset/ground_truth/" + f'{experiment}_blender.csv'
calibration_file = "dataset/ground_truth/" + 'calibration.json'

# Read the csv files with he dataset
df_lh2_in = pd.read_csv(lh2_file)
df_camera_in = pd.read_csv(camera_file)

# Load the file with the location of the calibration corners.
pts_src_px, pts_src_lh2, pts_dst = read_calibration_file(calibration_file)

####################################################################################
###                            Process Data                                      ###
####################################################################################

# Convert the 4k camera pixel data and the LH2 pixel data to the world coordinate frame of reference.
pts_cm_lh2 = camera_to_world_homography(df_lh2_in, pts_src_lh2, pts_dst)
pts_cm_camera = camera_to_world_homography(df_camera_in, pts_src_px, pts_dst)

# Reorganize data into easy to use dictionaries, with epoch timestamps
camera_data = reorganize_data(pts_cm_camera, df_camera_in['timestamp'])
lh2_data    = reorganize_data(pts_cm_lh2, df_lh2_in['timestamp'])

# Interpolate camera data to LH2 timebase
interp_data = interpolate_camera_to_lh2(camera_data, lh2_data)

# Calculate the L2 distance error
error = np.sqrt((lh2_data['x'] - interp_data['x'])**2 + (lh2_data['y'] - interp_data['y'])**2)

####################################################################################
###                                 Plot Results                                 ###
####################################################################################

plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time)


