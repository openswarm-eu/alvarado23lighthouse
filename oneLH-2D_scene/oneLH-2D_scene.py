# import the necessary packages
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lib.functions import read_calibration_file, camera_to_world_homography, reorganize_data, interpolate_camera_to_lh2

####################################################################################
###                     Open files and prepare Data                              ###
####################################################################################
# Define which of the 6 experimetns you want to plot
experiment = 3
# Define the input data files
lh2_file = "dataset/lighthouse/" + f'{experiment}_lh2.csv'
camera_file = "dataset/ground_truth/" + f'{experiment}_blender.csv'
calibration_file = "dataset/ground_truth/" + 'calibration.json'

# Read the csv data files
df_lh2_in = pd.read_csv(lh2_file)#, index_col=0)
df_camera_in = pd.read_csv(camera_file)#, index_col=0)

# Load the file with the location of the calibration corner.
pts_src_px, pts_src_lh2, pts_dst = read_calibration_file(calibration_file)

####################################################################################
###            Convert Camera pixel to Ground Centimeters                        ###
####################################################################################
                                        
# Convert the 4k camera pixel data and the LH2 pixel data to the world coordinate frame of reference.
pts_cm_lh2 = camera_to_world_homography(df_lh2_in, pts_src_lh2, pts_dst)
pts_cm_camera = camera_to_world_homography(df_camera_in, pts_src_px, pts_dst)

####################################################################################
###                     Timestamp to miliseconds                                 ###
####################################################################################

camera_data = reorganize_data(pts_cm_camera, df_camera_in['timestamp'])
lh2_data    = reorganize_data(pts_cm_lh2, df_lh2_in['timestamp'])


####################################################################################
###                             Interpolate data                                 ###
####################################################################################

interp_data = interpolate_camera_to_lh2(camera_data, lh2_data)

difference_lh2 = np.sqrt((lh2_data['x'] - interp_data['x'])**2 + (lh2_data['y'] - interp_data['y'])**2)

####################################################################################
###                      Find points closest to a time                           ###
####################################################################################

t = camera_data['time'][0] + 5.5 # seconds
# Find the closest point to this time
camera_idx = np.abs(camera_data['time'] - t).argmin()
lh2_idx = np.abs(lh2_data['time'] - t).argmin()
# ekf_idx = np.abs(ekf_np['time'] - t).argmin()

# Extract it in easy to plot variables
point_camera = [camera_data['x'][camera_idx], camera_data['y'][camera_idx]]
point_lh2 = [lh2_data['x'][lh2_idx], lh2_data['y'][lh2_idx]]
# point_ekf = [ekf_np['x'][ekf_idx], ekf_np['y'][ekf_idx]]


####################################################################################
###                      Custom start and End Times                         ###
####################################################################################
# 22 + 10
t_i = camera_data['time'][0] + 2 # seconds
t_o = t_i + 99
# Find the closest point to this time
c_sta = np.abs(camera_data['time'] - t_i).argmin()
l_sta = np.abs(lh2_data['time'] - t_i).argmin()
# e_sta = np.abs(ekf_np['time'] - t_i).argmin()

# Find the closest point to this time
c_sto = np.abs(camera_data['time'] - t_o).argmin()
l_sto = np.abs(lh2_data['time'] - t_o).argmin()
# e_sto = np.abs(ekf_np['time'] - t_o).argmin()




####################################################################################
###                                 Plot Results                                 ###
####################################################################################



# Plot the results
fig = plt.figure(layout="constrained", figsize=(5,4))
gs = GridSpec(6, 3, figure = fig)
# Define individual subplots
xy_ax    = fig.add_subplot(gs[0:4, 0:3])
error_ax = fig.add_subplot(gs[4:6, :])
axs = (xy_ax, error_ax)


# X vs. Y plots
xy_ax.plot(lh2_data['x'][l_sta:l_sto], lh2_data['y'][l_sta:l_sto], '--',color='b', lw=1, label="lighthouse")
xy_ax.plot(camera_data['x'][c_sta:c_sto], camera_data['y'][c_sta:c_sto], '-',color='k', lw=1, label="ground truth")
xy_ax.scatter([80,120,120,80], [80,80,120,120], edgecolor='r', facecolor='red', lw=1, label="markers")
# Plot one synchronized point to check for a delay.
# xy_ax.scatter(point_camera[0], point_camera[1], edgecolor='k', facecolor='xkcd:gray', lw=1)
# xy_ax.scatter(point_lh2[0], point_lh2[1], edgecolor='k', facecolor='xkcd:blue', lw=1)

error_ax.plot(lh2_data['time'][l_sta:l_sto] - t_i, difference_lh2[l_sta:l_sto], '-',color='b', lw=1, label="LH error")

# Add labels and grids
for ax in axs:
    ax.grid()
    ax.legend()
xy_ax.axis('equal')
# 
xy_ax.set_xlabel('X [cm]')
xy_ax.set_ylabel('Y [cm]')
#
xy_ax.set_xlim([60, 160])

error_ax.set_xlabel('Time [s]')
error_ax.set_ylabel('Error [cm]')
#
error_ax.set_xlim([0, lh2_data['time'][l_sto] - lh2_data['time'][l_sta]])


plt.savefig('Result-A-1lh_2d-example.pdf')

plt.show()


