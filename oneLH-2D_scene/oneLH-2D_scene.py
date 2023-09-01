# import the necessary packages
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

####################################################################################
###                     Open files and prepare Data                              ###
####################################################################################
experiment = 3
# Define the input data files
# pre = "2 - experiment 2 - 230406 - ekf + lh2/"
pre = ""
# ekf_file = pre + f'{experiment}_ekf.csv'
lh2_file = pre + f'{experiment}_lh2.csv'
camera_file = pre + f'video_track/{experiment}_blender.csv'

# Read the csv files
# df_ekf_in = pd.read_csv(ekf_file)#, index_col=0)
df_lh2_in = pd.read_csv(lh2_file)#, index_col=0)
df_camera_in = pd.read_csv(camera_file)#, index_col=0)

# Load the file with the corner locations
with open(pre + 'video_track/calibration.json') as f:
    data = json.load(f)

c_px  = data['GX010145.MP4']['corners_px']
c_cm  = data['GX010145.MP4']['corners_cm']
c_lh2 = data['GX010145.MP4']['corners_lh2']

# Define Source and Destination points for the Homography calculations
pts_src_px  = np.array([c_px['tl'], c_px['tr'], c_px['br'], c_px['bl']]) # pixels
pts_src_lh2 = np.array([c_lh2['tl'], c_lh2['tr'], c_lh2['br'], c_lh2['bl']]) # pixels
pts_dst     = np.array([c_cm['tl'], c_cm['tr'], c_cm['br'], c_cm['bl']]) # centimeters

####################################################################################
###            Convert Camera pixel to Ground Centimeters                        ###
####################################################################################

# Calculate the Homography Matrix
h_px_cm,  status = cv2.findHomography(pts_src_px, pts_dst)
h_lh2_cm, status = cv2.findHomography(pts_src_lh2, pts_dst)

# Prepare pixel points to convert
pts_pixel = df_camera_in[['x', 'y']].to_numpy().reshape((1,-1,2))
pts_lh2   = df_lh2_in[['x', 'y']].to_numpy().reshape((1,-1,2))
# pts_pixel = np.array([[[200, 400], [1000, 1500], [3000, 2000]]], dtype=float)  # Shape of the input array must be (1, n_points, 2), note the double square brackets before and after the points.

# Run the transform
pts_cm_camera = cv2.perspectiveTransform(pts_pixel, h_px_cm)
pts_cm_camera = pts_cm_camera.reshape((-1, 2))                   # We can reshape the output so that the points look like [[3,4], [1,4], [5,1]]
                                                   # They are easier to work with like this, without all that double square bracket non-sense
pts_cm_lh2 = cv2.perspectiveTransform(pts_lh2, h_lh2_cm)
pts_cm_lh2 = pts_cm_lh2.reshape((-1, 2))                   # We can reshape the output so that the points look like [[3,4], [1,4], [5,1]]
                                                   # They are easier to work with like this, without all that double square bracket non-sense

# print(pts_cm_camera[0:10])


####################################################################################
###                     Timestamp to miliseconds                                 ###
####################################################################################
# Create a new column called "time_s" with the posix epoch as floats of the timestamps (in seconds, with the decimal part as precises to 1us)
for df in [df_camera_in, df_lh2_in]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_s'] = df['timestamp'].apply(lambda x: x.timestamp() )


####################################################################################
###                             Interpolate data                                 ###
####################################################################################
# Reorganize the data to make it easier to manipulate in numpy (it's easier to do linear interpolation in numpy, instead of pandas.)
# ekf_np = {'time': df_ekf_in['time_s'].to_numpy(),
#           'x':    df_ekf_in['x'].to_numpy(),
#           'y':    df_ekf_in['y'].to_numpy(),}


camera_np = {'time': df_camera_in['time_s'].to_numpy(),
             'x':    pts_cm_camera[:,0],
             'y':    pts_cm_camera[:,1],}


lh2_np = {'time': df_lh2_in['time_s'].to_numpy(),
             'x':    pts_cm_lh2[:,0],
             'y':    pts_cm_lh2[:,1],}


# Offset the camera timestamp to get rid of the communication delay.
camera_np['time'] += 318109e-6 # seconds

# Interpolate the camera data against the ekf
# camera_np['x_interp_ekf'] = np.interp(ekf_np['time'], camera_np['time'],  camera_np['x'])
# camera_np['y_interp_ekf'] = np.interp(ekf_np['time'], camera_np['time'],  camera_np['y'])
# Interpolate the camera data against the lh2
camera_np['x_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['x'])
camera_np['y_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['y'])

# difference_ekf = np.sqrt((ekf_np['x'] - camera_np['x_interp_ekf'])**2 + (ekf_np['y'] - camera_np['y_interp_ekf'])**2)
difference_lh2 = np.sqrt((lh2_np['x'] - camera_np['x_interp_lh2'])**2 + (lh2_np['y'] - camera_np['y_interp_lh2'])**2)

####################################################################################
###                      Find points closest to a time                           ###
####################################################################################

t = camera_np['time'][0] + 5.5 # seconds
# Find the closest point to this time
camera_idx = np.abs(camera_np['time'] - t).argmin()
lh2_idx = np.abs(lh2_np['time'] - t).argmin()
# ekf_idx = np.abs(ekf_np['time'] - t).argmin()

# Extract it in easy to plot variables
point_camera = [camera_np['x'][camera_idx], camera_np['y'][camera_idx]]
point_lh2 = [lh2_np['x'][lh2_idx], lh2_np['y'][lh2_idx]]
# point_ekf = [ekf_np['x'][ekf_idx], ekf_np['y'][ekf_idx]]


####################################################################################
###                      Custom start and End Times                         ###
####################################################################################
# 22 + 10
t_i = camera_np['time'][0] + 2 # seconds
t_o = t_i + 99
# Find the closest point to this time
c_sta = np.abs(camera_np['time'] - t_i).argmin()
l_sta = np.abs(lh2_np['time'] - t_i).argmin()
# e_sta = np.abs(ekf_np['time'] - t_i).argmin()

# Find the closest point to this time
c_sto = np.abs(camera_np['time'] - t_o).argmin()
l_sto = np.abs(lh2_np['time'] - t_o).argmin()
# e_sto = np.abs(ekf_np['time'] - t_o).argmin()




####################################################################################
###                                 Plot Results                                 ###
####################################################################################



# Plot the results
fig = plt.figure(layout="constrained", figsize=(5,4))
# fig = plt.figure(layout="tight", figsize=(5.2,4))
gs = GridSpec(6, 3, figure = fig)
# Define individual subplots
xy_ax    = fig.add_subplot(gs[0:4, 0:3])
error_ax = fig.add_subplot(gs[4:6, :])
# theta_ax = fig.add_subplot(gs[0, 3:])
# gyro_ax = fig.add_subplot(gs[1, 3:])
# v_ax = fig.add_subplot(gs[2, 3:])
# Join them together to iterate over them
# axs = (xy_ax, theta_ax, gyro_ax, v_ax)
axs = (xy_ax, error_ax)


# X vs. Y plots
# xy_ax.plot(ekf_np['x'][e_sta:e_sto], ekf_np['y'][e_sta:e_sto], '--',color='g', lw=1, label="ekf")
xy_ax.plot(lh2_np['x'][l_sta:l_sto], lh2_np['y'][l_sta:l_sto], '--',color='b', lw=1, label="lighthouse")
xy_ax.plot(camera_np['x'][c_sta:c_sto], camera_np['y'][c_sta:c_sto], '-',color='k', lw=1, label="ground truth")
xy_ax.scatter([80,120,120,80], [80,80,120,120], edgecolor='r', facecolor='red', lw=1, label="markers")
# Plot one synchronized point to check for a delay.
# xy_ax.scatter(point_camera[0], point_camera[1], edgecolor='k', facecolor='xkcd:gray', lw=1)
# xy_ax.scatter(point_lh2[0], point_lh2[1], edgecolor='k', facecolor='xkcd:blue', lw=1)
# xy_ax.scatter(point_ekf[0], point_ekf[1], edgecolor='k', facecolor='xkcd:green', lw=1)

# error_ax.plot(ekf_np['time'][e_sta:e_sto] - t_i, difference_ekf[e_sta:e_sto], '-',color='g', lw=1, label="ekf error")
error_ax.plot(lh2_np['time'][l_sta:l_sto] - t_i, difference_lh2[l_sta:l_sto], '-',color='b', lw=1, label="LH error")

# # Theta vs. time
# theta_ax.step(lh2_angle[:, -1], lh2_angle[:,0], '-',color='r', lw=1, label="LH2 measurements")
# theta_ax.plot(ekf_state[:, -1], np.rad2deg(ekf_state[:,2]), '-',color='k', lw=1, label="ekf")

# # W vs time
# gyro_ax.plot(state[:, -1], np.rad2deg(state[:, 5]), '-',color='b', lw=1, label="Ground Truth")
# gyro_ax.plot(ekf_state[:, -1], np.rad2deg(ekf_state[:,4]), '-',color='k', lw=1, label="ekf")

# # V vs time
# v_ax.plot(state[:, -1], np.sqrt(state[:, 1]**2 + state[:, 3]**2), '-',color='b', lw=1, label="Ground Truth")
# v_ax.plot(ekf_state[:, -1], ekf_state[:,3], '-',color='k', lw=1, label="ekf")
# v_ax.plot(lh2_speed[:, -1], lh2_speed[:,0], '-',color='r', lw=1, label="LH2 measurements")

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
# xy_ax.set_ylim([ymin, ymax])
#
# theta_ax.set_title('Angle position plot')
# theta_ax.set_xlabel('Time [s]')
# theta_ax.set_ylabel('Theta [deg]')
#
error_ax.set_xlabel('Time [s]')
error_ax.set_ylabel('Error [cm]')
#
error_ax.set_xlim([0, 20])
# v_ax.set_xlabel('Time [s]')
# v_ax.set_ylabel('Linear speed [cm/s]')

plt.savefig('Result-A-1lh_2d-example.pdf')

plt.show()


