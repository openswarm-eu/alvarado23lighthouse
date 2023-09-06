import numpy as np

# Fix to avoid Type 3 fonts on the figures
# http://phyletica.org/matplotlib-fonts/
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_trajectory_and_error(lh2_data, camera_data, error, start_time, end_time):
    """
    """

    # Find the indexes of the start and end times for the plots
    t_i = camera_data['time'][0] + start_time # seconds
    t_o = t_i + end_time
    # Find the closest point to this time
    c_sta = np.abs(camera_data['time'] - t_i).argmin()
    l_sta = np.abs(lh2_data['time'] - t_i).argmin()
    # e_sta = np.abs(ekf_np['time'] - t_i).argmin()

    # Find the closest point to this time
    c_sto = np.abs(camera_data['time'] - t_o).argmin()
    l_sto = np.abs(lh2_data['time'] - t_o).argmin()


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

    error_ax.plot(lh2_data['time'][l_sta:l_sto] - t_i, error[l_sta:l_sto], '-',color='b', lw=1, label="LH error")

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