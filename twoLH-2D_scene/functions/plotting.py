import numpy as np
import seaborn as sns

# Fix to avoid Type 3 fonts on the figures
# http://phyletica.org/matplotlib-fonts/
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def plot_error_histogram(lh_data, exp_data):
    """ 
    Calculate and plot a histogram  of the error of the reconstructed points, vs. 
    the ground truth.
    """
    # Extract needed data from the main dataframe
    points = lh_data[['LH_x', 'LH_y', 'LH_z']].to_numpy()
    ground_truth = exp_data[['Rt_x', 'Rt_y', 'Rt_z']].to_numpy()

    # Calculate distance between points and their ground truth
    errors =  np.linalg.norm(ground_truth - points, axis=1) * 10 # x10 To have the errors in milimeters
    # print the mean and standard deviation
    print(f"Mean Absolute Error = {errors.mean()} mm")
    print(f"Root Mean Square Error = {np.sqrt((errors**2).mean())} mm")
    print(f"Error Standard Deviation = {errors.std()} mm")

    # prepare the plot
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 3, figure = fig)
    hist_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = (hist_ax,)

    # Plot the error histogram
    n, bins, patches = hist_ax.hist(errors, 50, density=False)
    hist_ax.axvline(x=errors.mean(), color='red', label="Mean")

    for ax in axs:
        ax.grid()
        ax.legend()
    
    hist_ax.set_xlabel('Distance Error [mm]')
    hist_ax.set_ylabel('Measurements')

    plt.show()

    return

def plot_reconstructed_3D_scene(point3D, t_star, R_star, calib_data=None, exp_data=None):
    """
    Plot a 3D scene with the traingulated points previously calculated
    ---
    input:
    point3D - array [3,N] - triangulated points of the positions of the LH2 receveier
    t_star  - array [3,1] - Translation vector between the first and the second lighthouse basestation
    R_star  - array [3,3] - Rotation matrix between the first and the second lighthouse basestation
    point3D - array [3,N] - second set of pointstriangulated points of the positions of the LH2 receveier
    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    # First lighthouse:
    arrow_size = 10
    ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:blue',lw=3)
    ax.quiver(0,0,0, 0,arrow_size,0, color='xkcd:red',lw=3)
    ax.quiver(0,0,0, 0,0,-arrow_size, color='xkcd:green',lw=3)

    # Second lighthouse:
    t_star_rotated = np.array([t_star.item(0), t_star.item(1), t_star.item(2)])
    # print(R_star)
    # print(t_star_rotated)
    x_axis = np.array([-arrow_size,0,0])@np.linalg.inv(R_star)
    y_axis = np.array([0,arrow_size,0])@np.linalg.inv(R_star)
    z_axis = np.array([0,0,-arrow_size])@np.linalg.inv(R_star)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],x_axis[0],x_axis[1],x_axis[2], color='xkcd:blue',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],y_axis[0],y_axis[1],y_axis[2],color='xkcd:red',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[1],t_star_rotated[2],z_axis[0],z_axis[1],z_axis[2],color='xkcd:green',lw=3)
    ax.scatter(point3D[:,0],point3D[:,1],point3D[:,2], alpha=0.1, label="lh2")

    # Plot the calibration points in the LHA reference frame
    if calib_data is not None:
        calib_lh2 = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl','tr','bl','br']]).reshape((4,3)) # originally it came out as shape=(3,1,4), I'm removing th uneeded dimension
        # ax.scatter(calib_lh2[:,0],calib_lh2[:,1],calib_lh2[:,2], alpha=0.5, color="xkcd:red")
        ax.scatter(calib_lh2[0,0],calib_lh2[0,1],calib_lh2[0,2], alpha=1, color="xkcd:red")
        ax.scatter(calib_lh2[1,0],calib_lh2[1,1],calib_lh2[1,2], alpha=1, color="xkcd:cyan")
        ax.scatter(calib_lh2[2,0],calib_lh2[2,1],calib_lh2[2,2], alpha=1, color="xkcd:green")
        ax.scatter(calib_lh2[3,0],calib_lh2[3,1],calib_lh2[3,2], alpha=1, color="xkcd:black")

    # Plot the Camera points, calibration and data
    if calib_data is not None and exp_data is not None:
        calib_cam = np.array([calib_data['corners_px_Rt'][corner] for corner in ['tl','tr','bl','br']])
        # ax.scatter(calib_cam[:,0],calib_cam[:,1],calib_cam[:,2], alpha=0.5, color="xkcd:orange")

        ax.scatter(calib_cam[0,0],calib_cam[0,1],calib_cam[0,2], alpha=1, color="xkcd:red")
        ax.scatter(calib_cam[1,0],calib_cam[1,1],calib_cam[1,2], alpha=1, color="xkcd:cyan")
        ax.scatter(calib_cam[2,0],calib_cam[2,1],calib_cam[2,2], alpha=1, color="xkcd:green")
        ax.scatter(calib_cam[3,0],calib_cam[3,1],calib_cam[3,2], alpha=1, color="xkcd:black")

        # cam_point3D = exp_data[['x','y','z']].values
        # ax.scatter(cam_point3D[:,0], cam_point3D[:,1], cam_point3D[:,2], alpha=1, color="xkcd:gray", label="camera")

        cam_point3D_Rt = exp_data[['Rt_x','Rt_y','Rt_z']].values
        ax.scatter(cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], cam_point3D_Rt[:,2], alpha=0.1, color="xkcd:red", label="camera")


    # R_1 = np.eye(3,dtype='float64')
    # t_1 = np.zeros((3,1),dtype='float64')

    # # Plot the lighthouse orientation
    # arrow = np.array([0,0,1]).reshape((-1,1))
    # ax.quiver(t_1[0],t_1[1],t_1[2], (R_1 @ arrow)[0], (R_1 @ arrow)[1], (R_1 @ arrow)[2], length=0.2, color='xkcd:orange' )
    # ax.quiver(t_star[0],t_star[1],t_star[2], (R_star @ arrow)[0], (R_star @ arrow)[1], (R_star @ arrow)[2], length=0.2, color='xkcd:orange' )

    # ax.scatter(point3D[:,0],point3D[:,1],point3D[:,2], color='xkcd:green', alpha=0.5, label='triangulated')
    # ax.scatter(t_1[0],t_1[1],t_1[2], color='xkcd:orange', label='triang LH1')
    # ax.scatter(t_star[0],t_star[1],t_star[2], color='xkcd:orange', label='triang LH2')    

    # Plot the real 
    ax.text(-0.18,-0.1,0,s='LHA')
    ax.text(t_star_rotated[0], t_star_rotated[1], t_star_rotated[2],s='LHB')

    ax.axis('equal')
    ax.legend()
    ax.set_title('2D solved scene - 3D triangulated Points')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')   

    plt.show()

def plot_ALL_reconstructed_3D_scene(df, exp_data, t_star, R_star, calib_data, experiment_file=None):
    """

    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green

    fig = plt.figure(layout="constrained", figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    # First lighthouse:
    arrow_size = 10
    # ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:blue',lw=3)   # Previous before changin the the reference frame to get a prettier looking plot
    # ax.quiver(0,0,0, 0,-arrow_size,0, color='xkcd:red',lw=3)
    # ax.quiver(0,0,0, 0,0,arrow_size, color='xkcd:green',lw=3)
    ax.quiver(0,0,0, -arrow_size,0,0, color='xkcd:green',lw=3)
    ax.quiver(0,0,0, 0,-arrow_size,0, color='xkcd:blue',lw=3)
    ax.quiver(0,0,0, 0,0,arrow_size, color='xkcd:red',lw=3)

    # Second lighthouse:
    t_star_rotated = np.array([t_star.item(0), t_star.item(1), t_star.item(2)])
    # print(R_star)
    # print(t_star_rotated)
    x_axis = np.array([-arrow_size,0,0])@np.linalg.inv(R_star)
    y_axis = np.array([0,arrow_size,0])@np.linalg.inv(R_star)
    z_axis = np.array([0,0,-arrow_size])@np.linalg.inv(R_star)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],x_axis[2],x_axis[0],x_axis[1], color='xkcd:blue',lw=3)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],y_axis[2],y_axis[0],y_axis[1],color='xkcd:red',lw=3)
    ax.quiver(t_star_rotated[2],t_star_rotated[0],t_star_rotated[1],z_axis[2],z_axis[0],z_axis[1],color='xkcd:green',lw=3)

    # Plot the calibration points in the LHA reference frame
    if calib_data is not None:
        calib_lh2 = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl','tr','bl','br']]).reshape((4,3)) # originally it came out as shape=(3,1,4), I'm removing th uneeded dimension
        # ax.scatter(calib_lh2[:,0],calib_lh2[:,1],calib_lh2[:,2], alpha=0.5, color="xkcd:red")
        # ax.scatter(calib_lh2[0,2],calib_lh2[0,0],calib_lh2[0,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[1,2],calib_lh2[1,0],calib_lh2[1,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[2,2],calib_lh2[2,0],calib_lh2[2,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_lh2[3,2],calib_lh2[3,0],calib_lh2[3,1], alpha=1, s=4, color="xkcd:black")

    # Plot the Camera points, calibration and data
    if calib_data is not None and exp_data is not None:
        calib_cam = np.array([calib_data['corners_px_Rt'][corner] for corner in ['tl','tr','bl','br']])
        # ax.scatter(calib_cam[:,2],calib_cam[:,0],calib_cam[:,1], alpha=0.5, color="xkcd:orange")

        # ax.scatter(calib_cam[0,2],calib_cam[0,0],calib_cam[0,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[1,2],calib_cam[1,0],calib_cam[1,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[2,2],calib_cam[2,0],calib_cam[2,1], alpha=1, s=4, color="xkcd:black")
        # ax.scatter(calib_cam[3,2],calib_cam[3,0],calib_cam[3,1], alpha=1, s=4, color="xkcd:black")

        # cam_point3D = exp_data[['x','y','z']].values
        # ax.scatter(cam_point3D[:,0], cam_point3D[:,1], cam_point3D[:,2], alpha=1, color="xkcd:gray", label="camera")


    for color in ['R', 'G', 'B']:

        point3D = df[color][['LH_x', 'LH_y','LH_z']].values
        cam_point3D_Rt = exp_data[color][['Rt_x','Rt_y','Rt_z']].values

        if color == 'R': 
            c = 'xkcd:red'
            label_lh = "Robot #1"
            label_c  = None
        if color == 'G': 
            c = 'xkcd:green'
            label_lh = "Robot #2"
            label_c  = None
        if color == 'B': 
            c = 'xkcd:blue'
            label_lh = "Robot #3"
            label_c  = "Ground Truth"

        # The BLUE triangle does 2 laps, for clarity, we plot only one.
        if '4' in experiment_file and color == 'B':
            ax.scatter(point3D[60:,2],point3D[60:,0],point3D[60:,1], alpha=0.5, color=c, s=4, label=label_lh)
            ax.plot(cam_point3D_Rt[60:,2], cam_point3D_Rt[60:,0], cam_point3D_Rt[60:,1], alpha=0.5, color="xkcd:black", label=label_c)
        else:
            ax.scatter(point3D[:,2],point3D[:,0],point3D[:,1], alpha=0.5, color=c, s=4, label=label_lh)
            ax.plot(cam_point3D_Rt[:,2], cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], alpha=0.5, color="xkcd:black", label=label_c)



        # ax.scatter(cam_point3D_Rt[:,0], cam_point3D_Rt[:,1], cam_point3D_Rt[:,2], alpha=0.2, color="xkcd:black", label=label_c)


    # Plot the real 
    ax.text(-0.18,-0.1,0,s='LHA')
    ax.text(t_star_rotated[2], t_star_rotated[0], t_star_rotated[1],s='LHB')


    ax.legend()
    ax.axis('equal')
    # ax.set_title('2D solved scene - 3D triangulated Points')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')   

    # Set Viewing orientation and zoom of the plot
    ax.view_init(-142,38, 0)
    ax.set_xlim3d((-144.424, -26.672))
    ax.set_ylim3d((-105, 13.6))
    ax.set_zlim3d((-20, 80))

    plt.savefig('Result-C-2lh_2d-solvedscene.pdf')
    plt.show()

def plot_projected_LH_views(pts_a, pts_b):
    """
    Plot the projected views from each of the lighthouse
    """

    fig = plt.figure(layout="constrained")
    gs = GridSpec(6, 3, figure = fig)
    lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
    lh2_ax = fig.add_subplot(gs[3:6, 0:3])
    axs = (lh1_ax, lh2_ax)

    # 2D plots - LH2 perspective
    lh1_ax.scatter(pts_a[:,0], pts_a[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
    lh2_ax.scatter(pts_b[:,0], pts_b[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")

    # Add labels and grids
    for ax in axs:
        ax.grid()
        ax.legend()
    lh1_ax.axis('equal')
    lh2_ax.axis('equal')
    # 
    lh1_ax.set_xlabel('U [px]')
    lh1_ax.set_ylabel('V [px]')
    #
    lh2_ax.set_xlabel('U [px]')
    lh2_ax.set_ylabel('V [px]')
    #
    lh1_ax.invert_yaxis()
    lh2_ax.invert_yaxis()

    plt.show()
