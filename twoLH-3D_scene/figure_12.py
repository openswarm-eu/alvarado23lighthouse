import pandas as pd
from functions.data_processing import   LH2_count_to_pixels, \
                                        solve_3d_scene, \
                                        scale_scene_to_real_size, \
                                        compute_distance_between_grid_points, \
                                        correct_perspective

from functions.plotting import plot_error_histogram

#############################################################################
###                                Options                                ###
#############################################################################

# file with the data to analyze
# data_file = './dataset/data_1point.csv'
data_file = './dataset/data_all.csv'

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    # Project sweep angles on to the z=1 image plane
    pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
    pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)

    # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
    df['LHA_proj_x'] = pts_lighthouse_A[:,0]
    df['LHA_proj_y'] = pts_lighthouse_A[:,1]
    df['LHB_proj_x'] = pts_lighthouse_B[:,0]
    df['LHB_proj_y'] = pts_lighthouse_B[:,1]

    # Solve the 3D scene with recoverPose and Triangulate points
    point3D, t_star, R_star = solve_3d_scene(pts_lighthouse_A, pts_lighthouse_B)

    # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
    # This will help correlate which point are supposed to go where.
    df['LH_x'] = point3D[:,0]
    df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
    df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

    # Scale the scene to real size
    df = scale_scene_to_real_size(df)

    # Compute the Euclidean Error between gridpoints
    x_dist, y_dist, z_dist = compute_distance_between_grid_points(df)

    # Bring reconstructed data to the origin for easier comparison
    df = correct_perspective(df)

    #############################################################################
    ###                             Plotting                                  ###
    #############################################################################

    # Plot the Euclidean error histogram
    plot_error_histogram(df)
