# This code grabs 8 or more (defined in code) random points from the grid position of the dataset and attempts to solve the scene with them.
# it does this several time to get statistics of how efective is to add more points to the scene reconstruction.
#
import pandas as pd
import numpy as np
from functions.data_processing import   LH2_count_to_pixels, \
                                        solve_3d_scene_get_Rt, \
                                        solve_3d_scene_triangulate_points, \
                                        scale_scene_to_real_size, \
                                        correct_perspective, \
                                        compute_errors, \
                                        is_coplanar, \
                                        compute_mad

#############################################################################
###                                Options                                ###
#############################################################################

# Max number of points to use for scene solving
N_POINTS = 100 
# How many random iteration to try per number of points
N_ITERATIONS = 300

# file with the data to analyze
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


    # Create dataframe to store the results of the analisys
    df_plot = pd.DataFrame(columns = ['n_points', "MAE", "RMS", 'STD', 'Coplanar', 'MAD'])

    # Set random seed.
    np.random.seed(1)

    # Get all unique grid points.
    points_grids = np.unique(df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy(), axis=0)

    # Get a averaged set of LH2 points
    pts_ave_A = np.empty((points_grids.shape[0],2), dtype=float)
    pts_ave_B = np.empty((points_grids.shape[0],2), dtype=float)
    for i in range(points_grids.shape[0]):
        pts_ave_A[i] = df.loc[(df['real_x_mm'] == points_grids[i,0])  & (df['real_y_mm'] == points_grids[i,1]) & (df['real_z_mm'] == points_grids[i,2]), ['LHA_proj_x', 'LHA_proj_y']].values.mean(axis=0)
        pts_ave_B[i] = df.loc[(df['real_x_mm'] == points_grids[i,0])  & (df['real_y_mm'] == points_grids[i,1]) & (df['real_z_mm'] == points_grids[i,2]), ['LHB_proj_x', 'LHB_proj_y']].values.mean(axis=0)


    # Start iterating over all Number of points used for the scene reconstruction
    for npoints in range (8,N_POINTS+1):
        print(f"n_points: {npoints}")
        # Start the for loop of how many iterations you're going to have.
        for iteration in range(N_ITERATIONS):

            if iteration % 20 == 0: print(f"\titeration: {iteration}")
            # extract N_POINTS unique points.
            calib_idx = np.random.choice(points_grids.shape[0], npoints, replace=False)
            calib_points_gt = points_grids[calib_idx]
            calib_lha = pts_ave_A[calib_idx]
            calib_lhb = pts_ave_B[calib_idx]
            # Get R_star and t_star from the unique points

            t_star, R_star = solve_3d_scene_get_Rt(calib_lha, calib_lhb)
            # Triangulate all points
            point3D = solve_3d_scene_triangulate_points(pts_lighthouse_A, pts_lighthouse_B, t_star, R_star)

            # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
            # This will help correlate which point are supposed to go where.
            df['LH_x'] = point3D[:,0]
            df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
            df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

            # Scale the scene to real size
            scale_scene_to_real_size(df)

            # Bring reconstructed data to the origin for easier comparison
            df = correct_perspective(df)
            # Calculate all relevant errors
            mae, rmse, std = compute_errors(df)
            # Measure complanarity
            coplanar = is_coplanar(calib_points_gt)
            # measure median average deviation between the chosen points
            mad = compute_mad(calib_points_gt)

            # add errors to the dataframe where we are accumulating them.
            df_plot.loc[len(df_plot)] = [npoints, mae, rmse, std, coplanar, mad]


    #############################################################################
    ###                             Save results                              ###
    #############################################################################


    #  Remove solutions of complanar points
    df_plot = df_plot.loc[ (df_plot['Coplanar'] > 30) & (df_plot['MAE'] < 200)]
    # Save the result
    df_plot.to_csv('./figure_13_dataset.csv', index=True)

    
