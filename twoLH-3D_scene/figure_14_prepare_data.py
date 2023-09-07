import pandas as pd
import numpy as np
from functions.data_processing import   LH2_count_to_pixels, \
                                        solve_3d_scene_get_Rt, \
                                        solve_3d_scene_triangulate_points, \
                                        scale_scene_to_real_size, \
                                        correct_perspective, \
                                        compute_errors, \
                                        compute_mad
#############################################################################
###                                Options                                ###
#############################################################################
# file with the data to analyze
data_file = 'dataset/data_all.csv'

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
    df_plot = pd.DataFrame(columns = ['MAD', "MAE", "RMS", 'STD'])

    # Iterate through all the different sized of cubes you can as calibration points
    # x_d, is the size of the x-aligned sides of the cube.
    for x_d in [40, 80, 120, 160, 200, 240]:
        for y_d in [40, 80, 120]:
            for z_d in [40, 80, 120, 160]:

                print(f'box = [{x_d}, {y_d}, {z_d}]')
                # Now iterate over all the possible starting points for these calibration cubes.
                for x_i in range(0, 240 + 40 - x_d, 40):
                    for y_i in range(0, 120 + 40 - y_d, 40):
                        for z_i in range(0, 160 + 40 - z_d, 40):

                            # Now generate the 8 points definning the calibration cube
                            points_calib = np.array([[x_i      , y_i      , z_i      ],
                                                     [x_i + x_d, y_i      , z_i      ],
                                                     [x_i      , y_i + y_d, z_i      ],
                                                     [x_i      , y_i      , z_i + z_d],
                                                     [x_i + x_d, y_i + y_d, z_i      ],
                                                     [x_i + x_d, y_i      , z_i + z_d],
                                                     [x_i      , y_i + y_d, z_i + z_d],
                                                     [x_i + x_d, y_i + y_d, z_i + z_d]])
                            
                            mad = compute_mad(points_calib)

                            # Get the average projections of the chosen calibration points
                            pts_ave_A = np.empty((0,2), dtype=float)
                            pts_ave_B = np.empty((0,2), dtype=float)
                            for i in range(points_calib.shape[0]):
                                pts_ave_A = np.vstack([pts_ave_A, df.loc[(df['real_x_mm'] == points_calib[i,0])  & (df['real_y_mm'] == points_calib[i,1]) & (df['real_z_mm'] == points_calib[i,2]), ['LHA_proj_x', 'LHA_proj_y']].values])
                                pts_ave_B = np.vstack([pts_ave_B, df.loc[(df['real_x_mm'] == points_calib[i,0])  & (df['real_y_mm'] == points_calib[i,1]) & (df['real_z_mm'] == points_calib[i,2]), ['LHB_proj_x', 'LHB_proj_y']].values])
                                # pts_ave_A = np.vstack([pts_ave_A, df.loc[(df['real_x_mm'] == points_calib[i,0])  & (df['real_y_mm'] == points_calib[i,1]) & (df['real_z_mm'] == points_calib[i,2]), ['LHA_proj_x', 'LHA_proj_y']].values.mean(axis=0)])
                                # pts_ave_B = np.vstack([pts_ave_B, df.loc[(df['real_x_mm'] == points_calib[i,0])  & (df['real_y_mm'] == points_calib[i,1]) & (df['real_z_mm'] == points_calib[i,2]), ['LHB_proj_x', 'LHB_proj_y']].values.mean(axis=0)])


                            # Get R_star and t_star from the unique points
                            t_star, R_star = solve_3d_scene_get_Rt(pts_ave_A, pts_ave_B)
                            # Triangulate all points
                            point3D = solve_3d_scene_triangulate_points(pts_lighthouse_A, pts_lighthouse_B, t_star, R_star)

                            # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
                            # This will help correlate which point are supposed to go where.
                            df['LH_x'] = point3D[:,0]
                            df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
                            df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

                            # Scale the scene to real size
                            if 'experimental' in data_file:
                                df = scale_scene_to_real_size(df)

                            # Bring reconstructed data to the origin for easier comparison
                            df = correct_perspective(df)
                            # Calculate all relevant errors
                            mae, rmse, std = compute_errors(df)

                            # add errors to the dataframe where we are accumulating them.
                            df_plot.loc[len(df_plot)] = [mad, mae, rmse, std]

    #############################################################################
    ###                             Results                                   ###
    #############################################################################
    # Save results
    df_plot.to_csv('./figure_14_dataset.csv', index=True)

    
