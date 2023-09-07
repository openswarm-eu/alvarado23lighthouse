import numpy as np
from functions.data_processing import   import_data, \
                                        LH2_count_to_pixels, \
                                        process_calibration, \
                                        solve_point_plane, \
                                        solve_2d_scene_get_Rtn, \
                                        scale_LH2_to_real_size, \
                                        scale_cam_to_real_size, \
                                        interpolate_cam_data, \
                                        correct_perspective

from functions.plotting import  plot_ALL_reconstructed_3D_scene, \
                                plot_error_histogram, \
                                plot_projected_LH_views, \
                                plot_reconstructed_3D_scene



#############################################################################
###                                Options                                ###
#############################################################################

# Choose which experiment to plot
experiment_number = 1

#############################################################################
###                                  Main                                 ###
#############################################################################

# file with the data to analyze
data_file = './dataset/lighthouse/data.csv'
experiment_file = f'./dataset/ground_truth/{experiment_number}_blender.csv'  
calib_file = './dataset/ground_truth/calibration.json'

if __name__ == "__main__":

    # Import data
    cam_data, data, calib_data = import_data(data_file, experiment_file, calib_file)

    # Separate the data from the 3 DotBots
    df = {  'R': data.loc[data['source'] == 'R'],
            'B': data.loc[data['source'] == 'B'],
            'G': data.loc[data['source'] == 'G']}
    
    # Separate the data from the 3 DotBots
    exp_data = {'R': cam_data.loc[cam_data['source'] == 'R'],
                'B': cam_data.loc[cam_data['source'] == 'B'],
                'G': cam_data.loc[cam_data['source'] == 'G']}
    
    # Iterate over all the available colors on the dataframe, ignore colors that do not appear in the experiment.
    for color in [c for c in df.keys() if not df[c].empty]:
        
        # COnvert the LH polynomial count into LH camera pixels
        pts_A = LH2_count_to_pixels(df[color]['LHA_count_1'].values, df[color]['LHA_count_2'].values, 0)
        pts_B = LH2_count_to_pixels(df[color]['LHB_count_1'].values, df[color]['LHB_count_2'].values, 1)
    
        # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
        df[color].loc[:,'LHA_proj_x'] = pts_A[:,0]
        df[color].loc[:,'LHA_proj_y'] = pts_A[:,1]
        df[color].loc[:,'LHB_proj_x'] = pts_B[:,0]
        df[color].loc[:,'LHB_proj_y'] = pts_B[:,1]

        # Solve the scene to find the transformation R,t from LHA to LHB
        solution_1, solution_2, zeta = solve_2d_scene_get_Rtn(pts_A, pts_B)
        t_star, R_star, n_star = solution_1

        # Transform LH projected points into 3D points
        point3D = solve_point_plane(n_star, zeta, pts_A)

        # Convert the for 4 calibration points from a LH projection to a 3D point
        calib_data = process_calibration(n_star, zeta, calib_data)

        # Scale up the LH2 points
        lh2_scale, calib_data, point3D = scale_LH2_to_real_size(calib_data, point3D)
        # Scale up the camera points
        calib_data, exp_data[color] = scale_cam_to_real_size(calib_data, exp_data[color])

        df[color]['LH_x'] = point3D[:,0]
        df[color]['LH_y'] = point3D[:,1]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
        df[color]['LH_z'] = point3D[:,2]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height
        
        # Interpolate Camera data to match the time base of the LH2 data
        exp_data[color] = interpolate_cam_data(df[color], exp_data[color]) 

        # Find the transform that superimposes one dataset over the ground truth.
        exp_data[color] = correct_perspective(calib_data, exp_data[color])

        #############################################################################
        ###                             Plotting                                  ###
        #############################################################################
        # Plot Error Histogram
        plot_error_histogram(df[color], exp_data[color])

        # Plot 3D reconstructed scene
        plot_reconstructed_3D_scene(point3D, t_star * lh2_scale, R_star, calib_data, exp_data[color])

        # Plot projected views of the lighthouse
        plot_projected_LH_views(pts_A, pts_B)

    # Plot all the trahectories at the same time.
    plot_ALL_reconstructed_3D_scene(df, exp_data, t_star * lh2_scale, R_star, calib_data, experiment_file=experiment_file)
