import json
import numpy as np
import pandas as pd
import cv2


def read_calibration_file(calib_filename):
    """
    Reads the coordinates for the 4 calibration points in:
    - Overhead 4K camera frame
    - lighthouse camera frame
    - Actual world coordinates in centimeters.
    """

    # Load the file with the corner locations
    with open(calib_filename) as f:
        data = json.load(f)

    # Read the corner locations
    c_px  = data['GX010145.MP4']['corners_px']
    c_cm  = data['GX010145.MP4']['corners_cm']
    c_lh2 = data['GX010145.MP4']['corners_lh2']

    # Define Source and Destination points for the Homography calculations
    pts_src_px  = np.array([c_px['tl'], c_px['tr'], c_px['br'], c_px['bl']])     # pixels
    pts_src_lh2 = np.array([c_lh2['tl'], c_lh2['tr'], c_lh2['br'], c_lh2['bl']]) # pixels
    pts_dst     = np.array([c_cm['tl'], c_cm['tr'], c_cm['br'], c_cm['bl']])     # centimeters

    return pts_src_px, pts_src_lh2, pts_dst


def camera_to_world_homography(input_df, src_corners, dst_corners):
    """
    Calculate the homography transformation between src_corners and dst_corners.
    And apply that transformation to input_df
    
    Parameters
    ----------
    input_df : dataframe with {'x', 'y'} columns
        points to transform.
    src_corners : array_like, shape(4,2)
        4 corresponding points in input_df' frame of reference
    dst_corners : array_like, shape(4,2)
        4 corresponding points in the target frame of reference

    Returns
    -------
    output_points : array_like, shape (N,2)
        points transformed to dst_corners frame of reference
    """

    # Calculate the Homography Matrix
    H, status = cv2.findHomography(src_corners, dst_corners)

    # Prepare pixel points to convert
    input_points = input_df[['x', 'y']].to_numpy().reshape((1,-1,2))
    # pts_example = np.array([[[200, 400], [1000, 1500], [3000, 2000]]], dtype=float)  # Shape of the input array must be (1, n_points, 2), note the double square brackets before and after the points.

    # Run the transformation
    output_points = cv2.perspectiveTransform(input_points, H)
    output_points = output_points.reshape((-1, 2))                  # We can reshape the output so that the points look like [[3,4], [1,4], [5,1]]
                                                                    # They are easier to work with like this, without all that double square bracket non-sense
    return output_points


def reorganize_data(xy_data, timestamp):
    """
    Create a dictionary of arrays to easily manipulate the data later on.
    With the following keys ('x', 'y', 't'), where 't' is in unix epoch

    Parameters
    ----------
    xy_data : array_like, float (N,2)
        X-Y position data
    timestamp : dataframe
        datarame coulmn with the data timestamp as strings of UTC datetimes

    Returns
    -------
    data : dictionary
        values:
            'x': array, float (N,)
                Original data X axis
            'y': array, float (N,)
                Original data Y axis
            'time': array, int (N,)
                Original data timestamps in unix epoch (microseconds)

    """
    # Convert the dataframe timestamp to an array of unix epochs
    time_s = pd.to_datetime(timestamp)
    time_s = time_s.apply(lambda x: x.timestamp())
    time_s = time_s.to_numpy()

    # Reorganize the data to make it easier to manipulate in numpy (it's easier to do linear interpolation in numpy, instead of pandas.)
    data = {'time':     time_s,
                'x':    xy_data[:,0],
                'y':    xy_data[:,1],}
    
    return data









def interpolate_camera_to_lh2(camera_data, lh2_data):
    """
    Interpolate the camera data to the lh2 timebase,
    so that a 1-to-1 accuracy comparison is possible. 

    Parameters
    ----------
    camera_data : array, shape (N,2)
        camera X-Y points
    camera_timebase : array, (N,)
        timestamps of the camera data in unisx epoch (microseconds)
    lh2_timebase : array, (M,)
        timestamps of the lh2 data in unisx epoch (microseconds)

    Returns
    -------
    interp_data: Dict
        Dictionary of numpy arrays with the interpolated LH2 data.
        values:
            'x': array, float (N,)
                Interpolated data X axis
            'y': array, float (N,)
                Interpolated data Y axis
            'time': array, int (N,)
                interpolated data timestamps in unix epoch (microseconds)

    """

    # Offset the camera timestamp to get rid of the communication delay.
    camera_data['time'] += 318109e-6 # seconds

    # Interpolate the camera data against the lh2
    interpolated_x = np.interp(lh2_data['time'], camera_data['time'],  camera_data['x'])
    interpolated_y = np.interp(lh2_data['time'], camera_data['time'],  camera_data['y'])

    # Put the interpolated data in a dictionary matching the structure of the input data.
    interp_data = {'time':  lh2_data['time'],
                    'x':    interpolated_x,
                    'y':    interpolated_y,}
    
    return interp_data