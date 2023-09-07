import json
import numpy as np
import pandas as pd
from dateutil import parser
import cv2

def import_data(data_file, experiment_file, calib_file):

    # Read the files.
    data = pd.read_csv(data_file, index_col=0, parse_dates=['timestamp'])
    exp_data = pd.read_csv(experiment_file, parse_dates=['timestamp'])
    with open(calib_file, 'r') as json_file:
        calib_data = json.load(json_file)
    lh2_calib_time = calib_data["GX010146.MP4"]["timestamps_lh2"]

    # Add a Z=0 axis to the Camera (X,Y) coordinates.
    exp_data['z'] = 0.0

    # Convert the strings to datetime objects
    for key in lh2_calib_time:
        lh2_calib_time[key] = [parser.parse(ts) for ts in lh2_calib_time[key]]

    # Get a timestamp column of the datetime.
    for df in [data, exp_data]:
        df['time_s'] = df['timestamp'].apply(lambda x: x.timestamp() )

    # Convert pixel corners to numpy arrays
    for corner in ['tl', 'tr', 'bl', 'br']:
        calib_data["GX010146.MP4"]['corners_px'][corner] = np.array(calib_data["GX010146.MP4"]['corners_px'][corner])

    # Slice the calibration data and add it to the  data dataframe.
    tl = data.loc[ (data['timestamp'] > lh2_calib_time["tl"][0]) & (data['timestamp'] < lh2_calib_time["tl"][1])].mean(axis=0, numeric_only=True)
    tr = data.loc[ (data['timestamp'] > lh2_calib_time["tr"][0]) & (data['timestamp'] < lh2_calib_time["tr"][1])].mean(axis=0, numeric_only=True)
    bl = data.loc[ (data['timestamp'] > lh2_calib_time["bl"][0]) & (data['timestamp'] < lh2_calib_time["bl"][1])].mean(axis=0, numeric_only=True)
    br = data.loc[ (data['timestamp'] > lh2_calib_time["br"][0]) & (data['timestamp'] < lh2_calib_time["br"][1])].mean(axis=0, numeric_only=True)
    
    # Slice the lh2 data to match the timestamps on the blender experiment
    start = exp_data['timestamp'].iloc[0 + 18]  # Use the a point about 250ms later than the start of the dataset, to address the time delay correction that we will do later when we interpolate the data.
    end   = exp_data['timestamp'].iloc[-1]
    data = data.loc[ (data['timestamp'] > start) & (data['timestamp'] < end)]

    # Save the calibration data.
    calib_data["GX010146.MP4"]['corners_lh2_count'] = {'tl':tl,
                                 'tr':tr,
                                 'bl':bl,
                                 'br':br,
                                 }

    return exp_data, data, calib_data["GX010146.MP4"]

def LH2_count_to_pixels(count_1, count_2, mode):
    """
    Convert the sweep count from a single lighthouse into pixel projected onto the LH2 image plane
    ---
    count_1 - int - polinomial count of the first sweep of the lighthouse
    count_2 - int - polinomial count of the second sweep of the lighthouse
    mode - int [0,1] - mode of the LH2, let's you know which polynomials are used for the LSFR. and at which speed the LH2 is rotating.
    """
    periods = [959000, 957000]

    # Translate points into position from each camera
    a1 = (count_1*8/periods[mode])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
    a2 = (count_2*8/periods[mode])*2*np.pi   

    # Transfor sweep angles to azimuth and elevation coordinates
    azimuth   = (a1+a2)/2 
    elevation = np.pi/2 - np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6)) 

    # Project the angles into the z=1 image plane
    pts_lighthouse = np.zeros((len(count_1),2))
    for i in range(len(count_1)):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(a2[i]/2-a1[i]/2-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

    # Return the projected points
    return pts_lighthouse

def solve_2d_scene_get_Rtn(pts_a, pts_b):

    # gg=cv2.findHomography(pts_a,pts_b, method = cv2.RANSAC, ransacReprojThreshold = 3)
    gg=cv2.findHomography(pts_a, pts_b, cv2.FM_LMEDS)

    homography_mat = gg[0]
    U, S, V = np.linalg.svd(homography_mat)

    V = V.T

    s1 = S[0]/S[1]
    s3 = S[2]/S[1]
    zeta = s1-s3
    a1 = np.sqrt(1-s3**2)
    b1 = np.sqrt(s1**2-1)

    def unitize(x_in,y_in):
        magnitude = np.sqrt(x_in**2 + y_in**2)
        x_out = x_in/magnitude
        y_out = y_in/magnitude

        return x_out, y_out  

    a, b = unitize(a1,b1)
    c, d = unitize(1+s1*s3,a1*b1)
    e, f = unitize(-b/s1,-a/s3)

    v1 = np.array(V[:,0])
    v3 = np.array(V[:,2])
    n1 = b*v1-a*v3
    n2 = b*v1+a*v3

    R1 = np.matmul(np.matmul(U,np.array([[c,0,d], [0,1,0], [-d,0,c]])),V.T)
    R2 = np.matmul(np.matmul(U,np.array([[c,0,-d], [0,1,0], [d,0,c]])),V.T)
    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    # print(R1)

    t1 = (e*v1+f*v3).reshape((3,1))
    t2 = (e*v1-f*v3).reshape((3,1))

    # if (n1[2]<0):
    #     t1 = -t1
    #     n1 = -n1
    # elif (n2[2]<0):
    #     t2 = -t2
    #     n2 = -n2

    fil = {'R1' : R1,
           'R2' : R2,
           'n1' : n1,
           'n2' : n2,
           't1' : t1,
           't2' : t2,
           'zeta': zeta}

    # Check which of the solutions is the correct one. 

    solution_1 = (t1, R1, n1)   
    solution_2 = (t2, R2, n2)   
    

    return solution_1, solution_2, zeta

def solve_point_plane(n_star, zeta, pts):
    
    # Extend the points to homogeneous coordinates.
    pts_hom = np.hstack((pts, np.ones((len(pts),1))))

    # Get the scaling factor for every point in the LH image plane.
    scales = (1/zeta) / (n_star @ pts_hom.T)
    # scale the points
    scales_matrix = np.vstack((scales,scales,scales))
    final_points = scales_matrix*pts_hom.T
    final_points = final_points.T

    return final_points

def process_calibration(n_star, zeta, calib_data):
    
    # Create the nested dictionary structure needed
    calib_data['corners_lh2_proj'] = {}
    calib_data['corners_lh2_proj']['LHA'] = {}
    calib_data['corners_lh2_proj']['LHB'] = {}

    calib_data['corners_lh2_3D'] = {}


    # Project calibration points 
    for corner in ['tl','tr','bl','br']:
        # Project the points
        c1a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_1']])
        c2a = np.array([calib_data['corners_lh2_count'][corner]['LHA_count_2']])
        c1b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_1']])
        c2b = np.array([calib_data['corners_lh2_count'][corner]['LHB_count_2']])
        pts_A = LH2_count_to_pixels(c1a, c2a, 0)
        pts_B = LH2_count_to_pixels(c1b, c2b, 1)

        # Add it back to the calib dictionary
        calib_data['corners_lh2_proj']['LHA'][corner] = pts_A
        calib_data['corners_lh2_proj']['LHB'][corner] = pts_B

        # Reconstruct the 3D points
        point3D = solve_point_plane(n_star, zeta, pts_A)

        # Add the 3D points back to the  dictionary
        calib_data['corners_lh2_3D'][corner] = point3D

    return calib_data

def scale_LH2_to_real_size(calib_data, point3D):
    
    # Grab the point at top-left and bottom-right and scale them to be the corners of a 40cm square and use them to calibrate/scale the system.
    scale_p1 = calib_data['corners_lh2_3D']['tl']
    scale_p2 = calib_data['corners_lh2_3D']['br']
    scale = np.sqrt(2) * 40 / np.linalg.norm(scale_p2 - scale_p1)
    # Scale all the 3D points
    point3D *= scale
    
    # scale the calibration points
    calib_data['corners_lh2_3D_scaled'] = {}
    calib_data['corners_lh2_3D_scaled']['tl'] = calib_data['corners_lh2_3D']['tl'] * scale
    calib_data['corners_lh2_3D_scaled']['tr'] = calib_data['corners_lh2_3D']['tr'] * scale
    calib_data['corners_lh2_3D_scaled']['bl'] = calib_data['corners_lh2_3D']['bl'] * scale
    calib_data['corners_lh2_3D_scaled']['br'] = calib_data['corners_lh2_3D']['br'] * scale

    # Return scaled up scene
    return scale, calib_data, point3D

def scale_cam_to_real_size(calib_data, exp_data):
    
    pts_src = np.array([calib_data['corners_px']['tl'], calib_data['corners_px']['tr'], calib_data['corners_px']['br'], calib_data['corners_px']['bl']])
    pts_dst = np.array([[0.0, 40.0, 0.0], [40.0, 40.0, 0.0], [40.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # h, status = cv2.getPerspectiveTransform(pts_src, pts_dst)
    h, status = cv2.findHomography(pts_src[:,:2], pts_dst[:,:2]) # Only grab X,Y axis, because findhomography needs 2d points

    homo_pts = cv2.perspectiveTransform(exp_data[['x', 'y']].values.reshape((-1,1,2)), h).reshape((-1,2))

    # Update data frame of experiment data.
    exp_data['x'] = homo_pts[:,0]
    exp_data['y'] = homo_pts[:,1]
    
    # scale the calibration points
    calib_data['corners_px_scaled'] = {}
    calib_data['corners_px_scaled']['tl'] = pts_dst[0]
    calib_data['corners_px_scaled']['tr'] = pts_dst[1]
    calib_data['corners_px_scaled']['br'] = pts_dst[2]
    calib_data['corners_px_scaled']['bl'] = pts_dst[3]

    # Return scaled up scene
    return calib_data, exp_data

def interpolate_cam_data(lh_data, exp_data):


    camera_np = {'time': exp_data['time_s'].to_numpy(),
                'x':    exp_data['x'].to_numpy(),
                'y':    exp_data['y'].to_numpy(),}


    lh2_np = {'time':   lh_data['time_s'].to_numpy(),
                'x':    lh_data['LH_x'],
                'y':    lh_data['LH_y'],}


    # Offset the camera timestamp to get rid of the communication delay.
    camera_np['time'] += 265000e-6 # seconds
    

    camera_np['x_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['x'])
    camera_np['y_interp_lh2'] = np.interp(lh2_np['time'], camera_np['time'],  camera_np['y'])


    exp_data_interp = pd.DataFrame({
                          'time_s' : lh2_np['time'],

                          'source': exp_data['source'].iloc[0],

                          'x': camera_np['x_interp_lh2'],

                          'y': camera_np['y_interp_lh2'],

                          'z': 0.0}
                          )

    return exp_data_interp

def correct_perspective(calib_data, exp_data):
    """
    THE SVD TECHNIQUE  FAILED, SO I HARDCODED T,R THAT I COMPUTED WITH MONTECARLO
    Create a rotation and translation vector to move the reconstructed grid onto the origin for better comparison.
    Using an SVD, according to: https://nghiaho.com/?page_id=671
    """
    
    A = np.array([calib_data['corners_px_scaled'][corner] for corner in ['tl', 'tr', 'bl', 'br']]).reshape((4,3))
    B = np.array([calib_data['corners_lh2_3D_scaled'][corner] for corner in ['tl', 'tr', 'bl', 'br']]).reshape((4,3))

    # Get  all the reconstructed points
    A2 = exp_data[['x','y','z']].to_numpy().T

    # Convert the point to column vectors,
    # to match twhat the SVD algorithm expects
    A = A.T
    B = B.T


    # Get the centroids
    A_centroid = A.mean(axis=1).reshape((-1,1))
    B_centroid = B.mean(axis=1).reshape((-1,1))

    # Get H
    H = (A - A_centroid) @ (B - B_centroid).T

    # Do the SVD
    U, S, V = np.linalg.svd(H)

    # Get the rotation matrix
    R = V @ U.T

    # check for errors, and run the correction
    if np.linalg.det(R) < 0:
        U, S, V = np.linalg.svd(R)
        V[:,2] = -1*V[:,2]
        R = V @ U.T

    # Get the ideal translation
    t = B_centroid - R @ A_centroid

    # The Kabsch algorithm failed to find the correct transformation for a particular scenario.
    # So we found the correct  transformation by Monte Carlo and override the results with the following matrices
    R1 = np.array(  [[ 0.98919328, -0.12556392,  0.07569918,],
                    [ 0.1027148 ,  0.22505302, -0.96891734,],
                    [ 0.10462473,  0.96622194,  0.23551821,]])
    
    R2 = np.array([[ 0.99979044,  0.01958034,  0.00597359],
                   [-0.01957252,  0.99980751, -0.00136582],
                   [-0.00599918,  0.00124862,  0.99998123]])
    
    R3 = np.array([[ 9.99999540e-01, -1.49610272e-04, -9.47844489e-04],
                   [ 1.49625869e-04,  9.99999989e-01,  1.63844060e-05],
                   [ 9.47842027e-04, -1.65262205e-05,  9.99999551e-01]])

    R = R3 @ R2 @ R1
    
    t = B_centroid - R @ A_centroid

    correct_points = (R@A2 + t)
    correct_points = correct_points.T

    correct_corners = (R@A + t).T
    # correct_corners = (A).T

    # Update dataframe
    exp_data['Rt_x'] = correct_points[:,0]
    exp_data['Rt_y'] = correct_points[:,1]
    exp_data['Rt_z'] = correct_points[:,2]

    # Add information to the calib data dictionary
    calib_data['corners_px_Rt'] = {}
    calib_data['corners_px_Rt']['tl'] = correct_corners[0]
    calib_data['corners_px_Rt']['tr'] = correct_corners[1]
    calib_data['corners_px_Rt']['bl'] = correct_corners[2]
    calib_data['corners_px_Rt']['br'] = correct_corners[3]

    return exp_data
