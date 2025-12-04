import pandas as pd
import itertools as it
import numpy as np
from scipy import stats
import ast
from os import listdir
import math

def file_read(file):
    """Reads in a single csv file, with 3 columns: time, pose (position of the insect, structured as 
    [top, middle, bottom]), and finally arduino data (stimulation side, frequency 
    of the stimulation and duration of the stimulation) """


    df = pd.read_csv(file)
    # Read in time, pose and arduino data. 
    time = df.get('time')
    differences = time.diff()
    fps = 1/differences.mean()


    pose_raw = df.get('pose')
    arduino_data = df.get('arduino_data')
    stim_deets = []     # Will contain a list of lists: [stimulation side, frequency] 
                        # or [None, None] if no stimulation has occured.
    stim_occur = []     #Simply a binary list denoting whether a stimulation has occured at specific time j. 

    pose = []
    # We iterate through both pose and arduino data lists. 
    for i, j in zip(pose_raw, arduino_data):

        # Convert the string representation of a list into an actual list
        i_list = ast.literal_eval(i)

        pose.append(i_list)
        # Check if arduino data is NOT an empty entry
        if isinstance(j, str) and j.strip():
            # If not, then append the stimulation information
            # print(file)
            try:
                # direction = j[0]
                # number = j[1:]
                # if direction == 'E': 
                #     direction = 'Both'
                # elif direction == 'A': 
                #     direction = 'Left'
                # elif direction == 'B': 
                #     direction ='Right'
                direction, number = j.split(", ")
                freq = int(number[:2])
                freq = int(number)
                stim_deets.append([direction, freq])
                stim_occur.append(1)
            except ValueError: 
                # If empty, no stimulation: [None, None, None]
                stim_deets.append([None, None])
                stim_occur.append(0)
        else: 
            # If empty, no stimulation: [None, None, None]
            stim_deets.append([None, None])
            stim_occur.append(0)


    # Return relevant lists. 
    return pose, stim_deets, stim_occur, fps



def exp_weighted_ma(part, alpha):
    """An application of an exponential weighted moving average filter to 
    smooth data - alpha close to 1 means minimal smoothing"""
    # Initialize empty lists to store x and y coordinates separately
    partx = []
    party = []
    
    # Separate x and y coordinates from the input list
    for i in part: 
        partx.append(i[0])
        party.append(i[1])
    
    # Convert lists to pandas Series for easier manipulation
    partx = pd.Series(partx)
    party = pd.Series(party)
    
    # Apply exponential weighted moving average to x coordinates
    # Round to 5 decimal places and convert back to list
    partx = round(partx.ewm(alpha, adjust=False).mean(), 5)
    partx = partx.tolist()

    # Apply exponential weighted moving average to y coordinates
    # Round to 5 decimal places and convert back to list
    party = round(party.ewm(alpha, adjust=False).mean(), 5)
    party = party.tolist()

    # Combine smoothed x and y coordinates back into a single list
    smooth_data = []
    for i in range(len(partx)):
        smooth_data.append([partx[i], party[i]])

    # Return the smoothed data
    return smooth_data

def remove_outliers_and_smooth(data, alpha=0.1, z_thresh=2):
    """
    Removes outliers from 2D data and applies EWMA smoothing.
    - data: list of [x, y] points
    - alpha: EWMA smoothing factor (0 < alpha <= 1)
    - z_thresh: z-score threshold for outlier detection
    """
    # Convert to numpy array for easier math
    arr = np.array(data)
    x, y = arr[:, 0], arr[:, 1]
    
    # Outlier detection using z-score
    z_x = np.abs(stats.zscore(x, nan_policy='omit'))
    z_y = np.abs(stats.zscore(y, nan_policy='omit'))
    mask = (z_x < z_thresh) & (z_y < z_thresh)
    
    # Remove outliers
    # x_clean, y_clean = x[mask], y[mask]
    
    # Or maybe we can replace outliers with NaN and interpolate 
    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[~mask] = np.nan
    y_clean[~mask] = np.nan
    x_clean = pd.Series(x_clean).interpolate().bfill().ffill().values
    y_clean = pd.Series(y_clean).interpolate().bfill().ffill().values
    
    # Apply EWMA smoothing
    x_smooth = pd.Series(x_clean).ewm(alpha=alpha, adjust=False).mean().values
    y_smooth = pd.Series(y_clean).ewm(alpha=alpha, adjust=False).mean().values
    
    # Combine back to [x, y] pairs
    smoothed_data = np.column_stack((x_smooth, y_smooth)).tolist()
    return smoothed_data # mask shows which points were kept


def remove_outliers_and_smooth_1d(data, alpha=0.1, z_thresh=2):
    """
    Removes outliers from 1D data and applies EWMA smoothing.
    - data: list of numeric values
    - alpha: EWMA smoothing factor (0 < alpha <= 1)
    - z_thresh: z-score threshold for outlier detection
    """
    arr = np.array(data, dtype=float)  # convert to float for NaN support

    # Outlier detection using z-score
    z = np.abs(stats.zscore(arr, nan_policy='omit'))
    mask = z < z_thresh

    # Replace outliers with NaN and interpolate
    clean = arr.copy()
    clean[~mask] = np.nan
    clean = pd.Series(clean).interpolate().bfill().ffill().values

    # Apply EWMA smoothing
    smooth = pd.Series(clean).ewm(alpha=alpha, adjust=False).mean().values

    return smooth.tolist()


def body_vel(pos, angles, fps):
    """Calculate the in-line and transverse velocities of the beetle.
    
    Args:
        pos: position as [[x, y], [x1, y1], .....]
        fps (int): frames per second that the data has been recorded at. 
    
    Returns:
        tuple: A tuple containing lists of in-line velocity and signed transverse velocity.
               Transverse velocity is negative in one direction and positive in the opposite direction.
    """

    body_v_in_line = []
    body_v_transverse = []

    for i in range(1, len(pos)):
        # Velocity vector of middle point
        delta = np.subtract(pos[i], pos[i-1])

        
        fwd_vector = [np.cos(np.radians(angles[i])), np.sin(np.radians(angles[i]))]
        norm = np.linalg.norm(fwd_vector)
        body_axis_unit = fwd_vector / norm

        # Calculate perpendicular vector to body axis (rotated 90 degrees CCW)
        perp_body_axis_unit = np.array([-body_axis_unit[1], body_axis_unit[0]])

        # Calculate velocities
        in_line_velocity = np.dot(delta, body_axis_unit)
        transverse_velocity = np.dot(delta, perp_body_axis_unit)

        pixels_per_mm = 4.1033
        scale_factor = fps/pixels_per_mm
        body_v_in_line.append(in_line_velocity * scale_factor)
        body_v_transverse.append(transverse_velocity * scale_factor)

        # Lateral velocity with sign using dot product with perpendicular axis
        lateral_velocity_signed = np.dot(delta, perp_body_axis_unit) * scale_factor
        body_v_transverse.append(lateral_velocity_signed)

    # Exponential smoothing
    alpha = 0.25
    body_v_in_line = pd.Series(body_v_in_line).ewm(alpha=alpha, adjust=False).mean()
    body_v_in_line = round(body_v_in_line, 5).tolist()
    
    body_v_transverse = pd.Series(body_v_transverse).ewm(alpha=alpha, adjust=False).mean()
    body_v_transverse = round(body_v_transverse, 5).tolist()
    
    # Normalization (baseline subtraction)
    ref_idx = int(0.1 * fps)
    if ref_idx >= len(body_v_in_line):
        ref_idx = 0

    ref_in_line = body_v_in_line[ref_idx]
    ref_transverse = body_v_transverse[ref_idx]

    body_v_in_line = [v - ref_in_line for v in body_v_in_line]
    body_v_transverse = [v - ref_transverse for v in body_v_transverse]

    return body_v_in_line, body_v_transverse



def get_body_angles(angles, fps):
    # Initialize normalized angles list with the first angle
    
    normalized_angles = [angles[0]] 
    
    # Normalize subsequent angles to avoid large jumps
    for i in range(1, len(angles)):
        # Calculate the difference between current and previous angle
        delta = angles[i] - angles[i - 1]
        
        # Adjust for jumps greater than 180 degrees
        # This ensures the smallest angle difference is always used
        delta = (delta + 180) % 360 - 180
        
        # Add the adjusted delta to the previous normalized angle
        normalized_angles.append(normalized_angles[-1] + delta)

    reference = reference = normalized_angles[int(0.15 * fps)]
    # Return the list of normalized angles
    return [angle - reference for angle in normalized_angles]


def get_ang_vel(angles, fps):
    """Calculate angular velocity (degs/s) from a list of angles over uniform time intervals (specify fps)."""
    if len(angles) < 2:
        return []  # Not enough data to calculate velocity

    angular_velocities = []
    time_interval = 1 / fps  # Time interval between measurements in seconds (100fps)

    for i in range(2, len(angles)):
        delta_angle = angles[i] - angles[i - 1]  # Change in angle
        angular_velocity = delta_angle / time_interval  # Angular velocity = delta_angle / delta_time
        angular_velocities.append(angular_velocity)

    return angular_velocities



def get_post_stim(pose, stim_deets, stim_occur, fps):
    """
    Extracts data occurring just before and after a stimulation.
    
    This function should be used BEFORE applying EWMA filters to extract moments of interest.
    The post_frames and pre_frames variables can be adjusted to change the extraction window.
    
    Args:
    pose (list): List containing x, y, angle details.
    stim_deets (list): List containing stimulation details.
    stim_occur (list): List indicating stimulation occurrences (1 for stimulation, 0 otherwise).
    fps (int): Frames per second of the recording.

    Returns:
    tuple: A tuple containing stimulation details and extracted body part coordinates.
    """
    #Define the dictionary
    stim_dict = {}

    # Define the extraction window
    post_frames = int(fps * 1.25) 
    pre_frames = int(fps * 0.15)   
    
    # Find indices where stimulation occurred
    stim_index = [i for i, x in enumerate(stim_occur) if x == 1]
    
    
    # Extract data for the last stimulation
    for stim in stim_index:
        start = stim - pre_frames
        end = stim + post_frames
        if start < 0 or end > len(pose): 
            continue
        # Extract body part coordinates within the defined window
        pose_sect = pose[stim-pre_frames:stim+post_frames]
        
        
        # Get stimulation details
        side = stim_deets[stim][0]
        freq = stim_deets[stim][1]

        if (side, freq) not in stim_dict: 
            stim_dict[(side, freq)] = []

        stim_dict[(side, freq)].append(pose_sect)

    return stim_dict


def statistical_significance(data1, data2): 
    # Convert data to numpy arrays
    array1 = np.array(data1)
    array2 = np.array(data2)

    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(array1, array2)

    # Print results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05  # Set your significance level
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the two groups.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two groups.")


            
def success_rate(forward_velocity, body_angles, vel_thresh=3, angle_thresh=4):
    """
    Args:
        forward_velocity, body_angles: dicts of lists (trials) per key
        vel_thresh: threshold for velocity increase (units)
        angle_thresh: threshold for angle change (degrees)
    Returns:
        fwd_vel_success, body_angle_success: lists of 1 (success) or 0 (fail) for each trial
    """
    fwd_vel_success = []
    body_angle_success = []

    # Helper function to compute baseline and during-stim medians
    def get_medians(trace):
        during_stim = trace[int(0.15/1.15*len(trace)):int(0.65/1.15*len(trace))]
        dur_stim = [abs(x) for x in during_stim]
        return 0, max(dur_stim) 

    # Forward velocity
    for key, trials in forward_velocity.items():
        for trial in trials:
            if key[1] == 40 or key[1] == 50:
                continue
            if key[0] == "Both":
                base, stim = get_medians(trial)
                
                if stim - base >= vel_thresh:
                    
                    fwd_vel_success.append(1)
                else:
                    fwd_vel_success.append(0)
            else: 
                continue

    # Body angle
    for key, trials in body_angles.items():
        for trial in trials:
            if key[0] == "Right" or key[0] == "Left":
                base, stim = get_medians(trial)
                if abs(stim - base) >= angle_thresh:
                    body_angle_success.append(1)
                else:
                    body_angle_success.append(0)
            else: 
                continue

    return fwd_vel_success, body_angle_success


def angle_interpolate(values): 
    angles_deg = [math.degrees(x) if x is not None else float('nan') for x in values]
    arr = np.array(angles_deg, dtype = np.float64)
    nans = np.isnan(arr)
    if nans.any(): 
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

    return arr.tolist()


def pos_interpolate(pos):
    """
    pos: list of [x, y] (with values or None)
    Returns a list of [x, y] with None values interpolated.
    """
    pos_array = np.array([
        [v if v is not None else np.nan for v in pt]
        for pt in pos
    ], dtype=np.float64)  # shape (n, 2)

    # Interpolate x and y independently
    for i in range(2):  # For x and y
        col = pos_array[:, i]
        nans = np.isnan(col)
        if nans.any() and (~nans).any():
            col[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                col[~nans]
            )
        pos_array[:, i] = col

    # Convert back to list of lists
    return pos_array.tolist()


def trial_is_outlier(angles, fwd_vel, key): 
    for i in range(len(angles) - 5): 
        if abs(angles[i] - angles[i+5]) > 40: 
            return True 
    if np.isnan(angles).all(): 
        return True
    
    if key[0] == "Right": 
        if angles[-1] > 5: 
            return True

    if key[0] == "Left": 
        if angles[-1] < -5:
            return True
        
    if key[0] == "Both": 
        if min(fwd_vel) < -5: 
            return True
    
    return False

def turning_fail(angles, key): 
    
    if key[0] == "Right": 
        if min(angles) > -3.2: 
            return True

    if key[0] == "Left": 
        if max(angles) < 3.2:
            return True

    return False


def elytra_fail(fwd_vel, key): 
    if key[0] == "Both": 
        if max(fwd_vel) < 18.2: 
            return True 
        
    return False


def up_down_antenna_edited(files): 
    # Declare empty lists to store data (optional if you want to store the results later)

    """ Angular bins assignment:
        Bin 1: [-45, 45)         # angles from -45 degrees (inclusive) up to 45 degrees (exclusive)
        Bin 2: [45, 135)         # angles from 45 degrees (inclusive) up to 135 degrees (exclusive)
        Bin 3: [-135, -45)       # angles from -135 degrees (inclusive) up to -45 degrees (exclusive)
        Bin 4: [135, 180) and [-180, -135)
        angles from 135 degrees (inclusive) up to 180 degrees (exclusive)
        AND angles from -180 degrees (inclusive) up to -135 degrees (exclusive)
        These two regions are combined into a single bin to handle angular wrapping."""
    bin_edges = [-180, -160, -20, 20, 160, 180]

    induced_angle = {}
    up_down_induced = {}

    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)
    
        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)


                initial_angle = angles[int(0.15 * fps)]
                
                disc_bin = np.digitize(initial_angle, bin_edges, right = False) - 1

                
                if disc_bin == 1:
                    if key[0] == "Right": 
                        dir = "up"
                    elif key[0] == "Left": 
                        dir = "down" 
                    elif key[0] == "Both": 
                        continue
                elif disc_bin == 3: 
                    if key[0] == "Right": 
                        dir = "down"
                    elif key[0] == "Left": 
                        dir = "up" 
                    elif key[0] == "Both": 
                        continue
                elif disc_bin == 0 or disc_bin == 2 or disc_bin == 4:
                    continue

                dir_key = (key[1], dir)

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)

                if trial_is_outlier(body_angle, key): 
                    continue

                during_stim = body_angle
                
                if key not in induced_angle: 
                    induced_angle[key] = []
                
                if dir_key not in up_down_induced: 
                    up_down_induced[dir_key]  = []

                if key[0] == "Right": 
                    induced_angle[key].append(np.abs(min(during_stim)))
                    up_down_induced[dir_key].append(np.abs(min(during_stim)))
                elif key[0] == "Left":
                    induced_angle[key].append(max(during_stim))
                    up_down_induced[dir_key].append(max(during_stim))
 
    return induced_angle, up_down_induced

def up_down_antenna(files):
    # Declare empty lists to store data (optional if you want to store the results later)
    num_bins = 2
    angle_min = -180
    angle_max = 180
    bin_edges = np.linspace(angle_min, angle_max, num_bins + 1)
    induced_angle = {}
    up_down_induced = {}

    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)
    
        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)


                initial_angle = angles[int(0.15 * fps)]
                
                disc_bin = (np.digitize([initial_angle], bin_edges) - 1).item()

                key = key + (disc_bin,)

                if -180 < initial_angle < 0: 
                    if key[0] == "Right": 
                        dir = "up"
                    elif key[0] == "Left": 
                        dir = "down"
                    elif key[0] == "Both": 
                        continue
                else:
                    if key[0] == "Right": 
                        dir = "up"
                    elif key[0] == "Left": 
                        dir = "down"
                    elif key[0] == "Both": 
                        continue

                dir_key = (key[1], dir)

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)

                if trial_is_outlier(body_angle, key): 
                    continue

                during_stim = body_angle
                
                if key not in induced_angle: 
                    induced_angle[key] = []
                
                if dir_key not in up_down_induced: 
                    up_down_induced[dir_key]  = []

                if key[0] == "Right": 
                    induced_angle[key].append(np.abs(min(during_stim)))
                    up_down_induced[dir_key].append(np.abs(min(during_stim)))
                elif key[0] == "Left":
                    induced_angle[key].append(max(during_stim))
                    up_down_induced[dir_key].append(max(during_stim))
 
    return induced_angle, up_down_induced


def up_down_elytra(files):
    # Declare empty lists to store data (optional if you want to store the results later)
    induced_angle = {}
    up_down_induced = {}

    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)
    
        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)
                #TODO: Clean up outlier trial

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)

                pos = [[item[0], item[1]] for item in pose_lst]
                pos = pos_interpolate(pos)
                pos = remove_outliers_and_smooth(pos, alpha=0.2, z_thresh=2.5)


                initial_angle = angles[int(0.15 / 1.15 * len(angles))]
                
                # TODO: Include detection of "Both" stimulation and bin to up / down dictionary
                if key[0] != "Both":
                    continue
                if (-220 < initial_angle <= -90) or (130 <= initial_angle <= 180):
                    dir = "down"
                elif(0 < initial_angle <= 50) or (-50 <= initial_angle <= 0):
                    dir = "up"
                else: 
                    continue

                dir_key = (key[1], dir)

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                in_line_vel, transv_vel = body_vel(pos, angles, fps)

                if trial_is_outlier(body_angle, in_line_vel, key): 
                    continue

                during_stim = in_line_vel[int(0.15 / 1.15 * len(in_line_vel)):int(0.65/1.15*len(in_line_vel))]
            
                
                if dir_key not in up_down_induced: 
                    up_down_induced[dir_key]  = []

                if key[0] == "Both": 
                    up_down_induced[dir_key].append(np.abs(max(during_stim)))

 
    return up_down_induced


def spread_freq(body_angle): 
    def iqr(values): 
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        """was 5 and 85 with return q3"""
        iqr = q3 - q1
        return iqr
    

    # Declare empty lists to store data (optional if you want to store the results later)
    body_angle_max = {}

    for key, value in body_angle.items(): 
        for lst in value: 
            if key[1] not in body_angle_max: 
                body_angle_max[key[1]] = []

            if key[0] == "Right": 
                body_angle_max[key[1]].append(np.abs(min(lst)))
            elif key[0] == "Left": 
                body_angle_max[key[1]].append(np.abs(max(lst)))
            elif key[0] == "Both": 
                continue 

    
    intra_freq_dist = {}

    for key, values in body_angle_max.items(): 
        if key not in intra_freq_dist: 
            intra_freq_dist[key] = []
        
        intra_freq_dist[key] = (np.median(values), iqr(values))

    return intra_freq_dist

def spread_freq_both(body_vel): 
    def iqr(values): 
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        """was 5 and 85 with return q3"""
        iqr = q3 - q1
        return iqr
    

    # Declare empty lists to store data (optional if you want to store the results later)
    body_vel_max = {}

    for key, value in body_vel.items(): 
        for lst in value: 
            if key[1] not in body_vel_max: 
                body_vel_max[key[1]] = []

            during_list = lst[int(0.15*len(lst)):int(0.65*len(lst))]
            if key[0] == "Right": 
                continue
            elif key[0] == "Left": 
                continue
            elif key[0] == "Both": 
                body_vel_max[key[1]].append(np.abs(max(during_list)))
                

    
    intra_freq_dist = {}

    for key, values in body_vel_max.items(): 
        if key not in intra_freq_dist: 
            intra_freq_dist[key] = []
        
        intra_freq_dist[key] = (np.median(values), iqr(values))
        
    return intra_freq_dist



def polar_dual_binning(files, num_bins, iqr_scale):


    bins = np.linspace(-180, 180, num_bins + 1)
    polar_dict_ratio = {}


    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)
    

                initial_angle = angles[int(0.15 * fps)]
                
                disc_bin = np.digitize(initial_angle, bins, right = False) - 1
                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)

                during_stim = body_angle[int(0.15*fps):int(0.65*fps)]

                if turning_fail(body_angle, key):
                    continue

                if trial_is_outlier(body_angle, body_angle, key): 
                    continue

                polar_key_max = (key[0], disc_bin)

                if polar_key_max not in polar_dict_ratio: 
                    polar_dict_ratio[polar_key_max]  = []
                
                
                if key[0] == "Right": 
                    induced_angle = np.abs(min(body_angle))
                    induced_stim = np.abs(min(during_stim))

                    ratio = induced_stim / induced_angle
                    polar_dict_ratio[polar_key_max].append(ratio)

                elif key[0] == "Left": 
                    induced_angle = np.abs(max(body_angle))
                    induced_stim = np.abs(max(during_stim))

                    ratio = induced_stim / induced_angle

                    polar_dict_ratio[polar_key_max].append(ratio)
                
                elif key[0] == "Both": 
                    continue

                
    return polar_dict_ratio


def polar_binning(files, num_bins, iqr_scale): 
    bins = np.linspace(-180, 180, num_bins + 1)
    polar_dict = {}
    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)


                initial_angle = angles[int(0.15 * fps)]
                
                disc_bin = np.digitize(initial_angle, bins, right = False) - 1

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                
                if turning_fail(body_angle, key):
                    continue

                if trial_is_outlier(body_angle, body_angle, key): 
                    continue


                
                polar_key = (key[0], disc_bin)
                
                if polar_key not in polar_dict: 
                    polar_dict[polar_key]  = []
                
                if key[0] == "Right": 
                    induced_angle = np.abs(min(body_angle))
                    median, iqr = iqr_scale[key[1]]
                    scaled = (induced_angle - median) / iqr
                    polar_dict[polar_key].append(scaled)

                elif key[0] == "Left": 
                    induced_angle = np.abs(max(body_angle))
                    median, iqr = iqr_scale[key[1]]
                    scaled = (induced_angle - median) / iqr
                    polar_dict[polar_key].append(scaled)
                
                elif key[0] == "Both": 
                    continue

                
    return polar_dict

def polar_binning_scatter(files, num_bins, iqr_scale): 
    bins = np.linspace(-180, 180, num_bins + 1)
    polar_dict = {}
    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)


                initial_angle = angles[int(0.15 * fps)]
                disc_bin = np.digitize(initial_angle, bins, right = False) - 1

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                
                if turning_fail(body_angle, key):
                    continue

                if trial_is_outlier(body_angle, body_angle, key): 
                    continue


                
                polar_key = (key[0], initial_angle)
                

                if polar_key not in polar_dict: 
                    polar_dict[polar_key]  = []
                
                if key[0] == "Right": 
                    induced_angle = np.abs(min(body_angle))
                    median, iqr = iqr_scale[key[1]]
                    scaled = (induced_angle - median) / iqr
                    polar_dict[polar_key].append(scaled)

                elif key[0] == "Left": 
                    induced_angle = np.abs(max(body_angle))
                    median, iqr = iqr_scale[key[1]]
                    scaled = (induced_angle - median) / iqr
                    
                
                    polar_dict[polar_key].append(scaled)
                
                elif key[0] == "Both": 
                    continue

                
    return polar_dict

def polar_binning_elytra(files, num_bins, iqr_scale): 
    bins = np.linspace(-180, 180, num_bins + 1)
    polar_dict = {}
    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)
                pos = [[item[0], item[1]] for item in pose_lst]
                pos = pos_interpolate(pos)


                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)


                initial_angle = angles[int(0.15 * fps)]
                
                # disc_bin = np.digitize(initial_angle, bins, right = False) - 1

                if -90 <= initial_angle <= 90: 
                    disc_bin = 0 
                else: 
                    disc_bin = 1
                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                in_line_vel, transv_vel = body_vel(pos, angles, fps)


                if turning_fail(body_angle, key):
                    continue

                if trial_is_outlier(body_angle, in_line_vel, key): 
                    continue


                during_stim = in_line_vel[int(0.15/1.15*len(in_line_vel)):int(0.65/1.15*len(in_line_vel))]
                polar_key = (key[0], disc_bin)
                
                if polar_key not in polar_dict: 
                    polar_dict[polar_key]  = []
                
                if key[0] == "Right": 
                    continue

                elif key[0] == "Left": 
                    continue
                
                elif key[0] == "Both": 
                    induced_speed = np.abs(max(in_line_vel))
                    median, iqr = iqr_scale[key[1]]
                    scaled = (induced_speed - median) / iqr
                    polar_dict[polar_key].append(scaled)

                
    return polar_dict


def stim_ratio(files):
    ratio_list = []
    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)

                angles = remove_outliers_and_smooth_1d(angles, alpha=0.5, z_thresh=2.5)
    
                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)

                during_stim = body_angle[int(0.15*fps):int(0.65*fps)]

                if turning_fail(body_angle, key):
                    continue

                if trial_is_outlier(body_angle, body_angle, key): 
                    continue
                
                
                if key[0] == "Right": 
                    induced_angle = np.abs(min(body_angle))
                    induced_stim = np.abs(min(during_stim))

                    ratio = induced_stim / induced_angle
                    ratio_list.append(ratio)

                elif key[0] == "Left": 
                    induced_angle = np.abs(max(body_angle))
                    induced_stim = np.abs(max(during_stim))

                    ratio = induced_stim / induced_angle

                    ratio_list.append(ratio)
                
                elif key[0] == "Both": 
                    continue

                
    return ratio_list