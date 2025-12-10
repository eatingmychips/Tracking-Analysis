#analysis\core_analysis.py
import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from scipy import signal, interpolate
from analysis.analysis_functions import *
from analysis.plotting_funcs import *
import statistics as stat
import matplotlib.patches as mpatches
import math
import json
from os import listdir
frequencies = [10, 20, 30, 40, 50]
######## Here we import the files necessary for analysis, we also import the representative files for gait plotting ########

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def run_analysis(directory):
    files = [directory + "\\" + x
            for x in find_csv_filenames(directory)]
    

    lateral_velocity = {}
    forward_velocity = {}
    body_angles = {}
    angular_velocity = {}

    dictionaries = [lateral_velocity, forward_velocity, body_angles, angular_velocity]
    #Declare trial_no variable
    turning_succ_no = 0
    turning_fail_no = 0 
    elytra_succ_no = 0 
    elytra_fail_no = 0 

    for file in files: 
        print(file)
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)
    
        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)
                
                
                pos = [[item[0], item[1]] for item in pose_lst]
                pos = pos_interpolate(pos)
                #Get middle and bottom points
                pos = remove_outliers_and_smooth(pos, alpha=0.2, z_thresh=2.5)
                angles = remove_outliers_and_smooth_1d(angles, alpha=0.2, z_thresh=2.5)

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                ang_vel = get_ang_vel(body_angle, fps)
                in_line_vel, transv_vel = body_vel(pos, angles, fps)

                for dict in dictionaries: 
                    if key not in dict: 
                        dict[key] = []

                # if turning_fail(body_angle, key):
                #     turning_fail_no += 1
                #     continue

                if trial_is_outlier(body_angle, in_line_vel, key): 
                    continue
                
                # if elytra_fail(in_line_vel, key): 
                #     elytra_fail_no += 1
                #     continue

                lateral_velocity[key].append(transv_vel)
                forward_velocity[key].append(in_line_vel)
                body_angles[key].append(body_angle)
                angular_velocity[key].append(ang_vel)
                
                if key[0] == "Both": 
                    elytra_succ_no += 1
                elif key[0] == "Right" or key[0] == "Left": 
                    turning_succ_no += 1

    lateral_max, fwd_max, angles_max, ang_vel_max = get_max_values(lateral_velocity, forward_velocity, body_angles, angular_velocity)
    print(elytra_succ_no)
    return {
        "body_angles": body_angles, 
        "angles_max": angles_max, 
        "forward_velocity": forward_velocity, 
        "fwd_max": fwd_max
    }

