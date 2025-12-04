import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from scipy import signal, interpolate
from analysis import *
import statistics as stat
import matplotlib.patches as mpatches
from os import listdir
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt


def antenna_time_plot(data_dict, frequencies, title):

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def process_list(data):
        data = np.array(data, dtype=object)
        max_len = max(len(sublist) for sublist in data)

        resampled_data = np.array([
        resample_1d_list(sublist, max_len) for sublist in data
            ], dtype=float)
        means = np.nanmean(resampled_data, axis=0)
        stds = np.nanstd(resampled_data, axis=0)
        lower = means - stds     # One std dev below the mean
        upper = means + stds     # One std dev above the mean
        return max_len, means, lower, upper

    axes_flat = axes.flatten()

    for idx, freq in enumerate(frequencies):
        ax = axes_flat[idx]

        
        # Use .get() with default empty list if key not found
        list1 = data_dict.get(("Right", freq), [])
        list2 = data_dict.get(("Left", freq), [])


        if len(list1) < 1 and len(list2) < 1:
            # If no data at all for this frequency, just create empty plot
            ax.set_title(f'Freq: {freq} Hz (No Data)', fontsize=18)
            ax.set_xlim(0, 1.15)
            ax.set_ylabel(title, fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue
    
        # Only process and plot if data exists
        if len(list1) > 0:
            max_len1, medians1, lower_quartiles1, upper_quartiles1 = process_list(list1)
            x1 = np.linspace(0, 1.15, max_len1)
            mask1 = (x1 >= 0.15) & (x1 <= 0.65)
            ax.fill_between(x1, lower_quartiles1, upper_quartiles1, color='lightgrey', alpha=0.3)
            ax.plot(x1, medians1, color='black', linewidth=2)
            ax.fill_between(x1[mask1], lower_quartiles1[mask1], upper_quartiles1[mask1], color='lightcoral', alpha=0.3)
            ax.plot(x1[mask1], medians1[mask1], color='red', linewidth=2, label='Right Stimulation')

        if len(list2) > 0:
            max_len2, medians2, lower_quartiles2, upper_quartiles2 = process_list(list2)
            x2 = np.linspace(0, 1.15, max_len2)
            mask2 = (x2 >= 0.15) & (x2 <= 0.65)
            ax.fill_between(x2, lower_quartiles2, upper_quartiles2, color='lightgrey', alpha=0.3)
            ax.plot(x2, medians2, color='black', linewidth=2)
            ax.fill_between(x2[mask2], lower_quartiles2[mask2], upper_quartiles2[mask2], color='lightgreen', alpha=0.3)
            ax.plot(x2[mask2], medians2[mask2], color='green', linewidth=2, label='Left Stimulation')

            


        # Formatting subplot
        ax.set_title(f'Freq: {freq} Hz', fontsize=18)
        ax.set_xlim(0, 1.1)
        ax.set_ylim(-45, 45)
        ax.set_ylabel(title, fontsize=16)
        if freq == 10:
            ax.legend(fontsize=14)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if len(frequencies) < len(axes_flat):
        fig.delaxes(axes_flat[-1])

    # Set x-label only to bottom row plots
    for i in range(len(axes_flat)):
        if i >= len(axes_flat) - 4:
            axes_flat[i].set_xlabel('Time (s)', fontsize=20)
                        # Set custom x-ticks and labels for this subplot
            xtick_positions = np.arange(0, 1.05, 0.2)  # Tick positions every 0.2 seconds
            xtick_labels = [f"{tick:.1f}" for tick in xtick_positions]  # Labels as strings
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)
    
    plt.tight_layout(h_pad=0.35)
    plt.show()
    
def resample_1d_list(original_list, new_len):
    # Remove invalid (non-float) entries
    filtered = [x for x in original_list if isinstance(x, (float, int, np.float32, np.float64))]
    old_len = len(filtered)
    if old_len == 0:
        # Return a list of np.nan if no valid numbers remain
        return [np.nan] * new_len
    if old_len == 1:
        return [filtered] * new_len
    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, new_len)
    return np.interp(new_idx, old_idx, filtered).tolist()


def get_max_values(lateral_vel, fwd_vel, body_angle, ang_vel):


    #fig, axes = plt.subplots(len(4), 1, figsize=(12, 25), sharex=True)

    all_measures = [lateral_vel, fwd_vel, body_angle, ang_vel]

    lateral_max = {}
    fwd_vel_max = {}
    body_angle_max = {}
    ang_vel_max = {}

    max_induced_dicts = [lateral_max, fwd_vel_max, body_angle_max, ang_vel_max]

    
    for unit, dict in zip(all_measures, max_induced_dicts): 
        for key, value in unit.items():
            for list in value: 
                during_stim = list[int(0.15/1.15*len(list)):int(0.65/1.15*len(list))]
                if key not in dict: 
                    dict[key]  = []
                if key[0] == "Right":           
                    dict[key].append(min(during_stim)) 
                elif key[0] == "Left": 
                    dict[key].append(max(during_stim))
                elif key[0] == "Both":
                    if dict is fwd_vel_max:
                        dict[key].append(abs(max(during_stim)))
                    else:
                        dict[key].append(max(during_stim, key=abs))

    return lateral_max, fwd_vel_max, body_angle_max, ang_vel_max


def frequency_plot(data_dict, frequencies, title):
    # Create a single figure for the boxplot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for boxplots
    box_data = []       # Combined data for all frequencies
    positions = []      # X-axis positions for boxplots
    colors = []         # Colors for each boxplot

    # Iterate through frequencies and collect data
    for idx, freq in enumerate(frequencies):

        list1 = data_dict.get(("Right", freq), [])
        list2 = data_dict.get(("Left", freq), [])
        if len(list1) > 0:  # Add "Right" data if available
            box_data.append(list1)
            positions.append(freq)  # X-axis position corresponds to frequency
            colors.append("red")  # Color for "Right"

        if len(list2) > 0:  # Add "Left" data if available
            box_data.append(list2)
            positions.append(freq)  # X-axis position corresponds to frequency
            colors.append("green")   # Color for "Left"

    # Plot the boxplots
    boxplots = ax.boxplot(box_data, positions=positions, patch_artist=True, widths = 3.5)
    ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 2)
    # Customize boxplot colors
    for patch, color in zip(boxplots['boxes'], colors):
        patch.set_facecolor(color)

    # Customize x-axis and labels
    ax.set_xticks(frequencies)  # Set x-ticks to frequencies
    ax.set_xticklabels(frequencies, fontsize=14)
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel(title, fontsize=16)
    ax.set_title("Boxplot by Frequency", fontsize=18)

    # Add a legend for "Right" and "Left"
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="red", lw=4, label="Right"),
            plt.Line2D([0], [0], color="green", lw=4, label="Left"),
        ],
        title="Side",
        loc="upper right",
        fontsize=12,
    )


    plt.tight_layout()
    plt.show()



def elytra_time_plot(data_dict, frequencies, title):

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def process_list(data):
            data = np.array(data, dtype=object)
            max_len = max(len(sublist) for sublist in data)

            resampled_data = np.array([
            resample_1d_list(sublist, max_len) for sublist in data
                ], dtype=float)
            means = np.nanmean(resampled_data, axis=0)
            stds = np.nanstd(resampled_data, axis=0)
            lower = means - stds     # One std dev below the mean
            upper = means + stds     # One std dev above the mean
            return max_len, means, lower, upper

    axes_flat = axes.flatten()
    for idx, freq in enumerate(frequencies):
        ax = axes_flat[idx]

        
        # Use .get() with default empty list if key not found
        list1 = data_dict.get(("Both", freq), [])

        if len(list1) < 1:
            # If no data at all for this frequency, just create empty plot
            ax.set_title(f'Freq: {freq} Hz (No Data)', fontsize=18)
            ax.set_xlim(0, 1.05)
            ax.set_ylabel('Lateral velocity\n(mm/s)', fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue

        if len(list1) > 0: 
            max_len1, medians1, lower_quartiles1, upper_quartiles1 = process_list(list1)

            x = np.linspace(0, 1.15, max_len1)
            mask = (x >= 0.1) & (x <= 0.6)

            # Right stimulation plot
            ax.fill_between(x[:len(medians1)], lower_quartiles1, upper_quartiles1,
                            color='lightgrey', alpha=0.3)
            ax.plot(x[:len(medians1)], medians1, color='black', linewidth=2)
            ax.fill_between(x[:len(medians1)][mask], lower_quartiles1[mask], upper_quartiles1[mask],
                            color='lightcoral', alpha=0.3)
            ax.plot(x[:len(medians1)][mask], medians1[mask],
                    color='red', linewidth=2, label='Both Elytra Stimulation')

        # Formatting subplot
        ax.set_title(f'Freq: {freq} Hz', fontsize=18)
        ax.set_xlim(0, 1.05)
        ax.set_ylabel(title, fontsize=16)
        if freq == 10:
            ax.legend(fontsize=14)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if len(frequencies) < len(axes_flat):
        fig.delaxes(axes_flat[-1])

    # Set x-label only to bottom row plots
    for i in range(len(axes_flat)):
        if i >= len(axes_flat) - 4:
            axes_flat[i].set_xlabel('Time (s)', fontsize=20)
                        # Set custom x-ticks and labels for this subplot
            xtick_positions = np.arange(0, 1.05, 0.2)  # Tick positions every 0.2 seconds
            xtick_labels = [f"{tick:.1f}" for tick in xtick_positions]  # Labels as strings
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)

    plt.tight_layout(h_pad=0.35)
    plt.show()


def frequency_plot_elytra(data_dict, frequencies, title):
    fig, ax = plt.subplots(figsize=(12, 8))

    box_data = []
    positions = []
    colors = []

    for freq in frequencies:
        list1 = data_dict.get(("Both", freq), [])
        if freq == 10: 
            print(np.percentile(list1, 10))
        if len(list1) > 0:
            box_data.append(list1)
            positions.append(freq)
            colors.append("grey")

    # Explicitly handle empty data scenario
    if len(box_data) == 0:
        print("No data available for any frequency.")
        return

    # Plot boxplots
    boxplots = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=5)

    # Customize colors
    for patch, color in zip(boxplots['boxes'], colors):
        patch.set_facecolor(color)

    # Set ticks to match positions exactly
    ax.set_xticks(positions)
    ax.set_xticklabels(positions, fontsize=14)

    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel(title, fontsize=16)
    ax.set_title("Boxplot by Frequency", fontsize=18)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="grey", lw=4, label="Both Elytra Stimulation")
        ],
        loc="upper right",
        fontsize=12,
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def antenna_trials_plot(data_dict, frequencies, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flatten()

    for idx, freq in enumerate(frequencies):
        ax = axes_flat[idx]
        # Get data for each frequency and side
        list1 = data_dict.get(("Right", freq), [])
        list2 = data_dict.get(("Left", freq), [])

        if len(list1) == 0 and len(list2) == 0:
            ax.set_title(f'Freq: {freq} Hz (No Data)', fontsize=18)
            ax.set_xlim(0, 1.05)
            ax.set_ylabel(title, fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue

        # Plot all Right stimulation trials (red)
        for trial in list1:
            trial = np.array(trial)
            x = np.linspace(0, 1.15, len(trial))
            ax.plot(x, trial, color='red', alpha=0.5, linewidth=1, label='Right Stimulation' if 'Right Stimulation' not in ax.get_legend_handles_labels()[1] else "")

        # Plot all Left stimulation trials (green)
        for trial in list2:
            trial = np.array(trial)
            x = np.linspace(0, 1.15, len(trial))
            ax.plot(x, trial, color='green', alpha=0.5, linewidth=1, label='Left Stimulation' if 'Left Stimulation' not in ax.get_legend_handles_labels()[1] else "")

        # Formatting subplot
        ax.set_title(f'Freq: {freq} Hz', fontsize=18)
        ax.set_xlim(0, 1.05)
        ax.set_ylabel(title, fontsize=16)
        if freq == 10:  # Show legend on one plot only
            ax.legend(fontsize=14)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Remove unused subplots if fewer than 6 frequencies
    if len(frequencies) < len(axes_flat):
        for i in range(len(frequencies), len(axes_flat)):
            fig.delaxes(axes_flat[i])

    # Set x-label only to bottom row plots
    for i in range(len(axes_flat)):
        if i >= len(axes_flat) - 3:
            axes_flat[i].set_xlabel('Time (s)', fontsize=20)
            xtick_positions = np.arange(0, 1.05, 0.2)
            xtick_labels = [f"{tick:.1f}" for tick in xtick_positions]
            axes_flat[i].set_xticks(xtick_positions)
            axes_flat[i].set_xticklabels(xtick_labels)

    plt.tight_layout(h_pad=0.35)
    plt.show()


def elytra_trials_plot(data_dict, frequencies, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flatten()

    for idx, freq in enumerate(frequencies):
        ax = axes_flat[idx]
        # Get data for each frequency
        list1 = data_dict.get(("Both", freq), [])

        if len(list1) == 0:
            ax.set_title(f'Freq: {freq} Hz (No Data)', fontsize=18)
            ax.set_xlim(0, 1.05)
            ax.set_ylabel(title, fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            continue

        # Plot all Both Elytra stimulation trials (blue)
        for trial in list1:
            trial = np.array(trial)
            x = np.linspace(0, 1.15, len(trial))
            # Only add label to the first line for the legend
            ax.plot(x, trial, color='blue', alpha=0.5, linewidth=1,
                    label='Both Elytra Stimulation' if 'Both Elytra Stimulation' not in ax.get_legend_handles_labels()[1] else "")

        # Formatting subplot
        ax.set_title(f'Freq: {freq} Hz', fontsize=18)
        ax.set_xlim(0, 1.05)
        ax.set_ylabel(title, fontsize=16)
        if freq == 10:
            ax.legend(fontsize=14)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Remove unused subplots if fewer than 6 frequencies
    if len(frequencies) < len(axes_flat):
        for i in range(len(frequencies), len(axes_flat)):
            fig.delaxes(axes_flat[i])

    # Set x-label only to bottom row plots
    for i in range(len(axes_flat)):
        if i >= len(axes_flat) - 3:
            axes_flat[i].set_xlabel('Time (s)', fontsize=20)
            xtick_positions = np.arange(0, 1.05, 0.2)
            xtick_labels = [f"{tick:.1f}" for tick in xtick_positions]
            axes_flat[i].set_xticks(xtick_positions)
            axes_flat[i].set_xticklabels(xtick_labels)

    plt.tight_layout(h_pad=0.35)
    plt.show()


