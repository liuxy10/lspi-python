import numpy as np
import os
from lspi.sample import Sample

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians

def plot_leg_sagittal_plane(knee_angles_, shank_angles_, st_ratio, thigh_length=0.4, shank_length=0.4, skip_rate= 1):
    """
    Plot leg positions in the sagittal plane with fixed ankle.
    
    Parameters:
    -----------
    knee_angles : array-like
        Array of knee angles in degrees for each frame (angle between thigh and shank)
    shank_angles : array-like
        Array of shank angles in degrees for each frame (angle between shank and vertical)
    thigh_length : float
        Length of the thigh segment in meters
    shank_length : float
        Length of the shank segment in meters
    
    Returns:
    --------
    hip_angles : array
        Calculated hip angles for each frame
    """
    # Ensure inputs are numpy arrays
    # num_pts = len(knee_angles) // skip_rate
    knee_angles = np.array(knee_angles_)[::skip_rate] 
    shank_angles = -np.array(shank_angles_)[::skip_rate]
    
    # Number of frames
    frames = len(knee_angles)

    base_hip = True
    
    if base_hip:
        # Fixed hip position
        hip_x = 0
        hip_y = 0
        hip_positions = [(0.,0.) for _ in range(frames)]
        # Lists to store positions and angles
        ankle_positions = []
        knee_positions = []
        hip_angles = []
        # Calculate positions and angles for each frame
        for i in range(0, frames):
            # Convert angles to radians
            knee_angle_rad = radians(knee_angles[i])
            shank_angle_rad = radians(shank_angles[i])
           
            thigh_angle_rad = shank_angle_rad + knee_angle_rad
            
            # Store hip angle (from vertical)
            hip_angle = np.degrees(thigh_angle_rad)
            hip_angles.append(hip_angle)
            # Calculate knee position
            knee_x = hip_x + thigh_length * sin(thigh_angle_rad)
            knee_y = hip_y - thigh_length * cos(thigh_angle_rad)
             
            # Calculate ankle position from ankle and shank angle
            ankle_x = knee_x + shank_length * sin(shank_angle_rad)
            ankle_y = knee_y - shank_length * cos(shank_angle_rad)
      
            
            # Store positions
            ankle_positions.append((ankle_x, ankle_y))
            knee_positions.append((knee_x, knee_y))

    else:

        # Fixed ankle position
        ankle_x = 0
        ankle_y = 0
        ankle_positions = [(0.,0.) for _ in range(frames)]
        # Lists to store positions and angles
        hip_positions = []
        knee_positions = []
        hip_angles = []
        # Calculate positions and angles for each frame
        for i in range(0, frames):
            # Convert angles to radians
            knee_angle_rad = radians(knee_angles[i])
            shank_angle_rad = radians(shank_angles[i])
            
            # Calculate knee position from ankle and shank angle
            knee_x = ankle_x + shank_length * sin(shank_angle_rad)
            knee_y = ankle_y + shank_length * cos(shank_angle_rad)
            
            # Calculate thigh angle from vertical
            # For an interior knee angle and shank angle from vertical,
            # calculate the thigh angle from vertical
            thigh_angle_rad = shank_angle_rad - (np.pi - knee_angle_rad)
            
            # Calculate hip position
            hip_x = knee_x + thigh_length * sin(thigh_angle_rad)
            hip_y = knee_y + thigh_length * cos(thigh_angle_rad)
            
            # Store hip angle (from vertical)
            hip_angle = np.degrees(thigh_angle_rad)
            hip_angles.append(hip_angle)
            
            # Store positions
            hip_positions.append((hip_x, hip_y))
            knee_positions.append((knee_x, knee_y))
        
    
    # First, visualize the spatial relationship between hip angle and frame
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(frames), hip_angles, 'b-', marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Hip Angle (degrees)')
    plt.title('Hip Angle Per Frame')
    plt.grid(True)
    
    # Fig 1 visualization of hip angle per frame
    plt.subplot(1, 2, 2)

    
    # Plot leg for each frame with color gradient
    for i in range(frames):
        hip_x, hip_y = hip_positions[i]
        ankle_x, ankle_y = ankle_positions[i]
        knee_x, knee_y = knee_positions[i]
        
        # Calculate color gradient from dark grey to light grey
        color_intensity = (0.75 - i / frames /2)  # Normalized intensity (0 to 1)
        # st_sw_id = frames /2
        if i < st_ratio * frames:
            color = (1, color_intensity, color_intensity)  # stance is marked by red
        else:
            color = (color_intensity, color_intensity, color_intensity)
        # Plot leg segments
        plt.plot([hip_x, knee_x], [hip_y, knee_y], color=color, linewidth=1)  # Thigh
        plt.plot([knee_x, ankle_x], [knee_y, ankle_y], color=color, linewidth=1)  # Shank
        plt.text(ankle_x + 0.1 * shank_length, ankle_y + 0.1 * shank_length, str(i), fontsize=8)
        # Plot foot position
        shank_angle_rad = radians(shank_angles[i])
        heel_x = ankle_x - 0.05 * cos(shank_angle_rad)  # Assuming foot length is 0.2m
        heel_y = ankle_y - 0.05 * sin(shank_angle_rad)
        toe_x = ankle_x + 0.1 * cos(shank_angle_rad)
        toe_y = ankle_y + 0.1 * sin(shank_angle_rad)
        
        # Plot foot as a line segment
        plt.plot([heel_x, toe_x], [heel_y, toe_y], color=color, linewidth=1)

    
    # Connect hip positions to show trajectory
    hip_x_values = [pos[0] for pos in hip_positions]
    hip_y_values = [pos[1] for pos in hip_positions]
    plt.plot(hip_x_values, hip_y_values, 'r-', linewidth=0.5)
    
    # Connect knee positions to show trajectory
    knee_x_values = [pos[0] for pos in knee_positions]
    knee_y_values = [pos[1] for pos in knee_positions]
    plt.plot(knee_x_values, knee_y_values, 'g-', linewidth=0.5)
    
    plt.axis('equal')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title(f'Leg Positions in Sagittal Plane with Fixed {"Ankle" if not base_hip else "Hip"}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return np.array(hip_angles)


if __name__ == "__main__":
    # Example data: 10 frames of a simple movement pattern
    # knee_angles = 180 - np.array([120, 130, 140, 150, 160, 170, 160, 150, 140, 130])  # Knee angle (degrees)
    # shank_angles = np.linspace(5, -20, 10)          # Shank angle from vertical (degrees)
    knee_angles =  np.load("knee_angle.npy") -20
    shank_angles = np.load("shank_angle.npy")
    # Call the function
    hip_angles = plot_leg_sagittal_plane(knee_angles, shank_angles, 0.6, skip_rate=5)