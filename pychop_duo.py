"""
Script for Synchronizing and Combining Frames from Two Videos Based on Drift and Timestamps

Description:
This script processes two videos recorded from different cameras (e.g., a SeaViewer and a GoPro)
and synchronizes their frames based on drift (distance traveled) and timestamps.
It extracts frames every 5 meters of drift, along with frames before and after the drift point,
and combines them into a single image for comparison.

Parameters to Modify:
- `shiplog_file`: Path to the ship log data file.
- `video1_path`: Path to the first video file (SeaViewer).
- `video2_path`: Path to the second video file (GoPro).
- `video2_timeshift`: Time shift in seconds to adjust the GoPro video's timestamps for synchronization.

Note:
- Time from Video 1 is considered accurate and well-synchronized with the ship log.
- Time from Video 2 (GoPro) may not be well-synchronized and can be adjusted using `video2_timeshift`.

Disclaimer:
This script was written with the help of an AI assistant and should not be used for machine learning education purposes.
"""

import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyproj import Transformer
import subprocess
import json
import os
from read_shiplog import read_shiplog
from recording_start_time import get_recording_start_time




# Parameters to modify
shiplog_file = "./data/EMB345_1s_16072024-01082024.dat"
video1_path = './data/T0_C_SeaViewer/20240717-140600MA.AVI'
video2_path = './data/T0_C_GOPRO/GH013919.MP4'
video2_timeshift = 38  # Time shift in seconds for Video 2 (GoPro)



drift_interval = 1  # meters
delta_frame = 10  # Number of frames before and after the drift point

# --- Step 1: Load Ship Log Data and Define Paths ---
# Load the ship log data
df = read_shiplog(shiplog_file)

# Ensure 'date time' is in datetime format and set as index
df['date time'] = pd.to_datetime(df['date time'])
df = df.set_index('date time').sort_index()

print('====> Read ship log data successfully.')

# --- Step 2: Extract Start Times for Both Videos ---

# Video 1: Start time from filename
video1_name = os.path.basename(video1_path)
start_time_str = video1_name[:15]  # '20240717-140600'
video1_start_time = datetime.strptime(start_time_str, '%Y%m%d-%H%M%S')

# Video 2: Start time from metadata using ffprobe
video2_name = os.path.basename(video2_path)

video2_start_time = get_recording_start_time(video2_path)

# Apply time shift to Video 2
video2_start_time += timedelta(seconds=video2_timeshift)

print(f"Video 1 Start Time: {video1_start_time}")
print(f"Video 2 Start Time: {video2_start_time}")
print(f"Time Shift for Video 2: {video2_timeshift} seconds")
print('====>  Extracted start times successfully.')
# --- Step 3: Determine Earliest Start Time ---

# Determine the earliest start time
earliest_start_time = min(video1_start_time, video2_start_time)

# Calculate time offsets for each video
video1_time_offset = (video1_start_time - earliest_start_time).total_seconds()
video2_time_offset = (video2_start_time - earliest_start_time).total_seconds()

# --- Step 4: Process Videos and Calculate Drift Based on Common Time Base ---

def process_video(video_path, time_offset, df, video_name):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare frame timestamps adjusted by time offset
    frame_numbers = range(total_frames)
    frame_times = [earliest_start_time + timedelta(seconds=(i / fps) + time_offset) for i in frame_numbers]
    
    # Create a DataFrame of frame timestamps
    frame_timestamps = pd.DataFrame({'timestamp': frame_times})
    
    # Reindex your DataFrame to include frame timestamps and interpolate
    combined_index = df.index.union(frame_times)
    df_interpolated = df.reindex(combined_index).sort_index().interpolate(method='time')
    
    # Extract interpolated coordinates and heading at frame timestamps
    coords = df_interpolated.loc[frame_times, ['SYS_STR_PosLat_dec', 'SYS_STR_PosLon_dec', 'Gyro_GPS_GPHDT_Heading']].reset_index()
    coords.columns = ['timestamp', 'latitude', 'longitude', 'heading']
    
    # Coordinate conversion to UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    coords['x'], coords['y'] = transformer.transform(coords['longitude'].values, coords['latitude'].values)
    
    # Apply heading correction
    coords['heading_rad'] = np.deg2rad(coords['heading'])
    coords['delta_x'] = -33.2 * np.sin(coords['heading_rad'])
    coords['delta_y'] = -33.2 * np.cos(coords['heading_rad'])
    coords['x_new'] = coords['x'] + coords['delta_x']
    coords['y_new'] = coords['y'] + coords['delta_y']
    
    # Add frame numbers
    coords['frame_number'] = pd.Series(frame_numbers, dtype=int)
    
    # Close the video capture
    cap.release()
    
    return coords, fps

# Process both videos
coords1, fps1 = process_video(video1_path, video1_time_offset, df, video1_name)
coords2, fps2 = process_video(video2_path, video2_time_offset, df, video2_name)
print('====> Processed videos successfully.')
# --- Step 5: Combine Coordinates from Both Videos and Calculate Drift ---

# Concatenate coordinates from both videos
combined_coords = pd.concat([coords1[['timestamp', 'x_new', 'y_new']],
                             coords2[['timestamp', 'x_new', 'y_new']]]).drop_duplicates().sort_values('timestamp')

# Calculate distances between consecutive points
combined_coords['delta_x_new'] = combined_coords['x_new'].diff()
combined_coords['delta_y_new'] = combined_coords['y_new'].diff()
combined_coords['distance'] = np.sqrt(combined_coords['delta_x_new']**2 + combined_coords['delta_y_new']**2)
combined_coords['distance'].fillna(0, inplace=True)
combined_coords['drift'] = combined_coords['distance'].cumsum()

# --- Step 6: Map Drift Back to Each Video's Frames ---

# Map drift back to coords1 and coords2 based on timestamps
coords1 = pd.merge_asof(coords1.sort_values('timestamp'), combined_coords[['timestamp', 'drift']], on='timestamp', direction='nearest')
coords2 = pd.merge_asof(coords2.sort_values('timestamp'), combined_coords[['timestamp', 'drift']], on='timestamp', direction='nearest')

# Reset indices for position-based indexing
coords1 = coords1.reset_index(drop=True)
coords2 = coords2.reset_index(drop=True)

print('====> Mapped drift back to each video successfully.')
# --- Step 7: Select Frames Every 5 Meters of Drift ---


max_index1 = len(coords1) - 1
max_index2 = len(coords2) - 1

# Maximum drift to consider
max_drift = combined_coords['drift'].max()


drift_points = np.arange(0, max_drift + drift_interval, drift_interval)

selected_frames = []

for point in drift_points:
    # Find the closest frame in coords1
    idx1_list = coords1[coords1['drift'] >= point].index
    if idx1_list.empty:
        continue  # No more frames in video 1
    idx1 = idx1_list.min()
    idx1 = int(idx1)
    
    idx_prev1 = idx1 - delta_frame if idx1 - delta_frame >= 0 else idx1
    idx_next1 = idx1 + delta_frame if idx1 + delta_frame <= max_index1 else idx1
    
    frames1 = coords1.iloc[[idx_prev1, idx1, idx_next1]]
    
    # Find the closest frame in coords2
    idx2_list = coords2[coords2['drift'] >= point].index
    if idx2_list.empty:
        continue  # No more frames in video 2
    idx2 = idx2_list.min()
    idx2 = int(idx2)
    
    idx_prev2 = idx2 - delta_frame if idx2 - delta_frame >= 0 else idx2
    idx_next2 = idx2 + delta_frame if idx2 + delta_frame <= max_index2 else idx2
    
    frames2 = coords2.iloc[[idx_prev2, idx2, idx_next2]]
    
    # Append to the list
    selected_frames.append({
        'point': point,
        'frames1': frames1,
        'frames2': frames2
    })
print(f"====> Selected frames every {drift_interval} meters of drift successfully.")
# --- Step 8: Read Frames and Create Combined Images ---

# Directories to save frames
output_dir = f"combined_{video1_name}_{video2_name}".replace('.', '_').replace(' ', '_')
os.makedirs(output_dir, exist_ok=True)

print(f"====> Saving combined images to: \n            {output_dir} ")
# Open video captures again
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

combined_frame_data = []
counter = 0
for item in selected_frames:
    counter += 1
    if counter % 10 == 0:
        print(f"Processing frame {counter} of {len(selected_frames)}")
        print()

    point = item['point']
    frames1 = item['frames1']
    frames2 = item['frames2']
    
    images = []
    
    # Process frames from Video 1
    for idx in frames1['frame_number']:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap1.read()
        if not ret:
            # Use the middle frame
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frames1['frame_number'].iloc[1])
            ret, frame = cap1.read()
            if not ret:
                # Create a black image if frame still not available
                frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        images.append(frame)
    
    # Process frames from Video 2
    for idx in frames2['frame_number']:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap2.read()
        if not ret:
            # Use the middle frame
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frames2['frame_number'].iloc[1])
            ret, frame = cap2.read()
            if not ret:
                # Create a black image if frame still not available
                frame_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        images.append(frame)
    
    # Ensure all frames are read
    if any(img is None for img in images):
        continue  # Skip if any frame is missing

    # Resize frames to match dimensions
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]
    
    # Arrange images in 2 rows and 3 columns
    first_row = np.hstack(resized_images[:3])
    second_row = np.hstack(resized_images[3:])
    combined_image = np.vstack([second_row, first_row])
    
    # Save combined image with specified naming convention
    timestamp_str = frames1.iloc[1]['timestamp'].strftime('%Y%m%d_%H%M%S')
    filename = f"{int(point):06d}m_{timestamp_str}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, combined_image)
    
    # Collect data
    combined_frame_info = {
        'drift_point': point,
        'filename': filename,
        'video1_frames': frames1['frame_number'].tolist(),
        'video2_frames': frames2['frame_number'].tolist(),
        'timestamp': frames1.iloc[1]['timestamp'],  # Use the middle frame's timestamp
        'latitude': frames1.iloc[1]['latitude'],  # Use the middle frame's latitude
        'longitude': frames1.iloc[1]['longitude'],  # Use the middle frame's longitude
        'heading': frames1.iloc[1]['heading'],  # Use the middle frame's heading
        'x': frames1.iloc[1]['x_new'],  # Use the middle frame's x coordinate
        'y': frames1.iloc[1]['y_new'],  # Use the middle frame's y coordinate
        'drift': frames1.iloc[1]['drift']  # Use the middle frame's drift
    }
    combined_frame_data.append(combined_frame_info)

# Release video captures
cap1.release()
cap2.release()

print('====> Combined images saved successfully.')

# --- Step 9: Save Frame Data to CSV ---

# Create DataFrame
combined_frame_df = pd.DataFrame(combined_frame_data)

# Save to CSV with video names included
csv_filename = f"combined_frames_{video1_name}_{video2_name}.csv".replace('.', '_').replace(' ', '_')
combined_frame_df.to_csv(csv_filename+'.csv', index=False)