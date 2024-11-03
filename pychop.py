import cv2 as cv
from read_shiplog import read_shiplog





df = read_shiplog("./data/EMB345_1s_16072024-01082024.dat")

import cv2
import pandas as pd
from datetime import datetime, timedelta

# Video file path
video_path = './data/T0_C_SeaViewer/20240717-140600MA.AVI'

# Extract start time from the video file name
video_name = video_path.split('/')[-1]
start_time_str = video_name[:15]  # '20240717-140600'
start_time = datetime.strptime(start_time_str, '%Y%m%d-%H%M%S')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load your DataFrame (replace with your actual data loading method)
# df = pd.read_csv('your_data.csv', parse_dates=['date time'])

# Ensure 'date time' is in datetime format and set as index
df['date time'] = pd.to_datetime(df['date time'])
df = df.set_index('date time').sort_index()

# Prepare frame timestamps
frame_numbers = range(total_frames)
frame_times = [start_time + timedelta(seconds=(i / fps)) for i in frame_numbers]

# Create a DataFrame of frame timestamps
frame_timestamps = pd.DataFrame({'timestamp': frame_times})

# Reindex your DataFrame to include frame timestamps and interpolate
combined_index = df.index.union(frame_times)
df_interpolated = df.reindex(combined_index).sort_index().interpolate(method='time')

# Extract interpolated coordinates at frame timestamps
coords = df_interpolated.loc[frame_times, ['SYS_STR_PosLat_dec', 'SYS_STR_PosLon_dec', 'Gyro_GPS_GPHDT_Heading']].reset_index()
coords.columns = ['timestamp', 'latitude', 'longitude', 'heading']

from pyproj import Transformer

# Step 1: Convert coordinates from lat/lon to UTM Zone 32N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

# Apply the transformation
coords['x'], coords['y'] = transformer.transform(coords['longitude'].values, coords['latitude'].values)

import numpy as np

# Step 2: Apply correction using 'heading'
# Convert heading to radians
coords['heading_rad'] = np.deg2rad(coords['heading'])

# Calculate delta X and delta Y
coords['delta_x'] = -33.2 * np.sin(coords['heading_rad'])
coords['delta_y'] = -33.2 * np.cos(coords['heading_rad'])

# Apply corrections to get new x and y
coords['x_new'] = coords['x'] + coords['delta_x']
coords['y_new'] = coords['y'] + coords['delta_y']

# Step 3: Calculate distances between consecutive points
coords['delta_x_new'] = coords['x_new'].diff()
coords['delta_y_new'] = coords['y_new'].diff()
coords['distance'] = np.sqrt(coords['delta_x_new']**2 + coords['delta_y_new']**2)

# Replace NaN in the first distance with zero
coords['distance'].fillna(0, inplace=True)

# Step 4: Calculate running sum of distances to get 'drift'
coords['drift'] = coords['distance'].cumsum()


import os

# Step 5: Select frames every 5 meters of drift
drift_interval = 5  # meters
max_drift = coords['drift'].max()
drift_points = np.arange(0, max_drift + drift_interval, drift_interval)

selected_indices = []

for point in drift_points:
    # Find the index where drift crosses the point
    indices = coords.index[coords['drift'] >= point]
    if not indices.empty:
        idx = indices[0]
        selected_indices.append(idx)

# Include one frame before and after each selected index, delta=5
additional_indices = []
frame_delta = 5
for idx in selected_indices:
    if idx > 0:
        additional_indices.append(idx - 5)
    additional_indices.append(idx)
    if idx < len(coords) - 5:
        additional_indices.append(idx + 5)

# Remove duplicates and sort
all_indices = sorted(set(additional_indices))

# Create a directory to save selected frames
output_dir = 'selected_frames'
os.makedirs(output_dir, exist_ok=True)

# Prepare a list to store data of selected frames
selected_frame_data = []

# Reset the video to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize frame counter
frame_counter = 0

# Loop over each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process only if the current frame is in all_indices
    if frame_counter in all_indices:
        # Get corresponding data
        row = coords.iloc[frame_counter]
        timestamp = row['timestamp']
        latitude = row['latitude']
        longitude = row['longitude']
        heading = row['heading']
        drift = row['drift']

        # Create a descriptive filename
        frame_filename = f"{video_name}_frame_{frame_counter:06d}_{timestamp.strftime('%Y%m%d%H%M%S')}.png"
        frame_path = os.path.join(output_dir, frame_filename)

        # Save the frame as PNG
        cv2.imwrite(frame_path, frame)

        # Collect frame data
        frame_info = {
            'frame_number': frame_counter,
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'heading': heading,
            'drift': drift,
            'frame_filename': frame_filename
        }
        selected_frame_data.append(frame_info)

    frame_counter += 1

# Release the video capture object
cap.release()


'''











import os

# Create a directory to save frames
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Prepare a list to store frame data
frame_data = []

# Reset the video to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Loop over each frame
for i in range(10100):#frame_numbers:
    ret, frame = cap.read()
    if not ret:
        break
    if i<=10000:
        continue

    # Get timestamp and coordinates
    timestamp = coords.iloc[i]['timestamp']
    latitude = coords.iloc[i]['latitude']
    longitude = coords.iloc[i]['longitude']

    # Save frame as PNG
    frame_filename = f'frame_{i:06d}.png'
    frame_path = os.path.join(output_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    # Append frame data
    frame_info = {
        'frame_number': i,
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude,
        'frame_filename': frame_filename
    }
    frame_data.append(frame_info)

# Release the video capture object
cap.release()



# Convert frame data to DataFrame
frame_df = pd.DataFrame(frame_data)

# Save frame data to a CSV file if needed
frame_df.to_csv('frame_data.csv', index=False)

# Display the first few rows
print(frame_df.head())

'''