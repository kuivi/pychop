# pychop

Synchronizing and Combining Underwater Video Frames Based on Drift and Timestamps

## Introduction

This script processes two underwater videos recorded from different cameras (e.g., a SeaViewer and a GoPro) during an experimental campaign off Kühlungsborn in the southwestern Baltic Sea. The aim is to synchronize and combine frames from both videos based on drift (distance traveled) and timestamps to investigate the short-term effects of bottom trawling on sediment integrity, biogeochemistry, and benthic communities.

## Description

The script extracts frames every e.g. 1 or 5 meters of drift, along with frames before and after each drift point, and combines them into a single image for comparison. This allows for coarse estimates of the density of the ocean quahog Arctica islandica and helps relate hydroacoustic backscatter patterns to the occurrence of mussel shell clusters, trawling marks, and other seafloor features.

![000268m_20240717_141533](https://github.com/user-attachments/assets/8e3f33be-8e17-4d7d-9894-98c6915b435e)

## Key Features:

	•	Synchronization Based on Drift and Time:
	•	Aligns videos using a common time base and drift calculation.
	•	Adjusts for time discrepancies in the GoPro video using a customizable time shift parameter.
	•	Data Processing:
	•	Reads ship log data to interpolate positions and headings.
	•	Applies coordinate transformations and heading corrections to obtain accurate positions.
	•	Frame Extraction and Combination:
	•	Extracts frames every 5 meters of drift, including frames before and after the drift point.
	•	Generates combined images with frames from both videos arranged in a 2x3 grid.
	•	Output:
	•	Saves combined images and frame data for further analysis.

Note: The script also includes small functions to read ship log data and extract initial timestamps from GoPro videos.

## Background

This experimental campaign aimed to investigate the short-term effects of bottom trawling on sediment integrity and biogeochemistry, as well as the composition and function of benthic communities in the southwestern Baltic Sea. The research is part of the second phase of the BMBF-funded DAM pilot-mission project MGF-OSTSEE II.

To evaluate changes in benthic macrofauna communities, Van Veen grab and multicorer samples were taken before and after trawling at various time points. Underwater video transects were conducted using a hand-held underwater video system to examine habitat properties, large mobile species, and visual effects of disturbance.

## Applications of the Script:

	•	Estimating the density of Arctica islandica by observing their siphons at the sediment-water boundary.
	•	Comparing underwater video images with hydroacoustic backscatter data to relate backscatter patterns to seafloor features like mussel shells, trawl marks, and other disturbances.
	•	Observing macrofaunal density and composition changes immediately after trawling disturbances.

## Usage Instructions

Parameters to Modify
```python 
shiplog_file: Path to the ship log data file.
video1_path: Path to the first video file (SeaViewer).
video2_path: Path to the second video file (GoPro).
video2_timeshift: Time shift in seconds to adjust the GoPro video’s timestamps for synchronization.
```

# Example

## Parameters to modify
``` python
shiplog_file = "./data/EMB345_1s_16072024-01082024.dat"
video1_path = './data/T0_C_SeaViewer/20240717-140600MA.AVI'
video2_path = './data/T0_C_GOPRO/GH013919.MP4'
video2_timeshift = 38  # Adjust as needed
```

## Running the Script

	### 1.	Install Dependencies:
```bash
pip install opencv-python pandas numpy pyproj
```

	### 2.	Execute the Script:
Run the script in your Python environment after adjusting the parameters.
	### 3.	Outputs:
	•	Combined images saved in a directory named combined_<video1_name>_<video2_name>.
	•	Images named using the drift in meters and the timestamp from Video 1, e.g., 25m_20240717_140625.png.
	•	A CSV file containing information about the combined frames.

## Dependencies

	•	Python 3.x
	•	OpenCV
	•	Pandas
	•	NumPy
	•	PyProj
	•	FFmpeg (for ffprobe to extract video metadata)

## Acknowledgments

This script was developed with the assistance of an AI language model.

Note: The script includes auxiliary functions to read ship log data and extract initial timestamps from GoPro videos, ensuring accurate synchronization between datasets.

License

MIT License

Contact

For questions or feedback, please contact [@kuivi](https://github.com/kuivi) and Mayya Gogina, IOW. 

This script was developed as part of the DAM pilot-mission project MGF-OSTSEE II to support the investigation of bottom trawling effects on marine environments.
