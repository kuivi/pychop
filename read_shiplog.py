import pandas as pd
import chardet
import numpy as np
import re

def read_shiplog(file_path):
    """
    Reads and processes the shiplog.txt file.

    Parameters:
    - file_path (str): Path to the shiplog.txt file.

    Returns:
    - pandas.DataFrame: The processed DataFrame, or None if an error occurs.
    """
    
    def detect_encoding(file_path, num_bytes=100000):
        """
        Detects the encoding of a file using chardet.
        
        Parameters:
        - file_path (str): Path to the file.
        - num_bytes (int): Number of bytes to read for detection.
        
        Returns:
        - str: Detected encoding.
        """
        with open(file_path, 'rb') as f:
            rawdata = f.read(num_bytes)
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {encoding} with confidence {confidence}")
        return encoding
    
    def parse_coordinate(coord_str):
        """
        Converts a coordinate string like "54° 06,692' N" to decimal degrees.
        
        Parameters:
        - coord_str (str): The coordinate string.
        
        Returns:
        - float or np.nan: The coordinate in decimal degrees or NaN if invalid.
        """
        if pd.isna(coord_str):
            return np.nan
        coord_str = coord_str.strip()
        try:
            pattern = r"(\d+)°\s*(\d+[\.,]?\d*)'\s*([NSEW])"
            match = re.match(pattern, coord_str)
            if not match:
                return np.nan
            degrees, minutes, direction = match.groups()
            degrees = float(degrees)
            minutes = float(minutes.replace(',', '.'))
            decimal_degrees = degrees + minutes / 60
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            return decimal_degrees
        except Exception as e:
            print(f"Error parsing coordinate '{coord_str}': {e}")
            return np.nan
    
    def read_csv_file(file_path, encoding, skiprows):
        """
        Reads the CSV file with the specified encoding and skipped rows.
        
        Parameters:
        - file_path (str): Path to the file.
        - encoding (str): File encoding.
        - skiprows (list): List of row indices to skip.
        
        Returns:
        - pandas.DataFrame: The read DataFrame.
        """
        try:
            df = pd.read_csv(
                file_path,
                delimiter='\t',
                skiprows=skiprows,
                encoding=encoding,
                dtype=str
            )
            print("File read successfully.")
            return df
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
        return None
    
    # Step 1: Detect encoding
    encoding = detect_encoding(file_path)
    
    # Step 2: Read the file, skipping lines 1 and 2 (0-based indexing)
    header_lines_to_skip = [1, 2]
    df = read_csv_file(file_path, encoding, header_lines_to_skip)
    
    if df is not None:
        # Step 3: Replace placeholders with NaN without using inplace=True
        placeholders = ['9999', '-', 'spot', '']
        df = df.replace(placeholders, np.nan).infer_objects(copy=False)
        
        # Step 4: Parse coordinate columns
        coordinate_columns = [
            'SYS.STR.PosLat',
            'SYS.STR.PosLon',
            'GPS1.GPGLL.Latitude',
            'GPS1.GPRMC.Latitude',
            'GPS1.GPGLL.Longitude',
            'GPS1.GPRMC.Longitude'
        ]
            #'SYS.CALC.NextWP',                    
        for col in coordinate_columns:
            if col in df.columns:
                df[col + '_dec'] = df[col].apply(parse_coordinate)
            else:
                print(f"Coordinate column '{col}' not found in DataFrame.")
        
        # Step 5: Convert numeric columns
        numeric_columns = [
            'SYS.STR.Course',
            'SYS.STR.DPT',
            'SYS.STR.HDG',
            'SYS.STR.Speed',
            'GPS1.GPVTG.CourseM',
            'GPS1.GPRMC.CourseT',
            'GPS1.GPVTG.CourseT',
            'GPS1.GPRMC.Date',
            'Navlot.SDDPT.Depth',
            'SMB.SMBRA.Chl',
            'SMB.SMBRA.S_SBE45',
            'SMB.SMBRA.T_SBE45',
            'Weatherstation_DWD.PEUMA.Air_pressure2',
            'Weatherstation_DWD.PEUMA.Air_pressure',
            'Weatherstation_DWD.PEUMA.Air_temperature',
            'Weatherstation_DWD.PEUMA.Humidity',
            'Weatherstation_DWD.PEUMA.Water_temperature',
            'Weatherstation_DWD.PEUMA.Absolute_wind_direction',
            'Weatherstation_DWD.PEUMA.Relative_wind_direction',
            'Weatherstation_DWD.PEUMA.Absolute_wind_speed_bf',
            'Weatherstation_DWD.PEUMA.Absolute_wind_speed',
            'Weatherstation_DWD.PEUMA.Relative_wind_speed_bf',
            'Weatherstation_DWD.PEUMA.Relative_wind_speed',
            'Gyro.HEHDT.Heading',
            'Gyro_GPS.GPHDT.Heading'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Numeric column '{col}' not found in DataFrame.")
        
        # Step 6: Rename columns to replace dots with underscores for easier access
        df.columns = df.columns.str.replace('.', '_')
        
        # Step 7: Return the processed DataFrame
        return df
    else:
        print("Failed to read the shiplog.txt file due to encoding or formatting issues.")
        return None

# Example usage:
# df = read_shiplog("./data/shiplog.txt")
# print(df.head())