import subprocess
import json
import datetime

def get_recording_start_time(video_path):

    # Command to extract metadata
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    # Run ffprobe command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    metadata = json.loads(result.stdout)

    # Extract creation_time and timecode from metadata
    creation_time = metadata['format']['tags']['creation_time']
    timecode = metadata['streams'][0]['tags']['timecode']

    # Extract just the date from creation_time
    creation_date = creation_time.split("T")[0]

    # Convert creation_date to a date object
    creation_date_obj = datetime.datetime.strptime(creation_date, '%Y-%m-%d').date()

    # Parse the timecode for hours, minutes, and seconds, ignoring frames
    timecode_parts = timecode.split(':')
    hours = int(timecode_parts[0])
    minutes = int(timecode_parts[1])
    seconds = int(timecode_parts[2])

    # Combine creation date with timecode time to create a single datetime object
    combined_datetime = datetime.datetime.combine(creation_date_obj, datetime.time(hours, minutes, seconds))

    return combined_datetime

# Example Usage
if __name__ == "__main__":
    video_file_path = "/path/to/your/video.mp4"  # Replace with your video file path

    try:
        recording_start = get_recording_start_time(video_file_path)
        print(f"Recording Start Time (UTC): {recording_start}")
    except Exception as e:
        print(f"Error: {e}")