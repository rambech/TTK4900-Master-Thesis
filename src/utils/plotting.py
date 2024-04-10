import json
import os


def load_last_file(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Filter out non-JSON files
    json_files = [file for file in files if file.endswith('.json')]

    if not json_files:
        print("No JSON files found in the folder")
        return None

    # Sort files by modification time
    json_files.sort(key=lambda x: os.path.getmtime(
        os.path.join(folder_path, x)), reverse=True)

    # Get the path of the last modified JSON file
    last_json_file = os.path.join(folder_path, json_files[0])

    # Load JSON file into a dictionary
    with open(last_json_file, 'r') as f:
        data = json.load(f)

    return data
