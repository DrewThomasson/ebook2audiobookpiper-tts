import json
import requests
import os
from tqdm import tqdm

# Load the JSON data
with open("voices.json", "r") as f:
    voices_data = json.load(f)

# Base URL for downloading model files
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

# Function to download voice files
def download_voice(voice_key, voice_info):
    files = voice_info["files"]
    
    download_dir = os.path.join(os.getcwd(), voice_key)
    os.makedirs(download_dir, exist_ok=True)
    
    for file_path, file_info in files.items():
        url = BASE_URL + file_path
        local_file_path = os.path.join(download_dir, os.path.basename(file_path))
        
        # Download the file with tqdm progress bar in the terminal
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(local_file_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(file_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                file.write(data)
                bar.update(len(data))
    
    print(f"Downloaded {len(files)} files for {voice_key}.")

# Iterate over all voice keys and download the corresponding files
for voice_key, voice_info in voices_data.items():
    print(f"Downloading files for {voice_info['name']}...")
    download_voice(voice_key, voice_info)

print("All voices have been downloaded.")
