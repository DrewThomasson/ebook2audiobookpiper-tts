import gradio as gr
import json
import requests
import os
from tqdm import tqdm

# Load the JSON data
with open("voices.json", "r") as f:
    voices_data = json.load(f)

# Base URL for downloading model files
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

# Function to download selected voice files
def download_voice(voice_key):
    voice_info = voices_data[voice_key]
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
    
    return f"Downloaded {len(files)} files for {voice_key}."

# Function to get details of the selected voice
def get_voice_details(voice_key):
    voice_info = voices_data[voice_key]
    details = f"""
    **Name:** {voice_info['name']}
    **Language:** {voice_info['language']['name_english']} ({voice_info['language']['code']})
    **Country:** {voice_info['language']['country_english']}
    **Quality:** {voice_info['quality']}
    **Number of Speakers:** {voice_info.get('num_speakers', 'N/A')}
    """
    return details

# List of voice keys for dropdown
voice_keys = list(voices_data.keys())

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Piper Voices Downloader")
    
    voice_selector = gr.Dropdown(label="Select Voice", choices=voice_keys)
    
    voice_details = gr.Markdown()
    download_button = gr.Button("Download Voice Files")
    
    voice_selector.change(get_voice_details, voice_selector, voice_details)
    download_button.click(download_voice, voice_selector, None)

demo.launch()

