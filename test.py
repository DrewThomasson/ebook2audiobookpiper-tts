import os
import sys
import time
import wave
import requests
from tqdm import tqdm
from piper import PiperVoice

# URLs for the model and config files
model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx"
config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/norman/medium/en_US-norman-medium.onnx.json"

# Local paths for the model and config files
model_path = "/app/en_US-norman-medium.onnx"
config_path = "/app/en_en_US_norman_medium_en_US-norman-medium.onnx.json"

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}")
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong with the download.")
    else:
        print(f"Downloaded {os.path.basename(output_path)} successfully.")

def ensure_files_exist():
    if not os.path.exists(model_path):
        print(f"Model file not found locally. Downloading from {model_url}...")
        download_file(model_url, model_path)
    
    if not os.path.exists(config_path):
        print(f"Config file not found locally. Downloading from {config_url}...")
        download_file(config_url, config_path)

def generate_audio(text, output_file_path):
    start_time = time.time()
    
    with wave.open(output_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Assuming mono channel
        wav_file.setsampwidth(2)  # Assuming 16-bit samples
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text, wav_file)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Audio generated and saved to {output_file_path} in {duration:.2f} seconds")

if __name__ == "__main__":
    # Ensure model and config files are available
    ensure_files_exist()

    # Load the model and configuration once
    print("Loading model...")
    voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
    print("Model loaded successfully.")

    text = "This is a test synthesis."
    output_file_path = "test.wav"
    loop_count = 10  # Set the number of times you want to generate the audio

    for i in range(loop_count):
        print(f"Iteration {i + 1}/{loop_count}")
        generate_audio(text, output_file_path)
