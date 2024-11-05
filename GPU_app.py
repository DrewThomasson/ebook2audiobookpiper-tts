print("starting...")

import os
import shutil
import subprocess
import re
from pydub import AudioSegment
import tempfile
import nltk
from nltk.tokenize import sent_tokenize
import sys
from tqdm import tqdm
import gradio as gr
from gradio import Progress
import urllib.request
import zipfile
import wave
import time
import torch
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import csv
import json
import requests

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

import subprocess

print("oippooop")

# Function to execute Piper-TTS via subprocess with CUDA enabled
def piper_to_tts(text_to_generate, output_audio_name, model_name):
    # Construct the Piper-TTS command with CUDA enabled
    #cmd = f"echo '{text_to_generate}' | piper --model {model_name} --output_file {output_audio_name} --cuda True"
    cmd = ["echo", text_to_generate, "|", "piper", "--model", model_name, "--output_file", output_audio_name, "--cuda", "True"]
    print(f"Executing command: {cmd}")
    
    # Execute the command
    try:
        #subprocess.run(cmd, shell=True)
        subprocess.run(cmd)
        print(f"Audio generated and saved to {output_audio_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Piper-TTS execution: {e}")
        raise



# Example usage
#piper_to_tts("Welcome to the world of speech, I am a bot!", "welcome.wav", "en_US-lessac-medium")
#piper_to_tts("Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!Welcome to the world of speech, I am a bot!", "welcome.wav", "en_US-lessac-medium")
subprocess.run(
    "echo 'Welcome to the world of speech I am a bot!' | piper --model en_US-lessac-medium --output_file welcome.wav",
    shell=True
)
subprocess.run(
    "echo 'Welcome to the world of speech I am a bot!' | piper --model en_US-lessac-medium --output_file welcome.wav",
    shell=True
)
subprocess.run(
    "echo 'Welcome to the world of speech I am a bot!' | piper --model en_US-lessac-medium --output_file welcome.wav",
    shell=True
)


piper_to_tts("Welcome to the world of speech, I am a bot!", "welcome.wav", "cs_CZ-jirka-low")




# Function to download and extract ZIP files with progress
def download_and_extract_zip(url, extract_to='.'):
    try:
        os.makedirs(extract_to, exist_ok=True)
        
        zip_path = os.path.join(extract_to, 'model.zip')
        
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading Model") as t:
            def reporthook(blocknum, blocksize, totalsize):
                if t.total != totalsize:
                    t.total = totalsize
                t.update(blocknum * blocksize - t.n)

            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        print(f"Downloaded zip file to {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            with tqdm(total=len(files), unit="file", desc="Extracting Files") as t:
                for file in files:
                    if not file.endswith('/'):  # Skip directories
                        extracted_path = zip_ref.extract(file, extract_to)
                        base_file_path = os.path.join(extract_to, os.path.basename(file))
                        os.rename(extracted_path, base_file_path)
                    t.update(1)
        
        os.remove(zip_path)
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"Extracted files to {extract_to}")
        
        required_files = ['model.pth', 'config.json', 'vocab.json_']
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(extract_to, file))]
        
        if not missing_files:
            print("All required files (model.pth, config.json, vocab.json_) found.")
        else:
            print(f"Missing files: {', '.join(missing_files)}")
    
    except Exception as e:
        print(f"Failed to download or extract zip file: {e}")

# Function to check if a folder is empty
def is_folder_empty(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return not bool(os.listdir(folder_path))
    else:
        print(f"The path {folder_path} is not a valid folder.")
        return None

# Function to remove a folder and its contents
def remove_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed {folder_path} and all of its contents.")
    except Exception as e:
        print(f"Error removing {folder_path}: {e}")

# Function to wipe all contents from a folder
def wipe_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory and its contents: {item_path}")
    
    print(f"All contents wiped from {folder_path}.")

# Function to create M4B from chapters
def create_m4b_from_chapters(input_dir, ebook_file, output_dir):
    def sort_key(chapter_file):
        numbers = re.findall(r'\d+', chapter_file)
        return int(numbers[0]) if numbers else 0

    def extract_metadata_and_cover(ebook_path):
        try:
            cover_path = ebook_path.rsplit('.', 1)[0] + '.jpg'
            subprocess.run(['ebook-meta', ebook_path, '--get-cover', cover_path], check=True)
            if os.path.exists(cover_path):
                return cover_path
        except Exception as e:
            print(f"Error extracting eBook metadata or cover: {e}")
        return None

    def combine_wav_files(chapter_files, output_path):
        combined_audio = AudioSegment.empty()
        for chapter_file in chapter_files:
            audio_segment = AudioSegment.from_wav(chapter_file)
            combined_audio += audio_segment
        combined_audio.export(output_path, format='wav')
        print(f"Combined audio saved to {output_path}")

    def generate_ffmpeg_metadata(chapter_files, metadata_file):
        with open(metadata_file, 'w') as file:
            file.write(';FFMETADATA1\n')
            start_time = 0
            for index, chapter_file in enumerate(chapter_files):
                duration_ms = len(AudioSegment.from_wav(chapter_file))
                file.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_time}\n')
                file.write(f'END={start_time + duration_ms}\ntitle=Chapter {index + 1}\n')
                start_time += duration_ms

    def create_m4b(combined_wav, metadata_file, cover_image, output_m4b):
        os.makedirs(os.path.dirname(output_m4b), exist_ok=True)
        
        ffmpeg_cmd = ['ffmpeg', '-i', combined_wav, '-i', metadata_file]
        if cover_image:
            ffmpeg_cmd += ['-i', cover_image, '-map', '0:a', '-map', '2:v']
        else:
            ffmpeg_cmd += ['-map', '0:a']
        
        ffmpeg_cmd += ['-map_metadata', '1', '-c:a', 'aac', '-b:a', '192k']
        if cover_image:
            ffmpeg_cmd += ['-c:v', 'png', '-disposition:v', 'attached_pic']
        ffmpeg_cmd += [output_m4b]
        
        subprocess.run(ffmpeg_cmd, check=True)

    chapter_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')],
        key=sort_key
    )
    temp_dir = tempfile.gettempdir()
    temp_combined_wav = os.path.join(temp_dir, 'combined.wav')
    metadata_file = os.path.join(temp_dir, 'metadata.txt')
    cover_image = extract_metadata_and_cover(ebook_file)
    output_m4b = os.path.join(output_dir, os.path.splitext(os.path.basename(ebook_file))[0] + '.m4b')

    combine_wav_files(chapter_files, temp_combined_wav)
    generate_ffmpeg_metadata(chapter_files, metadata_file)
    create_m4b(temp_combined_wav, metadata_file, cover_image, output_m4b)

    if os.path.exists(temp_combined_wav):
        os.remove(temp_combined_wav)
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if cover_image and os.path.exists(cover_image):
        os.remove(cover_image)

# Function to create chapter-labeled book
def create_chapter_labeled_book(ebook_file_path):
    def ensure_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

    ensure_directory(os.path.join(".", 'Working_files', 'Book'))

    def convert_to_epub(input_path, output_path):
        try:
            subprocess.run(['ebook-convert', input_path, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting the eBook: {e}")
            return False
        return True

    def save_chapters_as_text(epub_path):
        directory = os.path.join(".", "Working_files", "temp_ebook")
        ensure_directory(directory)

        book = epub.read_epub(epub_path)

        previous_filename = ''
        chapter_counter = 0

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()

                if text.strip():
                    if len(text) < 2300 and previous_filename:
                        with open(previous_filename, 'a', encoding='utf-8') as file:
                            file.write('\n' + text)
                    else:
                        previous_filename = os.path.join(directory, f"chapter_{chapter_counter}.txt")
                        chapter_counter += 1
                        with open(previous_filename, 'w', encoding='utf-8') as file:
                            file.write(text)
                            print(f"Saved chapter: {previous_filename}")

    input_ebook = ebook_file_path
    output_epub = os.path.join(".", "Working_files", "temp.epub")

    if os.path.exists(output_epub):
        os.remove(output_epub)
        print(f"File {output_epub} has been removed.")
    else:
        print(f"The file {output_epub} does not exist.")

    if convert_to_epub(input_ebook, output_epub):
        save_chapters_as_text(output_epub)

    def process_chapter_files(folder_path, output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker', 'Chapter'])

            chapter_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            for filename in chapter_files:
                if filename.startswith('chapter_') and filename.endswith('.txt'):
                    chapter_number = int(filename.split('_')[1].split('.')[0])
                    file_path = os.path.join(folder_path, filename)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            if text:
                                text = "NEWCHAPTERABC" + text
                            sentences = nltk.tokenize.sent_tokenize(text)
                            for sentence in sentences:
                                start_location = text.find(sentence)
                                end_location = start_location + len(sentence)
                                writer.writerow([sentence, start_location, end_location, 'True', 'Narrator', chapter_number])
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

    folder_path = os.path.join(".", "Working_files", "temp_ebook")
    output_csv = os.path.join(".", "Working_files", "Book", "Other_book.csv")

    process_chapter_files(folder_path, output_csv)

    def sort_key(filename):
        match = re.search(r'chapter_(\d+)\.txt', filename)
        return int(match.group(1)) if match else 0

    def combine_chapters(input_folder, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        sorted_files = sorted(files, key=sort_key)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, filename in enumerate(sorted_files):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    if i < len(sorted_files) - 1:
                        outfile.write("\nNEWCHAPTERABC\n")

    input_folder = os.path.join(".", 'Working_files', 'temp_ebook')
    output_file = os.path.join(".", 'Working_files', 'Book', 'Chapter_Book.txt')

    combine_chapters(input_folder, output_file)
    ensure_directory(os.path.join(".", "Working_files", "Book"))

# Function to check if Calibre is installed
def calibre_installed():
    try:
        subprocess.run(['ebook-convert', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Calibre is not installed. Please install Calibre for this functionality.")
        return False

# Function to combine WAV files
def combine_wav_files(input_directory, output_directory, file_name):
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, file_name)
    combined_audio = AudioSegment.empty()

    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        combined_audio += audio_segment

    combined_audio.export(output_file_path, format='wav')
    print(f"Combined audio saved to {output_file_path}")

# Function to split long sentences
def split_long_sentence(sentence, max_length=249, max_pauses=10):
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        if possible_splits:
            split_at = possible_splits[-1] + 1
        else:
            split_at = max_length
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    parts.append(sentence)
    return parts

# Function to convert chapters to audio using subprocess and Piper-TTS
def convert_chapters_to_audio_standard_model(chapters_dir, output_audio_dir, target_voice_path=None, language=None, piper_model_name=None):
    if not piper_model_name:
        print("No Piper model specified.")
        return
    
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)

    for chapter_file in sorted(os.listdir(chapters_dir)):
        if chapter_file.endswith('.txt'):
            match = re.search(r"chapter_(\d+).txt", chapter_file)
            if match:
                chapter_num = int(match.group(1))
            else:
                print(f"Skipping file {chapter_file} as it does not match the expected format.")
                continue

            chapter_path = os.path.join(chapters_dir, chapter_file)
            output_file_name = f"audio_chapter_{chapter_num}.wav"
            output_file_path = os.path.join(output_audio_dir, output_file_name)
            temp_audio_directory = os.path.join(".", "Working_files", "temp")
            os.makedirs(temp_audio_directory, exist_ok=True)
            temp_count = 0

            with open(chapter_path, 'r', encoding='utf-8') as file:
                chapter_text = file.read()
                sentences = sent_tokenize(chapter_text, language='italian' if language == 'it' else 'english')
                for sentence in tqdm(sentences, desc=f"Chapter {chapter_num}"):
                    fragments = split_long_sentence(sentence, max_length=249 if language == "en" else 213, max_pauses=10)
                    for fragment in fragments:
                        if fragment != "":
                            print(f"Generating fragment: {fragment}...")
                            fragment_file_path = os.path.join(temp_audio_directory, f"{temp_count}.wav")
                            piper_to_tts(fragment, fragment_file_path, piper_model_name)
                            temp_count += 1

            combine_wav_files(temp_audio_directory, output_audio_dir, output_file_name)
            wipe_folder(temp_audio_directory)
            print(f"Converted chapter {chapter_num} to audio.")

# Function to convert eBook to audio
def convert_ebook_to_audio(ebook_file, target_voice_file, voice_key, progress=gr.Progress()):
    ebook_file_path = ebook_file.name
    target_voice = target_voice_file.name if target_voice_file else None

    working_files = os.path.join(".", "Working_files", "temp_ebook")
    full_folder_working_files = os.path.join(".", "Working_files")
    chapters_directory = os.path.join(".", "Working_files", "temp_ebook")
    output_audio_directory = os.path.join(".", 'Chapter_wav_files')
    remove_folder_with_contents(full_folder_working_files)
    remove_folder_with_contents(output_audio_directory)

    try:
        progress(0, desc="Starting conversion")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    if not calibre_installed():
        return "Calibre is not installed."
    
    try:
        progress(0.1, desc="Creating chapter-labeled book")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    create_chapter_labeled_book(ebook_file_path)
    audiobook_output_path = os.path.join(".", "Audiobooks")
    
    try:
        progress(0.3, desc="Converting chapters to audio")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    convert_chapters_to_audio_standard_model(
        chapters_directory, 
        output_audio_directory, 
        target_voice, 
        piper_model_name=voice_key
    )
    
    try:
        progress(0.9, desc="Creating M4B from chapters")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    create_m4b_from_chapters(output_audio_directory, ebook_file_path, audiobook_output_path)
    
    m4b_filename = os.path.splitext(os.path.basename(ebook_file_path))[0] + '.m4b'
    m4b_filepath = os.path.join(audiobook_output_path, m4b_filename)

    try:
        progress(1.0, desc="Conversion complete")
    except Exception as e:
        print(f"Error updating progress: {e}")
    print(f"Audiobook created at {m4b_filepath}")
    return f"Audiobook created at {m4b_filepath}", m4b_filepath

# Function to list audiobook files
def list_audiobook_files(audiobook_folder):
    files = []
    for filename in os.listdir(audiobook_folder):
        if filename.endswith('.m4b'):
            files.append(os.path.join(audiobook_folder, filename))
    return files

# Function to download audiobooks
def download_audiobooks():
    audiobook_output_path = os.path.join(".", "Audiobooks")
    return list_audiobook_files(audiobook_output_path)

# Load voices from JSON
with open("voices.json", "r") as f:
    voices_data = json.load(f)

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

# Function to download voice files
def download_voice(voice_key):
    voice_info = voices_data[voice_key]
    files = voice_info["files"]
    
    download_dir = os.path.join(os.getcwd(), voice_key)
    os.makedirs(download_dir, exist_ok=True)
    
    downloaded_files = 0
    
    for file_path, file_info in files.items():
        local_file_path = os.path.join(download_dir, os.path.basename(file_path))
        
        if os.path.exists(local_file_path):
            print(f"File '{os.path.basename(file_path)}' already exists. Skipping download.")
        else:
            url = BASE_URL + file_path
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
            downloaded_files += 1
    
    if downloaded_files == 0:
        return f"All files for {voice_key} already exist. No files were downloaded."
    else:
        return f"Downloaded {downloaded_files} files for {voice_key}."

# Function to download and convert eBook
def download_and_convert(ebook_file, target_voice_file, voice_key, progress=gr.Progress()):
    download_message = download_voice(voice_key)
    conversion_result, m4b_filepath = convert_ebook_to_audio(ebook_file, target_voice_file, voice_key, progress)
    return f"{download_message}\n{conversion_result}", m4b_filepath

# Function to get voice details
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

# Define Gradio theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="blue",
    text_size=gr.themes.sizes.text_md,
)

# Initialize Gradio interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
    """
    # Piper-TTS eBook to Audiobook Converter
    
    Transform your eBooks into audiobooks using Piper-TTS.

    This interface is based on [Ebook2audiobookPiperTTS](https://github.com/DrewThomasson/ebook2audiobookpiper-tts).
    """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            ebook_file = gr.File(label="eBook File")
            target_voice_file = gr.File(label="Target Voice File (Optional)", visible=False)

    voice_selector = gr.Dropdown(label="Select Voice", choices=voice_keys)
    voice_details = gr.Markdown()
    
    convert_btn = gr.Button("Convert to Audiobook", variant="primary")
    output = gr.Textbox(label="Conversion Status")
    audio_player = gr.Audio(label="Audiobook Player", type="filepath")
    download_btn = gr.Button("Download Audiobook Files")
    download_files = gr.File(label="Download Files", interactive=False)
    
    convert_btn.click(
        download_and_convert,
        inputs=[ebook_file, target_voice_file, voice_selector],
        outputs=[output, audio_player]
    )
    
    download_btn.click(
        download_audiobooks,
        outputs=[download_files]
    )
    
    download_voice_btn = gr.Button("Download Voice Files", visible=False)
    
    voice_selector.change(get_voice_details, voice_selector, voice_details)
    download_voice_btn.click(download_voice, voice_selector, None)

demo.launch(share=True)

# Example usage of Piper-TTS without Gradio (optional)
# Uncomment the lines below if you want to generate 'welcome.wav' outside the Gradio interface

# model_name = "en_US-lessac-medium"  # Replace with your model folder name
# welcome_text = "Welcome to the world of speech synthesis I am a bot! " * 12
# piper_to_tts(welcome_text, "welcome.wav", model_name)
