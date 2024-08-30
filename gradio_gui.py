print("starting...")

import os
import shutil
import subprocess
import re
from pydub import AudioSegment
import tempfile
from pydub import AudioSegment
import os
import nltk
from nltk.tokenize import sent_tokenize
import sys
from tqdm import tqdm
import gradio as gr
from gradio import Progress
import urllib.request
import zipfile

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device selected is: {device}")

nltk.download('punkt')  # Make sure to download the necessary models



import os
import wave
import time
from piper import PiperVoice

def load_piper_tts(folder_name):
    model_file = os.path.join(folder_name, f"{folder_name}.onnx")
    config_file = os.path.join(folder_name, f"{folder_name}.json")

    # Automatically rename the config file if it's named incorrectly
    onnx_config_file = os.path.join(folder_name, f"{folder_name}.onnx.json")
    if os.path.exists(onnx_config_file) and not os.path.exists(config_file):
        os.rename(onnx_config_file, config_file)
        print(f"Renamed {onnx_config_file} to {config_file}")

    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print(f"Model file exists: {os.path.exists(model_file)}")
        print(f"Config file exists: {os.path.exists(config_file)}")
        print(f"Contents of {folder_name}:")
        for item in os.listdir(folder_name):
            print(item)
        raise FileNotFoundError(f"Model or config file not found in {folder_name}.")

    global voice
    voice = PiperVoice.load(model_file, config_path=config_file, use_cuda=False)
    print("Model loaded successfully.")


def piper_to_tts(text_to_generate, output_audio_name):
    if 'voice' not in globals():
        raise RuntimeError("Piper TTS model is not loaded. Please load it using load_piper_tts first.")

    start_time = time.time()

    with wave.open(output_audio_name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Assuming mono channel
        wav_file.setsampwidth(2)  # Assuming 16-bit samples
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize(text_to_generate, wav_file)

    end_time = time.time()
    print(f"Audio generated and saved to {output_audio_name} in {end_time - start_time:.2f} seconds")


def download_and_extract_zip(url, extract_to='.'):
    try:
        # Ensure the directory exists
        os.makedirs(extract_to, exist_ok=True)
        
        zip_path = os.path.join(extract_to, 'model.zip')
        
        # Download with progress bar
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading Model") as t:
            def reporthook(blocknum, blocksize, totalsize):
                t.total = totalsize
                t.update(blocknum * blocksize - t.n)

            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        print(f"Downloaded zip file to {zip_path}")
        
        # Unzipping with progress bar
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            with tqdm(total=len(files), unit="file", desc="Extracting Files") as t:
                for file in files:
                    if not file.endswith('/'):  # Skip directories
                        # Extract the file to the temporary directory
                        extracted_path = zip_ref.extract(file, extract_to)
                        # Move the file to the base directory
                        base_file_path = os.path.join(extract_to, os.path.basename(file))
                        os.rename(extracted_path, base_file_path)
                    t.update(1)
        
        # Cleanup: Remove the ZIP file and any empty folders
        os.remove(zip_path)
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"Extracted files to {extract_to}")
        
        # Check if all required files are present
        required_files = ['model.pth', 'config.json', 'vocab.json_']
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(extract_to, file))]
        
        if not missing_files:
            print("All required files (model.pth, config.json, vocab.json_) found.")
        else:
            print(f"Missing files: {', '.join(missing_files)}")
    
    except Exception as e:
        print(f"Failed to download or extract zip file: {e}")



def is_folder_empty(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List directory contents
        if not os.listdir(folder_path):
            return True  # The folder is empty
        else:
            return False  # The folder is not empty
    else:
        print(f"The path {folder_path} is not a valid folder.")
        return None  # The path is not a valid folder

def remove_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed {folder_path} and all of its contents.")
    except Exception as e:
        print(f"Error removing {folder_path}: {e}")




def wipe_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all the items in the given folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # If it's a file, remove it and print a message
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")
        # If it's a directory, remove it recursively and print a message
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory and its contents: {item_path}")
    
    print(f"All contents wiped from {folder_path}.")


# Example usage
# folder_to_wipe = 'path_to_your_folder'
# wipe_folder(folder_to_wipe)


def create_m4b_from_chapters(input_dir, ebook_file, output_dir):
    # Function to sort chapters based on their numeric order
    def sort_key(chapter_file):
        numbers = re.findall(r'\d+', chapter_file)
        return int(numbers[0]) if numbers else 0

    # Extract metadata and cover image from the eBook file
    def extract_metadata_and_cover(ebook_path):
        try:
            cover_path = ebook_path.rsplit('.', 1)[0] + '.jpg'
            subprocess.run(['ebook-meta', ebook_path, '--get-cover', cover_path], check=True)
            if os.path.exists(cover_path):
                return cover_path
        except Exception as e:
            print(f"Error extracting eBook metadata or cover: {e}")
        return None
    # Combine WAV files into a single file
    def combine_wav_files(chapter_files, output_path):
        # Initialize an empty audio segment
        combined_audio = AudioSegment.empty()

        # Sequentially append each file to the combined_audio
        for chapter_file in chapter_files:
            audio_segment = AudioSegment.from_wav(chapter_file)
            combined_audio += audio_segment
        # Export the combined audio to the output file path
        combined_audio.export(output_path, format='wav')
        print(f"Combined audio saved to {output_path}")

    # Function to generate metadata for M4B chapters
    def generate_ffmpeg_metadata(chapter_files, metadata_file):
        with open(metadata_file, 'w') as file:
            file.write(';FFMETADATA1\n')
            start_time = 0
            for index, chapter_file in enumerate(chapter_files):
                duration_ms = len(AudioSegment.from_wav(chapter_file))
                file.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_time}\n')
                file.write(f'END={start_time + duration_ms}\ntitle=Chapter {index + 1}\n')
                start_time += duration_ms

    # Generate the final M4B file using ffmpeg
    def create_m4b(combined_wav, metadata_file, cover_image, output_m4b):
        # Ensure the output directory exists
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



    # Main logic
    chapter_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')], key=sort_key)
    temp_dir = tempfile.gettempdir()
    temp_combined_wav = os.path.join(temp_dir, 'combined.wav')
    metadata_file = os.path.join(temp_dir, 'metadata.txt')
    cover_image = extract_metadata_and_cover(ebook_file)
    output_m4b = os.path.join(output_dir, os.path.splitext(os.path.basename(ebook_file))[0] + '.m4b')

    combine_wav_files(chapter_files, temp_combined_wav)
    generate_ffmpeg_metadata(chapter_files, metadata_file)
    create_m4b(temp_combined_wav, metadata_file, cover_image, output_m4b)

    # Cleanup
    if os.path.exists(temp_combined_wav):
        os.remove(temp_combined_wav)
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if cover_image and os.path.exists(cover_image):
        os.remove(cover_image)

# Example usage
# create_m4b_from_chapters('path_to_chapter_wavs', 'path_to_ebook_file', 'path_to_output_dir')






#this code right here isnt the book grabbing thing but its before to refrence in ordero to create the sepecial chapter labeled book thing with calibre idk some systems cant seem to get it so just in case but the next bit of code after this is the book grabbing code with booknlp 
import os
import subprocess
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import csv
import nltk

# Only run the main script if Value is True
def create_chapter_labeled_book(ebook_file_path):
    # Function to ensure the existence of a directory
    def ensure_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

    ensure_directory(os.path.join(".", 'Working_files', 'Book'))

    def convert_to_epub(input_path, output_path):
        # Convert the ebook to EPUB format using Calibre's ebook-convert
        try:
            subprocess.run(['ebook-convert', input_path, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting the eBook: {e}")
            return False
        return True

    def save_chapters_as_text(epub_path):
        # Create the directory if it doesn't exist
        directory = os.path.join(".", "Working_files", "temp_ebook")
        ensure_directory(directory)

        # Open the EPUB file
        book = epub.read_epub(epub_path)

        previous_chapter_text = ''
        previous_filename = ''
        chapter_counter = 0

        # Iterate through the items in the EPUB file
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Use BeautifulSoup to parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()

                # Check if the text is not empty
                if text.strip():
                    if len(text) < 2300 and previous_filename:
                        # Append text to the previous chapter if it's short
                        with open(previous_filename, 'a', encoding='utf-8') as file:
                            file.write('\n' + text)
                    else:
                        # Create a new chapter file and increment the counter
                        previous_filename = os.path.join(directory, f"chapter_{chapter_counter}.txt")
                        chapter_counter += 1
                        with open(previous_filename, 'w', encoding='utf-8') as file:
                            file.write(text)
                            print(f"Saved chapter: {previous_filename}")

    # Example usage
    input_ebook = ebook_file_path  # Replace with your eBook file path
    output_epub = os.path.join(".", "Working_files", "temp.epub")


    if os.path.exists(output_epub):
        os.remove(output_epub)
        print(f"File {output_epub} has been removed.")
    else:
        print(f"The file {output_epub} does not exist.")

    if convert_to_epub(input_ebook, output_epub):
        save_chapters_as_text(output_epub)

    # Download the necessary NLTK data (if not already present)
    nltk.download('punkt')

    def process_chapter_files(folder_path, output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker', 'Chapter'])

            # Process each chapter file
            chapter_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
            for filename in chapter_files:
                if filename.startswith('chapter_') and filename.endswith('.txt'):
                    chapter_number = int(filename.split('_')[1].split('.')[0])
                    file_path = os.path.join(folder_path, filename)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            # Insert "NEWCHAPTERABC" at the beginning of each chapter's text
                            if text:
                                text = "NEWCHAPTERABC" + text
                            sentences = nltk.tokenize.sent_tokenize(text)
                            for sentence in sentences:
                                start_location = text.find(sentence)
                                end_location = start_location + len(sentence)
                                writer.writerow([sentence, start_location, end_location, 'True', 'Narrator', chapter_number])
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

    # Example usage
    folder_path = os.path.join(".", "Working_files", "temp_ebook")
    output_csv = os.path.join(".", "Working_files", "Book", "Other_book.csv")

    process_chapter_files(folder_path, output_csv)

    def sort_key(filename):
        """Extract chapter number for sorting."""
        match = re.search(r'chapter_(\d+)\.txt', filename)
        return int(match.group(1)) if match else 0

    def combine_chapters(input_folder, output_file):
        # Create the output folder if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # List all txt files and sort them by chapter number
        files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        sorted_files = sorted(files, key=sort_key)

        with open(output_file, 'w', encoding='utf-8') as outfile:  # Specify UTF-8 encoding here
            for i, filename in enumerate(sorted_files):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:  # And here
                    outfile.write(infile.read())
                    # Add the marker unless it's the last file
                    if i < len(sorted_files) - 1:
                        outfile.write("\nNEWCHAPTERABC\n")

    # Paths
    input_folder = os.path.join(".", 'Working_files', 'temp_ebook')
    output_file = os.path.join(".", 'Working_files', 'Book', 'Chapter_Book.txt')


    # Combine the chapters
    combine_chapters(input_folder, output_file)

    ensure_directory(os.path.join(".", "Working_files", "Book"))


#create_chapter_labeled_book()




import os
import subprocess
import sys

# Check if Calibre's ebook-convert tool is installed
def calibre_installed():
    try:
        subprocess.run(['ebook-convert', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Calibre is not installed. Please install Calibre for this functionality.")
        return False


import os
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
# Assuming split_long_sentence and wipe_folder are defined elsewhere in your code

default_target_voice_path = "default_voice.wav"  # Ensure this is a valid path
default_language_code = "en"
def combine_wav_files(input_directory, output_directory, file_name):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, file_name)

    # Initialize an empty audio segment
    combined_audio = AudioSegment.empty()

    # Get a list of all .wav files in the specified input directory and sort them
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Sequentially append each file to the combined_audio
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        combined_audio += audio_segment

    # Export the combined audio to the output file path
    combined_audio.export(output_file_path, format='wav')

    print(f"Combined audio saved to {output_file_path}")

# Function to split long strings into parts
def split_long_sentence(sentence, max_length=249, max_pauses=10):
    """
    Splits a sentence into parts based on length or number of pauses without recursion.
    
    :param sentence: The sentence to split.
    :param max_length: Maximum allowed length of a sentence.
    :param max_pauses: Maximum allowed number of pauses in a sentence.
    :return: A list of sentence parts that meet the criteria.
    """
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        if possible_splits:
            # Find the best place to split the sentence, preferring the last possible split to keep parts longer
            split_at = possible_splits[-1] + 1
        else:
            # If no punctuation to split on within max_length, split at max_length
            split_at = max_length
        
        # Split the sentence and add the first part to the list
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    
    # Add the remaining part of the sentence
    parts.append(sentence)
    return parts

#convert chapters into audio with piper-tts

def convert_chapters_to_audio_standard_model(chapters_dir, output_audio_dir, target_voice_path=None, language=None, piper_model_name=None):
    load_piper_tts(piper_model_name)  # Load the Piper TTS model with the given name
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
                            piper_to_tts(fragment, fragment_file_path)
                            temp_count += 1

            combine_wav_files(temp_audio_directory, output_audio_dir, output_file_name)
            wipe_folder(temp_audio_directory)
            print(f"Converted chapter {chapter_num} to audio.")




# Define the functions to be used in the Gradio interface
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
    
    convert_chapters_to_audio_standard_model(chapters_directory, output_audio_directory, target_voice, piper_model_name=voice_key)
    try:
        progress(0.9, desc="Creating M4B from chapters")
    except Exception as e:
        print(f"Error updating progress: {e}")
    
    create_m4b_from_chapters(output_audio_directory, ebook_file_path, audiobook_output_path)
    
    # Get the name of the created M4B file
    m4b_filename = os.path.splitext(os.path.basename(ebook_file_path))[0] + '.m4b'
    m4b_filepath = os.path.join(audiobook_output_path, m4b_filename)

    try:
        progress(1.0, desc="Conversion complete")
    except Exception as e:
        print(f"Error updating progress: {e}")
    print(f"Audiobook created at {m4b_filepath}")
    return f"Audiobook created at {m4b_filepath}", m4b_filepath





def list_audiobook_files(audiobook_folder):
    # List all files in the audiobook folder
    files = []
    for filename in os.listdir(audiobook_folder):
        if filename.endswith('.m4b'):  # Adjust the file extension as needed
            files.append(os.path.join(audiobook_folder, filename))
    return files

def download_audiobooks():
    audiobook_output_path = os.path.join(".", "Audiobooks")
    return list_audiobook_files(audiobook_output_path)



















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

# Function to download selected voice files if missing
def download_voice(voice_key):
    voice_info = voices_data[voice_key]
    files = voice_info["files"]
    
    download_dir = os.path.join(os.getcwd(), voice_key)
    os.makedirs(download_dir, exist_ok=True)
    
    report = []
    files_to_download = []

    # Check for each file's existence
    for file_path, file_info in files.items():
        local_file_path = os.path.join(download_dir, os.path.basename(file_path))
        if os.path.exists(local_file_path):
            report.append(f"Found: {os.path.basename(file_path)} - No download needed.")
        else:
            report.append(f"Missing: {os.path.basename(file_path)} - Will download.")
            files_to_download.append((file_path, local_file_path))
    
    # If all files are present, return the report
    if not files_to_download:
        report.append("All files are present. No downloads were necessary.")
        return "\n".join(report)
    
    # Download missing files
    for file_path, local_file_path in files_to_download:
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
        report.append(f"Downloaded: {os.path.basename(file_path)}")
    
    report.append(f"Downloaded {len(files_to_download)} missing files for {voice_key}.")
    return "\n".join(report)


# Define a new function to chain the download and conversion processes
def download_and_convert(ebook_file, target_voice_file, voice_key, progress=gr.Progress()):
    # First, download the voice files
    download_message = download_voice(voice_key)
    
    # Then, proceed with the conversion
    conversion_result, m4b_filepath = convert_ebook_to_audio(ebook_file, target_voice_file, voice_key, progress)
    
    # Combine the download message with the conversion result
    return f"{download_message}\n{conversion_result}", m4b_filepath

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

# Integrating with the existing Gradio UI
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="blue",
    text_size=gr.themes.sizes.text_md,
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
    """
    # eBook to Audiobook Converter
    
    Transform your eBooks into immersive audiobooks.
    """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            ebook_file = gr.File(label="eBook File")
            target_voice_file = gr.File(label="Target Voice File (Optional)")

    voice_selector = gr.Dropdown(label="Select Voice", choices=voice_keys)  # Define voice_selector here

    convert_btn = gr.Button("Convert to Audiobook", variant="primary")
    output = gr.Textbox(label="Conversion Status")
    audio_player = gr.Audio(label="Audiobook Player", type="filepath")
    download_btn = gr.Button("Download Audiobook Files")
    download_files = gr.File(label="Download Files", interactive=False)

    convert_btn.click(
        download_and_convert,
        inputs=[ebook_file, target_voice_file, voice_selector],  # Now voice_selector is defined
        outputs=[output, audio_player]
    )

    download_btn.click(
        download_audiobooks,
        outputs=[download_files]
    )
    
    # Adding the Download Voices Section
    gr.Markdown("## Download Additional Voices")
    
    voice_details = gr.Markdown()
    download_voice_btn = gr.Button("Download Voice Files")
    
    voice_selector.change(get_voice_details, voice_selector, voice_details)
    download_voice_btn.click(download_voice, voice_selector, None)

demo.launch(share=True)



