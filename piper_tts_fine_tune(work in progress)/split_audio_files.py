from pydub import AudioSegment
import os
import soundfile as sf
import numpy as np

def convert_to_wav(input_file, output_file, sample_rate=16000):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(input_file)
    
    # Convert to mono and the desired sample rate
    audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    
    # Export the audio as a wav file
    audio.export(output_file, format="wav")
    return output_file

def split_audio(input_file, output_dir, chunk_length=10, sample_rate=16000):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the input file to a temporary wav file
    temp_wav = os.path.join(output_dir, "temp.wav")
    convert_to_wav(input_file, temp_wav, sample_rate=sample_rate)

    # Load the temporary wav file
    audio, sr = sf.read(temp_wav)

    # Calculate the number of samples per chunk
    chunk_samples = int(chunk_length * sample_rate)

    # Split the audio into chunks
    total_samples = len(audio)
    num_chunks = int(np.ceil(total_samples / chunk_samples))

    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, total_samples)
        chunk = audio[start_sample:end_sample]

        # Create the output filename
        output_file = os.path.join(output_dir, f'{i+1}.wav')

        # Save the chunk as a wav file
        sf.write(output_file, chunk, sample_rate, subtype='PCM_16')

        print(f"Saved: {output_file}")

    # Remove the temporary wav file
    os.remove(temp_wav)

if __name__ == "__main__":
    input_audio_path = "input.m4b"  # Replace with the path to your input audio
    output_directory = "output_chunks"          # Replace with the desired output folder
    split_audio(input_audio_path, output_directory)

