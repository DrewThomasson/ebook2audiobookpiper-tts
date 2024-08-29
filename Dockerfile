# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local test.py file into the container
COPY test.py /app/

# Install piper-tts
RUN pip install piper-tts
RUN pip install requests 
RUN pip install tqdm

# Command to run piper-tts with the specified model and text
CMD echo 'Welcome to the world of speech synthesis!' | piper --model /app/en_US-norman-medium.onnx --output_file /app/welcome.wav
