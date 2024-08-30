# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for Calibre and ffmpeg
RUN apt-get update && \
    apt-get install -y calibre ffmpeg git nano wget unzip git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

#No do a git clone instead
RUN git clone https://github.com/DrewThomasson/ebook2audiobookpiper-tts.git

# Install Python packages
RUN pip install --no-cache-dir piper-tts pydub nltk beautifulsoup4 ebooklib tqdm spacy gradio

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm


# Replace the NLTK folder with the backup
RUN echo "Replacing the nltk folder with the nltk folder backup I pulled from a docker image, just in case the nltk servers ever mess up." && \
    ZIP_URL="https://github.com/DrewThomasson/VoxNovel/blob/main/readme_files/nltk.zip?raw=true" && \
    TARGET_DIR="/usr/local/lib/python3.10/site-packages" && \
    TEMP_DIR=$(mktemp -d) && \
    wget -q -O "$TEMP_DIR/nltk.zip" "$ZIP_URL" && \
    unzip -q "$TEMP_DIR/nltk.zip" -d "$TEMP_DIR" && \
    rm -rf "$TARGET_DIR/nltk" && \
    mv "$TEMP_DIR/nltk" "$TARGET_DIR/nltk" && \
    rm -rf "$TEMP_DIR" && \
    echo "NLTK Files Replacement complete."


# Set the working directory
WORKDIR /app/ebook2audiobookpiper-tts

# NO USE THIS Default command
CMD ["python", "gradio_gui.py"]
#To run this docker on your computer run docker run -it athomasson2/ebook2audiobookpiper-tts
