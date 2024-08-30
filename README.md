# 📚 eBook to Audiobook Converter with Piper-tts

Convert eBooks to audiobooks effortlessly using a Docker container. This project leverages Calibre for eBook conversion and Piper-tts for text-to-speech, all wrapped in a Gradio interface.

## 🌟 Features

- **eBook Conversion:** Converts various eBook formats to text using Calibre.
- **Text-to-Speech:** High-quality TTS using eSpeak-ng with customizable voice, speed, and pitch.
- **Language Support:** Includes multiple languages and accents for TTS.
- **User-Friendly Interface:** Easy-to-use Gradio interface accessible via your web browser.
- **Dockerized:** Run the entire setup with a single Docker command.
- **High Speed** Get High speed audiobook generation on even a low end laptop.


## 🌐 Supported Languages

- **Arabic** (ar_JO)
- **Catalan** (ca_ES)
- **Czech** (cs_CZ)
- **Welsh** (cy_GB)
- **Danish** (da_DK)
- **German** (de_DE)
- **Greek** (el_GR)
- **English** (en_GB, en_US)
- **Spanish** (es_ES, es_MX)
- **Finnish** (fi_FI)
- **French** (fr_FR)
- **Hungarian** (hu_HU)
- **Icelandic** (is_IS)
- **Italian** (it_IT)
- **Georgian** (ka_GE)
- **Kazakh** (kk_KZ)
- **Luxembourgish** (lb_LU)
- **Nepali** (ne_NP)
- **Dutch** (nl_BE, nl_NL)
- **Norwegian** (no_NO)
- **Polish** (pl_PL)
- **Portuguese** (pt_BR, pt_PT)
- **Romanian** (ro_RO)
- **Russian** (ru_RU)
- **Serbian** (sr_RS)
- **Swedish** (sv_SE)
- **Swahili** (sw_CD)
- **Turkish** (tr_TR)
- **Ukrainian** (uk_UA)
- **Vietnamese** (vi_VN)
- **Chinese** (zh_CN)

## 🎥 Demo


https://github.com/user-attachments/assets/7d2328b9-ac65-4485-b1b3-fe1006f041c6



## 🖥️ Gradio Web Gui
<img width="1363" alt="Screenshot 2024-08-30 at 12 17 41 AM" src="https://github.com/user-attachments/assets/8515b9b2-1db2-4944-b12b-c3a6bfde1535">

<img width="1365" alt="Screenshot 2024-08-30 at 12 17 51 AM" src="https://github.com/user-attachments/assets/0dc196a9-5853-4194-9151-46fd92eff811">




## 🚀 Quick Start

To quickly get started with this eBook to Audiobook converter, simply run the following Docker command:

```bash
docker run -it --rm -p 7860:7860 athomasson2/ebook2audiobookpiper-tts:latest
```

This will start the Gradio interface on port `7860`. You can access it by navigating to `http://localhost:7860` in your web browser.

## 🌐 Offline Mode

For a fully offline experience with all Piper TTS voice models bundled, use the following Docker command:

```bash
docker run -it --rm -p 7860:7860 athomasson2/ebook2audiobookpiper-tts:latest_large
```

This version comes preloaded with every Piper TTS voice model, ensuring that you can convert eBooks to audiobooks without needing an internet connection. Perfect for uninterrupted usage in any environment!


## 🎛️ Gradio Interface Customizations

In the Gradio interface, you can customize the following settings for your audiobook conversion:

- **Speed:** Adjust the reading speed from 80 to 450 words per minute (default is 170).
- **Pitch:** Modify the pitch of the voice from 0 to 99 (default is 50).
- **Voice Selection:** Choose from a variety of voices and accents available in eSpeak-ng.
- **Audiobook Player:** Listen to the converted audiobook directly in the interface.
- **Download Audiobook:** Download the generated audiobook files directly to your device.

## 🛠️ Building the Docker Image

If you prefer to build the Docker image yourself, use the following Dockerfile:

```dockerfile
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
```

To build the image, run:

```bash
docker build -t athomasson2/ebook2audiobookpiper-tts:latest .
```

## 🎉 Enjoy

Explore the full power of this converter by running your Docker container. Customize the settings in the Gradio interface to suit your needs.

For more details and updates, visit the [DockerHub repository](https://hub.docker.com/repository/docker/athomasson2/ebook2audiobookpiper-tts).


## 🙏 Special Thanks

- **Piper-tts**: [Piper-tts on GitHub]([https://github.com/espeak-ng/espeak-ng](https://github.com/rhasspy/piper))
- **Calibre**: [Calibre Website](https://calibre-ebook.com)



