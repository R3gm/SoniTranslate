# 🎥 SoniTranslate 🈷️

🎬 Video Translation with Synchronized Audio 🌐

SonyTranslate is a powerful and user-friendly web application that allows you to easily translate videos into different languages. This repository hosts the code for the SonyTranslate web UI, which is built with the Gradio library to provide a seamless and interactive user experience.


| Description | Link |
| ----------- | ---- |
| 📙 Colab Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb) |
| 🎉 Repository | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/R3gm/SoniTranslate/) |
| 🚀 Online DEMO | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/r3gm/SoniTranslate_translate_audio_of_a_video_content) |

## SonyTranslate's web UI, which features a browser interface built on the Gradio library.
![image](https://github.com/R3gm/SoniTranslate/assets/114810545/53800b08-3a18-4f8a-be15-8710dc9102ec)


## Supported languages for translation 

| Language Code | Language   |
|---------------|------------|
| en            | English    |
| fr            | French     |
| de            | German     |
| es            | Spanish    |
| it            | Italian    |
| ja            | Japanese   |
| zh            | Chinese    |
| nl            | Dutch      |
| uk            | Ukrainian  |
| pt            | Portuguese |
| ar            | Arabic     |
| cs            | Czech      |
| da            | Danish     |
| fi            | Finnish    |
| el            | Greek      |
| he            | Hebrew     |
| hu            | Hungarian  |
| ko            | Korean     |
| fa            | Persian    |
| pl            | Polish     |
| ru            | Russian    |
| tr            | Turkish    |
| ur            | Urdu       |
| hi            | Hindi      |
| vi            | Vietnamese |
| id            | Indonesian |
| bn            | Bengali    |
| te            | Telugu     |
| mr            | Marathi    |
| ta            | Tamil      |
| jw (or jv)    | Javanese   |
| ca            | Catalan    |
| ne            | Nepali     |
| th            | Thai       |

## Example:

### Original audio

https://github.com/R3gm/SoniTranslate/assets/114810545/db9e78c0-b228-4e81-9704-e62d5cc407a3



### Translated audio

https://github.com/R3gm/SoniTranslate/assets/114810545/6a8ddc65-a46f-4653-9726-6df2615f0ef9


## Colab Runtime

To run SoniTranslate using Colab Runtime: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R3gm/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb)

## Install Locally (Installation tested in Linux)

### Before You Start

Before you start installing and using SoniTranslate, there are a few things you need to do:

1. Install the NVIDIA drivers for CUDA 11.8.0, NVIDIA CUDA is a parallel computing platform and programming model that enables developers to use the power of NVIDIA graphics processing units (GPUs) to speed up compute-intensive tasks. You can find the drivers [here](https://developer.nvidia.com/cuda-toolkit-archive). Follow the instructions on the website to download and install the drivers.
2. Accept the license agreement for using Pyannote. You need to have an account on Hugging Face and `accept the license to use the models`: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation
3. Create a [huggingface token](https://huggingface.co/settings/tokens). Hugging Face is a natural language processing platform that provides access to state-of-the-art models and tools. You will need to create a token in order to use some of the automatic model download features in SoniTranslate. Follow the instructions on the Hugging Face website to create a token.
4. Install [Anaconda](https://www.anaconda.com/). Anaconda is a free and open-source distribution of Python and R. It includes a package manager called conda that makes it easy to install and manage Python environments and packages. Follow the instructions on the Anaconda website to download and install Anaconda on your system.
5. Install Git for your system. Git is a version control system that helps you track changes to your code and collaborate with other developers. You can install Git with Anaconda by running `conda install -c anaconda git -y` in your terminal. If you have trouble installing Git via Anaconda, you can use the following link instead:
   - [Git for Linux](https://git-scm.com/download/linux)

Once you have completed these steps, you will be ready to install SoniTranslate.

### Getting Started

To install SoniTranslate, follow these steps:

1. Create a suitable anaconda environment for SoniTranslate and activate it:

```
conda create -n sonitr python=3.10 -y
conda activate sonitr
```

2. Clone this github repository and navigate to it:
```
git clone https://github.com/r3gm/SoniTranslate.git
cd SoniTranslate
```

3. Install required packages:

```
pip install -r requirements_colab.txt -v
pip install -r requirements_extra.txt -v
pip install onnxruntime-gpu
```

4. Install [ffmpeg](https://ffmpeg.org/download.html). FFmpeg is a free software project that produces libraries and programs for handling multimedia data. You will need it to process audio and video files. You can install ffmpeg with Anaconda by running `conda install -y ffmpeg` in your terminal. If you have trouble installing ffmpeg via Anaconda, you can use the following link instead: (https://ffmpeg.org/ffmpeg.html). Once it is installed, make sure it is in your PATH by running `ffmpeg -h` in your terminal. If you don't get an error message, you're good to go.

5. Optional install:

After installing FFmpeg, you can install these optional packages.


[Piper TTS](https://github.com/rhasspy/piper) is a fast, local neural text to speech system that sounds great and is optimized for the Raspberry Pi 4. Piper is used in a variety of projects. Voices are trained with VITS and exported to the onnxruntime.

```
pip install -q piper-tts==1.2.0
```

[Coqui XTTS](https://github.com/coqui-ai/TTS) is a text-to-speech (TTS) model that lets you generate realistic voices in different languages. It can clone voices with just a short audio clip, even speak in a different language! It's like having a personal voice mimic for any text you need spoken.

```
pip install -q -r requirements_xtts.txt
pip install -q TTS==0.21.1  --no-deps
```


### Running SoniTranslate

To run SoniTranslate locally, make sure the `sonitr` conda environment is active:

```
conda activate sonitr
```

Then navigate to the `SoniTranslate` folder and run either the `app_rvc.py`

```
python app_rvc.py
```

### Stop and close SoniTranslate.

In most environments, you can stop the execution by pressing Ctrl+C in the terminal where you launched the script `app_rvc.py`. This will interrupt the program and stop the Gradio app.
To deactivate the Conda environment, you can use the following command:

```
conda deactivate
```

This will deactivate the currently active Conda environment sonitr, and you'll return to the base environment or the global Python environment.

### Starting Over

If you need to start over from scratch, you can delete the `SoniTranslate` folder and remove the `sonitr` conda environment with the following set of commands:

```
conda deactivate
conda env remove -n sonitr
```

With the `sonitr` environment removed, you can start over with a fresh installation.



## 📖 News

🔥 2024/01/16: Expanded language support (Thai, Nepali, Catalan, Javanese, Tamil, Marathi, Telugu, Bengali and Indonesian), the introduction of whisper large v3, configurable GUI options, integration of BARK, Facebook-mms, Coqui XTTS, and Piper-TTS. Additional features included audio separation utilities, XTTS WAV creation, use an SRT file as a base for translation, document translation, manual speaker editing, and flexible output options (video, audio, subtitles).

🔥 2023/10/29: Edit the translated subtitle, download it, adjust volume and speed options.

🔥 2023/08/03: Changed default options and added directory view of downloads.

🔥 2023/08/02: Added support for Arabic, Czech, Danish, Finnish, Greek, Hebrew, Hungarian, Korean, Persian, Polish, Russian, Turkish, Urdu, Hindi, and Vietnamese languages. 🌐

🔥 2023/08/01: Add options for use RVC models.

🔥 2023/07/27: Fix some bug processing the video and audio.

🔥 2023/07/26: New UI and add mix options.


## Contributing

Welcome to contributions from the community! If you have any ideas, bug reports, or feature requests, please open an issue or submit a pull request. For more information, please refer to the contribution guidelines.



## License
Although the code is licensed under Apache 2, the models or weights may have commercial restrictions, as seen with pyannote diarization.
