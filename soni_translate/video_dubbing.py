#%cd SoniTranslate
import numpy as np
import gradio as gr
import whisperx
from whisperx.utils import LANGUAGES as LANG_TRANSCRIPT
from whisperx.utils import get_writer
from IPython.utils import capture
import torch
from gtts import gTTS
import librosa
import edge_tts
import asyncio
import gc
from pydub import AudioSegment
from tqdm import tqdm
from deep_translator import GoogleTranslator
import os
from .audio_segments import create_translated_audio
from .text_to_speech import make_voice_gradio, audio_segmentation_to_voice
from .translate_segments import translate_text
from .preprocessor import audio_video_preprocessor
from .utils import print_tree_directory, remove_files, select_zip_and_rar_files, download_list, manual_download, upload_model_list
from urllib.parse import unquote
from .speech_segmentation import transcribe_speech, align_speech, diarize_speech
import copy, logging, rarfile, zipfile, shutil, time, json, subprocess
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)

class ProgressUpdate:
    def __call__(self, progress, desc=""):
        #print(f"Progress: {progress * 100:.2f}% - {desc}")
        pass

### voices

directories = ['downloads', 'logs', 'weights']
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

from ..voice_main import ClassVoices
voices_conversion = ClassVoices()

LANGUAGES = {
    'Automatic detection': 'Automatic detection',
    'Arabic (ar)': 'ar',
    'Chinese (zh)': 'zh',
    'Czech (cs)': 'cs',
    'Danish (da)': 'da',
    'Dutch (nl)': 'nl',
    'English (en)': 'en',
    'Finnish (fi)': 'fi',
    'French (fr)': 'fr',
    'German (de)': 'de',
    'Greek (el)': 'el',
    'Hebrew (he)': 'he',
    'Hungarian (hu)': 'hu',
    'Italian (it)': 'it',
    'Japanese (ja)': 'ja',
    'Korean (ko)': 'ko',
    'Persian (fa)': 'fa',
    'Polish (pl)': 'pl',
    'Portuguese (pt)': 'pt',
    'Russian (ru)': 'ru',
    'Spanish (es)': 'es',
    'Turkish (tr)': 'tr',
    'Ukrainian (uk)': 'uk',
    'Urdu (ur)': 'ur',
    'Vietnamese (vi)': 'vi',
    'Hindi (hi)': 'hi',
}

def translate_from_video(
    video,
    link_video,
    directory_input,
    YOUR_HF_TOKEN,
    preview=False,
    WHISPER_MODEL_SIZE="large-v1",
    batch_size=16,
    compute_type="float16",
    SOURCE_LANGUAGE= "Automatic detection",
    TRANSLATE_AUDIO_TO="English (en)",
    min_speakers=1,
    max_speakers=2,
    tts_voice00="en-AU-WilliamNeural-Male",
    tts_voice01="en-CA-ClaraNeural-Female",
    tts_voice02="en-GB-ThomasNeural-Male",
    tts_voice03="en-GB-SoniaNeural-Female",
    tts_voice04="en-NZ-MitchellNeural-Male",
    tts_voice05="en-GB-MaisieNeural-Female",
    video_output="video_dub.mp4",
    AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
    max_accelerate_audio = 2.1,
    volume_original_audio = 0.25,
    volume_translated_audio = 1.80,
    output_format_subtitle = "srt",
    get_translated_text = False,
    get_video_from_text_json = False,
    text_json = "{}",
    enable_custom_voices = False,
    is_gui = False,
    progress=ProgressUpdate(),
    ):

    if YOUR_HF_TOKEN == "" or YOUR_HF_TOKEN == None:
      YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
      if YOUR_HF_TOKEN == None:
        print('No valid token')
        return "No valid token"
      else:
        os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

    if video is None:
        if os.path.exists(directory_input):
            video = directory_input
        else:
            video = link_video
    video = video if isinstance(video, str) else video.name
    print(video)

    if "SET_LIMIT" == os.getenv("DEMO"):
      preview=True
      print("DEMO; set preview=True; The generation is **limited to 10 seconds** to prevent errors with the CPU. If you use a GPU, you won't have any of these limitations.")
      AUDIO_MIX_METHOD='Adjusting volumes and mixing audio'
      print("DEMO; set Adjusting volumes and mixing audio")
      WHISPER_MODEL_SIZE="medium"
      print("DEMO; set whisper model to medium")

    TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
    SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]

    global result_diarize, result, align_language, deep_copied_result

    if not os.path.exists('audio'):
        os.makedirs('audio')

    if not os.path.exists('audio2/audio'):
        os.makedirs('audio2/audio')

    if tts_voice00[:2] != TRANSLATE_AUDIO_TO[:2]:
        print("WARNING: Make sure to select a 'TTS Speaker' suitable for the translation language to avoid errors with the TTS.")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else compute_type

    OutputFile = 'Video.mp4'
    audio_wav = "audio.wav"
    Output_name_file = "audio_dub_solo.ogg"
    mix_audio = "audio_mix.mp3"

    if not get_video_from_text_json:
        progress(0.15, desc="Processing video...")
        audio_video_preprocessor(preview, video, OutputFile, audio_wav)
        print("Set file complete.")

        progress(0.30, desc="Transcribing...")
        SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE
        audio, result = transcribe_speech(audio_wav, WHISPER_MODEL_SIZE, compute_type, batch_size, SOURCE_LANGUAGE)
        print("Transcript complete")

        progress(0.45, desc="Aligning...")
        align_language = result["language"]
        result = align_speech(audio, result)
        print("Align complete")
        if result['segments'] == []:
            print('No active speech found in audio')
            return

        progress(0.60, desc="Diarizing...")
        result_diarize = diarize_speech(audio_wav, result, min_speakers, max_speakers, YOUR_HF_TOKEN)
        print("Diarize complete")
        deep_copied_result = copy.deepcopy(result_diarize)

        progress(0.75, desc="Translating...")
        if TRANSLATE_AUDIO_TO == "zh":
            TRANSLATE_AUDIO_TO = "zh-CN"
        if TRANSLATE_AUDIO_TO == "he":
            TRANSLATE_AUDIO_TO = "iw"
        result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO)
        print("Translation complete")

    if get_translated_text:
        json_data = []
        for segment in result_diarize['segments']:
            start = segment['start']
            text = segment['text']
            json_data.append({'start': start, 'text': text})

        # Convert the list of dictionaries to a JSON string with indentation
        json_string = json.dumps(json_data, indent=2)
        #segments[line]['text'] = translated_line
        return json_string.encode().decode('unicode_escape')

    if get_video_from_text_json:
        # with open('text_json.json', 'r') as file:
        text_json_loaded = json.loads(text_json)
        for i, segment in enumerate(result_diarize['segments']):
            segment['text'] = text_json_loaded[i]['text']


    progress(0.85, desc="Text_to_speech...")
    audio_files, speakers_list = audio_segmentation_to_voice(
        result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, is_gui,
        tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05
    )

    # custom voice
    if enable_custom_voices:
        progress(0.90, desc="Applying customized voices...")
        voices_conversion(speakers_list, audio_files)

    # replace files with the accelerates
    os.system("mv -f audio2/audio/*.ogg audio/")

    os.system(f"rm {Output_name_file}")

    progress(0.95, desc="Creating final translated video...")

    create_translated_audio(result_diarize, audio_files, Output_name_file)

    os.system(f"rm {mix_audio}")

    # TYPE MIX AUDIO
    if AUDIO_MIX_METHOD == 'Adjusting volumes and mixing audio':
        # volume mix
        os.system(f'ffmpeg -y -i {audio_wav} -i {Output_name_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')
    else:
        try:
            # background mix
            os.system(f'ffmpeg -i {audio_wav} -i {Output_name_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio}')
        except:
            # volume mix except
            os.system(f'ffmpeg -y -i {audio_wav} -i {Output_name_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

    os.system(f"rm {video_output}")
    os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output}")

    # Write subtitle
    #output_format_subtitle = ["srt", "vtt", "txt", "tsv", "json", "aud"]
    #output_format_subtitle = "vtt"
    name_ori = "sub_ori."
    name_tra = "sub_tra."
    deep_copied_result["language"] = align_language
    result_diarize["language"] = "ja" if TRANSLATE_AUDIO_TO in ["ja", "zh-CN"] else align_language

    writer = get_writer(output_format_subtitle, output_dir=".")
    word_options = {
        "highlight_words": False,
        "max_line_count" : None,
        "max_line_width" : None,
    }

    if os.path.exists(name_ori+output_format_subtitle): os.remove(name_ori+output_format_subtitle)
    if os.path.exists(name_tra+output_format_subtitle): os.remove(name_tra+output_format_subtitle)
    # original lang
    # for segment in deep_copied_result["segments"]:
    #     for dictionary in segment:
    #         dictionary.pop('speaker', None)

    #deep_copied_result["segments"][0].pop('speaker')
    subs_copy_result = copy.deepcopy(deep_copied_result)
    for i in range(len(subs_copy_result["segments"])):
        if 'speaker' in subs_copy_result["segments"][i]:
            subs_copy_result["segments"][i].pop('speaker')

    writer(
        subs_copy_result,
        name_ori[:-1]+".mp3",
        word_options,
    )
    # translated lang
    # result_diarize.pop('word_segments')
    # result_diarize["segments"][0].pop('speaker')
    # result_diarize["segments"][0].pop('chars')
    # result_diarize["segments"][0].pop('words')
    subs_tra_copy_result = copy.deepcopy(result_diarize)
    subs_tra_copy_result.pop('word_segments')
    for i in range(len(subs_tra_copy_result["segments"])):
        subs_tra_copy_result["segments"][i].pop('speaker')
        subs_tra_copy_result["segments"][i].pop('chars')
        subs_tra_copy_result["segments"][i].pop('words')
    writer(
        subs_tra_copy_result,
        name_tra[:-1]+".mp3",
        word_options,
    )

    return video_output