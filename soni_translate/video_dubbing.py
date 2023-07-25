import numpy as np
import gradio as gr
import whisperx
import torch
from gtts import gTTS
import librosa
import edge_tts
import gc
from pydub import AudioSegment
from tqdm import tqdm
from deep_translator import GoogleTranslator
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import make_voice
from soni_translate.translate_segments import translate_text
import time

def translate_from_video(
    video,
    YOUR_HF_TOKEN,
    preview=False,
    WHISPER_MODEL_SIZE="large-v1",
    batch_size=16,
    compute_type="float16",
    SOURCE_LANGUAGE= "Automatic detection",
    TRANSLATE_AUDIO_TO="en",
    min_speakers=1,
    max_speakers=2,
    tts_voice00="en-AU-WilliamNeural-Male",
    tts_voice01="en-CA-ClaraNeural-Female",
    tts_voice02="en-GB-ThomasNeural-Male",
    tts_voice03="en-GB-SoniaNeural-Female",
    tts_voice04="en-NZ-MitchellNeural-Male",
    tts_voice05="en-GB-MaisieNeural-Female",
    video_output="video_dub.mp4"
    ):

    if YOUR_HF_TOKEN == "" or YOUR_HF_TOKEN == None:
      YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")    

    if not os.path.exists('audio'):
        os.makedirs('audio')

    if not os.path.exists('audio2/audio'):
        os.makedirs('audio2/audio')

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else compute_type

    OutputFile = 'Video.mp4'
    audio_wav = "audio.wav"
    Output_name_file = "audio_dub_solo.ogg"
    mix_audio = "audio_mix.mp3"

    os.system("rm Video.mp4")
    os.system("rm audio.webm")
    os.system("rm audio.wav")

    if os.path.exists(video):
        if preview:
            print('Creating preview video, 10 seconds')
            os.system(f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4')
        else:
            os.system(f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4')

        os.system("ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav")
    else:
        if preview:
            print('Creating preview from link, 10 seconds')
            #https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
            os.system(mp4_)
            os.system(wav_)
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f'python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}'

            os.system(wav_)

            for i in range (120):
                time.sleep(1)
                print('process audio')
                if os.path.exists(audio_wav) and not os.path.exists('audio.webm'):
                    time.sleep(1)
                    os.system(mp4_)
                    break
                if i == 119:
                  print('Error donwloading the audio')
                  return

    print("Set file complete.")

    SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(
        WHISPER_MODEL_SIZE,
        device,
        compute_type=compute_type,
        language= SOURCE_LANGUAGE,
        )
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect(); torch.cuda.empty_cache(); del model
    print("Transcript complete")

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
        )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=True,
        )
    gc.collect(); torch.cuda.empty_cache(); del model_a
    print("Align complete")

    if result['segments'] == []:
      print('No active speech found in audio')
      return

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
    diarize_segments = diarize_model(
        audio_wav,
        min_speakers=min_speakers,
        max_speakers=max_speakers)
    result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
    gc.collect(); torch.cuda.empty_cache(); del diarize_model
    print("Diarize complete")

    result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO)
    print("Translation complete")

    audio_files = []

    # Mapping speakers to voice variables
    speaker_to_voice = {
        'SPEAKER_00': tts_voice00,
        'SPEAKER_01': tts_voice01,
        'SPEAKER_02': tts_voice02,
        'SPEAKER_03': tts_voice03,
        'SPEAKER_04': tts_voice04,
        'SPEAKER_05': tts_voice05
    }

    for segment in tqdm(result_diarize['segments']):

        text = segment['text']
        start = segment['start']
        end = segment['end']

        try:
            speaker = segment['speaker']
        except KeyError:
            segment['speaker'] = "SPEAKER_99"
            speaker = segment['speaker']
            print("NO SPEAKER DETECT IN SEGMENT")

        # make the tts audio
        filename = f"audio/{start}.ogg"

        if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
            make_voice(text, speaker_to_voice[speaker], filename, TRANSLATE_AUDIO_TO)
        elif speaker == "SPEAKER_99":
            try:
                tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Using GTTS')
            except:
                tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Error: Audio will be replaced.')

        # duration
        duration_true = end - start
        duration_tts = librosa.get_duration(filename=filename)

        # porcentaje
        porcentaje = duration_tts / duration_true

        if porcentaje > 2.1:
            porcentaje = 2.1
        elif porcentaje <= 1.2 and porcentaje >= 0.8:
            porcentaje = 1.0
        elif porcentaje <= 0.79:
            porcentaje = 0.8

        # Smoth and round
        porcentaje = round(porcentaje+0.0, 1)

        # apply aceleration or opposite to the audio file in audio2 folder
        os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")

        duration_create = librosa.get_duration(filename=f"audio2/{filename}")
        audio_files.append(filename)

    # replace files with the accelerates
    os.system("mv -f audio2/audio/*.ogg audio/")

    os.system(f"rm {Output_name_file}")
    create_translated_audio(result_diarize, audio_files, Output_name_file)

    os.system(f"rm {mix_audio}")
    os.system(f'ffmpeg -i {audio_wav} -i {Output_name_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio}')

    os.system(f"rm {video_output}")
    os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output}")

    return video_output
