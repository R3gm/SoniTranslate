from pydub import AudioSegment
from tqdm import tqdm
import os

def create_translated_audio(result_diarize, audio_files, Output_name_file):
    total_duration = result_diarize['segments'][-1]['end'] # in seconds

    # silent audio with total_duration
    combined_audio = AudioSegment.silent(duration=int(total_duration * 1000))
    print(round((total_duration / 60),2), 'minutes of video')

    for line, audio_file in tqdm(zip(result_diarize['segments'], audio_files)):
        start = float(line['start'])

        # Overlay each audio at the corresponding time
        try:
          audio = AudioSegment.from_file(audio_file)
          ###audio_a = audio.speedup(playback_speed=1.5)
          start_time = start * 1000  # to ms
          combined_audio = combined_audio.overlay(audio, position=start_time)
        except:
          print(f'ERROR AUDIO FILE {audio_file}')

    os.system("rm -rf audio/*")

    # combined audio as a file
    combined_audio.export(Output_name_file, format="wav") # best than ogg, change if the audio is anomalous
