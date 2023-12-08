from gtts import gTTS
import edge_tts, asyncio, nest_asyncio
from tqdm import tqdm
import librosa, os, subprocess
from .language_configuration import fix_code_language

def edge_tts_voices_list():
    completed_process = subprocess.run(
        ["edge-tts", "--list-voices"], capture_output=True, text=True
    )
    lines = completed_process.stdout.strip().split("\n")

    voices = []
    for line in lines:
        if line.startswith("Name: "):
            voice_entry = {}
            voice_entry["Name"] = line.split(": ")[1]
        elif line.startswith("Gender: "):
            voice_entry["Gender"] = line.split(": ")[1]
            voices.append(voice_entry)

    formatted_voices = [f"{entry['Name']}-{entry['Gender']}" for entry in voices]

    if not formatted_voices:
        print("The list of Edge TTS voices could not be obtained, switching to an alternative method")
        tts_voice_list = asyncio.new_event_loop().run_until_complete(edge_tts.list_voices())
        formatted_voices = sorted([f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list])

    return formatted_voices

def speech_segment_text_to_tts(tts_text, tts_voice, filename, language, is_gui=False):
    print(tts_text, filename)
    try:
        nest_asyncio.apply() if is_gui else None
        asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except:
      try:
        tts = gTTS(tts_text, lang=fix_code_language(language))
        tts.save(filename)
        print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
      except:
        tts = gTTS('a', lang=fix_code_language(language))
        tts.save(filename)
        print('Error: Audio will be replaced.')


def audio_segmentation_to_voice(
    result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, is_gui,
    tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05
    ):

    audio_files = []
    speakers_list = []

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
            print(f"NO SPEAKER DETECT IN SEGMENT: TTS auxiliary will be used in the segment time {segment['start'], segment['text']}")

        # make the tts audio
        filename = f"audio/{start}.ogg"

        if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
            speech_segment_text_to_tts(text, speaker_to_voice[speaker], filename, TRANSLATE_AUDIO_TO, is_gui)
        elif speaker == "SPEAKER_99":
            try:
                tts = gTTS(text, lang=fix_code_language(TRANSLATE_AUDIO_TO))
                tts.save(filename)
                print('Using GTTS')
            except:
                tts = gTTS('a', lang=fix_code_language(TRANSLATE_AUDIO_TO))
                tts.save(filename)
                print('Error: Audio will be replaced.')

        # duration
        duration_true = end - start
        duration_tts = librosa.get_duration(filename=filename)

        # porcentaje
        porcentaje = duration_tts / duration_true

        if porcentaje > max_accelerate_audio:
            porcentaje = max_accelerate_audio
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
        speakers_list.append(speaker)

    return audio_files, speakers_list