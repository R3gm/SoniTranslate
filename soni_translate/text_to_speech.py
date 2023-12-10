from gtts import gTTS
import edge_tts, asyncio, nest_asyncio
from tqdm import tqdm
import librosa, os, re, subprocess
from .language_configuration import fix_code_language

class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)

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

    if not formatted_voices:
        print("Can't get EDGE TTS - list voices")

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


def segments_egde_tts(filtered_edge_segments, TRANSLATE_AUDIO_TO, is_gui):

    for segment in tqdm(filtered_edge_segments['segments']):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        # make the tts audio
        filename = f"audio/{start}.ogg"
        speech_segment_text_to_tts(text, tts_name, filename, TRANSLATE_AUDIO_TO, is_gui)

    #return audio_files, speakers_list

def audio_segmentation_to_voice(
    result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, is_gui,
    tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05
    ):

    # Mapping speakers to voice variables
    speaker_to_voice = {
        'SPEAKER_00': tts_voice00,
        'SPEAKER_01': tts_voice01,
        'SPEAKER_02': tts_voice02,
        'SPEAKER_03': tts_voice03,
        'SPEAKER_04': tts_voice04,
        'SPEAKER_05': tts_voice05
    }

    # Assign 'SPEAKER_00' to segments without a 'speaker' key
    for segment in result_diarize['segments']:
        if 'speaker' not in segment:
            segment['speaker'] = 'SPEAKER_00'
            print(f"NO SPEAKER DETECT IN SEGMENT: First TTS will be used in the segment time {segment['start'], segment['text']}")
         # Assign the TTS name
        segment['tts_name'] = speaker_to_voice[segment['speaker']]

    # Find TTS method
    pattern_edge = re.compile(r'.*-(Male|Female)$')

    speakers_edge = [speaker for speaker, voice in speaker_to_voice.items() if pattern_edge.match(voice)]

    # Filter method in segments
    filtered_edge = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_edge]}

    # Infer
    if speakers_edge:
        print(f"EDGE TTS: {speakers_edge}")
        segments_egde_tts(filtered_edge, TRANSLATE_AUDIO_TO, is_gui) # ogg

    [result.pop('tts_name', None) for result in result_diarize['segments']] # see if retain the tts_name in debug
    return accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge)


def accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge):

    print("Apply acceleration")
    audio_files = []
    speakers_list = []
    for segment in tqdm(result_diarize['segments']):

        text = segment['text']
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']

        # find name audio
        if speaker in speakers_edge:
            filename = f"audio/{start}.ogg"

        # duration
        duration_true = end - start
        duration_tts = librosa.get_duration(filename=filename)

        # Accelerate percentage
        acc_percentage = duration_tts / duration_true

        if acc_percentage > max_accelerate_audio:
            acc_percentage = max_accelerate_audio
        elif acc_percentage <= 1.2 and acc_percentage >= 0.8:
            acc_percentage = 1.0
        elif acc_percentage <= 0.79:
            acc_percentage = 0.8

        # Smoth and round
        acc_percentage = round(acc_percentage+0.0, 1)

        # apply aceleration or opposite to the audio file in audio2 folder
        os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={acc_percentage} audio2/{filename}")

        duration_create = librosa.get_duration(filename=f"audio2/{filename}")
        print(acc_percentage, duration_tts, duration_create)
        audio_files.append(filename)
        speakers_list.append(speaker)

    return audio_files, speakers_list
