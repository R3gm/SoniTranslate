from gtts import gTTS
import edge_tts, asyncio, nest_asyncio
from tqdm import tqdm
import librosa, os, re, subprocess, torch, gc
from language_configuration import fix_code_language
import numpy as np
#from scipy.io.wavfile import write as write_wav
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype_env = torch.float16 if torch.cuda.is_available() else torch.float32

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

def edge_request_tts(tts_text, tts_voice, filename, language, is_gui=False):
    print(tts_text, filename)
    try:
        #nest_asyncio.apply() if not is_gui else None
        asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except Exception as error:
      print(str(error))
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
        edge_request_tts(text, tts_name, filename, TRANSLATE_AUDIO_TO, is_gui)


bark_voices_list = {
    'de_speaker_0-Male BARK': 'v2/de_speaker_0',
    'de_speaker_1-Male BARK': 'v2/de_speaker_1',
    'de_speaker_2-Male BARK': 'v2/de_speaker_2',
    'de_speaker_3-Female BARK': 'v2/de_speaker_3',
    'de_speaker_4-Male BARK': 'v2/de_speaker_4',
    'de_speaker_5-Male BARK': 'v2/de_speaker_5',
    'de_speaker_6-Male BARK': 'v2/de_speaker_6',
    'de_speaker_7-Male BARK': 'v2/de_speaker_7',
    'de_speaker_8-Female BARK': 'v2/de_speaker_8',
    'de_speaker_9-Male BARK': 'v2/de_speaker_9',
    'en_speaker_0-Male BARK': 'v2/en_speaker_0',
    'en_speaker_1-Male BARK': 'v2/en_speaker_1',
    'en_speaker_2-Male BARK': 'v2/en_speaker_2',
    'en_speaker_3-Male BARK': 'v2/en_speaker_3',
    'en_speaker_4-Male BARK': 'v2/en_speaker_4',
    'en_speaker_5-Male BARK': 'v2/en_speaker_5',
    'en_speaker_6-Male BARK': 'v2/en_speaker_6',
    'en_speaker_7-Male BARK': 'v2/en_speaker_7',
    'en_speaker_8-Male BARK': 'v2/en_speaker_8',
    'en_speaker_9-Female BARK': 'v2/en_speaker_9',
    'es_speaker_0-Male BARK': 'v2/es_speaker_0',
    'es_speaker_1-Male BARK': 'v2/es_speaker_1',
    'es_speaker_2-Male BARK': 'v2/es_speaker_2',
    'es_speaker_3-Male BARK': 'v2/es_speaker_3',
    'es_speaker_4-Male BARK': 'v2/es_speaker_4',
    'es_speaker_5-Male BARK': 'v2/es_speaker_5',
    'es_speaker_6-Male BARK': 'v2/es_speaker_6',
    'es_speaker_7-Male BARK': 'v2/es_speaker_7',
    'es_speaker_8-Female BARK': 'v2/es_speaker_8',
    'es_speaker_9-Female BARK': 'v2/es_speaker_9',
    'fr_speaker_0-Male BARK': 'v2/fr_speaker_0',
    'fr_speaker_1-Female BARK': 'v2/fr_speaker_1',
    'fr_speaker_2-Female BARK': 'v2/fr_speaker_2',
    'fr_speaker_3-Male BARK': 'v2/fr_speaker_3',
    'fr_speaker_4-Male BARK': 'v2/fr_speaker_4',
    'fr_speaker_5-Female BARK': 'v2/fr_speaker_5',
    'fr_speaker_6-Male BARK': 'v2/fr_speaker_6',
    'fr_speaker_7-Male BARK': 'v2/fr_speaker_7',
    'fr_speaker_8-Male BARK': 'v2/fr_speaker_8',
    'fr_speaker_9-Male BARK': 'v2/fr_speaker_9',
    'hi_speaker_0-Female BARK': 'v2/hi_speaker_0',
    'hi_speaker_1-Female BARK': 'v2/hi_speaker_1',
    'hi_speaker_2-Male BARK': 'v2/hi_speaker_2',
    'hi_speaker_3-Female BARK': 'v2/hi_speaker_3',
    'hi_speaker_4-Female BARK': 'v2/hi_speaker_4',
    'hi_speaker_5-Male BARK': 'v2/hi_speaker_5',
    'hi_speaker_6-Male BARK': 'v2/hi_speaker_6',
    'hi_speaker_7-Male BARK': 'v2/hi_speaker_7',
    'hi_speaker_8-Male BARK': 'v2/hi_speaker_8',
    'hi_speaker_9-Female BARK': 'v2/hi_speaker_9',
    'it_speaker_0-Male BARK': 'v2/it_speaker_0',
    'it_speaker_1-Male BARK': 'v2/it_speaker_1',
    'it_speaker_2-Female BARK': 'v2/it_speaker_2',
    'it_speaker_3-Male BARK': 'v2/it_speaker_3',
    'it_speaker_4-Male BARK': 'v2/it_speaker_4',
    'it_speaker_5-Male BARK': 'v2/it_speaker_5',
    'it_speaker_6-Male BARK': 'v2/it_speaker_6',
    'it_speaker_7-Female BARK': 'v2/it_speaker_7',
    'it_speaker_8-Male BARK': 'v2/it_speaker_8',
    'it_speaker_9-Female BARK': 'v2/it_speaker_9',
    'ja_speaker_0-Female BARK': 'v2/ja_speaker_0',
    'ja_speaker_1-Female BARK': 'v2/ja_speaker_1',
    'ja_speaker_2-Male BARK': 'v2/ja_speaker_2',
    'ja_speaker_3-Female BARK': 'v2/ja_speaker_3',
    'ja_speaker_4-Female BARK': 'v2/ja_speaker_4',
    'ja_speaker_5-Female BARK': 'v2/ja_speaker_5',
    'ja_speaker_6-Male BARK': 'v2/ja_speaker_6',
    'ja_speaker_7-Female BARK': 'v2/ja_speaker_7',
    'ja_speaker_8-Female BARK': 'v2/ja_speaker_8',
    'ja_speaker_9-Female BARK': 'v2/ja_speaker_9',
    'ko_speaker_0-Female BARK': 'v2/ko_speaker_0',
    'ko_speaker_1-Male BARK': 'v2/ko_speaker_1',
    'ko_speaker_2-Male BARK': 'v2/ko_speaker_2',
    'ko_speaker_3-Male BARK': 'v2/ko_speaker_3',
    'ko_speaker_4-Male BARK': 'v2/ko_speaker_4',
    'ko_speaker_5-Male BARK': 'v2/ko_speaker_5',
    'ko_speaker_6-Male BARK': 'v2/ko_speaker_6',
    'ko_speaker_7-Male BARK': 'v2/ko_speaker_7',
    'ko_speaker_8-Male BARK': 'v2/ko_speaker_8',
    'ko_speaker_9-Male BARK': 'v2/ko_speaker_9',
    'pl_speaker_0-Male BARK': 'v2/pl_speaker_0',
    'pl_speaker_1-Male BARK': 'v2/pl_speaker_1',
    'pl_speaker_2-Male BARK': 'v2/pl_speaker_2',
    'pl_speaker_3-Male BARK': 'v2/pl_speaker_3',
    'pl_speaker_4-Female BARK': 'v2/pl_speaker_4',
    'pl_speaker_5-Male BARK': 'v2/pl_speaker_5',
    'pl_speaker_6-Female BARK': 'v2/pl_speaker_6',
    'pl_speaker_7-Male BARK': 'v2/pl_speaker_7',
    'pl_speaker_8-Male BARK': 'v2/pl_speaker_8',
    'pl_speaker_9-Female BARK': 'v2/pl_speaker_9',
    'pt_speaker_0-Male BARK': 'v2/pt_speaker_0',
    'pt_speaker_1-Male BARK': 'v2/pt_speaker_1',
    'pt_speaker_2-Male BARK': 'v2/pt_speaker_2',
    'pt_speaker_3-Male BARK': 'v2/pt_speaker_3',
    'pt_speaker_4-Male BARK': 'v2/pt_speaker_4',
    'pt_speaker_5-Male BARK': 'v2/pt_speaker_5',
    'pt_speaker_6-Male BARK': 'v2/pt_speaker_6',
    'pt_speaker_7-Male BARK': 'v2/pt_speaker_7',
    'pt_speaker_8-Male BARK': 'v2/pt_speaker_8',
    'pt_speaker_9-Male BARK': 'v2/pt_speaker_9',
    'ru_speaker_0-Male BARK': 'v2/ru_speaker_0',
    'ru_speaker_1-Male BARK': 'v2/ru_speaker_1',
    'ru_speaker_2-Male BARK': 'v2/ru_speaker_2',
    'ru_speaker_3-Male BARK': 'v2/ru_speaker_3',
    'ru_speaker_4-Male BARK': 'v2/ru_speaker_4',
    'ru_speaker_5-Female BARK': 'v2/ru_speaker_5',
    'ru_speaker_6-Female BARK': 'v2/ru_speaker_6',
    'ru_speaker_7-Male BARK': 'v2/ru_speaker_7',
    'ru_speaker_8-Male BARK': 'v2/ru_speaker_8',
    'ru_speaker_9-Female BARK': 'v2/ru_speaker_9',
    'tr_speaker_0-Male BARK': 'v2/tr_speaker_0',
    'tr_speaker_1-Male BARK': 'v2/tr_speaker_1',
    'tr_speaker_2-Male BARK': 'v2/tr_speaker_2',
    'tr_speaker_3-Male BARK': 'v2/tr_speaker_3',
    'tr_speaker_4-Female BARK': 'v2/tr_speaker_4',
    'tr_speaker_5-Female BARK': 'v2/tr_speaker_5',
    'tr_speaker_6-Male BARK': 'v2/tr_speaker_6',
    'tr_speaker_7-Male BARK': 'v2/tr_speaker_7',
    'tr_speaker_8-Male BARK': 'v2/tr_speaker_8',
    'tr_speaker_9-Male BARK': 'v2/tr_speaker_9',
    'zh_speaker_0-Male BARK': 'v2/zh_speaker_0',
    'zh_speaker_1-Male BARK': 'v2/zh_speaker_1',
    'zh_speaker_2-Male BARK': 'v2/zh_speaker_2',
    'zh_speaker_3-Male BARK': 'v2/zh_speaker_3',
    'zh_speaker_4-Female BARK': 'v2/zh_speaker_4',
    'zh_speaker_5-Male BARK': 'v2/zh_speaker_5',
    'zh_speaker_6-Female BARK': 'v2/zh_speaker_6',
    'zh_speaker_7-Female BARK': 'v2/zh_speaker_7',
    'zh_speaker_8-Male BARK': 'v2/zh_speaker_8',
    'zh_speaker_9-Female BARK': 'v2/zh_speaker_9'
}

def segments_bark_tts(filtered_bark_segments, TRANSLATE_AUDIO_TO, model_id_bark="suno/bark-small"):
    from transformers import AutoProcessor, AutoModel, BarkModel
    from optimum.bettertransformer import BetterTransformer

    # load model bark
    model = BarkModel.from_pretrained(model_id_bark, torch_dtype=torch_dtype_env).to(device)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id_bark, return_tensors="pt") # , padding=True
    if torch.cuda.is_available():
        # convert to bettertransformer
        model = BetterTransformer.transform(model, keep_original_model=False)
        # enable CPU offload
        #model.enable_cpu_offload()
    sampling_rate = model.generation_config.sample_rate

    #filtered_segments = filtered_bark_segments['segments']
    # Sorting the segments by 'tts_name'
    #sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    #print(sorted_segments)

    for segment in tqdm(filtered_bark_segments['segments']):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        inputs = processor(text, voice_preset=bark_voices_list[tts_name]).to(device)

        # make the tts audio
        filename = f"audio/{start}.ogg"
        print(text, filename)
        try:
            # Infer
            with torch.inference_mode():
                speech_output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8, pad_token_id=processor.tokenizer.pad_token_id)
            # Save file
            sf.write(
                file=filename,
                samplerate=sampling_rate,
                data=speech_output.cpu().numpy().squeeze().astype(np.float32),
                format='ogg', subtype='vorbis'
            )
        except Exception as error:
            print(f"Error: {str(error)}")
            try:
              tts = gTTS(tts_text, lang=fix_code_language(TRANSLATE_AUDIO_TO))
              tts.save(filename)
              print(f'For {tts_name} the TTS auxiliary will be used')
            except:
              sample_rate_aux = 22050
              duration = float(segment['end']) - float(segment['start'])
              data = np.zeros(int(sample_rate_aux * duration)).astype(np.float32)
              sf.write(filename, data, sample_rate_aux, format='ogg', subtype='vorbis')
              print('Error: Audio will be replaced -> [silent audio].')
        gc.collect(); torch.cuda.empty_cache()
    del processor; del model; gc.collect(); torch.cuda.empty_cache()

def audio_segmentation_to_voice(
    result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, is_gui,
    tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05,
    model_id_bark="suno/bark-small"
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
    pattern_bark = re.compile(r'.* BARK$')

    speakers_edge = [speaker for speaker, voice in speaker_to_voice.items() if pattern_edge.match(voice)]
    speakers_bark = [speaker for speaker, voice in speaker_to_voice.items() if pattern_bark.match(voice)]

    # Filter method in segments
    filtered_edge = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_edge]}
    filtered_bark = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_bark]}

    # Infer
    if filtered_edge["segments"]:
        print(f"EDGE TTS: {speakers_edge}")
        segments_egde_tts(filtered_edge, TRANSLATE_AUDIO_TO, is_gui) # ogg
    if filtered_bark["segments"]:
        print(f"BARK TTS: {speakers_bark}")
        segments_bark_tts(filtered_bark, TRANSLATE_AUDIO_TO, model_id_bark) # wav

    [result.pop('tts_name', None) for result in result_diarize['segments']] # see if retain the tts_name in debug
    return accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge, speakers_bark)


def accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge, speakers_bark):

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
        elif speaker in speakers_bark:
            filename = f"audio/{start}.ogg" # wav

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

if __name__ == '__main__':
    from segments import result_diarize

    audio_segmentation_to_voice(
        result_diarize,
        TRANSLATE_AUDIO_TO="en",
        max_accelerate_audio=2.1,
        is_gui= True,
        tts_voice00='es_speaker_8-Female BARK',
        tts_voice01="en-CA-ClaraNeural-Female",
        tts_voice02="en-GB-ThomasNeural-Male",
        tts_voice03="en-GB-SoniaNeural-Female",
        tts_voice04="en-NZ-MitchellNeural-Male",
        tts_voice05="en-GB-MaisieNeural-Female",
        )
