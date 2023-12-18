from gtts import gTTS
import edge_tts, asyncio, nest_asyncio
from tqdm import tqdm
import librosa, os, re, torch, gc, subprocess
from .language_configuration import fix_code_language, bark_voices_list, vits_voices_list
import numpy as np
from typing import Any, Dict
from pathlib import Path
#from scipy.io.wavfile import write as write_wav
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype_env = torch.float16 if torch.cuda.is_available() else torch.float32

class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)

def verify_saved_file_and_size(filename):
    if not os.path.exists(filename):
        raise TTS_OperationError(f"File '{filename}' was not saved.")
    if os.path.getsize(filename) == 0:
        raise TTS_OperationError(f"File '{filename}' has a zero size. Related to incorrect TTS for the target language")

def error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename):
    print(f"Error: {str(error)}")
    try:
        tts = gTTS(segment["text"], lang=fix_code_language(TRANSLATE_AUDIO_TO))
        tts.save(filename)
        print(f'TTS auxiliary will be utilized rather than TTS: {segment["tts_name"]}')
        verify_saved_file_and_size(filename)
    except Exception as error:
        print(f"Error: {str(error)}")
        sample_rate_aux = 22050
        duration = float(segment['end']) - float(segment['start'])
        data = np.zeros(int(sample_rate_aux * duration)).astype(np.float32)
        sf.write(filename, data, sample_rate_aux, format='ogg', subtype='vorbis')
        print('Error: Audio will be replaced -> [silent audio].')
        verify_saved_file_and_size(filename)

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

def segments_egde_tts(filtered_edge_segments, TRANSLATE_AUDIO_TO, is_gui):

    for segment in tqdm(filtered_edge_segments['segments']):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        # make the tts audio
        filename = f"audio/{start}.ogg"

        print(text, filename)
        try:
            #nest_asyncio.apply() if not is_gui else None
            asyncio.run(edge_tts.Communicate(text, "-".join(tts_name.split('-')[:-1])).save(filename))
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)

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
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect(); torch.cuda.empty_cache()
    try:
        del processor; del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass


def uromanize(input_string):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    #script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    if not os.path.exists("./uroman"):
        print("Clonning repository uroman https://github.com/isi-nlp/uroman.git for romanize the text")
        process = subprocess.Popen(["git", "clone", "https://github.com/isi-nlp/uroman.git"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    script_path = os.path.join("./uroman", "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

def segments_vits_tts(filtered_vits_segments, TRANSLATE_AUDIO_TO):
    from transformers import VitsModel, AutoTokenizer

    filtered_segments = filtered_vits_segments['segments']
    # Sorting the segments by 'tts_name'
    sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    print(sorted_segments)

    model_name_key = None
    for segment in tqdm(sorted_segments):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        if tts_name != model_name_key:
            model_name_key = tts_name
            model = VitsModel.from_pretrained(vits_voices_list[tts_name])
            tokenizer = AutoTokenizer.from_pretrained(vits_voices_list[tts_name])
            sampling_rate = model.config.sampling_rate

        if tokenizer.is_uroman:
            romanize_text = uromanize(text)
            print(f"Romanize text: {romanize_text}")
            inputs = tokenizer(romanize_text, return_tensors="pt")
        else:
            inputs = tokenizer(text, return_tensors="pt")

        # make the tts audio
        filename = f"audio/{start}.ogg"
        print(text, filename)
        try:
            # Infer
            with torch.no_grad():
              speech_output = model(**inputs).waveform
            # Save file
            sf.write(
                file=filename,
                samplerate=sampling_rate,
                data=speech_output.cpu().numpy().squeeze().astype(np.float32),
                format='ogg', subtype='vorbis'
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect(); torch.cuda.empty_cache()
    try:
        del tokenizer; del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass


def segments_coqui_tts(filtered_coqui_segments, TRANSLATE_AUDIO_TO, model_id_coqui="tts_models/multilingual/multi-dataset/xtts_v2", emotion=None):
    """ XTTS
    Install:
    pip install -q TTS==0.21.1
    pip install -q numpy==1.23.5

    Notes:
    - tts_name is the wav|mp3|ogg|m4a file for VC
    """
    from TTS.api import TTS

    TRANSLATE_AUDIO_TO = fix_code_language(TRANSLATE_AUDIO_TO, syntax="coqui")
    supported_lang_coqui = ['zh-cn', 'en', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'es', 'hu', 'ko', 'ja']
    if TRANSLATE_AUDIO_TO not in supported_lang_coqui:
        raise TTS_OperationError(f"'{TRANSLATE_AUDIO_TO}' is not a supported language for Coqui XTTS")
    #Emotion and speed can only be used with Coqui Studio models. Which is discontinued
    #emotions = ["Neutral", "Happy", "Sad", "Angry", "Dull"]

    # Init TTS
    model = TTS(model_id_coqui).to(device)
    sampling_rate = 24000

    #filtered_segments = filtered_coqui_segments['segments']
    # Sorting the segments by 'tts_name'
    #sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    #print(sorted_segments)

    for segment in tqdm(filtered_coqui_segments['segments']):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        # make the tts audio
        filename = f"audio/{start}.ogg"
        print(text, filename)
        try:
            # Infer
            wav = model.tts(text=text, speaker_wav=tts_name, language=TRANSLATE_AUDIO_TO)
            # Save file
            sf.write(
                file=filename,
                samplerate=sampling_rate,
                data=wav,
                format='ogg', subtype='vorbis'
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect(); torch.cuda.empty_cache()
    try:
        del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass


def load_piper_model(model: str, data_dir: list, download_dir: str = '', update_voices: bool = False):
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, find_voice, get_voices

    try:
        import onnxruntime as rt
        if rt.get_device() == 'GPU':
            print("onnxruntime device > GPU")
            cuda = True
        else:
            print("onnxruntime device > CPU") # try pip install onnxruntime-gpu
            cuda = False
    except Exception as error:
        raise TTS_OperationError(f"onnxruntime error: {str(error)}")

    if not download_dir:
        # Download to first data directory by default
        download_dir = data_dir[0]

    # Download voice if file doesn't exist
    model_path = Path(model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(download_dir, update_voices=update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(model, data_dir, download_dir, voices_info)
        model, config = find_voice(model, data_dir)

    # Load voice
    voice = PiperVoice.load(model, config_path=config, use_cuda=cuda)

    return voice

def synthesize_text_to_audio_np_array(voice, text, synthesize_args):
    audio_stream = voice.synthesize_stream_raw(text, **synthesize_args)

    # Collect the audio bytes into a single NumPy array
    audio_data = b''
    for audio_bytes in audio_stream:
        audio_data += audio_bytes

    # Ensure correct data type and convert audio bytes to NumPy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    return audio_np

def segments_vits_onnx_tts(filtered_onnx_vits_segments, TRANSLATE_AUDIO_TO):
    """
    Install:
    pip install -q piper-tts==1.2.0 onnxruntime-gpu
    """
    
    data_dir = [str(Path.cwd())] # "Data directory to check for downloaded models (default: current directory)"
    download_dir = "piper_vits_onnx_models"
    #model_name = "en_US-lessac-medium" tts_name in a dict like VITS
    update_voices = True # "Download latest voices.json during startup",

    synthesize_args = {
        "speaker_id": None,
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.0,
    }

    filtered_segments = filtered_onnx_vits_segments['segments']
    # Sorting the segments by 'tts_name'
    sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    print(sorted_segments)

    model_name_key = None
    for segment in tqdm(sorted_segments):

        speaker = segment['speaker']
        text = segment['text']
        start = segment['start']
        tts_name = segment['tts_name']

        if tts_name != model_name_key:
            model_name_key = tts_name
            model = load_piper_model(tts_name, data_dir, download_dir, update_voices)
            sampling_rate = model.config.sample_rate

        # make the tts audio
        filename = f"audio/{start}.ogg"
        print(text, filename)
        try:
            # Infer
            speech_output = synthesize_text_to_audio_np_array(model, text, synthesize_args)
            # Save file
            sf.write(
                file=filename,
                samplerate=sampling_rate,
                data=speech_output, #.cpu().numpy().squeeze().astype(np.float32),
                format='ogg', subtype='vorbis'
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect(); torch.cuda.empty_cache()
    try:
        del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass


def audio_segmentation_to_voice(
    result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, is_gui,
    tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05,
    model_id_bark="suno/bark-small",
    model_id_coqui="tts_models/multilingual/multi-dataset/xtts_v2"
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
    pattern_vits = re.compile(r'.* VITS$')
    pattern_coqui = re.compile(r'.+\.(wav|mp3|ogg|m4a)$')

    speakers_edge = [speaker for speaker, voice in speaker_to_voice.items() if pattern_edge.match(voice)]
    speakers_bark = [speaker for speaker, voice in speaker_to_voice.items() if pattern_bark.match(voice)]
    speakers_vits = [speaker for speaker, voice in speaker_to_voice.items() if pattern_vits.match(voice)]
    speakers_coqui = [speaker for speaker, voice in speaker_to_voice.items() if pattern_coqui.match(voice)]

    # Filter method in segments
    filtered_edge = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_edge]}
    filtered_bark = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_bark]}
    filtered_vits = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_vits]}
    filtered_coqui = {"segments": [segment for segment in result_diarize['segments'] if segment['speaker'] in speakers_coqui]}

    # Infer
    if filtered_edge["segments"]:
        print(f"EDGE TTS: {speakers_edge}")
        segments_egde_tts(filtered_edge, TRANSLATE_AUDIO_TO, is_gui) # mp3
    if filtered_bark["segments"]:
        print(f"BARK TTS: {speakers_bark}")
        segments_bark_tts(filtered_bark, TRANSLATE_AUDIO_TO, model_id_bark) # wav
    if filtered_vits["segments"]:
        print(f"VITS TTS: {speakers_vits}")
        segments_vits_tts(filtered_vits, TRANSLATE_AUDIO_TO) # wav
    if filtered_coqui["segments"]:
        print(f"Coqui TTS: {speakers_coqui}")
        segments_coqui_tts(filtered_vits, TRANSLATE_AUDIO_TO, model_id_coqui) # wav

    [result.pop('tts_name', None) for result in result_diarize['segments']]
    return accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge, speakers_bark, speakers_vits, speakers_coqui)


def accelerate_segments(result_diarize, max_accelerate_audio, speakers_edge, speakers_bark, speakers_vits, speakers_coqui):

    print("Apply acceleration")
    audio_files = []
    speakers_list = []
    for segment in tqdm(result_diarize['segments']):

        text = segment['text']
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']

        # find name audio
        #if speaker in speakers_edge:
        filename = f"audio/{start}.ogg"
        #elif speaker in speakers_bark + speakers_vits + speakers_coqui:
        #    filename = f"audio/{start}.wav" # wav

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
        tts_voice00='en-facebook-mms VITS',
        tts_voice01="en-CA-ClaraNeural-Female",
        tts_voice02="en-GB-ThomasNeural-Male",
        tts_voice03="en-GB-SoniaNeural-Female",
        tts_voice04="en-NZ-MitchellNeural-Male",
        tts_voice05="en-GB-MaisieNeural-Female",
        )
