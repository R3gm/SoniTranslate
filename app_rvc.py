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
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import audio_segmentation_to_voice, edge_tts_voices_list
from soni_translate.translate_segments import translate_text
from soni_translate.preprocessor import audio_video_preprocessor
from soni_translate.language_configuration import LANGUAGES, LANGUAGES_LIST, bark_voices_list, vits_voices_list
from soni_translate.utils import print_tree_directory, remove_files, select_zip_and_rar_files, download_list, manual_download, upload_model_list
from urllib.parse import unquote
from soni_translate.speech_segmentation import transcribe_speech, align_speech, diarize_speech, diarization_models
import copy, logging, rarfile, zipfile, shutil, time, json, subprocess
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)

from soni_translate.languages_gui import language_data
selected_language = os.getenv("SONITRANSLATE_LANGUAGE")
lg_conf = language_data.get(selected_language, language_data.get("english"))

device = "cuda" if torch.cuda.is_available() else "cpu"
list_compute_type = ['int8', 'float16', 'float32'] if torch.cuda.is_available() else ['int8', 'float32']
compute_type_default = 'float16' if torch.cuda.is_available() else 'float32'
whisper_model_default = 'large-v3' if torch.cuda.is_available() else 'medium'
print('Working in:', device)

# Custom voice
directories = ['downloads', 'logs', 'weights']
[os.mkdir(directory) for directory in directories if not os.path.exists(directory)]

def custom_model_voice_enable(enable_custom_voice):
    os.environ["VOICES_MODELS"] = 'ENABLE' if enable_custom_voice else 'DISABLE'

from voice_main import ClassVoices
voices_conversion = ClassVoices()

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
    diarization_model= "pyannote/speaker-diarization@2.1",
    translate_process =  "google_translator_batch",
    progress=gr.Progress(),
    ):

    if not YOUR_HF_TOKEN:
      YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
      if not YOUR_HF_TOKEN:
          raise ValueError("No valid token")
      else:
          os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

    TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
    SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]
    if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
        print("WARNING: Make sure to select a 'TTS Speaker' suitable for the translation language to avoid errors with the TTS.")

    if video is None:
        video = directory_input if os.path.exists(directory_input) else link_video
    video = video if isinstance(video, str) else video.name

    if "SET_LIMIT" == os.getenv("DEMO"):
        preview = True
        AUDIO_MIX_METHOD='Adjusting volumes and mixing audio'
        WHISPER_MODEL_SIZE="medium"
        print(
            "DEMO; set preview=True; Generation is limited to 10 seconds to prevent CPU errors. No limitations with GPU.\n"
            "DEMO; set Adjusting volumes and mixing audio\n"
            "DEMO; set whisper model to medium"
        )

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else compute_type

    global result_diarize, result, align_language, deep_copied_result

    os.makedirs('audio') if not os.path.exists('audio') else None
    os.makedirs('audio2/audio') if not os.path.exists('audio2/audio') else None
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
            raise ValueError('No active speech found in audio')

        progress(0.60, desc="Diarizing...")
        diarize_model_select = diarization_models[diarization_model]
        result_diarize = diarize_speech(audio_wav, result, min_speakers, max_speakers, YOUR_HF_TOKEN, diarize_model_select)
        print("Diarize complete")
        deep_copied_result = copy.deepcopy(result_diarize)

        progress(0.75, desc="Translating...")
        result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO, translate_process, chunk_size=1800)
        print("Translation complete")
        print(result_diarize)

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
        result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, True,
        tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05
    )

    # custom voice
    if os.getenv('VOICES_MODELS') == 'ENABLE':
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
    result_diarize["language"] = "ja" if TRANSLATE_AUDIO_TO in ["ja", "zh"] else align_language

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

title = "<center><strong><font size='7'>📽️ SoniTranslate 🈷️</font></strong></center>"
news = """ ## 📖 News
        🔥 2023/10/29: Edit the translated subtitle, download it, adjust volume and speed options.

        🔥 2023/08/03: Changed default options and added directory view of downloads..

        🔥 2023/08/02: Added support for Arabic, Czech, Danish, Finnish, Greek, Hebrew, Hungarian, Korean, Persian, Polish, Russian, Turkish, Urdu, Hindi, and Vietnamese languages. 🌐

        🔥 2023/08/01: Add options for use R.V.C. models.

        🔥 2023/07/27: Fix some bug processing the video and audio.

        🔥 2023/07/26: New UI and add mix options.
        """

def create_gui(theme, logs_in_gui=False):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(title)
        gr.Markdown(lg_conf["description"])

        with gr.Tab(lg_conf["tab_translate"]):
            with gr.Row():
                with gr.Column():
                    input_data_type = gr.inputs.Dropdown(["SUBMIT VIDEO", "URL", "Find Video Path"], default="SUBMIT VIDEO", label=lg_conf["video_source"])
                    def swap_visibility(data_type):
                        if data_type == "URL":
                            return gr.update(visible=False, value=None), gr.update(visible=True, value=''), gr.update(visible=False, value='')
                        elif data_type == "SUBMIT VIDEO":
                            return gr.update(visible=True, value=None), gr.update(visible=False, value=''), gr.update(visible=False, value='')
                        elif data_type == "Find Video Path":
                            return gr.update(visible=False, value=None), gr.update(visible=False, value=''), gr.update(visible=True, value='')
                    video_input = gr.File(label="VIDEO")
                    blink_input = gr.Textbox(visible=False, label="Media link.", info=lg_conf["link_info"], placeholder=lg_conf["link_ph"])
                    directory_input = gr.Textbox(visible=False, label="Video Path.", info=lg_conf["dir_info"], placeholder=lg_conf["dir_ph"])
                    input_data_type.change(fn=swap_visibility, inputs=input_data_type, outputs=[video_input, blink_input, directory_input])

                    link = gr.HTML()

                    SOURCE_LANGUAGE = gr.Dropdown(LANGUAGES_LIST, value='Automatic detection', label=lg_conf["sl_label"], info=lg_conf["sl_info"])
                    TRANSLATE_AUDIO_TO = gr.Dropdown(LANGUAGES_LIST[1:], value='English (en)', label=lg_conf["tat_label"], info=lg_conf["tat_info"])

                    line_ = gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["num_speakers"])
                    MAX_TTS = 6
                    min_speakers = gr.Slider(1, MAX_TTS, default=1, label=lg_conf["min_sk"], step=1, visible=False)
                    max_speakers = gr.Slider(1, MAX_TTS, value=2, step=1, label=lg_conf["max_sk"], interative=True)
                    gr.Markdown(lg_conf["tts_select"])
                    def submit(value):
                        visibility_dict = {
                            f'tts_voice{i:02d}': gr.update(visible=i < value) for i in range(6)
                        }
                        return [value for value in visibility_dict.values()]
                    tts_voice00 = gr.Dropdown(list_tts, value='en-AU-WilliamNeural-Male', label=lg_conf["sk1"], visible=True, interactive= True)
                    tts_voice01 = gr.Dropdown(list_tts, value='en-CA-ClaraNeural-Female', label=lg_conf["sk2"], visible=True, interactive= True)
                    tts_voice02 = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label=lg_conf["sk3"], visible=False, interactive= True)
                    tts_voice03 = gr.Dropdown(list_tts, value='en-GB-SoniaNeural-Female', label=lg_conf["sk4"], visible=False, interactive= True)
                    tts_voice04 = gr.Dropdown(list_tts, value='en-NZ-MitchellNeural-Male', label=lg_conf["sk4"], visible=False, interactive= True)
                    tts_voice05 = gr.Dropdown(list_tts, value='en-GB-MaisieNeural-Female', label=lg_conf["sk6"], visible=False, interactive= True)
                    max_speakers.change(submit, max_speakers, [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])

                    with gr.Column():
                          with gr.Accordion(lg_conf["extra_setting"], open=False):
                              audio_accelerate = gr.Slider(label = lg_conf["acc_max_label"], value=2.1, step=0.1, minimum=1.0, maximum=2.5, visible=True, interactive= True, info=lg_conf["acc_max_info"])

                              audio_mix_options = ['Mixing audio with sidechain compression', 'Adjusting volumes and mixing audio']
                              AUDIO_MIX = gr.Dropdown(audio_mix_options, value=audio_mix_options[1], label=lg_conf["aud_mix_label"], info=lg_conf["aud_mix_info"])
                              volume_original_mix = gr.Slider(label=lg_conf["vol_ori"], info='for <Adjusting volumes and mixing audio>', value=0.25, step=0.05, minimum=0.0, maximum=2.50, visible=True, interactive= True,)
                              volume_translated_mix = gr.Slider(label=lg_conf["vol_tra"], info='for <Adjusting volumes and mixing audio>', value=1.80, step=0.05, minimum=0.0, maximum=2.50, visible=True, interactive= True,)

                              gr.HTML("<hr></h2>")
                              sub_type_options = ["srt", "vtt", "txt", "tsv", "json", "aud"]
                              def get_subs_path(type_subs):
                                return f"sub_ori.{type_subs}", f"sub_tra.{type_subs}"
                              sub_type_output = gr.inputs.Dropdown(sub_type_options, default=sub_type_options[0], label=lg_conf["sub_type"])


                              gr.HTML("<hr></h2>")
                              gr.Markdown(lg_conf["whisper_title"])
                              whisper_model_options = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']
                              WHISPER_MODEL_SIZE = gr.inputs.Dropdown(whisper_model_options, default=whisper_model_default, label="Whisper model")
                              batch_size = gr.inputs.Slider(1, 32, default=16, label="Batch size", step=1)
                              compute_type = gr.inputs.Dropdown(list_compute_type, default=compute_type_default, label="Compute type")
                              pyannote_models_list = list(diarization_models.keys())
                              diarization_process_dropdown = gr.inputs.Dropdown(pyannote_models_list, default=pyannote_models_list[1], label="Diarization model")
                              valid_translate_process = ["google_translator_batch", "google_translator_iterative"]
                              translate_process_dropdown = gr.inputs.Dropdown(valid_translate_process, default=valid_translate_process[0], label="Translation process")

                              gr.HTML("<hr></h2>")
                              VIDEO_OUTPUT_NAME = gr.Textbox(label=lg_conf["out_name_label"] ,value="video_output.mp4", info=lg_conf["out_name_info"])
                              PREVIEW = gr.Checkbox(label="Preview", info=lg_conf["preview_info"])

                with gr.Column(variant='compact'):

                    edit_sub_check = gr.Checkbox(label=lg_conf["edit_sub_label"], info=lg_conf["edit_sub_info"])
                    dummy_false_check = gr.Checkbox(False, visible= False,)
                    def visible_component_subs(input_bool):
                        if input_bool:
                            return gr.update(visible=True), gr.update(visible=True)
                        else:
                            return gr.update(visible=False), gr.update(visible=False)
                    subs_button = gr.Button(lg_conf["button_subs"], visible= False,)
                    subs_edit_space = gr.Textbox(visible= False, lines=10, label=lg_conf["editor_sub_label"], info=lg_conf["editor_sub_info"], placeholder=lg_conf["editor_sub_ph"])
                    edit_sub_check.change(visible_component_subs, [edit_sub_check], [subs_button, subs_edit_space])

                    with gr.Row():
                        video_button = gr.Button(lg_conf["button_translate"])
                    with gr.Row():
                        video_output = gr.outputs.File(label=lg_conf["output_result_label"]) #gr.Video()
                    with gr.Row():
                        sub_ori_output = gr.outputs.File(label=lg_conf["sub_ori"])
                        sub_tra_output = gr.outputs.File(label=lg_conf["sub_tra"])

                    line_ = gr.HTML("<hr></h2>")
                    if os.getenv("YOUR_HF_TOKEN") == None or os.getenv("YOUR_HF_TOKEN") == "":
                      HFKEY = gr.Textbox(visible= True, label="HF Token", info=lg_conf["ht_token_info"], placeholder=lg_conf["ht_token_ph"])
                    else:
                      HFKEY = gr.Textbox(visible= False, label="HF Token", info=lg_conf["ht_token_info"], placeholder=lg_conf["ht_token_ph"])

                    gr.Examples(
                        examples=[
                            [
                                "./assets/Video_main.mp4",
                                "",
                                "",
                                "",
                                False,
                                "large-v3",
                                16,
                                "float16",
                                "Spanish (es)",
                                "English (en)",
                                1,
                                2,
                                'en-AU-WilliamNeural-Male',
                                'en-CA-ClaraNeural-Female',
                                'en-GB-ThomasNeural-Male',
                                'en-GB-SoniaNeural-Female',
                                'en-NZ-MitchellNeural-Male',
                                'en-GB-MaisieNeural-Female',
                                "video_output.mp4",
                                'Adjusting volumes and mixing audio',
                            ],
                            [
                                None,
                                "https://www.youtube.com/watch?v=5ZeHtRKHl7Y",
                                "",
                                "",
                                False,
                                "large-v3",
                                16,
                                "float16",
                                "Japanese (ja)",
                                "English (en)",
                                1,
                                2,
                                'en-CA-ClaraNeural-Female',
                                'en-AU-WilliamNeural-Male',
                                'en-GB-ThomasNeural-Male',
                                'en-GB-SoniaNeural-Female',
                                'en-NZ-MitchellNeural-Male',
                                'en-GB-MaisieNeural-Female',
                                "video_output.mp4",
                                'Adjusting volumes and mixing audio',
                            ],
                        ], # no update
                        fn=translate_from_video,
                        inputs=[
                        video_input,
                        blink_input,
                        directory_input,
                        HFKEY,
                        PREVIEW,
                        WHISPER_MODEL_SIZE,
                        batch_size,
                        compute_type,
                        SOURCE_LANGUAGE,
                        TRANSLATE_AUDIO_TO,
                        min_speakers,
                        max_speakers,
                        tts_voice00,
                        tts_voice01,
                        tts_voice02,
                        tts_voice03,
                        tts_voice04,
                        tts_voice05,
                        VIDEO_OUTPUT_NAME,
                        AUDIO_MIX,
                        audio_accelerate,
                        volume_original_mix,
                        volume_translated_mix,
                        sub_type_output,
                        ],
                        outputs=[video_output],
                        cache_examples=False,
                    )

        with gr.Tab("Custom voice R.V.C. (Optional)"):
            with gr.Column():
              with gr.Accordion("Get the R.V.C. Models", open=True):
                url_links = gr.Textbox(label="URLs", value="",info=lg_conf["cv_url_info"], placeholder="urls here...", lines=1)
                download_finish = gr.HTML()
                download_button = gr.Button("DOWNLOAD MODELS")

                def update_models():
                  models, index_paths = upload_model_list()
                  for i in range(8):
                    dict_models = {
                        f'model_voice_path{i:02d}': gr.update(choices=models) for i in range(8)
                    }
                    dict_index = {
                        f'file_index2_{i:02d}': gr.update(choices=index_paths) for i in range(8)
                    }
                    dict_changes = {**dict_models, **dict_index}
                    return [value for value in dict_changes.values()]

            with gr.Column():
              with gr.Accordion(lg_conf["replace_title"], open=False):
                with gr.Column(variant='compact'):
                  with gr.Column():
                    gr.Markdown(lg_conf["sec1_title"])
                    enable_custom_voice = gr.Checkbox(label="ENABLE", info=lg_conf["enable_replace"])
                    enable_custom_voice.change(custom_model_voice_enable, [enable_custom_voice], [])

                    gr.Markdown(lg_conf["sec2_title"])
                    gr.Markdown(lg_conf["sec2_subtitle"])
                    gr.Markdown(lg_conf["cv_tts1"])
                    with gr.Row():
                      model_voice_path00 = gr.Dropdown(models, label = 'Model-1', visible=True, interactive= True)
                      file_index2_00 = gr.Dropdown(index_paths, label = 'Index-1', visible=True, interactive= True)
                      name_transpose00 = gr.Number(label = 'Transpose-1', value=0, visible=True, interactive= True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_tts2"])
                    with gr.Row():
                      model_voice_path01 = gr.Dropdown(models, label='Model-2', visible=True, interactive=True)
                      file_index2_01 = gr.Dropdown(index_paths, label='Index-2', visible=True, interactive=True)
                      name_transpose01 = gr.Number(label='Transpose-2', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_tts3"])
                    with gr.Row():
                      model_voice_path02 = gr.Dropdown(models, label='Model-3', visible=True, interactive=True)
                      file_index2_02 = gr.Dropdown(index_paths, label='Index-3', visible=True, interactive=True)
                      name_transpose02 = gr.Number(label='Transpose-3', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_tts4"])
                    with gr.Row():
                      model_voice_path03 = gr.Dropdown(models, label='Model-4', visible=True, interactive=True)
                      file_index2_03 = gr.Dropdown(index_paths, label='Index-4', visible=True, interactive=True)
                      name_transpose03 = gr.Number(label='Transpose-4', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_tts5"])
                    with gr.Row():
                      model_voice_path04 = gr.Dropdown(models, label='Model-5', visible=True, interactive=True)
                      file_index2_04 = gr.Dropdown(index_paths, label='Index-5', visible=True, interactive=True)
                      name_transpose04 = gr.Number(label='Transpose-5', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_tts6"])
                    with gr.Row():
                      model_voice_path05 = gr.Dropdown(models, label='Model-6', visible=True, interactive=True)
                      file_index2_05 = gr.Dropdown(index_paths, label='Index-6', visible=True, interactive=True)
                      name_transpose05 = gr.Number(label='Transpose-6', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["cv_aux"])
                    with gr.Row():
                      model_voice_path06 = gr.Dropdown(models, label='Model-Aux', visible=True, interactive=True)
                      file_index2_06 = gr.Dropdown(index_paths, label='Index-Aux', visible=True, interactive=True)
                      name_transpose06 = gr.Number(label='Transpose-Aux', value=0, visible=True, interactive=True)
                    gr.HTML("<hr></h2>")
                    with gr.Row():
                      f0_methods_voice = ["pm", "harvest", "crepe", "rmvpe"]
                      f0_method_global = gr.Dropdown(f0_methods_voice, value='pm', label = 'Global F0 method', visible=True, interactive= True)

                with gr.Row(variant='compact'):
                  button_config = gr.Button(lg_conf["cv_button_apply"])

                  confirm_conf = gr.HTML()

                button_config.click(voices_conversion.apply_conf, inputs=[
                    f0_method_global,
                    model_voice_path00, name_transpose00, file_index2_00,
                    model_voice_path01, name_transpose01, file_index2_01,
                    model_voice_path02, name_transpose02, file_index2_02,
                    model_voice_path03, name_transpose03, file_index2_03,
                    model_voice_path04, name_transpose04, file_index2_04,
                    model_voice_path05, name_transpose05, file_index2_05,
                    model_voice_path06, name_transpose06, file_index2_06,
                    ], outputs=[confirm_conf])


              with gr.Column():
                    with gr.Accordion("Test R.V.C.", open=False):

                      with gr.Row(variant='compact'):
                        text_test = gr.Textbox(label="Text", value="This is an example",info="write a text", placeholder="...", lines=5)
                        with gr.Column():
                          tts_test = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS', visible=True, interactive= True)
                          model_voice_path07 = gr.Dropdown(models, label = 'Model', visible=True, interactive= True) #value=''
                          file_index2_07 = gr.Dropdown(index_paths, label = 'Index', visible=True, interactive= True) #value=''
                          transpose_test = gr.Number(label = 'Transpose', value=0, visible=True, interactive= True, info="integer, number of semitones, raise by an octave: 12, lower by an octave: -12")
                          f0method_test = gr.Dropdown(f0_methods_voice, value='pm', label = 'F0 method', visible=True, interactive= True)
                      with gr.Row(variant='compact'):
                        button_test = gr.Button("Test audio")

                      with gr.Column():
                        with gr.Row():
                          original_ttsvoice = gr.Audio()
                          ttsvoice = gr.Audio()

                        button_test.click(voices_conversion.make_test, inputs=[
                            text_test,
                            tts_test,
                            model_voice_path07,
                            file_index2_07,
                            transpose_test,
                            f0method_test,
                            ], outputs=[ttsvoice, original_ttsvoice])

                    download_button.click(download_list, [url_links], [download_finish]).then(update_models, [],
                                      [
                                        model_voice_path00, model_voice_path01, model_voice_path02, model_voice_path03, model_voice_path04, model_voice_path05, model_voice_path06, model_voice_path07,
                                        file_index2_00, file_index2_01, file_index2_02, file_index2_03, file_index2_04, file_index2_05, file_index2_06, file_index2_07
                                      ])


        with gr.Tab(lg_conf["tab_help"]):
            gr.Markdown(lg_conf["tutorial"])
            gr.Markdown(news)

        if logs_in_gui:
            print("Logs in gui need public url")
            import sys

            class Logger:
                def __init__(self, filename):
                    self.terminal = sys.stdout
                    self.log = open(filename, "w")
                def write(self, message):
                    self.terminal.write(message)
                    self.log.write(message)
                def flush(self):
                    self.terminal.flush()
                    self.log.flush()
                def isatty(self):
                    return False

            sys.stdout = Logger("output.log")

            def read_logs():
                sys.stdout.flush()
                with open("output.log", "r") as f:
                    return f.read()

            with gr.Accordion("Logs", open = False):
                logs = gr.Textbox(label=">>>")
                app.load(read_logs, None, logs, every=1)

        # run translate text
        subs_button.click(translate_from_video, inputs=[
            video_input,
            blink_input,
            directory_input,
            HFKEY,
            PREVIEW,
            WHISPER_MODEL_SIZE,
            batch_size,
            compute_type,
            SOURCE_LANGUAGE,
            TRANSLATE_AUDIO_TO,
            min_speakers,
            max_speakers,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            VIDEO_OUTPUT_NAME,
            AUDIO_MIX,
            audio_accelerate,
            volume_original_mix,
            volume_translated_mix,
            sub_type_output,
            edit_sub_check, # TRUE BY DEFAULT
            dummy_false_check, # dummy false
            subs_edit_space,
            diarization_process_dropdown,
            translate_process_dropdown,
            ], outputs=subs_edit_space)

        # run translate
        video_button.click(translate_from_video, inputs=[
            video_input,
            blink_input,
            directory_input,
            HFKEY,
            PREVIEW,
            WHISPER_MODEL_SIZE,
            batch_size,
            compute_type,
            SOURCE_LANGUAGE,
            TRANSLATE_AUDIO_TO,
            min_speakers,
            max_speakers,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            VIDEO_OUTPUT_NAME,
            AUDIO_MIX,
            audio_accelerate,
            volume_original_mix,
            volume_translated_mix,
            sub_type_output,
            dummy_false_check,
            edit_sub_check,
            subs_edit_space,
            diarization_process_dropdown,
            translate_process_dropdown,
            ], outputs=video_output).then(get_subs_path, [sub_type_output], [sub_ori_output, sub_tra_output])
    return app


import argparse

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theme", type=str, default="Taithrah/Minimal", help="Specify the theme; find themes in https://huggingface.co/spaces/gradio/theme-gallery; Example: --theme aliabid94/new-theme")
    parser.add_argument("--public_url", action="store_true", default=False, help="Enable public link")
    parser.add_argument("--logs_in_gui", action="store_true", default=False, help="Displays the operations performed in Logs")
    return parser

if __name__ == '__main__':
    list_edge = edge_tts_voices_list()
    list_bark = list(bark_voices_list.keys())
    list_vits = list(vits_voices_list.keys())
    list_tts = sorted(list_edge + list_bark + list_vits)
    models, index_paths = upload_model_list()
    os.environ["VOICES_MODELS"] = 'DISABLE'

    parser = create_parser()

    args = parser.parse_args()
    # Simulating command-line arguments
    #args_list = "--theme aliabid94/new-theme --public_url".split()
    #args = parser.parse_args(args_list)

    app = create_gui(args.theme, logs_in_gui=args.logs_in_gui)
    app.launch(
        share=args.public_url,
        show_error=True,
        enable_queue=True,
        quiet=False,
        debug=True
    )