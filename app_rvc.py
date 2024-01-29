import gradio as gr
import whisperx
import torch
import os
from tqdm import tqdm  # noqa
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
)
from soni_translate.translate_segments import translate_text
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.language_configuration import (
    LANGUAGES,
    LANGUAGES_LIST,
    bark_voices_list,
    vits_voices_list,
)
from soni_translate.utils import (
    remove_files,
    download_list,
    upload_model_list,
    download_manager,
    move_files,
    run_command,
    is_audio_file,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.logging_setup import logger, set_logging_level
from soni_translate.speech_segmentation import (
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    srt_file_to_segments,
    document_preprocessor,
    determine_chunk_size,
    plain_text_to_segments,
    segments_to_plain_text,
    process_subtitles,
)
from soni_translate.languages_gui import language_data
import copy
import logging
import json
from voice_main import ClassVoices
import argparse

try:
    from piper import PiperVoice  # noqa

    piper_enabled = True
    logger.info("PIPER TTS enabled")
except Exception as error:
    logger.warning(str(error))
    piper_enabled = False
    logger.info("PIPER TTS disabled")
try:
    from TTS.api import TTS  # noqa

    xtts_enabled = True
    logger.info("Coqui XTTS enabled")
    logger.info(
        "In this app, by using Coqui TTS (text-to-speech), you "
        "acknowledge and agree to the license.\n"
        "You confirm that you have read, understood, and agreed "
        "to the Terms and Conditions specified at the following link:\n"
        "https://coqui.ai/cpml.txt."
    )
    os.environ["COQUI_TOS_AGREED"] = "1"
except Exception as error:
    logger.warning(str(error))
    xtts_enabled = False
    logger.info("Coqui XTTS disabled")


class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(bark_voices_list.keys())
        self.list_vits = list(vits_voices_list.keys())
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = (
            piper_tts_voices_list() if self.piper_enabled else []
        )
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = (
            coqui_xtts_voices_list() if self.xtts_enabled else []
        )
        list_tts = sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_vits_onnx
            + self.list_coqui_xtts
        )
        return list_tts


modules = ["numba", "httpx", "markdown_it", "speechbrain", "fairseq"]
for module in modules:
    logging.getLogger(module).setLevel(logging.WARNING)


device = "cuda" if torch.cuda.is_available() else "cpu"
list_compute_type = (
    ["int8", "float16", "float32"]
    if torch.cuda.is_available()
    else ["int8", "float32"]
)
compute_type_default = "float16" if torch.cuda.is_available() else "float32"
whisper_model_default = "large-v3" if torch.cuda.is_available() else "medium"
logger.info(f"Working in: {device}")

# Custom voice
directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
]
[
    os.makedirs(directory)
    for directory in directories
    if not os.path.exists(directory)
]


def custom_model_voice_enable(enable_custom_voice):
    os.environ["VOICES_MODELS"] = (
        "ENABLE" if enable_custom_voice else "DISABLE"
    )


voices_conversion = ClassVoices()


def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)

def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)

class SoniTranslate:
    def __init__(self, dev=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None

    def multilingual_media_conversion(
        self,
        media_file,
        link_media,
        directory_input,
        YOUR_HF_TOKEN,
        preview=False,
        WHISPER_MODEL_SIZE="large-v3",
        batch_size=16,
        compute_type="float16",
        SOURCE_LANGUAGE="Automatic detection",
        TRANSLATE_AUDIO_TO="English (en)",
        min_speakers=1,
        max_speakers=2,
        tts_voice00="en-AU-WilliamNeural-Male",
        tts_voice01="en-CA-ClaraNeural-Female",
        tts_voice02="en-GB-ThomasNeural-Male",
        tts_voice03="en-GB-SoniaNeural-Female",
        tts_voice04="en-NZ-MitchellNeural-Male",
        tts_voice05="en-GB-MaisieNeural-Female",
        video_output_name="video_dub",
        AUDIO_MIX_METHOD="Adjusting volumes and mixing audio",
        max_accelerate_audio=2.1,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle="srt",
        get_translated_text=False,
        get_video_from_text_json=False,
        text_json="{}",
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        subtitle_file=None,
        output_type="video",
        voiceless_track=False,
        voice_imitation=False,
        voice_imitation_max_segments=3,
        voice_imitation_vocals_dereverb=False,
        voice_imitation_remove_previous=True,
        dereverb_automatic_xtts=True,
        is_gui=False,
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
            wrn_lang = "Make sure to select a 'TTS Speaker' suitable for the translation language to avoid errors with the TTS." # noqa
            warn_disp(wrn_lang, is_gui)

        if "_XTTS_" in tts_voice00 and voice_imitation:
            wrn_lang = "When you select XTTS, it is advisable to disable Voice Imitation." # noqa
            warn_disp(wrn_lang, is_gui)

        if os.getenv("VOICES_MODELS") == "ENABLE" and voice_imitation:
            wrn_lang = "When you use R.V.C. models, it is advisable to disable Voice Imitation." # noqa
            warn_disp(wrn_lang, is_gui)

        if media_file is None:
            media_file = (
                directory_input
                if os.path.exists(directory_input)
                else link_media
            )
        media_file = (
            media_file if isinstance(media_file, str) else media_file.name
        )

        if "SET_LIMIT" == os.getenv("DEMO"):
            preview = True
            AUDIO_MIX_METHOD = "Adjusting volumes and mixing audio"
            WHISPER_MODEL_SIZE = "medium"
            logger.info(
                "DEMO; set preview=True; Generation is limited to "
                "10 seconds to prevent CPU errors. No limitations with GPU.\n"
                "DEMO; set Adjusting volumes and mixing audio\n"
                "DEMO; set whisper model to medium"
            )

        # Check GPU
        compute_type = "float32" if self.device == "cpu" else compute_type

        base_video_file = "Video.mp4"
        base_audio_wav = "audio.wav"
        dub_audio_file = "audio_dub_solo.ogg"
        mix_audio_file = "audio_mix.mp3"
        video_output_file = f"{video_output_name}.mp4"

        if not get_video_from_text_json:
            self.result_diarize = (
                self.align_language
            ) = self.result_source_lang = None
            if is_audio_file(media_file):
                prog_disp(
                    "Processing audio...", 0.15, is_gui, progress=progress
                )
                audio_preprocessor(preview, media_file, base_audio_wav)
            else:
                prog_disp(
                    "Processing video...", 0.15, is_gui, progress=progress
                )
                audio_video_preprocessor(
                    preview, media_file, base_video_file, base_audio_wav
                )
            logger.debug("Set file complete.")

            if subtitle_file:
                prog_disp("From SRT file...", 0.30, is_gui, progress=progress)
                if SOURCE_LANGUAGE == "Automatic detection":
                    raise Exception(
                        "To use an SRT file, you need to specify its original "
                        "language (Source language)"
                    )
                subtitle_file = (
                    subtitle_file
                    if isinstance(subtitle_file, str)
                    else subtitle_file.name
                )
                audio = whisperx.load_audio(base_audio_wav)
                result = srt_file_to_segments(subtitle_file)
                result["language"] = SOURCE_LANGUAGE
            else:
                prog_disp("Transcribing...", 0.30, is_gui, progress=progress)
                SOURCE_LANGUAGE = (
                    None
                    if SOURCE_LANGUAGE == "Automatic detection"
                    else SOURCE_LANGUAGE
                )
                audio, result = transcribe_speech(
                    base_audio_wav,
                    WHISPER_MODEL_SIZE,
                    compute_type,
                    batch_size,
                    SOURCE_LANGUAGE,
                )
            logger.debug("Transcript complete")

            prog_disp("Aligning...", 0.45, is_gui, progress=progress)
            self.align_language = result["language"]
            result = align_speech(audio, result)
            logger.debug("Align complete")
            if result["segments"] == []:
                raise ValueError("No active speech found in audio")

            prog_disp("Diarizing...", 0.60, is_gui, progress=progress)
            diarize_model_select = diarization_models[diarization_model]
            self.result_diarize = diarize_speech(
                base_audio_wav,
                result,
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN,
                diarize_model_select,
            )
            logger.debug("Diarize complete")
            self.result_source_lang = copy.deepcopy(self.result_diarize)

            prog_disp("Translating...", 0.75, is_gui, progress=progress)
            self.result_diarize["segments"] = translate_text(
                self.result_diarize["segments"],
                TRANSLATE_AUDIO_TO,
                translate_process,
                chunk_size=1800,
            )
            logger.debug("Translation complete")
            logger.debug(self.result_diarize)

        if get_translated_text:
            json_data = []
            for segment in self.result_diarize["segments"]:
                start = segment["start"]
                text = segment["text"]
                speaker = int(segment.get("speaker", "SPEAKER_00")[-1]) + 1
                json_data.append(
                    {"start": start, "text": text, "speaker": speaker}
                )

            # Convert the list of dictionaries to a JSON string with indentation
            json_string = json.dumps(json_data, indent=2)
            return json_string.encode().decode("unicode_escape")

        if get_video_from_text_json:
            # with open('text_json.json', 'r') as file:
            text_json_loaded = json.loads(text_json)
            for i, segment in enumerate(self.result_diarize["segments"]):
                segment["text"] = text_json_loaded[i]["text"]
                segment["speaker"] = "SPEAKER_0" + str(
                    int(text_json_loaded[i]["speaker"]) - 1
                )

        # Write subtitle
        sub_file = process_subtitles(
            self.result_source_lang,
            self.align_language,
            self.result_diarize,
            output_format_subtitle,
            TRANSLATE_AUDIO_TO,
        )

        if output_type == "subtitle":
            return sub_file

        prog_disp("Text to speech...", 0.85, is_gui, progress=progress)
        audio_files, speakers_list = audio_segmentation_to_voice(
            self.result_diarize,
            TRANSLATE_AUDIO_TO,
            max_accelerate_audio,
            True,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            dereverb_automatic_xtts,
        )

        # Voice Imitation (Tone color converter)
        if voice_imitation:
            from soni_translate.text_to_speech import toneconverter

            toneconverter(
                copy.deepcopy(self.result_diarize),
                preprocessor_max_segments=voice_imitation_max_segments,
                remove_previous_process=voice_imitation_remove_previous,
                get_vocals_dereverb=voice_imitation_vocals_dereverb,
            )

        # custom voice
        if os.getenv("VOICES_MODELS") == "ENABLE":
            prog_disp(
                "Applying customized voices...",
                0.90,
                is_gui,
                progress=progress,
            )
            voices_conversion(speakers_list, audio_files)

        # replace files with the accelerates
        move_files("audio2/audio/", "audio/")

        prog_disp(
            "Creating final translated video...",
            0.95,
            is_gui,
            progress=progress,
        )
        remove_files([dub_audio_file, mix_audio_file])
        create_translated_audio(
            self.result_diarize, audio_files, dub_audio_file
        )

        # Voiceless track
        if voiceless_track:
            from soni_translate.mdx_net import process_uvr_task

            _, base_audio_wav, _, _, _ = process_uvr_task(
                orig_song_path=base_audio_wav,
                main_vocals=False,
                dereverb=False,
                song_id="voiceless",
                only_voiceless=True,
                remove_files_output_dir=False,
            )

        # TYPE MIX AUDIO
        command_volume_mix = f'ffmpeg -y -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio_file}'
        command_background_mix = f'ffmpeg -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio_file}'
        if AUDIO_MIX_METHOD == "Adjusting volumes and mixing audio":
            # volume mix
            run_command(command_volume_mix)
        else:
            try:
                # background mix
                run_command(command_background_mix)
            except Exception as error_mix:
                # volume mix except
                logger.error(str(error_mix))
                run_command(command_volume_mix)

        if output_type == "audio" or is_audio_file(media_file):
            return mix_audio_file

        # Merge new audio + video
        remove_files(video_output_file)
        run_command(
            f"ffmpeg -i {base_video_file} -i {mix_audio_file} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output_file}"
        )

        return video_output_file

    def multilingual_docs_conversion(
        self,
        string_text,  # string
        document=None,  # doc path gui
        directory_input="",  # doc path
        SOURCE_LANGUAGE="English (en)",
        TRANSLATE_AUDIO_TO="English (en)",
        tts_voice00="en-AU-WilliamNeural-Male",
        name_final_file="sample",
        translate_process="google_translator_iterative",
        output_type="audio",
        chunk_size=None,
        is_gui=False,
        progress=gr.Progress(),
    ):
        SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]
        if translate_process != "disable translation":
            TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
        else:
            TRANSLATE_AUDIO_TO = SOURCE_LANGUAGE
            logger.info("No translation")
        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            logger.debug(
                "Make sure to select a 'TTS Speaker' suitable for the "
                "translation language to avoid errors with the TTS."
            )

        is_string = False
        if document is None:
            if os.path.exists(directory_input):
                document = directory_input
            else:
                document = string_text
                is_string = True
        document = document if isinstance(document, str) else document.name
        if not document:
            raise Exception("No data found")

        # audio_wav = "audio.wav"
        final_wav_file = (
            "audio_book.wav"
            if not name_final_file
            else f"{name_final_file}.wav"
        )

        prog_disp("Processing text...", 0.15, is_gui, progress=progress)
        result_file_path, result_text = document_preprocessor(
            document, is_string
        )

        if (
            output_type == "text"
            and translate_process == "disable translation"
        ):
            return result_file_path

        if "SET_LIMIT" == os.getenv("DEMO"):
            result_text = result_text[:50]
            logger.info(
                "DEMO; Generation is limited to 50 characters to prevent "
                "CPU errors. No limitations with GPU.\n"
            )

        if translate_process != "disable translation":
            # chunks text for translation
            result_diarize = plain_text_to_segments(result_text, 1700)
            prog_disp("Translating...", 0.30, is_gui, progress=progress)
            # not or iterative with 1700 chars
            result_diarize["segments"] = translate_text(
                result_diarize["segments"],
                TRANSLATE_AUDIO_TO,
                translate_process,
                chunk_size=0,
            )

            txt_file_path, result_text = segments_to_plain_text(result_diarize)

            if output_type == "text":
                return txt_file_path

        # (TTS limits) plain text to result_diarize
        chunk_size = (
            chunk_size if chunk_size else determine_chunk_size(tts_voice00)
        )
        result_diarize = plain_text_to_segments(result_text, chunk_size)
        logger.debug(result_diarize)

        prog_disp("Text to speech...", 0.45, is_gui, progress=progress)
        audio_files, speakers_list = audio_segmentation_to_voice(
            result_diarize,
            TRANSLATE_AUDIO_TO,
            1.0,
            is_gui,
            tts_voice00,
            "",
            "",
            "",
            "",
            "",
        )

        # custom voice
        if os.getenv("VOICES_MODELS") == "ENABLE":
            prog_disp(
                "Applying customized voices...",
                0.80,
                is_gui,
                progress=progress,
            )
            voices_conversion(speakers_list, audio_files)

        # replace files with the accelerates and custom voice
        move_files("audio2/audio/", "audio/")

        prog_disp(
            "Creating final audio file...", 0.90, is_gui, progress=progress
        )
        remove_files(final_wav_file)
        create_translated_audio(
            result_diarize, audio_files, final_wav_file, True
        )

        return final_wav_file


SoniTr = SoniTranslate()

title = "<center><strong><font size='7'>📽️ SoniTranslate 🈷️</font></strong></center>"
news = """ ## 📖 News
        🔥 2024/01/16: Expanded language support, the introduction of whisper large v3, configurable GUI options, integration of BARK, Facebook-mms, Coqui XTTS, and Piper-TTS. Additional features included audio separation utilities, XTTS WAV creation,  use an SRT file as a base for translation, document translation, manual speaker editing, and flexible output options (video, audio, subtitles).

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
                    input_data_type = gr.inputs.Dropdown(
                        ["SUBMIT VIDEO", "URL", "Find Video Path"],
                        default="SUBMIT VIDEO",
                        label=lg_conf["video_source"],
                    )

                    def swap_visibility(data_type):
                        if data_type == "URL":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=True, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "SUBMIT VIDEO":
                            return (
                                gr.update(visible=True, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "Find Video Path":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=True, value=""),
                            )

                    video_input = gr.File(label="VIDEO")
                    blink_input = gr.Textbox(
                        visible=False,
                        label="Media link.",
                        info=lg_conf["link_info"],
                        placeholder=lg_conf["link_ph"],
                    )
                    directory_input = gr.Textbox(
                        visible=False,
                        label="Video Path.",
                        info=lg_conf["dir_info"],
                        placeholder=lg_conf["dir_ph"],
                    )
                    input_data_type.change(
                        fn=swap_visibility,
                        inputs=input_data_type,
                        outputs=[video_input, blink_input, directory_input],
                    )

                    link = gr.HTML()

                    SOURCE_LANGUAGE = gr.Dropdown(
                        LANGUAGES_LIST,
                        value="Automatic detection",
                        label=lg_conf["sl_label"],
                        info=lg_conf["sl_info"],
                    )
                    TRANSLATE_AUDIO_TO = gr.Dropdown(
                        LANGUAGES_LIST[1:],
                        value="English (en)",
                        label=lg_conf["tat_label"],
                        info=lg_conf["tat_info"],
                    )

                    line_ = gr.HTML("<hr></h2>")
                    gr.Markdown(lg_conf["num_speakers"])
                    MAX_TTS = 6
                    min_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        default=1,
                        label=lg_conf["min_sk"],
                        step=1,
                        visible=False,
                    )
                    max_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=2,
                        step=1,
                        label=lg_conf["max_sk"],
                        interative=True,
                    )
                    gr.Markdown(lg_conf["tts_select"])

                    def submit(value):
                        visibility_dict = {
                            f"tts_voice{i:02d}": gr.update(visible=i < value)
                            for i in range(6)
                        }
                        return [value for value in visibility_dict.values()]

                    tts_voice00 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-AU-WilliamNeural-Male",
                        label=lg_conf["sk1"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice01 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-CA-ClaraNeural-Female",
                        label=lg_conf["sk2"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice02 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-GB-ThomasNeural-Male",
                        label=lg_conf["sk3"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice03 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-GB-SoniaNeural-Female",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice04 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-NZ-MitchellNeural-Male",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice05 = gr.Dropdown(
                        tts_info.tts_list(),
                        value="en-GB-MaisieNeural-Female",
                        label=lg_conf["sk6"],
                        visible=False,
                        interactive=True,
                    )
                    max_speakers.change(
                        submit,
                        max_speakers,
                        [
                            tts_voice00,
                            tts_voice01,
                            tts_voice02,
                            tts_voice03,
                            tts_voice04,
                            tts_voice05,
                        ],
                    )

                    with gr.Column():
                        with gr.Accordion(
                            lg_conf["vc_title"],
                            open=False,
                        ):
                            gr.Markdown(lg_conf["vc_subtitle"])
                            voice_imitation_gui = gr.Checkbox(
                                False,
                                label=lg_conf["vc_active_label"],
                                info=lg_conf["vc_active_info"],
                            )
                            voice_imitation_max_segments_gui = gr.Slider(
                                label=lg_conf["vc_segments_label"],
                                info=lg_conf["vc_segments_info"],
                                value=1,
                                step=1,
                                minimum=1,
                                maximum=10,
                                visible=True,
                                interactive=True,
                            )
                            voice_imitation_vocals_dereverb_gui = gr.Checkbox(
                                False,
                                label=lg_conf["vc_dereverb_label"],
                                info=lg_conf["vc_dereverb_info"],
                            )
                            voice_imitation_remove_previous_gui = gr.Checkbox(
                                True,
                                label=lg_conf["vc_remove_label"],
                                info=lg_conf["vc_remove_info"],
                            )

                    if xtts_enabled:
                        with gr.Column():
                            with gr.Accordion(
                                lg_conf["xtts_title"],
                                open=False,
                            ):
                                gr.Markdown(lg_conf["xtts_subtitle"])
                                wav_speaker_file = gr.File(
                                    label=lg_conf["xtts_file_label"]
                                )
                                wav_speaker_name = gr.Textbox(
                                    label=lg_conf["xtts_name_label"],
                                    value="",
                                    info=lg_conf["xtts_name_info"],
                                    placeholder="default_name",
                                    lines=1,
                                )
                                wav_speaker_start = gr.Number(
                                    label="Time audio start",
                                    value=0,
                                    visible=False,
                                )
                                wav_speaker_end = gr.Number(
                                    label="Time audio end",
                                    value=0,
                                    visible=False,
                                )
                                wav_speaker_dir = gr.Textbox(
                                    label="Directory save",
                                    value="_XTTS_",
                                    visible=False,
                                )
                                wav_speaker_dereverb = gr.Checkbox(
                                    True,
                                    label=lg_conf["xtts_dereverb_label"],
                                    info=lg_conf["xtts_dereverb_info"]
                                )
                                wav_speaker_output = gr.HTML()
                                create_xtts_wav = gr.Button(
                                    lg_conf["xtts_button"]
                                )
                                gr.Markdown(lg_conf["xtts_footer"])
                    else:
                        wav_speaker_dereverb = gr.Checkbox(
                            False,
                            label=lg_conf["xtts_dereverb_label"],
                            info=lg_conf["xtts_dereverb_info"],
                            visible=False
                        )

                    with gr.Column():
                        with gr.Accordion(
                            lg_conf["extra_setting"], open=False
                        ):
                            audio_accelerate = gr.Slider(
                                label=lg_conf["acc_max_label"],
                                value=1.9,
                                step=0.1,
                                minimum=1.0,
                                maximum=2.5,
                                visible=True,
                                interactive=True,
                                info=lg_conf["acc_max_info"],
                            )

                            audio_mix_options = [
                                "Mixing audio with sidechain compression",
                                "Adjusting volumes and mixing audio",
                            ]
                            AUDIO_MIX = gr.Dropdown(
                                audio_mix_options,
                                value=audio_mix_options[1],
                                label=lg_conf["aud_mix_label"],
                                info=lg_conf["aud_mix_info"],
                            )
                            volume_original_mix = gr.Slider(
                                label=lg_conf["vol_ori"],
                                info="for <Adjusting volumes and mixing audio>",
                                value=0.25,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )
                            volume_translated_mix = gr.Slider(
                                label=lg_conf["vol_tra"],
                                info="for <Adjusting volumes and mixing audio>",
                                value=1.80,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )

                            gr.HTML("<hr></h2>")
                            sub_type_options = [
                                "srt",
                                "vtt",
                                "txt",
                                "tsv",
                                "json",
                                "aud",
                            ]

                            def get_subs_path(type_subs):
                                if os.path.exists(
                                    f"sub_ori.{type_subs}"
                                ) and os.path.exists(f"sub_tra.{type_subs}"):
                                    return (
                                        f"sub_ori.{type_subs}",
                                        f"sub_tra.{type_subs}",
                                    )
                                else:
                                    return None, None

                            sub_type_output = gr.inputs.Dropdown(
                                sub_type_options,
                                default=sub_type_options[0],
                                label=lg_conf["sub_type"],
                            )

                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["whisper_title"])
                            whisper_model_options = [
                                "tiny",
                                "base",
                                "small",
                                "medium",
                                "large-v1",
                                "large-v2",
                                "large-v3",
                            ]
                            WHISPER_MODEL_SIZE = gr.inputs.Dropdown(
                                whisper_model_options,
                                default=whisper_model_default,
                                label="Whisper model",
                            )
                            batch_size = gr.inputs.Slider(
                                1, 32, default=16, label="Batch size", step=1
                            )
                            compute_type = gr.inputs.Dropdown(
                                list_compute_type,
                                default=compute_type_default,
                                label="Compute type",
                            )
                            input_srt = gr.File(
                                label=lg_conf["srt_file_label"],
                                file_types=[".srt", ".ass"],
                                height=130,
                            )
                            pyannote_models_list = list(
                                diarization_models.keys()
                            )
                            diarization_process_dropdown = gr.inputs.Dropdown(
                                pyannote_models_list,
                                default=pyannote_models_list[1],
                                label="Diarization model",
                            )
                            valid_translate_process = [
                                "google_translator_batch",
                                "google_translator_iterative",
                                "disable_translation",
                            ]
                            translate_process_dropdown = gr.inputs.Dropdown(
                                valid_translate_process,
                                default=valid_translate_process[0],
                                label="Translation process",
                            )

                            gr.HTML("<hr></h2>")
                            main_output_type_opt = [
                                "video",
                                "audio",
                                "subtitle",
                            ]
                            main_output_type = gr.inputs.Dropdown(
                                main_output_type_opt,
                                default=main_output_type_opt[0],
                                label="Output type",
                            )
                            main_voiceless_track = gr.Checkbox(
                                label=lg_conf["voiceless_tk_label"],
                                info=lg_conf["voiceless_tk_info"],
                            )
                            VIDEO_OUTPUT_NAME = gr.Textbox(
                                label=lg_conf["out_name_label"],
                                value="video_output",
                                info=lg_conf["out_name_info"],
                            )
                            PREVIEW = gr.Checkbox(
                                label="Preview", info=lg_conf["preview_info"]
                            )
                            is_gui_dummy_check = gr.Checkbox(
                                True, visible=False
                            )

                with gr.Column(variant="compact"):
                    edit_sub_check = gr.Checkbox(
                        label=lg_conf["edit_sub_label"],
                        info=lg_conf["edit_sub_info"],
                    )
                    dummy_false_check = gr.Checkbox(
                        False,
                        visible=False,
                    )

                    def visible_component_subs(input_bool):
                        if input_bool:
                            return gr.update(visible=True), gr.update(
                                visible=True
                            )
                        else:
                            return gr.update(visible=False), gr.update(
                                visible=False
                            )

                    subs_button = gr.Button(
                        lg_conf["button_subs"],
                        variant="primary",
                        visible=False,
                    )
                    subs_edit_space = gr.Textbox(
                        visible=False,
                        lines=10,
                        label=lg_conf["editor_sub_label"],
                        info=lg_conf["editor_sub_info"],
                        placeholder=lg_conf["editor_sub_ph"],
                    )
                    edit_sub_check.change(
                        visible_component_subs,
                        [edit_sub_check],
                        [subs_button, subs_edit_space],
                    )

                    with gr.Row():
                        video_button = gr.Button(
                            lg_conf["button_translate"],
                            variant="primary",
                        )
                    with gr.Row():
                        video_output = gr.outputs.File(
                            label=lg_conf["output_result_label"]
                        )  # gr.Video()
                    with gr.Row():
                        sub_ori_output = gr.outputs.File(
                            label=lg_conf["sub_ori"]
                        )
                        sub_tra_output = gr.outputs.File(
                            label=lg_conf["sub_tra"]
                        )

                    line_ = gr.HTML("<hr></h2>")
                    if (
                        os.getenv("YOUR_HF_TOKEN") is None
                        or os.getenv("YOUR_HF_TOKEN") == ""
                    ):
                        HFKEY = gr.Textbox(
                            visible=True,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )
                    else:
                        HFKEY = gr.Textbox(
                            visible=False,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )

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
                                "en-AU-WilliamNeural-Male",
                                "en-CA-ClaraNeural-Female",
                                "en-GB-ThomasNeural-Male",
                                "en-GB-SoniaNeural-Female",
                                "en-NZ-MitchellNeural-Male",
                                "en-GB-MaisieNeural-Female",
                                "video_output",
                                "Adjusting volumes and mixing audio",
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
                                "en-CA-ClaraNeural-Female",
                                "en-AU-WilliamNeural-Male",
                                "en-GB-ThomasNeural-Male",
                                "en-GB-SoniaNeural-Female",
                                "en-NZ-MitchellNeural-Male",
                                "en-GB-MaisieNeural-Female",
                                "video_output",
                                "Adjusting volumes and mixing audio",
                            ],
                        ],  # no update
                        fn=SoniTr.multilingual_media_conversion,
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

        with gr.Tab(lg_conf["tab_docs"]):
            with gr.Column():
                with gr.Accordion("Docs", open=True):
                    with gr.Column(variant="compact"):
                        with gr.Column():
                            input_doc_type = gr.Dropdown(
                                [
                                    "WRITE TEXT",
                                    "SUBMIT DOCUMENT",
                                    "Find Document Path",
                                ],
                                default="SUBMIT DOCUMENT",
                                label=lg_conf["docs_input_label"],
                                info=lg_conf["docs_input_info"],
                            )

                            def swap_visibility(data_type):
                                if data_type == "WRITE TEXT":
                                    return (
                                        gr.update(visible=True, value=""),
                                        gr.update(visible=False, value=None),
                                        gr.update(visible=False, value=""),
                                    )
                                elif data_type == "SUBMIT DOCUMENT":
                                    return (
                                        gr.update(visible=False, value=""),
                                        gr.update(visible=True, value=None),
                                        gr.update(visible=False, value=""),
                                    )
                                elif data_type == "Find Document Path":
                                    return (
                                        gr.update(visible=False, value=""),
                                        gr.update(visible=False, value=None),
                                        gr.update(visible=True, value=""),
                                    )

                            text_docs = gr.Textbox(
                                label="Text",
                                value="This is an example",
                                info="Write a text",
                                placeholder="...",
                                lines=5,
                                visible=False,
                            )
                            input_docs = gr.File(
                                label="Document", visible=True
                            )
                            directory_input_docs = gr.Textbox(
                                visible=False,
                                label="Document Path",
                                info="Example: /home/my_doc.pdf",
                                placeholder="Path goes here...",
                            )
                            input_doc_type.change(
                                fn=swap_visibility,
                                inputs=input_doc_type,
                                outputs=[
                                    text_docs,
                                    input_docs,
                                    directory_input_docs,
                                ],
                            )

                            link = gr.HTML()

                            tts_documents = gr.Dropdown(
                                list(
                                    filter(
                                        lambda x: x != "_XTTS_/AUTOMATIC.wav",
                                        tts_info.tts_list(),
                                    )
                                ),
                                value="en-GB-ThomasNeural-Male",
                                label="TTS",
                                visible=True,
                                interactive=True,
                            )

                            link = gr.HTML()

                            docs_SOURCE_LANGUAGE = gr.Dropdown(
                                LANGUAGES_LIST[1:],
                                value="English (en)",
                                label=lg_conf["sl_label"],
                                info=lg_conf["docs_source_info"],
                            )
                            docs_TRANSLATE_TO = gr.Dropdown(
                                LANGUAGES_LIST[1:],
                                value="English (en)",
                                label=lg_conf["tat_label"],
                                info=lg_conf["tat_info"],
                            )

                            with gr.Column():
                                with gr.Accordion(
                                    lg_conf["extra_setting"], open=False
                                ):
                                    docs_valid_translate_process = [
                                        "google_translator_iterative",
                                        "disable_translation",
                                    ]
                                    docs_translate_process_dropdown = gr.inputs.Dropdown(
                                        docs_valid_translate_process,
                                        default=docs_valid_translate_process[
                                            0
                                        ],
                                        label="Translation process",
                                    )

                                    line_ = gr.HTML("<hr></h2>")

                                    docs_output_type_opt = [
                                        "audio",
                                        "text",
                                    ]  # Add DOCX and etc.
                                    docs_output_type = gr.inputs.Dropdown(
                                        docs_output_type_opt,
                                        default=docs_output_type_opt[0],
                                        label="Output type",
                                    )
                                    docs_OUTPUT_NAME = gr.Textbox(
                                        label="Final file name",
                                        value="final_sample",
                                        info=lg_conf["out_name_info"],
                                    )
                                    docs_chunk_size = gr.Number(
                                        label=lg_conf["chunk_size_label"],
                                        value=0,
                                        visible=True,
                                        interactive=True,
                                        info=lg_conf["chunk_size_info"],
                                    )
                                    docs_dummy_check = gr.Checkbox(
                                        True, visible=False
                                    )

                            with gr.Row():
                                docs_button = gr.Button(
                                    lg_conf["docs_button"],
                                    variant="primary",
                                )
                            with gr.Row():
                                docs_output = gr.outputs.File(label="Result")

        with gr.Tab("Custom voice R.V.C. (Optional)"):
            with gr.Column():
                with gr.Accordion("Get the R.V.C. Models", open=True):
                    url_links = gr.Textbox(
                        label="URLs",
                        value="",
                        info=lg_conf["cv_url_info"],
                        placeholder="urls here...",
                        lines=1,
                    )
                    download_finish = gr.HTML()
                    download_button = gr.Button("DOWNLOAD MODELS")

                    def update_models():
                        models, index_paths = upload_model_list()
                        for i in range(8):
                            dict_models = {
                                f"model_voice_path{i:02d}": gr.update(
                                    choices=models
                                )
                                for i in range(8)
                            }
                            dict_index = {
                                f"file_index2_{i:02d}": gr.update(
                                    choices=index_paths
                                )
                                for i in range(8)
                            }
                            dict_changes = {**dict_models, **dict_index}
                            return [value for value in dict_changes.values()]

            with gr.Column():
                with gr.Accordion(lg_conf["replace_title"], open=False):
                    with gr.Column(variant="compact"):
                        with gr.Column():
                            gr.Markdown(lg_conf["sec1_title"])
                            enable_custom_voice = gr.Checkbox(
                                label="ENABLE", info=lg_conf["enable_replace"]
                            )
                            enable_custom_voice.change(
                                custom_model_voice_enable,
                                [enable_custom_voice],
                                [],
                            )

                            gr.Markdown(lg_conf["sec2_title"])
                            gr.Markdown(lg_conf["sec2_subtitle"])
                            gr.Markdown(lg_conf["cv_tts1"])
                            with gr.Row():
                                model_voice_path00 = gr.Dropdown(
                                    models,
                                    label="Model-1",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_00 = gr.Dropdown(
                                    index_paths,
                                    label="Index-1",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose00 = gr.Number(
                                    label="Transpose-1",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_tts2"])
                            with gr.Row():
                                model_voice_path01 = gr.Dropdown(
                                    models,
                                    label="Model-2",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_01 = gr.Dropdown(
                                    index_paths,
                                    label="Index-2",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose01 = gr.Number(
                                    label="Transpose-2",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_tts3"])
                            with gr.Row():
                                model_voice_path02 = gr.Dropdown(
                                    models,
                                    label="Model-3",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_02 = gr.Dropdown(
                                    index_paths,
                                    label="Index-3",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose02 = gr.Number(
                                    label="Transpose-3",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_tts4"])
                            with gr.Row():
                                model_voice_path03 = gr.Dropdown(
                                    models,
                                    label="Model-4",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_03 = gr.Dropdown(
                                    index_paths,
                                    label="Index-4",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose03 = gr.Number(
                                    label="Transpose-4",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_tts5"])
                            with gr.Row():
                                model_voice_path04 = gr.Dropdown(
                                    models,
                                    label="Model-5",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_04 = gr.Dropdown(
                                    index_paths,
                                    label="Index-5",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose04 = gr.Number(
                                    label="Transpose-5",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_tts6"])
                            with gr.Row():
                                model_voice_path05 = gr.Dropdown(
                                    models,
                                    label="Model-6",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_05 = gr.Dropdown(
                                    index_paths,
                                    label="Index-6",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose05 = gr.Number(
                                    label="Transpose-6",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["cv_aux"])
                            with gr.Row():
                                model_voice_path06 = gr.Dropdown(
                                    models,
                                    label="Model-Aux",
                                    visible=True,
                                    interactive=True,
                                )
                                file_index2_06 = gr.Dropdown(
                                    index_paths,
                                    label="Index-Aux",
                                    visible=True,
                                    interactive=True,
                                )
                                name_transpose06 = gr.Number(
                                    label="Transpose-Aux",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )
                            gr.HTML("<hr></h2>")
                            with gr.Row():
                                f0_methods_voice = [
                                    "pm",
                                    "harvest",
                                    "crepe",
                                    "rmvpe",
                                ]
                                f0_method_global = gr.Dropdown(
                                    f0_methods_voice,
                                    value="pm",
                                    label="Global F0 method",
                                    visible=True,
                                    interactive=True,
                                )

                    with gr.Row(variant="compact"):
                        button_config = gr.Button(
                            lg_conf["cv_button_apply"],
                            variant="primary",
                        )

                        confirm_conf = gr.HTML()

                    button_config.click(
                        voices_conversion.apply_conf,
                        inputs=[
                            f0_method_global,
                            model_voice_path00,
                            name_transpose00,
                            file_index2_00,
                            model_voice_path01,
                            name_transpose01,
                            file_index2_01,
                            model_voice_path02,
                            name_transpose02,
                            file_index2_02,
                            model_voice_path03,
                            name_transpose03,
                            file_index2_03,
                            model_voice_path04,
                            name_transpose04,
                            file_index2_04,
                            model_voice_path05,
                            name_transpose05,
                            file_index2_05,
                            model_voice_path06,
                            name_transpose06,
                            file_index2_06,
                        ],
                        outputs=[confirm_conf],
                    )

                with gr.Column():
                    with gr.Accordion("Test R.V.C.", open=False):
                        with gr.Row(variant="compact"):
                            text_test = gr.Textbox(
                                label="Text",
                                value="This is an example",
                                info="write a text",
                                placeholder="...",
                                lines=5,
                            )
                            with gr.Column():
                                tts_test = gr.Dropdown(
                                    sorted(tts_info.list_edge),
                                    value="en-GB-ThomasNeural-Male",
                                    label="TTS",
                                    visible=True,
                                    interactive=True,
                                )
                                model_voice_path07 = gr.Dropdown(
                                    models,
                                    label="Model",
                                    visible=True,
                                    interactive=True,
                                )  # value=''
                                file_index2_07 = gr.Dropdown(
                                    index_paths,
                                    label="Index",
                                    visible=True,
                                    interactive=True,
                                )  # value=''
                                transpose_test = gr.Number(
                                    label="Transpose",
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                    info="integer, number of semitones, raise by an octave: 12, lower by an octave: -12",
                                )
                                f0method_test = gr.Dropdown(
                                    f0_methods_voice,
                                    value="pm",
                                    label="F0 method",
                                    visible=True,
                                    interactive=True,
                                )
                        with gr.Row(variant="compact"):
                            button_test = gr.Button("Test audio")

                        with gr.Column():
                            with gr.Row():
                                original_ttsvoice = gr.Audio()
                                ttsvoice = gr.Audio()

                            button_test.click(
                                voices_conversion.make_test,
                                inputs=[
                                    text_test,
                                    tts_test,
                                    model_voice_path07,
                                    file_index2_07,
                                    transpose_test,
                                    f0method_test,
                                ],
                                outputs=[ttsvoice, original_ttsvoice],
                            )

                    download_button.click(
                        download_list, [url_links], [download_finish]
                    ).then(
                        update_models,
                        [],
                        [
                            model_voice_path00,
                            model_voice_path01,
                            model_voice_path02,
                            model_voice_path03,
                            model_voice_path04,
                            model_voice_path05,
                            model_voice_path06,
                            model_voice_path07,
                            file_index2_00,
                            file_index2_01,
                            file_index2_02,
                            file_index2_03,
                            file_index2_04,
                            file_index2_05,
                            file_index2_06,
                            file_index2_07,
                        ],
                    )

        with gr.Tab(lg_conf["tab_help"]):
            gr.Markdown(lg_conf["tutorial"])
            gr.Markdown(news)

        if logs_in_gui:
            logger.info("Logs in gui need public url")
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

            with gr.Accordion("Logs", open=False):
                logs = gr.Textbox(label=">>>")
                app.load(read_logs, None, logs, every=1)

        if xtts_enabled:
            # Update tts list
            def update_tts_list():
                update_dict = {
                    f"tts_voice{i:02d}": gr.update(choices=tts_info.tts_list())
                    for i in range(6)
                }
                update_dict["tts_documents"] = gr.update(
                    choices=list(
                        filter(
                            lambda x: x != "_XTTS_/AUTOMATIC.wav",
                            tts_info.tts_list(),
                        )
                    )
                )
                return [value for value in update_dict.values()]

            create_xtts_wav.click(
                create_wav_file_vc,
                inputs=[
                    wav_speaker_name,
                    wav_speaker_file,
                    wav_speaker_start,
                    wav_speaker_end,
                    wav_speaker_dir,
                    wav_speaker_dereverb,
                ],
                outputs=[wav_speaker_output],
            ).then(
                update_tts_list,
                None,
                [
                    tts_voice00,
                    tts_voice01,
                    tts_voice02,
                    tts_voice03,
                    tts_voice04,
                    tts_voice05,
                    tts_documents,
                ],
            )

        # Run translate text
        subs_button.click(
            SoniTr.multilingual_media_conversion,
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
                edit_sub_check,  # TRUE BY DEFAULT
                dummy_false_check,  # dummy false
                subs_edit_space,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                wav_speaker_dereverb,
                is_gui_dummy_check,
            ],
            outputs=subs_edit_space,
        )

        # Run translate tts and complete
        video_button.click(
            SoniTr.multilingual_media_conversion,
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
                dummy_false_check,
                edit_sub_check,
                subs_edit_space,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                wav_speaker_dereverb,
                is_gui_dummy_check,
            ],
            outputs=video_output,
        ).then(
            get_subs_path, [sub_type_output], [sub_ori_output, sub_tra_output]
        )

        # Run docs process
        docs_button.click(
            SoniTr.multilingual_docs_conversion,
            inputs=[
                text_docs,
                input_docs,
                directory_input_docs,
                docs_SOURCE_LANGUAGE,
                docs_TRANSLATE_TO,
                tts_documents,
                docs_OUTPUT_NAME,
                docs_translate_process_dropdown,
                docs_output_type,
                docs_chunk_size,
                docs_dummy_check,
            ],
            outputs=docs_output,
        )

    return app


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Taithrah/Minimal",
        help="Specify the theme; find themes in https://huggingface.co/spaces/gradio/theme-gallery; Example: --theme aliabid94/new-theme",
    )
    parser.add_argument(
        "--public_url",
        action="store_true",
        default=False,
        help="Enable public link",
    )
    parser.add_argument(
        "--logs_in_gui",
        action="store_true",
        default=False,
        help="Displays the operations performed in Logs",
    )
    parser.add_argument(
        "--verbosity_level",
        type=str,
        default="info",
        help="Set logger verbosity level:  debug, info, warning, error or critical",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help=" Select the language of the interface: english, spanish",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()
    # Simulating command-line arguments
    # args_list = "--theme aliabid94/new-theme --public_url".split()
    # args = parser.parse_args(args_list)

    set_logging_level(args.verbosity_level)

    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

    tts_info = TTS_Info(piper_enabled, xtts_enabled)
    # list_tts = tts_info.tts_list()

    models, index_paths = upload_model_list()
    os.environ["VOICES_MODELS"] = "DISABLE"

    lg_conf = language_data.get(args.language, language_data.get("english"))

    app = create_gui(args.theme, logs_in_gui=args.logs_in_gui)
    app.launch(
        share=args.public_url,
        show_error=True,
        enable_queue=True,
        quiet=False,
        debug=True,
    )

