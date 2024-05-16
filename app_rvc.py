import gradio as gr
from soni_translate.logging_setup import (
    logger,
    set_logging_level,
    configure_logging_libs,
); configure_logging_libs() # noqa
import whisperx
import torch
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
    accelerate_segments,
)
from soni_translate.translate_segments import (
    translate_text,
    TRANSLATION_PROCESS_OPTIONS,
    DOCS_TRANSLATION_PROCESS_OPTIONS
)
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.postprocessor import (
    OUTPUT_TYPE_OPTIONS,
    DOCS_OUTPUT_TYPE_OPTIONS,
    sound_separate,
    get_no_ext_filename,
    media_out,
    get_subtitle_speaker,
)
from soni_translate.language_configuration import (
    LANGUAGES,
    UNIDIRECTIONAL_L_LIST,
    LANGUAGES_LIST,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
    OPENAI_TTS_MODELS,
)
from soni_translate.utils import (
    remove_files,
    download_list,
    upload_model_list,
    download_manager,
    run_command,
    is_audio_file,
    is_subtitle_file,
    copy_files,
    get_valid_files,
    get_link_list,
    remove_directory_contents,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.speech_segmentation import (
    ASR_MODEL_OPTIONS,
    COMPUTE_TYPE_GPU,
    COMPUTE_TYPE_CPU,
    find_whisper_models,
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    BORDER_COLORS,
    srt_file_to_segments,
    document_preprocessor,
    determine_chunk_size,
    plain_text_to_segments,
    segments_to_plain_text,
    process_subtitles,
    linguistic_level_segments,
    break_aling_segments,
    doc_to_txtximg_pages,
    page_data_to_segments,
    update_page_data,
    fix_timestamps_docs,
    create_video_from_images,
    merge_video_and_audio,
)
from soni_translate.languages_gui import language_data, news
import copy
import logging
import json
from pydub import AudioSegment
from voice_main import ClassVoices
import argparse
import time
import hashlib
import sys

directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
    "outputs",
]
[
    os.makedirs(directory)
    for directory in directories
    if not os.path.exists(directory)
]


class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = (
            piper_tts_voices_list() if self.piper_enabled else []
        )
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = (
            coqui_xtts_voices_list() if self.xtts_enabled else []
        )
        list_tts = self.list_coqui_xtts + sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_openai_tts
            + self.list_vits_onnx
        )
        return list_tts


def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)


def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)


class SoniTrCache:
    def __init__(self):
        self.cache = {
            'media': [[]],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_data = {
            'media': [],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_keys = list(self.cache.keys())
        self.first_task = self.cache_keys[0]
        self.last_task = self.cache_keys[-1]

        self.pre_step = None
        self.pre_params = []

    def set_variable(self, variable_name, value):
        setattr(self, variable_name, value)

    def task_in_cache(self, step: str, params: list, previous_step_data: dict):

        self.pre_step_cache = None

        if step == self.first_task:
            self.pre_step = None

        if self.pre_step:
            self.cache[self.pre_step] = self.pre_params

            # Fill data in cache
            self.cache_data[self.pre_step] = copy.deepcopy(previous_step_data)

        self.pre_params = params
        # logger.debug(f"Step: {str(step)}, Cache params: {str(self.cache)}")
        if params == self.cache[step]:
            logger.debug(f"In cache: {str(step)}")

            # Set the var needed for next step
            # Recovery from cache_data the current step
            for key, value in self.cache_data[step].items():
                self.set_variable(key, copy.deepcopy(value))
                logger.debug(
                    f"Chache load: {str(key)}"
                )

            self.pre_step = step
            return True

        else:
            logger.debug(f"Flush next and caching {str(step)}")
            selected_index = self.cache_keys.index(step)

            for idx, key in enumerate(self.cache.keys()):
                if idx >= selected_index:
                    self.cache[key] = []
                    self.cache_data[key] = {}

            # The last is now previous
            self.pre_step = step
            return False

    def clear_cache(self, media, force=False):

        self.cache["media"] = (
            self.cache["media"] if len(self.cache["media"]) else [[]]
        )

        if media != self.cache["media"][0] or force:

            # Clear cache
            self.cache = {key: [] for key in self.cache}
            self.cache["media"] = [[]]

            logger.info("Cache flushed")


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


def check_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "To use GPT for translation, please set up your OpenAI API key "
            "as an environment variable in Linux as follows: "
            "export OPENAI_API_KEY='your-api-key-here'. Or change the "
            "translation process in Advanced settings."
        )


class SoniTranslate(SoniTrCache):
    def __init__(self, cpu_mode=False):
        super().__init__()
        if cpu_mode:
            os.environ["SONITR_DEVICE"] = "cpu"
        else:
            os.environ["SONITR_DEVICE"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = os.environ.get("SONITR_DEVICE")
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None
        self.edit_subs_complete = False
        self.voiceless_id = None
        self.burn_subs_id = None

        self.vci = ClassVoices(only_cpu=cpu_mode)

        self.tts_voices = self.get_tts_voice_list()

        logger.info(f"Working in: {self.device}")

    def get_tts_voice_list(self):
        try:
            from piper import PiperVoice  # noqa

            piper_enabled = True
            logger.info("PIPER TTS enabled")
        except Exception as error:
            logger.debug(str(error))
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
                "to the Terms and Conditions specified at the following "
                "link:\nhttps://coqui.ai/cpml.txt."
            )
            os.environ["COQUI_TOS_AGREED"] = "1"
        except Exception as error:
            logger.debug(str(error))
            xtts_enabled = False
            logger.info("Coqui XTTS disabled")

        self.tts_info = TTS_Info(piper_enabled, xtts_enabled)

        return self.tts_info.tts_list()

    def batch_multilingual_media_conversion(self, *kwargs):
        # logger.debug(str(kwargs))

        media_file_arg = kwargs[0] if kwargs[0] is not None else []

        link_media_arg = kwargs[1]
        link_media_arg = [x.strip() for x in link_media_arg.split(',')]
        link_media_arg = get_link_list(link_media_arg)

        path_arg = kwargs[2]
        path_arg = [x.strip() for x in path_arg.split(',')]
        path_arg = get_valid_files(path_arg)

        edit_text_arg = kwargs[31]
        get_text_arg = kwargs[32]

        is_gui_arg = kwargs[-1]

        kwargs = kwargs[3:]

        media_batch = media_file_arg + link_media_arg + path_arg
        media_batch = list(filter(lambda x: x != "", media_batch))
        media_batch = media_batch if media_batch else [None]
        logger.debug(str(media_batch))

        remove_directory_contents("outputs")

        if edit_text_arg or get_text_arg:
            return self.multilingual_media_conversion(
                media_batch[0], "", "", *kwargs
            )

        if "SET_LIMIT" == os.getenv("DEMO"):
            media_batch = [media_batch[0]]

        result = []
        for media in media_batch:
            # Call the nested function with the parameters
            output_file = self.multilingual_media_conversion(
                media, "", "", *kwargs
            )

            if isinstance(output_file, str):
                output_file = [output_file]
            result.extend(output_file)

            if is_gui_arg and len(media_batch) > 1:
                gr.Info(f"Done: {os.path.basename(output_file[0])}")

        return result

    def multilingual_media_conversion(
        self,
        media_file=None,
        link_media="",
        directory_input="",
        YOUR_HF_TOKEN="",
        preview=False,
        transcriber_model="large-v3",
        batch_size=4,
        compute_type="auto",
        origin_language="Automatic detection",
        target_language="English (en)",
        min_speakers=1,
        max_speakers=1,
        tts_voice00="en-US-EmmaMultilingualNeural-Female",
        tts_voice01="en-US-AndrewMultilingualNeural-Male",
        tts_voice02="en-US-AvaMultilingualNeural-Female",
        tts_voice03="en-US-BrianMultilingualNeural-Male",
        tts_voice04="de-DE-SeraphinaMultilingualNeural-Female",
        tts_voice05="de-DE-FlorianMultilingualNeural-Male",
        tts_voice06="fr-FR-VivienneMultilingualNeural-Female",
        tts_voice07="fr-FR-RemyMultilingualNeural-Male",
        tts_voice08="en-US-EmmaMultilingualNeural-Female",
        tts_voice09="en-US-AndrewMultilingualNeural-Male",
        tts_voice10="en-US-EmmaMultilingualNeural-Female",
        tts_voice11="en-US-AndrewMultilingualNeural-Male",
        video_output_name="",
        mix_method_audio="Adjusting volumes and mixing audio",
        max_accelerate_audio=2.1,
        acceleration_rate_regulation=False,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle="srt",
        get_translated_text=False,
        get_video_from_text_json=False,
        text_json="{}",
        avoid_overlap=False,
        vocal_refinement=False,
        literalize_numbers=True,
        segment_duration_limit=15,
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        subtitle_file=None,
        output_type="video (mp4)",
        voiceless_track=False,
        voice_imitation=False,
        voice_imitation_max_segments=3,
        voice_imitation_vocals_dereverb=False,
        voice_imitation_remove_previous=True,
        voice_imitation_method="freevc",
        dereverb_automatic_xtts=True,
        text_segmentation_scale="sentence",
        divide_text_segments_by="",
        soft_subtitles_to_video=True,
        burn_subtitles_to_video=False,
        enable_cache=True,
        custom_voices=False,
        custom_voices_workers=1,
        is_gui=False,
        progress=gr.Progress(),
    ):
        if not YOUR_HF_TOKEN:
            YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
            if diarization_model == "disable" or max_speakers == 1:
                if YOUR_HF_TOKEN is None:
                    YOUR_HF_TOKEN = ""
            elif not YOUR_HF_TOKEN:
                raise ValueError("No valid Hugging Face token")
            else:
                os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

        if (
            "gpt" in translate_process
            or transcriber_model == "OpenAI_API_Whisper"
            or "OpenAI-TTS" in tts_voice00
        ):
            check_openai_api_key()

        if media_file is None:
            media_file = (
                directory_input
                if os.path.exists(directory_input)
                else link_media
            )
        media_file = (
            media_file if isinstance(media_file, str) else media_file.name
        )

        if is_subtitle_file(media_file):
            subtitle_file = media_file
            media_file = ""

        if media_file is None:
            media_file = ""

        if not origin_language:
            origin_language = "Automatic detection"

        if origin_language in UNIDIRECTIONAL_L_LIST and not subtitle_file:
            raise ValueError(
                f"The language '{origin_language}' "
                "is not supported for transcription (ASR)."
            )

        if get_translated_text:
            self.edit_subs_complete = False
        if get_video_from_text_json:
            if not self.edit_subs_complete:
                raise ValueError("Generate the transcription first.")

        if (
            ("sound" in output_type or output_type == "raw media")
            and (get_translated_text or get_video_from_text_json)
        ):
            raise ValueError(
                "Please disable 'edit generate subtitles' "
                f"first to acquire the {output_type}."
            )

        TRANSLATE_AUDIO_TO = LANGUAGES[target_language]
        SOURCE_LANGUAGE = LANGUAGES[origin_language]

        if (
            transcriber_model == "OpenAI_API_Whisper"
            and SOURCE_LANGUAGE == "zh-TW"
        ):
            logger.warning(
                "OpenAI API Whisper only supports Chinese (Simplified)."
            )
            SOURCE_LANGUAGE = "zh"

        if (
            text_segmentation_scale in ["word", "character"]
            and "subtitle" not in output_type
        ):
            wrn_lang = (
                "Text segmentation by words or characters is typically"
                " used for generating subtitles. If subtitles are not the"
                " intended output, consider selecting 'sentence' "
                "segmentation method to ensure optimal results."

            )
            warn_disp(wrn_lang, is_gui)

        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            wrn_lang = (
                "Make sure to select a 'TTS Speaker' suitable for"
                " the translation language to avoid errors with the TTS."
            )
            warn_disp(wrn_lang, is_gui)

        if "_XTTS_" in tts_voice00 and voice_imitation:
            wrn_lang = (
                "When you select XTTS, it is advisable "
                "to disable Voice Imitation."
            )
            warn_disp(wrn_lang, is_gui)

        if custom_voices and voice_imitation:
            wrn_lang = (
                "When you use R.V.C. models, it is advisable"
                " to disable Voice Imitation."
            )
            warn_disp(wrn_lang, is_gui)

        if not media_file and not subtitle_file:
            raise ValueError(
                "Specifify a media or SRT file in advanced settings"
            )

        if subtitle_file:
            subtitle_file = (
                subtitle_file
                if isinstance(subtitle_file, str)
                else subtitle_file.name
            )

        if subtitle_file and SOURCE_LANGUAGE == "Automatic detection":
            raise Exception(
                "To use an SRT file, you need to specify its "
                "original language (Source language)"
            )

        if not media_file and subtitle_file:
            diarization_model = "disable"
            media_file = "audio_support.wav"
            if not get_video_from_text_json:
                remove_files(media_file)
                srt_data = srt_file_to_segments(subtitle_file)
                total_duration = srt_data["segments"][-1]["end"] + 30.
                support_audio = AudioSegment.silent(
                    duration=int(total_duration * 1000)
                )
                support_audio.export(
                    media_file, format="wav"
                )
                logger.info("Supporting audio for the SRT file, created.")

        if "SET_LIMIT" == os.getenv("DEMO"):
            preview = True
            mix_method_audio = "Adjusting volumes and mixing audio"
            transcriber_model = "medium"
            logger.info(
                "DEMO; set preview=True; Generation is limited to "
                "10 seconds to prevent CPU errors. No limitations with GPU.\n"
                "DEMO; set Adjusting volumes and mixing audio\n"
                "DEMO; set whisper model to medium"
            )

        # Check GPU
        if self.device == "cpu" and compute_type not in COMPUTE_TYPE_CPU:
            logger.info("Compute type changed to float32")
            compute_type = "float32"

        base_video_file = "Video.mp4"
        base_audio_wav = "audio.wav"
        dub_audio_file = "audio_dub_solo.ogg"
        vocals_audio_file = "audio_Vocals_DeReverb.wav"
        voiceless_audio_file = "audio_Voiceless.wav"
        mix_audio_file = "audio_mix.mp3"
        vid_subs = "video_subs_file.mp4"
        video_output_file = "video_dub.mp4"

        if os.path.exists(media_file):
            media_base_hash = get_hash(media_file)
        else:
            media_base_hash = media_file
        self.clear_cache(media_base_hash, force=(not enable_cache))

        if not get_video_from_text_json:
            self.result_diarize = (
                self.align_language
            ) = self.result_source_lang = None
            if not self.task_in_cache("media", [media_base_hash, preview], {}):
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

            if "sound" in output_type:
                prog_disp(
                    "Separating sounds in the file...",
                    0.50,
                    is_gui,
                    progress=progress
                )
                separate_out = sound_separate(base_audio_wav, output_type)
                final_outputs = []
                for out in separate_out:
                    final_name = media_out(
                        media_file,
                        f"{get_no_ext_filename(out)}",
                        video_output_name,
                        "wav",
                        file_obj=out,
                    )
                    final_outputs.append(final_name)
                logger.info(f"Done: {str(final_outputs)}")
                return final_outputs

            if output_type == "raw media":
                output = media_out(
                    media_file,
                    "raw_media",
                    video_output_name,
                    "wav" if is_audio_file(media_file) else "mp4",
                    file_obj=base_audio_wav if is_audio_file(media_file) else base_video_file,
                )
                logger.info(f"Done: {output}")
                return output

            if not self.task_in_cache("refine_vocals", [vocal_refinement], {}):
                self.vocals = None
                if vocal_refinement:
                    try:
                        from soni_translate.mdx_net import process_uvr_task
                        _, _, _, _, file_vocals = process_uvr_task(
                            orig_song_path=base_audio_wav,
                            main_vocals=False,
                            dereverb=True,
                            remove_files_output_dir=True,
                        )
                        remove_files(vocals_audio_file)
                        copy_files(file_vocals, ".")
                        self.vocals = vocals_audio_file
                    except Exception as error:
                        logger.error(str(error))

            if not self.task_in_cache("transcript_align", [
                subtitle_file,
                SOURCE_LANGUAGE,
                transcriber_model,
                compute_type,
                batch_size,
                literalize_numbers,
                segment_duration_limit,
                (
                    "l_unit"
                    if text_segmentation_scale in ["word", "character"]
                    and subtitle_file
                    else "sentence"
                )
            ], {"vocals": self.vocals}):
                if subtitle_file:
                    prog_disp(
                        "From SRT file...", 0.30, is_gui, progress=progress
                    )
                    audio = whisperx.load_audio(
                        base_audio_wav if not self.vocals else self.vocals
                    )
                    self.result = srt_file_to_segments(subtitle_file)
                    self.result["language"] = SOURCE_LANGUAGE
                else:
                    prog_disp(
                        "Transcribing...", 0.30, is_gui, progress=progress
                    )
                    SOURCE_LANGUAGE = (
                        None
                        if SOURCE_LANGUAGE == "Automatic detection"
                        else SOURCE_LANGUAGE
                    )
                    audio, self.result = transcribe_speech(
                        base_audio_wav if not self.vocals else self.vocals,
                        transcriber_model,
                        compute_type,
                        batch_size,
                        SOURCE_LANGUAGE,
                        literalize_numbers,
                        segment_duration_limit,
                    )
                logger.debug(
                    "Transcript complete, "
                    f"segments count {len(self.result['segments'])}"
                )

                self.align_language = self.result["language"]
                if (
                    not subtitle_file
                    or text_segmentation_scale in ["word", "character"]
                ):
                    prog_disp("Aligning...", 0.45, is_gui, progress=progress)
                    try:
                        if self.align_language in ["vi"]:
                            logger.info(
                                "Deficient alignment for the "
                                f"{self.align_language} language, skipping the"
                                " process. It is suggested to reduce the "
                                "duration of the segments as an alternative."
                            )
                        else:
                            self.result = align_speech(audio, self.result)
                            logger.debug(
                                "Align complete, "
                                f"segments count {len(self.result['segments'])}"
                            )
                    except Exception as error:
                        logger.error(str(error))

            if self.result["segments"] == []:
                raise ValueError("No active speech found in audio")

            if not self.task_in_cache("break_align", [
                divide_text_segments_by,
                text_segmentation_scale,
                self.align_language
            ], {
                "result": self.result,
                "align_language": self.align_language
            }):
                if self.align_language in ["ja", "zh", "zh-TW"]:
                    divide_text_segments_by += "|!|?|...|。"
                if text_segmentation_scale in ["word", "character"]:
                    self.result = linguistic_level_segments(
                        self.result,
                        text_segmentation_scale,
                    )
                elif divide_text_segments_by:
                    try:
                        self.result = break_aling_segments(
                            self.result,
                            break_characters=divide_text_segments_by,
                        )
                    except Exception as error:
                        logger.error(str(error))

            if not self.task_in_cache("diarize", [
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN[:len(YOUR_HF_TOKEN)//2],
                diarization_model
            ], {
                "result": self.result
            }):
                prog_disp("Diarizing...", 0.60, is_gui, progress=progress)
                diarize_model_select = diarization_models[diarization_model]
                self.result_diarize = diarize_speech(
                    base_audio_wav if not self.vocals else self.vocals,
                    self.result,
                    min_speakers,
                    max_speakers,
                    YOUR_HF_TOKEN,
                    diarize_model_select,
                )
                logger.debug("Diarize complete")
            self.result_source_lang = copy.deepcopy(self.result_diarize)

            if not self.task_in_cache("translate", [
                TRANSLATE_AUDIO_TO,
                translate_process
            ], {
                "result_diarize": self.result_diarize
            }):
                prog_disp("Translating...", 0.70, is_gui, progress=progress)
                lang_source = (
                    self.align_language
                    if self.align_language
                    else SOURCE_LANGUAGE
                )
                self.result_diarize["segments"] = translate_text(
                    self.result_diarize["segments"],
                    TRANSLATE_AUDIO_TO,
                    translate_process,
                    chunk_size=1800,
                    source=lang_source,
                )
                logger.debug("Translation complete")
                logger.debug(self.result_diarize)

        if get_translated_text:

            json_data = []
            for segment in self.result_diarize["segments"]:
                start = segment["start"]
                text = segment["text"]
                speaker = int(segment.get("speaker", "SPEAKER_00")[-2:]) + 1
                json_data.append(
                    {"start": start, "text": text, "speaker": speaker}
                )

            # Convert list of dictionaries to a JSON string with indentation
            json_string = json.dumps(json_data, indent=2)
            logger.info("Done")
            self.edit_subs_complete = True
            return json_string.encode().decode("unicode_escape")

        if get_video_from_text_json:

            if self.result_diarize is None:
                raise ValueError("Generate the transcription first.")
            # with open('text_json.json', 'r') as file:
            text_json_loaded = json.loads(text_json)
            for i, segment in enumerate(self.result_diarize["segments"]):
                segment["text"] = text_json_loaded[i]["text"]
                segment["speaker"] = "SPEAKER_{:02d}".format(
                    int(text_json_loaded[i]["speaker"]) - 1
                )

        # Write subtitle
        if not self.task_in_cache("subs_and_edit", [
            copy.deepcopy(self.result_diarize),
            output_format_subtitle,
            TRANSLATE_AUDIO_TO
        ], {
            "result_diarize": self.result_diarize
        }):
            if output_format_subtitle == "disable":
                self.sub_file = "sub_tra.srt"
            elif output_format_subtitle != "ass":
                self.sub_file = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    output_format_subtitle,
                    TRANSLATE_AUDIO_TO,
                )

            # Need task
            if output_format_subtitle != "srt":
                _ = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    "srt",
                    TRANSLATE_AUDIO_TO,
                )

            if output_format_subtitle == "ass":
                convert_ori = "ffmpeg -i sub_ori.srt sub_ori.ass -y"
                convert_tra = "ffmpeg -i sub_tra.srt sub_tra.ass -y"
                self.sub_file = "sub_tra.ass"
                run_command(convert_ori)
                run_command(convert_tra)

        format_sub = (
            output_format_subtitle
            if output_format_subtitle != "disable"
            else "srt"
        )

        if output_type == "subtitle":

            out_subs = []
            tra_subs = media_out(
                media_file,
                TRANSLATE_AUDIO_TO,
                video_output_name,
                format_sub,
                file_obj=self.sub_file,
            )
            out_subs.append(tra_subs)

            ori_subs = media_out(
                media_file,
                self.align_language,
                video_output_name,
                format_sub,
                file_obj=f"sub_ori.{format_sub}",
            )
            out_subs.append(ori_subs)
            logger.info(f"Done: {out_subs}")
            return out_subs

        if output_type == "subtitle [by speaker]":
            output = get_subtitle_speaker(
                media_file,
                result=self.result_diarize,
                language=TRANSLATE_AUDIO_TO,
                extension=format_sub,
                base_name=video_output_name,
            )
            logger.info(f"Done: {str(output)}")
            return output

        if "video [subtitled]" in output_type:
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO + "_subtitled",
                video_output_name,
                "wav" if is_audio_file(media_file) else (
                    "mkv" if "mkv" in output_type else "mp4"
                ),
                file_obj=base_audio_wav if is_audio_file(media_file) else base_video_file,
                soft_subtitles=False if is_audio_file(media_file) else True,
                subtitle_files=output_format_subtitle,
            )
            msg_out = output[0] if isinstance(output, list) else output
            logger.info(f"Done: {msg_out}")
            return output

        if not self.task_in_cache("tts", [
            TRANSLATE_AUDIO_TO,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            tts_voice06,
            tts_voice07,
            tts_voice08,
            tts_voice09,
            tts_voice10,
            tts_voice11,
            dereverb_automatic_xtts
        ], {
            "sub_file": self.sub_file
        }):
            prog_disp("Text to speech...", 0.80, is_gui, progress=progress)
            self.valid_speakers = audio_segmentation_to_voice(
                self.result_diarize,
                TRANSLATE_AUDIO_TO,
                is_gui,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                dereverb_automatic_xtts,
            )

        if not self.task_in_cache("acc_and_vc", [
            max_accelerate_audio,
            acceleration_rate_regulation,
            voice_imitation,
            voice_imitation_max_segments,
            voice_imitation_remove_previous,
            voice_imitation_vocals_dereverb,
            voice_imitation_method,
            custom_voices,
            custom_voices_workers,
            copy.deepcopy(self.vci.model_config),
            avoid_overlap
        ], {
            "valid_speakers": self.valid_speakers
        }):
            audio_files, speakers_list = accelerate_segments(
                    self.result_diarize,
                    max_accelerate_audio,
                    self.valid_speakers,
                    acceleration_rate_regulation,
                )

            # Voice Imitation (Tone color converter)
            if voice_imitation:
                prog_disp(
                    "Voice Imitation...", 0.85, is_gui, progress=progress
                )
                from soni_translate.text_to_speech import toneconverter

                try:
                    toneconverter(
                        copy.deepcopy(self.result_diarize),
                        voice_imitation_max_segments,
                        voice_imitation_remove_previous,
                        voice_imitation_vocals_dereverb,
                        voice_imitation_method,
                    )
                except Exception as error:
                    logger.error(str(error))

            # custom voice
            if custom_voices:
                prog_disp(
                    "Applying customized voices...",
                    0.90,
                    is_gui,
                    progress=progress,
                )

                try:
                    self.vci(
                        audio_files,
                        speakers_list,
                        overwrite=True,
                        parallel_workers=custom_voices_workers,
                    )
                    self.vci.unload_models()
                except Exception as error:
                    logger.error(str(error))

            prog_disp(
                "Creating final translated video...",
                0.95,
                is_gui,
                progress=progress,
            )
            remove_files(dub_audio_file)
            create_translated_audio(
                self.result_diarize,
                audio_files,
                dub_audio_file,
                False,
                avoid_overlap,
            )

        # Voiceless track, change with file
        hash_base_audio_wav = get_hash(base_audio_wav)
        if voiceless_track:
            if self.voiceless_id != hash_base_audio_wav:
                from soni_translate.mdx_net import process_uvr_task

                try:
                    # voiceless_audio_file_dir = "clean_song_output/voiceless"
                    remove_files(voiceless_audio_file)
                    uvr_voiceless_audio_wav, _ = process_uvr_task(
                        orig_song_path=base_audio_wav,
                        song_id="voiceless",
                        only_voiceless=True,
                        remove_files_output_dir=False,
                    )
                    copy_files(uvr_voiceless_audio_wav, ".")
                    base_audio_wav = voiceless_audio_file
                    self.voiceless_id = hash_base_audio_wav

                except Exception as error:
                    logger.error(str(error))
            else:
                base_audio_wav = voiceless_audio_file

        if not self.task_in_cache("mix_aud", [
            mix_method_audio,
            volume_original_audio,
            volume_translated_audio,
            voiceless_track
        ], {}):
            # TYPE MIX AUDIO
            remove_files(mix_audio_file)
            command_volume_mix = f'ffmpeg -y -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio_file}'
            command_background_mix = f'ffmpeg -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio_file}'
            if mix_method_audio == "Adjusting volumes and mixing audio":
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

        if "audio" in output_type or is_audio_file(media_file):
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO,
                video_output_name,
                "wav" if "wav" in output_type else (
                    "ogg" if "ogg" in output_type else "mp3"
                ),
                file_obj=mix_audio_file,
                subtitle_files=output_format_subtitle,
            )
            msg_out = output[0] if isinstance(output, list) else output
            logger.info(f"Done: {msg_out}")
            return output

        hash_base_video_file = get_hash(base_video_file)

        if burn_subtitles_to_video:
            hashvideo_text = [
                hash_base_video_file,
                [seg["text"] for seg in self.result_diarize["segments"]]
            ]
            if self.burn_subs_id != hashvideo_text:
                try:
                    logger.info("Burn subtitles")
                    remove_files(vid_subs)
                    command = f"ffmpeg -i {base_video_file} -y -vf subtitles=sub_tra.srt -max_muxing_queue_size 9999 {vid_subs}"
                    run_command(command)
                    base_video_file = vid_subs
                    self.burn_subs_id = hashvideo_text
                except Exception as error:
                    logger.error(str(error))
            else:
                base_video_file = vid_subs

        if not self.task_in_cache("output", [
            hash_base_video_file,
            hash_base_audio_wav,
            burn_subtitles_to_video
        ], {}):
            # Merge new audio + video
            remove_files(video_output_file)
            run_command(
                f"ffmpeg -i {base_video_file} -i {mix_audio_file} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output_file}"
            )

        output = media_out(
            media_file,
            TRANSLATE_AUDIO_TO,
            video_output_name,
            "mkv" if "mkv" in output_type else "mp4",
            file_obj=video_output_file,
            soft_subtitles=soft_subtitles_to_video,
            subtitle_files=output_format_subtitle,
        )
        msg_out = output[0] if isinstance(output, list) else output
        logger.info(f"Done: {msg_out}")

        return output

    def hook_beta_processor(
        self,
        document,
        tgt_lang,
        translate_process,
        ori_lang,
        tts,
        name_final_file,
        custom_voices,
        custom_voices_workers,
        output_type,
        chunk_size,
        width,
        height,
        start_page,
        end_page,
        bcolor,
        is_gui,
        progress
    ):
        prog_disp("Processing pages...", 0.10, is_gui, progress=progress)
        doc_data = doc_to_txtximg_pages(document,  width, height, start_page, end_page, bcolor)
        result_diarize = page_data_to_segments(doc_data, 1700)

        prog_disp("Translating...", 0.20, is_gui, progress=progress)
        result_diarize["segments"] = translate_text(
            result_diarize["segments"],
            tgt_lang,
            translate_process,
            chunk_size=0,
            source=ori_lang,
        )
        chunk_size = (
            chunk_size if chunk_size else determine_chunk_size(tts)
        )
        doc_data = update_page_data(result_diarize, doc_data)

        prog_disp("Text to speech...", 0.30, is_gui, progress=progress)
        result_diarize = page_data_to_segments(doc_data, chunk_size)
        valid_speakers = audio_segmentation_to_voice(
            result_diarize,
            tgt_lang,
            is_gui,
            tts,
        )

        # fix format and set folder output
        audio_files, speakers_list = accelerate_segments(
                result_diarize,
                1.0,
                valid_speakers,
            )

        # custom voice
        if custom_voices:
            prog_disp(
                "Applying customized voices...",
                0.60,
                is_gui,
                progress=progress,
            )
            self.vci(
                audio_files,
                speakers_list,
                overwrite=True,
                parallel_workers=custom_voices_workers,
            )
            self.vci.unload_models()

        # Update time segments and not concat
        result_diarize = fix_timestamps_docs(result_diarize, audio_files)
        final_wav_file = "audio_book.wav"
        remove_files(final_wav_file)

        prog_disp("Creating audio file...", 0.70, is_gui, progress=progress)
        create_translated_audio(
            result_diarize, audio_files, final_wav_file, False
        )

        prog_disp("Creating video file...", 0.80, is_gui, progress=progress)
        video_doc = create_video_from_images(
                doc_data,
                result_diarize
        )

        # Merge video and audio
        prog_disp("Merging...", 0.90, is_gui, progress=progress)
        vid_out = merge_video_and_audio(video_doc, final_wav_file)

        # End
        output = media_out(
            document,
            tgt_lang,
            name_final_file,
            "mkv" if "mkv" in output_type else "mp4",
            file_obj=vid_out,
        )
        logger.info(f"Done: {output}")
        return output

    def multilingual_docs_conversion(
        self,
        string_text="",  # string
        document=None,  # doc path gui
        directory_input="",  # doc path
        origin_language="English (en)",
        target_language="English (en)",
        tts_voice00="en-US-EmmaMultilingualNeural-Female",
        name_final_file="",
        translate_process="google_translator",
        output_type="audio",
        chunk_size=None,
        custom_voices=False,
        custom_voices_workers=1,
        start_page=1,
        end_page=99999,
        width=1280,
        height=720,
        bcolor="dynamic",
        is_gui=False,
        progress=gr.Progress(),
    ):
        if "gpt" in translate_process:
            check_openai_api_key()

        SOURCE_LANGUAGE = LANGUAGES[origin_language]
        if translate_process != "disable_translation":
            TRANSLATE_AUDIO_TO = LANGUAGES[target_language]
        else:
            TRANSLATE_AUDIO_TO = SOURCE_LANGUAGE
            logger.info("No translation")
        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            logger.debug(
                "Make sure to select a 'TTS Speaker' suitable for the "
                "translation language to avoid errors with the TTS."
            )

        self.clear_cache(string_text, force=True)

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

        if "videobook" in output_type:
            if not document.lower().endswith(".pdf"):
                raise ValueError(
                    "Videobooks are only compatible with PDF files."
                )

            return self.hook_beta_processor(
                document,
                TRANSLATE_AUDIO_TO,
                translate_process,
                SOURCE_LANGUAGE,
                tts_voice00,
                name_final_file,
                custom_voices,
                custom_voices_workers,
                output_type,
                chunk_size,
                width,
                height,
                start_page,
                end_page,
                bcolor,
                is_gui,
                progress
            )

        # audio_wav = "audio.wav"
        final_wav_file = "audio_book.wav"

        prog_disp("Processing text...", 0.15, is_gui, progress=progress)
        result_file_path, result_text = document_preprocessor(
            document, is_string, start_page, end_page
        )

        if (
            output_type == "book (txt)"
            and translate_process == "disable_translation"
        ):
            return result_file_path

        if "SET_LIMIT" == os.getenv("DEMO"):
            result_text = result_text[:50]
            logger.info(
                "DEMO; Generation is limited to 50 characters to prevent "
                "CPU errors. No limitations with GPU.\n"
            )

        if translate_process != "disable_translation":
            # chunks text for translation
            result_diarize = plain_text_to_segments(result_text, 1700)
            prog_disp("Translating...", 0.30, is_gui, progress=progress)
            # not or iterative with 1700 chars
            result_diarize["segments"] = translate_text(
                result_diarize["segments"],
                TRANSLATE_AUDIO_TO,
                translate_process,
                chunk_size=0,
                source=SOURCE_LANGUAGE,
            )

            txt_file_path, result_text = segments_to_plain_text(result_diarize)

            if output_type == "book (txt)":
                return media_out(
                    result_file_path if is_string else document,
                    TRANSLATE_AUDIO_TO,
                    name_final_file,
                    "txt",
                    file_obj=txt_file_path,
                )

        # (TTS limits) plain text to result_diarize
        chunk_size = (
            chunk_size if chunk_size else determine_chunk_size(tts_voice00)
        )
        result_diarize = plain_text_to_segments(result_text, chunk_size)
        logger.debug(result_diarize)

        prog_disp("Text to speech...", 0.45, is_gui, progress=progress)
        valid_speakers = audio_segmentation_to_voice(
            result_diarize,
            TRANSLATE_AUDIO_TO,
            is_gui,
            tts_voice00,
        )

        # fix format and set folder output
        audio_files, speakers_list = accelerate_segments(
                result_diarize,
                1.0,
                valid_speakers,
            )

        # custom voice
        if custom_voices:
            prog_disp(
                "Applying customized voices...",
                0.80,
                is_gui,
                progress=progress,
            )
            self.vci(
                audio_files,
                speakers_list,
                overwrite=True,
                parallel_workers=custom_voices_workers,
            )
            self.vci.unload_models()

        prog_disp(
            "Creating final audio file...", 0.90, is_gui, progress=progress
        )
        remove_files(final_wav_file)
        create_translated_audio(
            result_diarize, audio_files, final_wav_file, True
        )

        output = media_out(
            result_file_path if is_string else document,
            TRANSLATE_AUDIO_TO,
            name_final_file,
            "mp3" if "mp3" in output_type else (
                "ogg" if "ogg" in output_type else "wav"
            ),
            file_obj=final_wav_file,
        )

        logger.info(f"Done: {output}")

        return output


title = "<center><strong><font size='7'>📽️ SoniTranslate 🈷️</font></strong></center>"


def create_gui(theme, logs_in_gui=False):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(title)
        gr.Markdown(lg_conf["description"])

        with gr.Tab(lg_conf["tab_translate"]):
            with gr.Row():
                with gr.Column():
                    input_data_type = gr.Dropdown(
                        ["SUBMIT VIDEO", "URL", "Find Video Path"],
                        value="SUBMIT VIDEO",
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

                    video_input = gr.File(
                        label="VIDEO",
                        file_count="multiple",
                        type="filepath",
                    )
                    blink_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["link_label"],
                        info=lg_conf["link_info"],
                        placeholder=lg_conf["link_ph"],
                    )
                    directory_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["dir_label"],
                        info=lg_conf["dir_info"],
                        placeholder=lg_conf["dir_ph"],
                    )
                    input_data_type.change(
                        fn=swap_visibility,
                        inputs=input_data_type,
                        outputs=[video_input, blink_input, directory_input],
                    )

                    gr.HTML()

                    SOURCE_LANGUAGE = gr.Dropdown(
                        LANGUAGES_LIST,
                        value=LANGUAGES_LIST[0],
                        label=lg_conf["sl_label"],
                        info=lg_conf["sl_info"],
                    )
                    TRANSLATE_AUDIO_TO = gr.Dropdown(
                        LANGUAGES_LIST[1:],
                        value="English (en)",
                        label=lg_conf["tat_label"],
                        info=lg_conf["tat_info"],
                    )

                    gr.HTML("<hr></h2>")

                    gr.Markdown(lg_conf["num_speakers"])
                    MAX_TTS = 12
                    min_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=1,
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
                    )
                    gr.Markdown(lg_conf["tts_select"])

                    def submit(value):
                        visibility_dict = {
                            f"tts_voice{i:02d}": gr.update(visible=i < value)
                            for i in range(MAX_TTS)
                        }
                        return [value for value in visibility_dict.values()]

                    tts_voice00 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk1"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice01 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk2"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice02 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AvaMultilingualNeural-Female",
                        label=lg_conf["sk3"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice03 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-BrianMultilingualNeural-Male",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice04 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-SeraphinaMultilingualNeural-Female",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice05 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-FlorianMultilingualNeural-Male",
                        label=lg_conf["sk6"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice06 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-VivienneMultilingualNeural-Female",
                        label=lg_conf["sk7"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice07 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-RemyMultilingualNeural-Male",
                        label=lg_conf["sk8"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice08 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk9"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice09 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk10"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice10 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk11"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice11 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk12"],
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
                            tts_voice06,
                            tts_voice07,
                            tts_voice08,
                            tts_voice09,
                            tts_voice10,
                            tts_voice11,
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
                            openvoice_models = ["openvoice", "openvoice_v2"]
                            voice_imitation_method_options = (
                                ["freevc"] + openvoice_models
                                if SoniTr.tts_info.xtts_enabled
                                else openvoice_models
                            )
                            voice_imitation_method_gui = gr.Dropdown(
                                voice_imitation_method_options,
                                value=voice_imitation_method_options[0],
                                label=lg_conf["vc_method_label"],
                                info=lg_conf["vc_method_info"],
                            )
                            voice_imitation_max_segments_gui = gr.Slider(
                                label=lg_conf["vc_segments_label"],
                                info=lg_conf["vc_segments_info"],
                                value=3,
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

                    if SoniTr.tts_info.xtts_enabled:
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
                            acceleration_rate_regulation_gui = gr.Checkbox(
                                False,
                                label=lg_conf["acc_rate_label"],
                                info=lg_conf["acc_rate_info"],
                            )
                            avoid_overlap_gui = gr.Checkbox(
                                False,
                                label=lg_conf["or_label"],
                                info=lg_conf["or_info"],
                            )

                            gr.HTML("<hr></h2>")

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
                                info="for Adjusting volumes and mixing audio",
                                value=0.25,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )
                            volume_translated_mix = gr.Slider(
                                label=lg_conf["vol_tra"],
                                info="for Adjusting volumes and mixing audio",
                                value=1.80,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )
                            main_voiceless_track = gr.Checkbox(
                                label=lg_conf["voiceless_tk_label"],
                                info=lg_conf["voiceless_tk_info"],
                            )

                            gr.HTML("<hr></h2>")
                            sub_type_options = [
                                "disable",
                                "srt",
                                "vtt",
                                "ass",
                                "txt",
                                "tsv",
                                "json",
                                "aud",
                            ]

                            sub_type_output = gr.Dropdown(
                                sub_type_options,
                                value=sub_type_options[1],
                                label=lg_conf["sub_type"],
                            )
                            soft_subtitles_to_video_gui = gr.Checkbox(
                                label=lg_conf["soft_subs_label"],
                                info=lg_conf["soft_subs_info"],
                            )
                            burn_subtitles_to_video_gui = gr.Checkbox(
                                label=lg_conf["burn_subs_label"],
                                info=lg_conf["burn_subs_info"],
                            )

                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["whisper_title"])
                            literalize_numbers_gui = gr.Checkbox(
                                True,
                                label=lg_conf["lnum_label"],
                                info=lg_conf["lnum_info"],
                            )
                            vocal_refinement_gui = gr.Checkbox(
                                False,
                                label=lg_conf["scle_label"],
                                info=lg_conf["scle_info"],
                            )
                            segment_duration_limit_gui = gr.Slider(
                                label=lg_conf["sd_limit_label"],
                                info=lg_conf["sd_limit_info"],
                                value=15,
                                step=1,
                                minimum=1,
                                maximum=30,
                            )
                            whisper_model_default = (
                                "large-v3"
                                if SoniTr.device == "cuda"
                                else "medium"
                            )

                            WHISPER_MODEL_SIZE = gr.Dropdown(
                                ASR_MODEL_OPTIONS + find_whisper_models(),
                                value=whisper_model_default,
                                label="Whisper ASR model",
                                info=lg_conf["asr_model_info"],
                                allow_custom_value=True,
                            )
                            com_t_opt, com_t_default = (
                                [COMPUTE_TYPE_GPU, "float16"]
                                if SoniTr.device == "cuda"
                                else [COMPUTE_TYPE_CPU, "float32"]
                            )
                            compute_type = gr.Dropdown(
                                com_t_opt,
                                value=com_t_default,
                                label=lg_conf["ctype_label"],
                                info=lg_conf["ctype_info"],
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=8,
                                label=lg_conf["batchz_label"],
                                info=lg_conf["batchz_info"],
                                step=1,
                            )
                            input_srt = gr.File(
                                label=lg_conf["srt_file_label"],
                                file_types=[".srt", ".ass", ".vtt"],
                                height=130,
                            )

                            gr.HTML("<hr></h2>")
                            text_segmentation_options = [
                                "sentence",
                                "word",
                                "character"
                            ]
                            text_segmentation_scale_gui = gr.Dropdown(
                                text_segmentation_options,
                                value=text_segmentation_options[0],
                                label=lg_conf["tsscale_label"],
                                info=lg_conf["tsscale_info"],
                            )
                            divide_text_segments_by_gui = gr.Textbox(
                                label=lg_conf["divide_text_label"],
                                value="",
                                info=lg_conf["divide_text_info"],
                            )

                            gr.HTML("<hr></h2>")
                            pyannote_models_list = list(
                                diarization_models.keys()
                            )
                            diarization_process_dropdown = gr.Dropdown(
                                pyannote_models_list,
                                value=pyannote_models_list[1],
                                label=lg_conf["diarization_label"],
                            )
                            translate_process_dropdown = gr.Dropdown(
                                TRANSLATION_PROCESS_OPTIONS,
                                value=TRANSLATION_PROCESS_OPTIONS[0],
                                label=lg_conf["tr_process_label"],
                            )

                            gr.HTML("<hr></h2>")
                            main_output_type = gr.Dropdown(
                                OUTPUT_TYPE_OPTIONS,
                                value=OUTPUT_TYPE_OPTIONS[0],
                                label=lg_conf["out_type_label"],
                            )
                            VIDEO_OUTPUT_NAME = gr.Textbox(
                                label=lg_conf["out_name_label"],
                                value="",
                                info=lg_conf["out_name_info"],
                            )
                            play_sound_gui = gr.Checkbox(
                                True,
                                label=lg_conf["task_sound_label"],
                                info=lg_conf["task_sound_info"],
                            )
                            enable_cache_gui = gr.Checkbox(
                                True,
                                label=lg_conf["cache_label"],
                                info=lg_conf["cache_info"],
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
                        video_output = gr.File(
                            label=lg_conf["output_result_label"],
                            file_count="multiple",
                            interactive=False,

                        )  # gr.Video()

                    gr.HTML("<hr></h2>")

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
                                ["./assets/Video_main.mp4"],
                                "",
                                "",
                                "",
                                False,
                                whisper_model_default,
                                4,
                                com_t_default,
                                "Spanish (es)",
                                "English (en)",
                                1,
                                2,
                                "en-CA-ClaraNeural-Female",
                                "en-AU-WilliamNeural-Male",
                            ],
                        ],  # no update
                        fn=SoniTr.batch_multilingual_media_conversion,
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
                                value="SUBMIT DOCUMENT",
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

                            gr.HTML()

                            tts_documents = gr.Dropdown(
                                list(
                                    filter(
                                        lambda x: x != "_XTTS_/AUTOMATIC.wav",
                                        SoniTr.tts_info.tts_list(),
                                    )
                                ),
                                value="en-US-EmmaMultilingualNeural-Female",
                                label="TTS",
                                visible=True,
                                interactive=True,
                            )

                            gr.HTML()

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
                                    docs_translate_process_dropdown = gr.Dropdown(
                                        DOCS_TRANSLATION_PROCESS_OPTIONS,
                                        value=DOCS_TRANSLATION_PROCESS_OPTIONS[
                                            0
                                        ],
                                        label="Translation process",
                                    )

                                    gr.HTML("<hr></h2>")

                                    docs_output_type = gr.Dropdown(
                                        DOCS_OUTPUT_TYPE_OPTIONS,
                                        value=DOCS_OUTPUT_TYPE_OPTIONS[2],
                                        label="Output type",
                                    )
                                    docs_OUTPUT_NAME = gr.Textbox(
                                        label="Final file name",
                                        value="",
                                        info=lg_conf["out_name_info"],
                                    )
                                    docs_chunk_size = gr.Number(
                                        label=lg_conf["chunk_size_label"],
                                        value=0,
                                        visible=True,
                                        interactive=True,
                                        info=lg_conf["chunk_size_info"],
                                    )
                                    gr.HTML("<hr></h2>")
                                    start_page_gui = gr.Number(
                                        step=1,
                                        value=1,
                                        minimum=1,
                                        maximum=99999,
                                        label="Start page",
                                    )
                                    end_page_gui = gr.Number(
                                        step=1,
                                        value=99999,
                                        minimum=1,
                                        maximum=99999,
                                        label="End page",
                                    )
                                    gr.HTML("<hr>Videobook config</h2>")
                                    videobook_width_gui = gr.Number(
                                        step=1,
                                        value=1280,
                                        minimum=100,
                                        maximum=4096,
                                        label="Width",
                                    )
                                    videobook_height_gui = gr.Number(
                                        step=1,
                                        value=720,
                                        minimum=100,
                                        maximum=4096,
                                        label="Height",
                                    )
                                    videobook_bcolor_gui = gr.Dropdown(
                                        BORDER_COLORS,
                                        value=BORDER_COLORS[0],
                                        label="Border color",
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
                                docs_output = gr.File(
                                    label="Result",
                                    interactive=False,
                                )

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
                        models_path, index_path = upload_model_list()

                        dict_models = {
                            f"fmodel{i:02d}": gr.update(
                                choices=models_path
                            )
                            for i in range(MAX_TTS+1)
                        }
                        dict_index = {
                            f"findex{i:02d}": gr.update(
                                choices=index_path, value=None
                            )
                            for i in range(MAX_TTS+1)
                        }
                        dict_changes = {**dict_models, **dict_index}
                        return [value for value in dict_changes.values()]

            with gr.Column():
                with gr.Accordion(lg_conf["replace_title"], open=False):
                    with gr.Column(variant="compact"):
                        with gr.Column():
                            gr.Markdown(lg_conf["sec1_title"])
                            enable_custom_voice = gr.Checkbox(
                                False,
                                label="ENABLE",
                                info=lg_conf["enable_replace"]
                            )
                            workers_custom_voice = gr.Number(
                                step=1,
                                value=1,
                                minimum=1,
                                maximum=50,
                                label="workers",
                                visible=False,
                            )

                            gr.Markdown(lg_conf["sec2_title"])
                            gr.Markdown(lg_conf["sec2_subtitle"])

                            PITCH_ALGO_OPT = [
                                "pm",
                                "harvest",
                                "crepe",
                                "rmvpe",
                                "rmvpe+",
                            ]

                            def model_conf():
                                return gr.Dropdown(
                                    models_path,
                                    # value="",
                                    label="Model",
                                    visible=True,
                                    interactive=True,
                                )

                            def pitch_algo_conf():
                                return gr.Dropdown(
                                    PITCH_ALGO_OPT,
                                    value=PITCH_ALGO_OPT[3],
                                    label="Pitch algorithm",
                                    visible=True,
                                    interactive=True,
                                )

                            def pitch_lvl_conf():
                                return gr.Slider(
                                    label="Pitch level",
                                    minimum=-24,
                                    maximum=24,
                                    step=1,
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )

                            def index_conf():
                                return gr.Dropdown(
                                    index_path,
                                    value=None,
                                    label="Index",
                                    visible=True,
                                    interactive=True,
                                )

                            def index_inf_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Index influence",
                                    value=0.75,
                                )

                            def respiration_filter_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label="Respiration median filtering",
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )

                            def envelope_ratio_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Envelope ratio",
                                    value=0.25,
                                    interactive=True,
                                )

                            def consonant_protec_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label="Consonant breath protection",
                                    value=0.5,
                                    interactive=True,
                                )

                            def button_conf(tts_name):
                                return gr.Button(
                                    lg_conf["cv_button_apply"]+" "+tts_name,
                                    variant="primary",
                                )

                            TTS_TABS = [
                                'TTS Speaker {:02d}'.format(i) for i in range(1, MAX_TTS+1)
                            ]

                            CV_SUBTITLES = [
                                lg_conf["cv_tts1"],
                                lg_conf["cv_tts2"],
                                lg_conf["cv_tts3"],
                                lg_conf["cv_tts4"],
                                lg_conf["cv_tts5"],
                                lg_conf["cv_tts6"],
                                lg_conf["cv_tts7"],
                                lg_conf["cv_tts8"],
                                lg_conf["cv_tts9"],
                                lg_conf["cv_tts10"],
                                lg_conf["cv_tts11"],
                                lg_conf["cv_tts12"],
                            ]

                            configs_storage = []

                            for i in range(MAX_TTS):  # Loop from 00 to 11
                                with gr.Accordion(CV_SUBTITLES[i], open=False):
                                    gr.Markdown(TTS_TABS[i])
                                    with gr.Column():
                                        tag_gui = gr.Textbox(
                                            value=TTS_TABS[i], visible=False
                                        )
                                        model_gui = model_conf()
                                        pitch_algo_gui = pitch_algo_conf()
                                        pitch_lvl_gui = pitch_lvl_conf()
                                        index_gui = index_conf()
                                        index_inf_gui = index_inf_conf()
                                        rmf_gui = respiration_filter_conf()
                                        er_gui = envelope_ratio_conf()
                                        cbp_gui = consonant_protec_conf()

                                        with gr.Row(variant="compact"):
                                            button_config = button_conf(
                                                TTS_TABS[i]
                                            )

                                            confirm_conf = gr.HTML()

                                        button_config.click(
                                            SoniTr.vci.apply_conf,
                                            inputs=[
                                                tag_gui,
                                                model_gui,
                                                pitch_algo_gui,
                                                pitch_lvl_gui,
                                                index_gui,
                                                index_inf_gui,
                                                rmf_gui,
                                                er_gui,
                                                cbp_gui,
                                            ],
                                            outputs=[confirm_conf],
                                        )

                                        configs_storage.append({
                                            "tag": tag_gui,
                                            "model": model_gui,
                                            "index": index_gui,
                                        })

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
                                    sorted(SoniTr.tts_info.list_edge),
                                    value="en-GB-ThomasNeural-Male",
                                    label="TTS",
                                    visible=True,
                                    interactive=True,
                                )
                                model_test = model_conf()
                                index_test = index_conf()
                                pitch_test = pitch_lvl_conf()
                                pitch_alg_test = pitch_algo_conf()
                        with gr.Row(variant="compact"):
                            button_test = gr.Button("Test audio")

                        with gr.Column():
                            with gr.Row():
                                original_ttsvoice = gr.Audio()
                                ttsvoice = gr.Audio()

                            button_test.click(
                                SoniTr.vci.make_test,
                                inputs=[
                                    text_test,
                                    tts_test,
                                    model_test,
                                    index_test,
                                    pitch_test,
                                    pitch_alg_test,
                                ],
                                outputs=[ttsvoice, original_ttsvoice],
                            )

                    download_button.click(
                        download_list,
                        [url_links],
                        [download_finish],
                        queue=False
                    ).then(
                        update_models,
                        [],
                        [
                            elem["model"] for elem in configs_storage
                        ] + [model_test] + [
                            elem["index"] for elem in configs_storage
                        ] + [index_test],
                    )

        with gr.Tab(lg_conf["tab_help"]):
            gr.Markdown(lg_conf["tutorial"])
            gr.Markdown(news)

            def play_sound_alert(play_sound):

                if not play_sound:
                    return None

                # silent_sound = "assets/empty_audio.mp3"
                sound_alert = "assets/sound_alert.mp3"

                time.sleep(0.25)
                # yield silent_sound
                yield None

                time.sleep(0.25)
                yield sound_alert

            sound_alert_notification = gr.Audio(
                value=None,
                type="filepath",
                format="mp3",
                autoplay=True,
                visible=False,
            )

        if logs_in_gui:
            logger.info("Logs in gui need public url")

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

        if SoniTr.tts_info.xtts_enabled:
            # Update tts list
            def update_tts_list():
                update_dict = {
                    f"tts_voice{i:02d}": gr.update(choices=SoniTr.tts_info.tts_list())
                    for i in range(MAX_TTS)
                }
                update_dict["tts_documents"] = gr.update(
                    choices=list(
                        filter(
                            lambda x: x != "_XTTS_/AUTOMATIC.wav",
                            SoniTr.tts_info.tts_list(),
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
                    tts_voice06,
                    tts_voice07,
                    tts_voice08,
                    tts_voice09,
                    tts_voice10,
                    tts_voice11,
                    tts_documents,
                ],
            )

        # Run translate text
        subs_button.click(
            SoniTr.batch_multilingual_media_conversion,
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
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                edit_sub_check,  # TRUE BY DEFAULT
                dummy_false_check,  # dummy false
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=subs_edit_space,
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # Run translate tts and complete
        video_button.click(
            SoniTr.batch_multilingual_media_conversion,
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
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                dummy_false_check,
                edit_sub_check,
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=video_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
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
                enable_custom_voice,
                workers_custom_voice,
                start_page_gui,
                end_page_gui,
                videobook_width_gui,
                videobook_height_gui,
                videobook_bcolor_gui,
                docs_dummy_check,
            ],
            outputs=docs_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

    return app


def get_language_config(language_data, language=None, base_key="english"):
    base_lang = language_data.get(base_key)

    if language not in language_data:
        logger.error(
            f"Language {language} not found, defaulting to {base_key}"
        )
        return base_lang

    lg_conf = language_data.get(language, {})
    lg_conf.update((k, v) for k, v in base_lang.items() if k not in lg_conf)

    return lg_conf


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Taithrah/Minimal",
        help=(
            "Specify the theme; find themes in "
            "https://huggingface.co/spaces/gradio/theme-gallery;"
            " Example: --theme aliabid94/new-theme"
        ),
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
        help=(
            "Set logger verbosity level: "
            "debug, info, warning, error, or critical"
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help=" Select the language of the interface: english, spanish",
    )
    parser.add_argument(
        "--cpu_mode",
        action="store_true",
        default=False,
        help="Enable CPU mode to run the program without utilizing GPU acceleration.",
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

    models_path, index_path = upload_model_list()

    SoniTr = SoniTranslate(cpu_mode=args.cpu_mode)

    lg_conf = get_language_config(language_data, language=args.language)

    app = create_gui(args.theme, logs_in_gui=args.logs_in_gui)

    app.queue()

    app.launch(
        max_threads=1,
        share=args.public_url,
        show_error=True,
        quiet=False,
        debug=(True if logger.isEnabledFor(logging.DEBUG) else False),
    )
