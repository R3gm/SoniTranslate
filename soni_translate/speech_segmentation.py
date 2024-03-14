from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
import whisperx
import torch
import gc
from IPython.utils import capture
from .language_configuration import EXTRA_ALIGN
from .logging_setup import logger

device = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_speech(
    audio_wav, WHISPER_MODEL_SIZE, compute_type, batch_size, SOURCE_LANGUAGE
):
    """
    Transcribe speech using a whisper model.

    Parameters:
    - audio_wav (str): Path to the audio file in WAV format.
    - WHISPER_MODEL_SIZE (str): The whisper model to be loaded.
    - compute_type (str): Type of compute to be used (e.g., 'int8', 'float16').
    - batch_size (int): Batch size for transcription.
    - SOURCE_LANGUAGE (str): Source language for transcription.

    Returns:
    - Tuple containing:
        - audio: Loaded audio file.
        - result: Transcription result as a dictionary.
    """
    with capture.capture_output() as cap:
        model = whisperx.load_model(
            WHISPER_MODEL_SIZE,
            device,
            compute_type=compute_type,
            language=SOURCE_LANGUAGE,
        )
        del cap
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.

    Parameters:
    - audio (array): The audio data in a suitable format for alignment.
    - result (dict): Metadata containing information about the segments and language.

    Returns:
    - result (dict): Updated metadata after aligning the segments with the audio.
        This includes character-level alignments if
        'return_char_alignments' is set to True.

    Notes:
    - This function uses language-specific models to align speech segments.
    - It performs language compatibility checks and selects the
        appropriate alignment model.
    - Cleans up memory by releasing resources after alignment.
    """
    DAMHF.update(DAMT)  # lang align
    if (
        not result["language"] in DAMHF.keys()
        and not result["language"] in EXTRA_ALIGN.keys()
    ):
        audio = result = None
        logger.warning(
            "Automatic detection: Source language not compatible with align"
        )
        raise ValueError(
            f"Detected language {result['language']}  incompatible, "
            "you can select the source language to avoid this error."
        )

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
        model_name=None
        if result["language"] in DAMHF.keys()
        else EXTRA_ALIGN[result["language"]],
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=True,
    )
    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    """
    Performs speaker diarization on speech segments.

    Parameters:
    - audio_wav (array): Audio data in WAV format to perform speaker diarization.
    - result (dict): Metadata containing information about speech segments
        and alignments.
    - min_speakers (int): Minimum number of speakers expected in the audio.
    - max_speakers (int): Maximum number of speakers expected in the audio.
    - YOUR_HF_TOKEN (str): Your Hugging Face API token for model authentication.
    - model_name (str): Name of the speaker diarization model to be used
        (default: "pyannote/speaker-diarization@2.1").

    Returns:
    - result_diarize (dict): Updated metadata after assigning speaker
        labels to segments.

    Notes:
    - This function utilizes a speaker diarization model to label speaker
        segments in the audio.
    - It assigns speakers to word-level segments based on diarization results.
    - Cleans up memory by releasing resources after diarization.
    - If only one speaker is specified, each segment is automatically assigned as
        the first speaker, eliminating the need for diarization inference.
    """

    if max(min_speakers, max_speakers) > 1 and model_name:
        try:
            with capture.capture_output() as cap:
                diarize_model = whisperx.DiarizationPipeline(
                    model_name=model_name,
                    use_auth_token=YOUR_HF_TOKEN,
                    device=device,
                )
                del cap
        except Exception as error:
            error_str = str(error)
            gc.collect()
            torch.cuda.empty_cache()  # noqa
            if "'NoneType' object has no attribute 'to'" in error_str:
                if model_name == diarization_models["pyannote_2.1"]:
                    raise ValueError(
                        "Accept the license agreement for using Pyannote 2.1."
                        " You need to have an account on Hugging Face and "
                        "accept the license to use the models: "
                        "https://huggingface.co/pyannote/speaker-diarization "
                        "and https://huggingface.co/pyannote/segmentation "
                        "Get your KEY TOKEN here: https://hf.co/settings/tokens"
                    )
                elif model_name == diarization_models["pyannote_3.1"]:
                    raise ValueError(
                        "New Licence Pyannote 3.1: You need to have an account"
                        " on Hugging Face and accept the license to use the "
                        "models: https://huggingface.co/pyannote/speaker-diarization-3.1"
                        " and https://huggingface.co/pyannote/segmentation-3.0 "
                    )
            else:
                raise error
        diarize_segments = diarize_model(
            audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
        )

        result_diarize = whisperx.assign_word_speakers(
            diarize_segments, result
        )
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()  # noqa
    else:
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"}
            for item in result_diarize["segments"]
        ]
    return result_diarize
