from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
import whisperx, torch, gc
from IPython.utils import capture


device = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_speech(audio_wav, WHISPER_MODEL_SIZE, compute_type, batch_size, SOURCE_LANGUAGE):
    with capture.capture_output() as cap:
        model = whisperx.load_model(
            WHISPER_MODEL_SIZE,
            device,
            compute_type=compute_type,
            language= SOURCE_LANGUAGE,
            )
        del cap
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect(); torch.cuda.empty_cache(); del model
    return audio, result


def align_speech(audio, result):
    # 2. Align whisper output

    DAMHF.update(DAMT) #lang align
    EXTRA_ALIGN = {
        "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    } # add new align models here
    #print(result['language'], DAM.keys(), EXTRA_ALIGN.keys())
    if not result['language'] in DAMHF.keys() and not result['language'] in EXTRA_ALIGN.keys():
        audio = result = None
        print("Automatic detection: Source language not compatible")
        print(f"Detected language {result['language']}  incompatible, you can select the source language to avoid this error.")
        return

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
        model_name = None if result["language"] in DAMHF.keys() else EXTRA_ALIGN[result["language"]]
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
    return result

diarization_models = {
    "pyannote_3.1" : "pyannote/speaker-diarization-3.1",
    "pyannote_2.1" : "pyannote/speaker-diarization@2.1"
}

def diarize_speech(audio_wav, result, min_speakers, max_speakers, YOUR_HF_TOKEN, model_name="pyannote/speaker-diarization@2.1"):
    # 3. Assign speaker labels
    with capture.capture_output() as cap:
        diarize_model = whisperx.DiarizationPipeline(model_name=model_name, use_auth_token=YOUR_HF_TOKEN, device=device)
        del cap
    diarize_segments = diarize_model(
        audio_wav,
        min_speakers=min_speakers,
        max_speakers=max_speakers)

    result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
    gc.collect(); torch.cuda.empty_cache(); del diarize_model
    return result_diarize
