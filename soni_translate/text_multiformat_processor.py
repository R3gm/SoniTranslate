from whisperx.utils import get_writer
from .utils import remove_files
import srt
import re
import os
import copy


def extract_from_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        srt_content = file.read()

    subtitle_generator = srt.parse(srt_content)
    srt_content_list = list(subtitle_generator)

    return srt_content_list


def srt_file_to_segments(file_path, speaker=False):
    srt_content_list = extract_from_srt(file_path)

    segments = []
    for segment in srt_content_list:
        segments.append(
            {
                "text": str(segment.content),
                "start": float(segment.start.total_seconds()),
                "end": float(segment.end.total_seconds()),
            }
        )

    if not segments:
        raise Exception("No data found in srt subtitle file")

    if speaker:
        segments = [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    return {"segments": segments}


# documents


def pdf_to_txt(pdf_file):
    import PyPDF2

    with open(pdf_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def docx_to_txt(docx_file):
    from docx import Document

    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def replace_multiple_elements(text, replacements):
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    replaced_text = pattern.sub(
        lambda match: replacements[match.group(0)], text
    )

    # Remove multiple spaces
    replaced_text = re.sub(r"\s+", " ", replaced_text)

    return replaced_text


def document_preprocessor(file_path, is_string):
    if not is_string:
        file_ext = os.path.splitext(file_path)[1].lower()

    if is_string:
        text = file_path
    elif file_ext == ".pdf":
        text = pdf_to_txt(file_path)
    elif file_ext == ".docx":
        text = docx_to_txt(file_path)
    elif file_ext == ".txt":
        with open(file_path, "r") as file:
            text = file.read()
    else:
        raise Exception("Unsupported file format")

    # Add space to break segments more easily later
    replacements = {
        "、": "、 ",
        "。": "。 ",
        # "\n": " ",
    }
    text = replace_multiple_elements(text, replacements)

    # Save text to a .txt file
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = "./text_preprocessor.txt"

    with open(txt_file_path, "w") as txt_file:
        txt_file.write(text)

    return txt_file_path, text


def split_text_into_chunks(text, chunk_size):
    words = re.findall(r"\b\w+\b", text)
    chunks = []
    current_chunk = ""
    for word in words:
        if (
            len(current_chunk) + len(word) + 1 <= chunk_size
        ):  # Adding 1 for the space between words
            if current_chunk:
                current_chunk += " "
            current_chunk += word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def determine_chunk_size(file_name):
    patterns = {
        re.compile(r".*-(Male|Female)$"): 600,  # by character
        re.compile(r".* BARK$"): 100,  # t 64 256
        re.compile(r".* VITS$"): 500,
        re.compile(
            r".+\.(wav|mp3|ogg|m4a)$"
        ): 150,  # t 250 400 api automatic split
        re.compile(r".* VITS-onnx$"): 250,  # automatic sentence split
    }

    for pattern, chunk_size in patterns.items():
        if pattern.match(file_name):
            return chunk_size

    # Default chunk size if the file doesn't match any pattern; max 1800
    return 100


def plain_text_to_segments(result_text=None, chunk_size=None):
    if not chunk_size:
        chunk_size = 100
    text_chunks = split_text_into_chunks(result_text, chunk_size)

    segments_chunks = []
    for num, chunk in enumerate(text_chunks):
        chunk_dict = {
            "text": chunk,
            "start": (1.0 + num),
            "end": (2.0 + num),
            "speaker": "SPEAKER_00",
        }
        segments_chunks.append(chunk_dict)

    result_diarize = {"segments": segments_chunks}

    return result_diarize


def segments_to_plain_text(result_diarize):
    complete_text = ""
    for seg in result_diarize["segments"]:
        complete_text += seg["text"] + " "  # issue

    # Save text to a .txt file
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = "./text_translation.txt"

    with open(txt_file_path, "w") as txt_file:
        txt_file.write(complete_text)

    return txt_file_path, complete_text


# subtitles


def process_subtitles(
    deep_copied_result,
    align_language,
    result_diarize,
    output_format_subtitle,
    TRANSLATE_AUDIO_TO,
):
    name_ori = "sub_ori."
    name_tra = "sub_tra."
    remove_files(
        [name_ori + output_format_subtitle, name_tra + output_format_subtitle]
    )

    writer = get_writer(output_format_subtitle, output_dir=".")
    word_options = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }

    # original lang
    subs_copy_result = copy.deepcopy(deep_copied_result)
    subs_copy_result["language"] = align_language
    for segment in subs_copy_result["segments"]:
        segment.pop("speaker", None)

    writer(
        subs_copy_result,
        name_ori[:-1] + ".mp3",
        word_options,
    )

    # translated lang
    subs_tra_copy_result = copy.deepcopy(result_diarize)
    subs_tra_copy_result["language"] = (
        "ja" if TRANSLATE_AUDIO_TO in ["ja", "zh"] else align_language
    )
    subs_tra_copy_result.pop("word_segments", None)
    for segment in subs_tra_copy_result["segments"]:
        for key in ["speaker", "chars", "words"]:
            segment.pop(key, None)

    writer(
        subs_tra_copy_result,
        name_tra[:-1] + ".mp3",
        word_options,
    )

    return name_tra + output_format_subtitle
