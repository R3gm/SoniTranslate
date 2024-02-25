from .utils import remove_files
from .logging_setup import logger
import unicodedata
import shutil
import os
import re


def get_no_ext_filename(file_path):
    file_name_with_extension = os.path.basename(rf"{file_path}")
    filename_without_extension, _ = os.path.splitext(file_name_with_extension)
    return filename_without_extension


def get_video_info(link):
    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info_dict = ydl.extract_info(link, download=False, process=False)
            video_id = info_dict.get("id", "youtube_video")
            video_title = info_dict.get("title", video_id)
    except Exception as error:
        logger.error(str(error))
        aux_name = link[:10] + link[-15:]
        video_title, video_id = f"video_{aux_name}", "NO_ID"
    return video_title, video_id


def sanitize_file_name(file_name):
    # Normalize the string to NFKD form to separate combined
    # characters into base characters and diacritics
    normalized_name = unicodedata.normalize("NFKD", file_name)
    # Replace any non-ASCII characters or special symbols with an underscore
    sanitized_name = re.sub(r"[^\w\s.-]", "_", normalized_name)
    return sanitized_name


def get_output_file(original_file, new_file_name):

    directory, filename = os.path.split(original_file)

    new_file_path = os.path.join(directory, new_file_name)
    remove_files(new_file_path)

    shutil.copy2(original_file, new_file_path)
    return os.path.join(os.getcwd(), new_file_path)


def media_out(
    media_file,
    lang_code,
    media_out_name="",
    extension="mp4",
    file_obj="video_dub.mp4",
):
    if not media_out_name:
        if os.path.exists(media_file):
            media_out_name = get_no_ext_filename(media_file)
        else:
            media_out_name, _ = get_video_info(media_file)

    f_name = f"{sanitize_file_name(media_out_name)}__{lang_code}.{extension}"

    return get_output_file(file_obj, f_name)
