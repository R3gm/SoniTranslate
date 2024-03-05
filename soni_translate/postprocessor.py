from .utils import remove_files, run_command
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
    aux_name = f"video_url_{link}"
    params_dlp = {"quiet": True, "no_warnings": True, "noplaylist": True}
    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL(params_dlp) as ydl:
            if link.startswith(("www.youtube.com/", "m.youtube.com/")):
                link = "https://" + link
            info_dict = ydl.extract_info(link, download=False, process=False)
            video_id = info_dict.get("id", aux_name)
            video_title = info_dict.get("title", video_id)
            if "youtube.com" in link and "&list=" in link:
                video_title = ydl.extract_info(
                    "https://m.youtube.com/watch?v="+video_id,
                    download=False,
                    process=False
                ).get("title", video_title)
    except Exception as error:
        logger.error(str(error))
        video_title, video_id = aux_name, "NO_ID"
    return video_title, video_id


def sanitize_file_name(file_name):
    # Normalize the string to NFKD form to separate combined
    # characters into base characters and diacritics
    normalized_name = unicodedata.normalize("NFKD", file_name)
    # Replace any non-ASCII characters or special symbols with an underscore
    sanitized_name = re.sub(r"[^\w\s.-]", "_", normalized_name)
    return sanitized_name


def get_output_file(
        original_file,
        new_file_name,
        soft_subtitles,
        output_directory="",
):
    directory, filename = os.path.split(original_file)

    if output_directory and os.path.isdir(output_directory):
        new_file_path = os.path.join(output_directory, new_file_name)
    else:
        new_file_path = os.path.join(directory, "outputs", new_file_name)
    remove_files(new_file_path)

    cm = None
    if soft_subtitles and original_file.endswith(".mp4"):
        if new_file_path.endswith(".mp4"):
            cm = f'ffmpeg -y -i "{original_file}" -i sub_tra.srt -i sub_ori.srt -map 0:v -map 0:a -map 1 -map 2 -c:v copy -c:a copy -c:s mov_text "{new_file_path}"'
        else:
            cm = f'ffmpeg -y -i "{original_file}" -i sub_tra.srt -i sub_ori.srt -map 0:v -map 0:a -map 1 -map 2 -c:v copy -c:a copy -c:s srt -movflags use_metadata_tags -map_metadata 0 "{new_file_path}"'
    elif new_file_path.endswith(".mkv"):
        cm = f'ffmpeg -i "{original_file}" -c:v copy -c:a copy "{new_file_path}"'
    elif new_file_path.endswith(".wav"):
        cm = f'ffmpeg -y -i "{original_file}" -acodec pcm_s16le -ar 44100 -ac 2 "{new_file_path}"'
    elif new_file_path.endswith(".ogg"):
        cm = f'ffmpeg -i "{original_file}" -c:a libvorbis "{new_file_path}"'

    if cm:
        try:
            run_command(cm)
        except Exception as error:
            logger.error(str(error))
            remove_files(new_file_path)
            shutil.copy2(original_file, new_file_path)
    else:
        shutil.copy2(original_file, new_file_path)

    return os.path.join(os.getcwd(), new_file_path)


def media_out(
    media_file,
    lang_code,
    media_out_name="",
    extension="mp4",
    file_obj="video_dub.mp4",
    soft_subtitles=False,
):
    if not media_out_name:
        if os.path.exists(media_file):
            base_name = get_no_ext_filename(media_file)
        else:
            base_name, _ = get_video_info(media_file)

        media_out_name = f"{base_name}__{lang_code}"

    f_name = f"{sanitize_file_name(media_out_name)}.{extension}"

    return get_output_file(file_obj, f_name, soft_subtitles)
