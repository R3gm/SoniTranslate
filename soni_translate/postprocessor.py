from .utils import remove_files, run_command
from .text_multiformat_processor import get_subtitle
from .logging_setup import logger
import unicodedata
import shutil
import copy
import os
import re

OUTPUT_TYPE_OPTIONS = [
    "video (mp4)",
    "video (mkv)",
    "audio (mp3)",
    "audio (ogg)",
    "audio (wav)",
    "subtitle",
    "subtitle [by speaker]",
    "video [subtitled] (mp4)",
    "video [subtitled] (mkv)",
    "audio [original vocal sound]",
    "audio [original background sound]",
    "audio [original vocal and background sound]",
    "audio [original vocal-dereverb sound]",
    "audio [original vocal-dereverb and background sound]",
    "raw media",
]

DOCS_OUTPUT_TYPE_OPTIONS = [
    "videobook (mp4)",
    "videobook (mkv)",
    "audiobook (wav)",
    "audiobook (mp3)",
    "audiobook (ogg)",
    "book (txt)",
]  # Add DOCX and etc.


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
    directory_base = "."  # default directory

    if output_directory and os.path.isdir(output_directory):
        new_file_path = os.path.join(output_directory, new_file_name)
    else:
        new_file_path = os.path.join(directory_base, "outputs", new_file_name)
    remove_files(new_file_path)

    cm = None
    if soft_subtitles and original_file.endswith(".mp4"):
        if new_file_path.endswith(".mp4"):
            cm = f'ffmpeg -y -i "{original_file}" -i sub_tra.srt -i sub_ori.srt -map 0:v -map 0:a -map 1 -map 2 -c:v copy -c:a copy -c:s mov_text "{new_file_path}"'
        else:
            cm = f'ffmpeg -y -i "{original_file}" -i sub_tra.srt -i sub_ori.srt -map 0:v -map 0:a -map 1 -map 2 -c:v copy -c:a copy -c:s srt -movflags use_metadata_tags -map_metadata 0 "{new_file_path}"'
    elif new_file_path.endswith(".mkv"):
        cm = f'ffmpeg -i "{original_file}" -c:v copy -c:a copy "{new_file_path}"'
    elif new_file_path.endswith(".wav") and not original_file.endswith(".wav"):
        cm = f'ffmpeg -y -i "{original_file}" -acodec pcm_s16le -ar 44100 -ac 2 "{new_file_path}"'
    elif new_file_path.endswith(".ogg"):
        cm = f'ffmpeg -i "{original_file}" -c:a libvorbis "{new_file_path}"'
    elif new_file_path.endswith(".mp3") and not original_file.endswith(".mp3"):
        cm = f'ffmpeg -y -i "{original_file}" -codec:a libmp3lame -qscale:a 2 "{new_file_path}"'

    if cm:
        try:
            run_command(cm)
        except Exception as error:
            logger.error(str(error))
            remove_files(new_file_path)
            shutil.copy2(original_file, new_file_path)
    else:
        shutil.copy2(original_file, new_file_path)

    return os.path.abspath(new_file_path)


def media_out(
    media_file,
    lang_code,
    media_out_name="",
    extension="mp4",
    file_obj="video_dub.mp4",
    soft_subtitles=False,
    subtitle_files="disable",
):
    if media_out_name:
        base_name = media_out_name + "_origin"
    else:
        if os.path.exists(media_file):
            base_name = get_no_ext_filename(media_file)
        else:
            base_name, _ = get_video_info(media_file)

        media_out_name = f"{base_name}__{lang_code}"

    f_name = f"{sanitize_file_name(media_out_name)}.{extension}"

    if subtitle_files != "disable":
        final_media = [get_output_file(file_obj, f_name, soft_subtitles)]
        name_tra = f"{sanitize_file_name(media_out_name)}.{subtitle_files}"
        name_ori = f"{sanitize_file_name(base_name)}.{subtitle_files}"
        tgt_subs = f"sub_tra.{subtitle_files}"
        ori_subs = f"sub_ori.{subtitle_files}"
        final_subtitles = [
            get_output_file(tgt_subs, name_tra, False),
            get_output_file(ori_subs, name_ori, False)
        ]
        return final_media + final_subtitles
    else:
        return get_output_file(file_obj, f_name, soft_subtitles)


def get_subtitle_speaker(media_file, result, language, extension, base_name):

    segments_base = copy.deepcopy(result)

    # Sub segments by speaker
    segments_by_speaker = {}
    for segment in segments_base["segments"]:
        if segment["speaker"] not in segments_by_speaker.keys():
            segments_by_speaker[segment["speaker"]] = [segment]
        else:
            segments_by_speaker[segment["speaker"]].append(segment)

    if not base_name:
        if os.path.exists(media_file):
            base_name = get_no_ext_filename(media_file)
        else:
            base_name, _ = get_video_info(media_file)

    files_subs = []
    for name_sk, segments in segments_by_speaker.items():

        subtitle_speaker = get_subtitle(
            language,
            {"segments": segments},
            extension,
            filename=name_sk,
        )

        media_out_name = f"{base_name}_{language}_{name_sk}"

        output = media_out(
            media_file,  # no need
            language,
            media_out_name,
            extension,
            file_obj=subtitle_speaker,
        )

        files_subs.append(output)

    return files_subs


def sound_separate(media_file, task_uvr):
    from .mdx_net import process_uvr_task

    outputs = []

    if "vocal" in task_uvr:
        try:
            _, _, _, _, vocal_audio = process_uvr_task(
                orig_song_path=media_file,
                main_vocals=False,
                dereverb=True if "dereverb" in task_uvr else False,
                remove_files_output_dir=True,
            )
            outputs.append(vocal_audio)
        except Exception as error:
            logger.error(str(error))

    if "background" in task_uvr:
        try:
            background_audio, _ = process_uvr_task(
                orig_song_path=media_file,
                song_id="voiceless",
                only_voiceless=True,
                remove_files_output_dir=False if "vocal" in task_uvr else True,
            )
            # copy_files(background_audio, ".")
            outputs.append(background_audio)
        except Exception as error:
            logger.error(str(error))

    if not outputs:
        raise Exception("Error in uvr process")

    return outputs
