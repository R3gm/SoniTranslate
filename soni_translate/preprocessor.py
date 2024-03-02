from .utils import remove_files
import os, shutil, subprocess, time, shlex, sys # noqa
from .logging_setup import logger


class OperationFailedError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)


def audio_preprocessor(preview, base_audio, audio_wav, use_cuda=False):
    base_audio = base_audio.strip()
    previous_files_to_remove = [audio_wav]
    remove_files(previous_files_to_remove)

    if preview:
        logger.warning(
            "Creating a preview video of 10 seconds, to disable "
            "this option, go to advanced settings and turn off preview."
        )
        wav_ = f'ffmpeg -y -i "{base_audio}" -ss 00:00:20 -t 00:00:10 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav'
    else:
        wav_ = f'ffmpeg -y -i "{base_audio}" -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav'

    # Run cmd process
    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }
    wav_ = shlex.split(wav_)
    result_convert_audio = subprocess.Popen(wav_, **sub_params)
    output, errors = result_convert_audio.communicate()
    time.sleep(1)
    if result_convert_audio.returncode in [1, 2] or not os.path.exists(
        audio_wav
    ):
        raise OperationFailedError(f"Error can't create the audio file:\n{errors.decode('utf-8')}")


def audio_video_preprocessor(
    preview, video, OutputFile, audio_wav, use_cuda=False
):
    video = video.strip()
    previous_files_to_remove = [OutputFile, "audio.webm", audio_wav]
    remove_files(previous_files_to_remove)

    if os.path.exists(video):
        if preview:
            logger.warning(
                "Creating a preview video of 10 seconds, "
                "to disable this option, go to advanced "
                "settings and turn off preview."
            )
            mp4_ = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4'
        else:
            # Check if the file ends with ".mp4" extension
            if video.endswith(".mp4"):
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
                time.sleep(0.5)
                if os.path.exists(OutputFile):
                    mp4_ = "ffmpeg -h"
                else:
                    mp4_ = f'ffmpeg -y -i "{video}" -c copy Video.mp4'
            else:
                logger.warning(
                    "File does not have the '.mp4' extension. Converting video."
                )
                mp4_ = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4'
    else:
        if preview:
            logger.warning(
                "Creating a preview from the link, 10 seconds "
                "to disable this option, go to advanced "
                "settings and turn off preview."
            )
            # https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f"python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}"

    # Run cmd process
    mp4_ = shlex.split(mp4_)
    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }

    if os.path.exists(video):
        logger.info("Process video...")
        result_convert_video = subprocess.Popen(mp4_, **sub_params)
        # result_convert_video.wait()
        output, errors = result_convert_video.communicate()
        time.sleep(1)
        if result_convert_video.returncode in [1, 2] or not os.path.exists(
            OutputFile
        ):
            raise OperationFailedError(f"Error processing video:\n{errors.decode('utf-8')}")
        logger.info("Process audio...")
        wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
        wav_ = shlex.split(wav_)
        result_convert_audio = subprocess.Popen(wav_, **sub_params)
        output, errors = result_convert_audio.communicate()
        time.sleep(1)
        if result_convert_audio.returncode in [1, 2] or not os.path.exists(
            audio_wav
        ):
            raise OperationFailedError(f"Error can't create the audio file:\n{errors.decode('utf-8')}")

    else:
        wav_ = shlex.split(wav_)
        if preview:
            result_convert_video = subprocess.Popen(mp4_, **sub_params)
            output, errors = result_convert_video.communicate()
            time.sleep(0.5)
            result_convert_audio = subprocess.Popen(wav_, **sub_params)
            output, errors = result_convert_audio.communicate()
            time.sleep(0.5)
            if result_convert_audio.returncode in [1, 2] or not os.path.exists(
                audio_wav
            ):
                raise OperationFailedError(
                    f"Error can't create the preview file:\n{errors.decode('utf-8')}"
                )
        else:
            logger.info("Process audio...")
            result_convert_audio = subprocess.Popen(wav_, **sub_params)
            output, errors = result_convert_audio.communicate()
            time.sleep(1)
            if result_convert_audio.returncode in [1, 2] or not os.path.exists(
                audio_wav
            ):
                raise OperationFailedError(f"Error can't download the audio:\n{errors.decode('utf-8')}")
            logger.info("Process video...")
            result_convert_video = subprocess.Popen(mp4_, **sub_params)
            output, errors = result_convert_video.communicate()
            time.sleep(1)
            if result_convert_video.returncode in [1, 2] or not os.path.exists(
                OutputFile
            ):
                raise OperationFailedError(f"Error can't download the video:\n{errors.decode('utf-8')}")


def old_audio_video_preprocessor(preview, video, OutputFile, audio_wav):
    previous_files_to_remove = [OutputFile, "audio.webm", audio_wav]
    remove_files(previous_files_to_remove)

    if os.path.exists(video):
        if preview:
            logger.warning(
                "Creating a preview video of 10 seconds, "
                "to disable this option, go to advanced "
                "settings and turn off preview."
            )
            command = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4'
            result_convert_video = subprocess.run(
                command, capture_output=True, text=True, shell=True
            )
        else:
            # Check if the file ends with ".mp4" extension
            if video.endswith(".mp4"):
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
                result_convert_video = {}
                result_convert_video = subprocess.run(
                    "echo Video copied",
                    capture_output=True,
                    text=True,
                    shell=True,
                )
            else:
                logger.warning(
                    "File does not have the '.mp4' extension. Converting video."
                )
                command = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4'
                result_convert_video = subprocess.run(
                    command, capture_output=True, text=True, shell=True
                )

        if result_convert_video.returncode in [1, 2]:
            raise OperationFailedError("Error can't convert the video")

        for i in range(120):
            time.sleep(1)
            logger.info("Process video...")
            if os.path.exists(OutputFile):
                time.sleep(1)
                command = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
                result_convert_audio = subprocess.run(
                    command, capture_output=True, text=True, shell=True
                )
                time.sleep(1)
                break
            if i == 119:
                # if not os.path.exists(OutputFile):
                raise OperationFailedError("Error processing video")

        if result_convert_audio.returncode in [1, 2]:
            raise OperationFailedError(
                f"Error can't create the audio file: {result_convert_audio.stderr}"
            )

        for i in range(120):
            time.sleep(1)
            logger.info("Process audio...")
            if os.path.exists(audio_wav):
                break
            if i == 119:
                raise OperationFailedError("Error can't create the audio file")

    else:
        video = video.strip()
        if preview:
            logger.warning(
                "Creating a preview from the link, 10 "
                "seconds to disable this option, go to "
                "advanced settings and turn off preview."
            )
            # https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
            result_convert_video = subprocess.run(
                mp4_, capture_output=True, text=True, shell=True
            )
            result_convert_audio = subprocess.run(
                wav_, capture_output=True, text=True, shell=True
            )
            if result_convert_audio.returncode in [1, 2]:
                raise OperationFailedError("Error can't download a preview")
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f"python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}"

            result_convert_audio = subprocess.run(
                wav_, capture_output=True, text=True, shell=True
            )

            if result_convert_audio.returncode in [1, 2]:
                raise OperationFailedError("Error can't download the audio")

            for i in range(120):
                time.sleep(1)
                logger.info("Process audio...")
                if os.path.exists(audio_wav) and not os.path.exists(
                    "audio.webm"
                ):
                    time.sleep(1)
                    result_convert_video = subprocess.run(
                        mp4_, capture_output=True, text=True, shell=True
                    )
                    break
                if i == 119:
                    raise OperationFailedError("Error downloading the audio")

            if result_convert_video.returncode in [1, 2]:
                raise OperationFailedError("Error can't download the video")
