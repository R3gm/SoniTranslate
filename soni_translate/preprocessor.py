
from .utils import print_tree_directory, remove_files, select_zip_and_rar_files, download_list, manual_download, upload_model_list
import os, shutil, subprocess

def audio_video_preprocessor(preview, video, OutputFile, audio_wav):

    previous_files_to_remove = [OutputFile, "audio.webm", audio_wav]
    remove_files(previous_files_to_remove)

    if os.path.exists(video):
        if preview:
            print('Creating a preview video of 10 seconds, to disable this option, go to advanced settings and turn off preview.')
            command = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4'
            result_convert_video = subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            # Check if the file ends with ".mp4" extension
            if video.endswith(".mp4"):
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
                result_convert_video = {}
                result_convert_video = subprocess.run("echo Video copied", capture_output=True, text=True, shell=True)
            else:
                print("File does not have the '.mp4' extension. Converting video.")
                command = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4'
                result_convert_video = subprocess.run(command, capture_output=True, text=True, shell=True)

        if result_convert_video.returncode in [1, 2]:
            print("Error can't convert the video")
            return

        for i in range (120):
            time.sleep(1)
            print('Process video...')
            if os.path.exists(OutputFile):
                time.sleep(1)
                command = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
                result_convert_audio = subprocess.run(command, capture_output=True, text=True, shell=True)
                time.sleep(1)
                break
            if i == 119:
                # if not os.path.exists(OutputFile):
                print('Error processing video')
                return

        if result_convert_audio.returncode in [1, 2]:
            print(f"Error can't create the audio file: {result_convert_audio.stderr}")
            return

        for i in range (120):
            time.sleep(1)
            print('process audio...')
            if os.path.exists(audio_wav):
                break
            if i == 119:
                print("Error can't create the audio file")
                return

    else:
        video = video.strip()
        if preview:
            print('Creating a preview from the link, 10 seconds to disable this option, go to advanced settings and turn off preview.')
            #https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
            result_convert_video = subprocess.run(mp4_, capture_output=True, text=True, shell=True)
            result_convert_audio = subprocess.run(wav_, capture_output=True, text=True, shell=True)
            if result_convert_audio.returncode in [1, 2]:
                print("Error can't download a preview")
                return
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f'python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}'

            result_convert_audio = subprocess.run(wav_, capture_output=True, text=True, shell=True)

            if result_convert_audio.returncode in [1, 2]:
                print("Error can't download the audio")
                return

            for i in range (120):
                time.sleep(1)
                print('process audio...')
                if os.path.exists(audio_wav) and not os.path.exists('audio.webm'):
                    time.sleep(1)
                    result_convert_video = subprocess.run(mp4_, capture_output=True, text=True, shell=True)
                    break
                if i == 119:
                    print('Error downloading the audio')
                    return

            if result_convert_video.returncode in [1, 2]:
                print("Error can't download the video")
                return
        