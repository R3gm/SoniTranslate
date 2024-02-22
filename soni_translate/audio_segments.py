from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger


def create_translated_audio(
    result_diarize, audio_files, final_file, concat=False
):
    total_duration = result_diarize["segments"][-1]["end"]  # in seconds

    if concat:
        """
        file .\audio\1.ogg
        file .\audio\2.ogg
        file .\audio\3.ogg
        file .\audio\4.ogg
        ...
        """

        # Write the file paths to list.txt
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                if i == len(audio_files) - 1:  # Check if it's the last item
                    file.write(f"file {audio_file}")
                else:
                    file.write(f"file {audio_file}\n")

        # command = f"ffmpeg -f concat -safe 0 -i list.txt {final_file}"
        command = (
            f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        )
        run_command(command)

    else:
        # silent audio with total_duration
        combined_audio = AudioSegment.silent(
            duration=int(total_duration * 1000)
        )
        logger.info(
            f"Audio duration: {total_duration // 60} "
            f"minutes and {int(total_duration % 60)} seconds"
        )

        for line, audio_file in tqdm(
            zip(result_diarize["segments"], audio_files)
        ):
            start = float(line["start"])

            # Overlay each audio at the corresponding time
            try:
                audio = AudioSegment.from_file(audio_file)
                # audio_a = audio.speedup(playback_speed=1.5)
                start_time = start * 1000  # to ms
                combined_audio = combined_audio.overlay(
                    audio, position=start_time
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")

        # combined audio as a file
        combined_audio.export(
            final_file, format="wav"
        )  # best than ogg, change if the audio is anomalous
