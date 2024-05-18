from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np


class Mixer:
    def __init__(self):
        self.parts = []

    def __len__(self):
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        return int(1000.0 * frame_count / seg.frame_rate)

    def overlay(self, sound, position=0):
        self.parts.append((position, sound))
        return self

    def _sync(self):
        positions, segs = zip(*self.parts)

        frame_rate = segs[0].frame_rate
        array_type = segs[0].array_type # noqa

        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)
        return list(zip(offsets, segs))

    def append(self, sound):
        self.overlay(sound, position=len(self))

    def to_audio_segment(self):
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels

        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        sample_count = int(frame_count * seg.channels)

        output = np.zeros(sample_count, dtype="int32")
        for offset, seg in parts:
            sample_offset = offset * channels
            samples = np.frombuffer(seg.get_array_of_samples(), dtype="int32")
            samples = np.int16(samples/np.max(np.abs(samples)) * 32767)
            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}).normalize(headroom=0.0)


def create_translated_audio(
    result_diarize, audio_files, final_file, concat=False, avoid_overlap=False,
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
        base_audio = AudioSegment.silent(
            duration=int(total_duration * 1000), frame_rate=41000
        )
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        logger.debug(
            f"Audio duration: {total_duration // 60} "
            f"minutes and {int(total_duration % 60)} seconds"
        )

        last_end_time = 0
        previous_speaker = ""
        for line, audio_file in tqdm(
            zip(result_diarize["segments"], audio_files)
        ):
            start = float(line["start"])

            # Overlay each audio at the corresponding time
            try:
                audio = AudioSegment.from_file(audio_file)
                # audio_a = audio.speedup(playback_speed=1.5)

                if avoid_overlap:
                    speaker = line["speaker"]
                    if (last_end_time - 0.500) > start:
                        overlap_time = last_end_time - start
                        if previous_speaker and previous_speaker != speaker:
                            start = (last_end_time - 0.500)
                        else:
                            start = (last_end_time - 0.200)
                        if overlap_time > 2.5:
                            start = start - 0.3
                        logger.info(
                              f"Avoid overlap for {str(audio_file)} "
                              f"with {str(start)}"
                        )

                    previous_speaker = speaker

                    duration_tts_seconds = len(audio) / 1000.0  # to sec
                    last_end_time = (start + duration_tts_seconds)

                start_time = start * 1000  # to ms
                combined_audio = combined_audio.overlay(
                    audio, position=start_time
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")

        # combined audio as a file
        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(
            final_file, format="wav"
        )  # best than ogg, change if the audio is anomalous
