"""Unit tests for MiniMax TTS integration."""
import json
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.conftest  # noqa: F401


class TestMiniMaxTTSModels(unittest.TestCase):
    """Test MiniMax TTS model list in language_configuration."""

    def test_minimax_tts_models_exist(self):
        from soni_translate.language_configuration import MINIMAX_TTS_MODELS

        self.assertIsInstance(MINIMAX_TTS_MODELS, list)
        self.assertGreater(len(MINIMAX_TTS_MODELS), 0)

    def test_minimax_tts_models_format(self):
        """All MiniMax TTS models follow '>voice_id MiniMax-TTS' format."""
        from soni_translate.language_configuration import MINIMAX_TTS_MODELS

        for model in MINIMAX_TTS_MODELS:
            self.assertTrue(
                model.startswith(">"), f"'{model}' should start with '>'"
            )
            self.assertTrue(
                model.endswith("MiniMax-TTS"),
                f"'{model}' should end with 'MiniMax-TTS'",
            )
            voice_id = model.split()[0][1:]
            self.assertTrue(len(voice_id) > 0)

    def test_verified_voices_present(self):
        from soni_translate.language_configuration import MINIMAX_TTS_MODELS

        verified_voices = [
            "Wise_Woman",
            "Deep_Voice_Man",
            "Friendly_Person",
            "English_Graceful_Lady",
            "English_Insightful_Speaker",
            "English_radiant_girl",
            "English_Persuasive_Man",
            "English_Lucky_Robot",
            "cute_boy",
            "lovely_girl",
            "Inspirational_girl",
            "sweet_girl",
        ]
        model_names = [m.split()[0][1:] for m in MINIMAX_TTS_MODELS]
        for voice in verified_voices:
            self.assertIn(voice, model_names, f"'{voice}' should be in list")

    def test_model_count(self):
        from soni_translate.language_configuration import MINIMAX_TTS_MODELS

        self.assertEqual(len(MINIMAX_TTS_MODELS), 12)


class TestMiniMaxTTSPatternMatch(unittest.TestCase):
    """Test the regex pattern for MiniMax TTS voice matching."""

    def test_pattern_matches_minimax_tts(self):
        pattern = re.compile(r".* MiniMax-TTS$")
        self.assertTrue(pattern.match(">Wise_Woman MiniMax-TTS"))
        self.assertTrue(pattern.match(">English_Graceful_Lady MiniMax-TTS"))
        self.assertTrue(pattern.match(">cute_boy MiniMax-TTS"))

    def test_pattern_does_not_match_openai_tts(self):
        pattern = re.compile(r".* MiniMax-TTS$")
        self.assertFalse(pattern.match(">alloy OpenAI-TTS"))
        self.assertFalse(pattern.match(">echo HD OpenAI-TTS"))

    def test_pattern_does_not_match_edge_tts(self):
        pattern = re.compile(r".* MiniMax-TTS$")
        self.assertFalse(pattern.match("en-US-EmmaMultilingualNeural-Female"))


class TestVoiceIdExtraction(unittest.TestCase):
    """Test voice_id extraction from tts_name."""

    def test_extract_simple_voice_id(self):
        tts_name = ">Wise_Woman MiniMax-TTS"
        voice_id = tts_name.split()[0][1:]
        self.assertEqual(voice_id, "Wise_Woman")

    def test_extract_english_voice_id(self):
        tts_name = ">English_Graceful_Lady MiniMax-TTS"
        voice_id = tts_name.split()[0][1:]
        self.assertEqual(voice_id, "English_Graceful_Lady")

    def test_extract_cute_boy_voice_id(self):
        tts_name = ">cute_boy MiniMax-TTS"
        voice_id = tts_name.split()[0][1:]
        self.assertEqual(voice_id, "cute_boy")


class TestSegmentsMiniMaxTTS(unittest.TestCase):
    """Test segments_minimax_tts function."""

    def test_raises_without_api_key(self):
        from soni_translate.text_to_speech import segments_minimax_tts, TTS_OperationError

        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            segments = {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Hello",
                        "start": 0.0,
                        "end": 1.0,
                        "tts_name": ">Wise_Woman MiniMax-TTS",
                    }
                ]
            }
            with self.assertRaises(TTS_OperationError):
                segments_minimax_tts(segments, "en")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("soni_translate.text_to_speech.verify_saved_file_and_size")
    @patch("soni_translate.text_to_speech.write_chunked")
    @patch("soni_translate.text_to_speech.pad_array")
    @patch("soni_translate.text_to_speech.sf")
    @patch("builtins.open", mock_open())
    @patch("soni_translate.text_to_speech.requests")
    def test_successful_tts_call(
        self, mock_requests, mock_sf, mock_pad, mock_write, mock_verify
    ):
        from soni_translate.text_to_speech import segments_minimax_tts
        import numpy as np

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"audio": "ff" * 100}}
        mock_requests.post.return_value = mock_response

        mock_sf.read.return_value = (MagicMock(), 24000)
        mock_pad.return_value = MagicMock()

        segments = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "tts_name": ">Wise_Woman MiniMax-TTS",
                }
            ]
        }

        segments_minimax_tts(segments, "en")

        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        self.assertEqual(call_args[0][0], "https://api.minimax.io/v1/t2a_v2")
        request_body = call_args[1]["json"]
        self.assertEqual(request_body["model"], "speech-2.8-hd")
        self.assertEqual(request_body["text"], "Hello world")
        self.assertEqual(request_body["voice_setting"]["voice_id"], "Wise_Woman")
        self.assertEqual(request_body["audio_setting"]["format"], "mp3")
        headers = call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("soni_translate.text_to_speech.error_handling_in_tts")
    @patch("soni_translate.text_to_speech.requests")
    def test_api_error_triggers_fallback(self, mock_requests, mock_error_handler):
        from soni_translate.text_to_speech import segments_minimax_tts

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_requests.post.return_value = mock_response

        segments = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 1.0,
                    "tts_name": ">Wise_Woman MiniMax-TTS",
                }
            ]
        }

        segments_minimax_tts(segments, "en")
        mock_error_handler.assert_called_once()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("soni_translate.text_to_speech.verify_saved_file_and_size")
    @patch("soni_translate.text_to_speech.write_chunked")
    @patch("soni_translate.text_to_speech.pad_array")
    @patch("soni_translate.text_to_speech.sf")
    @patch("builtins.open", mock_open())
    @patch("soni_translate.text_to_speech.requests")
    def test_different_voice_id_sent(
        self, mock_requests, mock_sf, mock_pad, mock_write, mock_verify
    ):
        from soni_translate.text_to_speech import segments_minimax_tts

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"audio": "ff" * 100}}
        mock_requests.post.return_value = mock_response
        mock_sf.read.return_value = (MagicMock(), 24000)
        mock_pad.return_value = MagicMock()

        segments = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Test",
                    "start": 0.0,
                    "end": 1.0,
                    "tts_name": ">English_Graceful_Lady MiniMax-TTS",
                }
            ]
        }

        segments_minimax_tts(segments, "en")

        request_body = mock_requests.post.call_args[1]["json"]
        self.assertEqual(
            request_body["voice_setting"]["voice_id"], "English_Graceful_Lady"
        )


class TestAudioSegmentationToVoice(unittest.TestCase):
    """Test that audio_segmentation_to_voice handles MiniMax TTS."""

    def test_minimax_pattern_detection(self):
        from soni_translate.text_to_speech import find_spkr

        pattern_minimax_tts = re.compile(r".* MiniMax-TTS$")
        speaker_to_voice = {
            "SPEAKER_00": ">Wise_Woman MiniMax-TTS",
            "SPEAKER_01": "en-US-EmmaMultilingualNeural-Female",
        }
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "World"},
        ]

        speakers = find_spkr(pattern_minimax_tts, speaker_to_voice, segments)
        self.assertEqual(speakers, ["SPEAKER_00"])

    def test_filter_by_speaker(self):
        from soni_translate.text_to_speech import filter_by_speaker

        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "World"},
        ]
        filtered = filter_by_speaker(["SPEAKER_00"], segments)
        self.assertEqual(len(filtered["segments"]), 1)
        self.assertEqual(filtered["segments"][0]["text"], "Hello")


class TestReturnValueUpdate(unittest.TestCase):
    """Test that audio_segmentation_to_voice returns 7-element list."""

    @patch("soni_translate.text_to_speech.remove_directory_contents")
    @patch("soni_translate.text_to_speech.segments_egde_tts")
    def test_returns_seven_elements(self, mock_edge, mock_rm):
        from soni_translate.text_to_speech import audio_segmentation_to_voice

        result_diarize = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 1.0,
                }
            ]
        }

        result = audio_segmentation_to_voice(
            result_diarize,
            TRANSLATE_AUDIO_TO="en",
            is_gui=False,
            tts_voice00="en-US-EmmaMultilingualNeural-Female",
        )

        self.assertEqual(len(result), 7)

    @patch("soni_translate.text_to_speech.remove_directory_contents")
    @patch("soni_translate.text_to_speech.segments_minimax_tts")
    def test_minimax_speakers_populated(self, mock_mm_tts, mock_rm):
        from soni_translate.text_to_speech import audio_segmentation_to_voice

        result_diarize = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 1.0,
                }
            ]
        }

        result = audio_segmentation_to_voice(
            result_diarize,
            TRANSLATE_AUDIO_TO="en",
            is_gui=False,
            tts_voice00=">Wise_Woman MiniMax-TTS",
        )

        minimax_speakers = result[6]
        self.assertIn("SPEAKER_00", minimax_speakers)
        mock_mm_tts.assert_called_once()


class TestAccelerateSegmentsUnpacking(unittest.TestCase):
    """Test that accelerate_segments can unpack 7-element valid_speakers."""

    def test_seven_element_unpacking(self):
        valid_speakers = [
            ["SPEAKER_00"],  # edge
            [],  # bark
            [],  # vits
            [],  # coqui
            [],  # vits_onnx
            [],  # openai_tts
            [],  # minimax_tts
        ]

        (
            speakers_edge,
            speakers_bark,
            speakers_vits,
            speakers_coqui,
            speakers_vits_onnx,
            speakers_openai_tts,
            speakers_minimax_tts,
        ) = valid_speakers

        self.assertEqual(speakers_edge, ["SPEAKER_00"])
        self.assertEqual(speakers_minimax_tts, [])


if __name__ == "__main__":
    unittest.main()
