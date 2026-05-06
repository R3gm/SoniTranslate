"""Unit tests for MiniMax translation integration."""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import conftest to mock heavy deps
import tests.conftest  # noqa: F401


class TestMiniMaxClientCreation(unittest.TestCase):
    """Test _create_minimax_client helper."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_create_client_with_valid_key(self):
        """Client is created with correct base_url and api_key."""
        from soni_translate.translate_segments import _create_minimax_client

        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = _create_minimax_client()
            mock_cls.assert_called_once_with(
                api_key="test-key-123",
                base_url="https://api.minimax.io/v1",
            )
            self.assertIsNotNone(client)

    def test_create_client_missing_key_raises(self):
        """ValueError raised when MINIMAX_API_KEY is not set."""
        from soni_translate.translate_segments import _create_minimax_client

        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_client()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))


class TestTranslationProcessOptions(unittest.TestCase):
    """Test that MiniMax models are in the translation options lists."""

    def test_minimax_in_translation_options(self):
        from soni_translate.translate_segments import TRANSLATION_PROCESS_OPTIONS

        self.assertIn("MiniMax-M2.5", TRANSLATION_PROCESS_OPTIONS)
        self.assertIn("MiniMax-M2.5_batch", TRANSLATION_PROCESS_OPTIONS)
        self.assertIn("MiniMax-M2.7", TRANSLATION_PROCESS_OPTIONS)
        self.assertIn("MiniMax-M2.7_batch", TRANSLATION_PROCESS_OPTIONS)

    def test_minimax_in_docs_translation_options(self):
        from soni_translate.translate_segments import DOCS_TRANSLATION_PROCESS_OPTIONS

        self.assertIn("MiniMax-M2.5", DOCS_TRANSLATION_PROCESS_OPTIONS)
        self.assertIn("MiniMax-M2.7", DOCS_TRANSLATION_PROCESS_OPTIONS)

    def test_disable_still_last(self):
        from soni_translate.translate_segments import TRANSLATION_PROCESS_OPTIONS

        self.assertEqual(
            TRANSLATION_PROCESS_OPTIONS[-1], "disable_translation"
        )

    def test_google_translator_still_first(self):
        from soni_translate.translate_segments import TRANSLATION_PROCESS_OPTIONS

        self.assertEqual(
            TRANSLATION_PROCESS_OPTIONS[0], "google_translator_batch"
        )

    def test_all_options_present(self):
        """Verify all expected options exist."""
        from soni_translate.translate_segments import TRANSLATION_PROCESS_OPTIONS

        expected = [
            "google_translator_batch",
            "google_translator",
            "gpt-3.5-turbo-0125_batch",
            "gpt-3.5-turbo-0125",
            "gpt-4-turbo-preview_batch",
            "gpt-4-turbo-preview",
            "MiniMax-M2.5_batch",
            "MiniMax-M2.5",
            "MiniMax-M2.7_batch",
            "MiniMax-M2.7",
            "disable_translation",
        ]
        self.assertEqual(TRANSLATION_PROCESS_OPTIONS, expected)


class TestTranslateTextRouting(unittest.TestCase):
    """Test that translate_text correctly routes to MiniMax."""

    @patch("soni_translate.translate_segments._create_minimax_client")
    @patch("soni_translate.translate_segments.gpt_sequential")
    def test_minimax_m25_routes_to_sequential(self, mock_seq, mock_client):
        from soni_translate.translate_segments import translate_text

        mock_client.return_value = MagicMock()
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]
        mock_seq.return_value = segments

        translate_text(segments, "es", translation_process="MiniMax-M2.5")

        mock_seq.assert_called_once()
        args, kwargs = mock_seq.call_args
        self.assertEqual(args[1], "MiniMax-M2.5")
        self.assertIn("client", kwargs)

    @patch("soni_translate.translate_segments._create_minimax_client")
    @patch("soni_translate.translate_segments.gpt_sequential")
    def test_minimax_m27_routes_to_sequential(self, mock_seq, mock_client):
        from soni_translate.translate_segments import translate_text

        mock_client.return_value = MagicMock()
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]
        mock_seq.return_value = segments

        translate_text(segments, "es", translation_process="MiniMax-M2.7")

        mock_seq.assert_called_once()
        args, kwargs = mock_seq.call_args
        self.assertEqual(args[1], "MiniMax-M2.7")

    @patch("soni_translate.translate_segments._create_minimax_client")
    @patch("soni_translate.translate_segments.gpt_batch")
    def test_minimax_m25_batch_routes_to_batch(self, mock_batch, mock_client):
        from soni_translate.translate_segments import translate_text

        mock_client.return_value = MagicMock()
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]
        mock_batch.return_value = segments

        translate_text(
            segments, "es", translation_process="MiniMax-M2.5_batch"
        )

        mock_batch.assert_called_once()
        args, kwargs = mock_batch.call_args
        self.assertEqual(args[1], "MiniMax-M2.5")  # _batch stripped
        self.assertIn("client", kwargs)

    @patch("soni_translate.translate_segments._create_minimax_client")
    @patch("soni_translate.translate_segments.gpt_batch")
    def test_minimax_m27_batch_routes_to_batch(self, mock_batch, mock_client):
        from soni_translate.translate_segments import translate_text

        mock_client.return_value = MagicMock()
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]
        mock_batch.return_value = segments

        translate_text(
            segments, "es", translation_process="MiniMax-M2.7_batch"
        )

        mock_batch.assert_called_once()
        args, kwargs = mock_batch.call_args
        self.assertEqual(args[1], "MiniMax-M2.7")

    @patch("soni_translate.translate_segments.translate_batch")
    def test_google_translator_still_works(self, mock_tb):
        from soni_translate.translate_segments import translate_text

        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]
        mock_tb.return_value = segments

        translate_text(segments, "es", translation_process="google_translator_batch")
        mock_tb.assert_called_once()


class TestGptSequentialWithClient(unittest.TestCase):
    """Test that gpt_sequential accepts an external client."""

    @patch("soni_translate.translate_segments.call_gpt_translate")
    def test_custom_client_passed_through(self, mock_call):
        from soni_translate.translate_segments import gpt_sequential

        mock_client = MagicMock()
        mock_call.return_value = "Hola"
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]

        result = gpt_sequential(segments, "MiniMax-M2.5", "es", client=mock_client)

        call_args = mock_call.call_args
        self.assertEqual(call_args[0][0], mock_client)
        self.assertEqual(call_args[0][1], "MiniMax-M2.5")
        self.assertEqual(result[0]["text"], "Hola")

    @patch("soni_translate.translate_segments.call_gpt_translate")
    def test_default_client_when_none(self, mock_call):
        from soni_translate.translate_segments import gpt_sequential

        mock_call.return_value = "Hola"
        segments = [{"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}]

        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            gpt_sequential(segments, "gpt-3.5-turbo-0125", "es")
            mock_cls.assert_called_once()


class TestGptBatchWithClient(unittest.TestCase):
    """Test that gpt_batch accepts an external client."""

    @patch("soni_translate.translate_segments.call_gpt_translate")
    def test_custom_client_passed_through(self, mock_call):
        from soni_translate.translate_segments import gpt_batch

        # Mock tiktoken
        import tiktoken
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        mock_client = MagicMock()
        mock_call.return_value = [{"A1": "Hola"}]
        segments = [
            {"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"},
        ]

        gpt_batch(segments, "MiniMax-M2.5", "es", client=mock_client)

        call_args = mock_call.call_args
        self.assertEqual(call_args[0][0], mock_client)
        self.assertEqual(call_args[0][1], "MiniMax-M2.5")


class TestCallGptTranslate(unittest.TestCase):
    """Test call_gpt_translate with MiniMax-like responses."""

    def test_sequential_json_response(self):
        from soni_translate.translate_segments import call_gpt_translate

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"translated_text": "Hola mundo"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_gpt_translate(
            mock_client,
            "MiniMax-M2.5",
            "Translate JSON output",
            "Translate: Hello world",
        )
        self.assertEqual(result, "Hola mundo")

    def test_batch_json_response(self):
        from soni_translate.translate_segments import call_gpt_translate

        mock_client = MagicMock()
        original_text = {
            "conversation": [{"A1": "Hello"}, {"B1": "How are you?"}]
        }
        response_json = json.dumps({
            "translated_conversation": [
                {"A1": "Hola"},
                {"B1": "Como estas?"},
            ]
        })
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=response_json))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_gpt_translate(
            mock_client,
            "MiniMax-M2.7",
            "Translate conversation",
            "Translate this",
            original_text=original_text,
            batch_lines=2,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["A1"], "Hola")

    def test_json_mode_used(self):
        """Verify response_format=json_object is sent."""
        from soni_translate.translate_segments import call_gpt_translate

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"text": "test"}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        call_gpt_translate(mock_client, "MiniMax-M2.5", "sys", "user")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(
            call_kwargs["response_format"], {"type": "json_object"}
        )


if __name__ == "__main__":
    unittest.main()
