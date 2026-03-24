"""Integration tests for MiniMax provider (require MINIMAX_API_KEY).

Run with: MINIMAX_API_KEY=<key> python -m pytest tests/test_minimax_integration.py -v
"""
import importlib
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_MSG = "MINIMAX_API_KEY not set; skipping integration tests"


def _real_import(name):
    """Import a real module, bypassing any conftest mocks."""
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_MSG)
class TestMiniMaxLLMIntegration(unittest.TestCase):
    """Integration tests for MiniMax LLM translation via OpenAI-compatible API."""

    def _create_client(self):
        openai = _real_import("openai")
        return openai.OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )

    @staticmethod
    def _clean_response(text):
        """Strip thinking tags and markdown code fences from LLM response."""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
        return text.strip()

    def test_minimax_chat_completion(self):
        """Test basic chat completion with MiniMax M2.5."""
        client = self._create_client()
        response = client.chat.completions.create(
            model="MiniMax-M2.5",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Machine translation. Output JSON with key translated_text.",
                },
                {
                    "role": "user",
                    "content": "Translate to Spanish: Hello",
                },
            ],
        )
        result = response.choices[0].message.content
        result = self._clean_response(result)
        parsed = json.loads(result)
        self.assertIn("translated_text", parsed)
        self.assertTrue(len(parsed["translated_text"]) > 0)

    def test_minimax_m27_chat_completion(self):
        """Test chat completion with MiniMax M2.7."""
        client = self._create_client()
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Translate. Output JSON with key translated_text.",
                },
                {
                    "role": "user",
                    "content": "Translate to French: Good morning",
                },
            ],
        )
        result = response.choices[0].message.content
        result = self._clean_response(result)
        parsed = json.loads(result)
        self.assertIn("translated_text", parsed)

    def test_minimax_batch_translation_format(self):
        """Test batch translation format with MiniMax (conversation JSON)."""
        client = self._create_client()
        batch = {"conversation": [{"A1": "Hello"}, {"B1": "Goodbye"}]}
        response = client.chat.completions.create(
            model="MiniMax-M2.5",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Machine translation. Output translated_conversation JSON with list of 2 items.",
                },
                {
                    "role": "user",
                    "content": f"Translate each text value to Spanish:\n{batch}",
                },
            ],
        )
        result = response.choices[0].message.content
        result = self._clean_response(result)
        parsed = json.loads(result)
        values = list(parsed.values())
        found_list = False
        for v in values:
            if isinstance(v, list) and len(v) >= 2:
                found_list = True
                break
        self.assertTrue(found_list, f"Expected list in response: {parsed}")


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_MSG)
class TestMiniMaxTTSIntegration(unittest.TestCase):
    """Integration tests for MiniMax TTS API."""

    def _get_requests(self):
        return _real_import("requests")

    def test_minimax_tts_api_call(self):
        """Test MiniMax TTS with Wise_Woman voice."""
        requests = self._get_requests()
        response = requests.post(
            "https://api.minimax.io/v1/t2a_v2",
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "speech-2.8-hd",
                "text": "Hello, this is a test.",
                "voice_setting": {"voice_id": "Wise_Woman", "speed": 1.0},
                "audio_setting": {"format": "mp3"},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("audio", data["data"])
        audio_bytes = bytes.fromhex(data["data"]["audio"])
        self.assertGreater(len(audio_bytes), 100)

    def test_minimax_tts_different_voice(self):
        """Test TTS with English_Graceful_Lady voice."""
        requests = self._get_requests()
        response = requests.post(
            "https://api.minimax.io/v1/t2a_v2",
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "speech-2.8-hd",
                "text": "Testing voice selection.",
                "voice_setting": {"voice_id": "English_Graceful_Lady", "speed": 1.0},
                "audio_setting": {"format": "mp3"},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

    def test_minimax_tts_audio_is_valid_mp3(self):
        """Verify the returned audio bytes are valid MP3."""
        requests = self._get_requests()
        response = requests.post(
            "https://api.minimax.io/v1/t2a_v2",
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "speech-2.8-hd",
                "text": "Valid audio check.",
                "voice_setting": {"voice_id": "Deep_Voice_Man", "speed": 1.0},
                "audio_setting": {"format": "mp3"},
            },
            timeout=30,
        )
        audio_bytes = bytes.fromhex(response.json()["data"]["audio"])
        self.assertTrue(
            audio_bytes[:3] == b"ID3" or audio_bytes[0] == 0xFF,
            "Audio should be valid MP3",
        )


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_MSG)
class TestThinkTagStripping(unittest.TestCase):
    """Test that thinking tags are properly stripped from responses."""

    def test_strip_think_tags(self):
        raw = '<think>Some reasoning</think>\n{"translated_text": "Hola"}'
        cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
        parsed = json.loads(cleaned)
        self.assertEqual(parsed["translated_text"], "Hola")

    def test_no_think_tags_passthrough(self):
        raw = '{"translated_text": "Hola"}'
        cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
        parsed = json.loads(cleaned)
        self.assertEqual(parsed["translated_text"], "Hola")

    def test_multiline_think_tags(self):
        raw = '<think>\nLine 1\nLine 2\n</think>\n\n{"text": "result"}'
        cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
        parsed = json.loads(cleaned)
        self.assertEqual(parsed["text"], "result")


if __name__ == "__main__":
    unittest.main()
