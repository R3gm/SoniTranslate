"""Pytest conftest - mock heavy dependencies before importing project modules."""
import sys
import types
from unittest.mock import MagicMock

# Mock heavy dependencies that are not needed for testing MiniMax integration
MOCK_MODULES = [
    "edge_tts",
    "gtts",
    "whisperx",
    "torch",
    "librosa",
    "soundfile",
    "pydub",
    "transformers",
    "optimum",
    "optimum.bettertransformer",
    "TTS",
    "TTS.api",
    "piper",
    "piper.download",
    "openvoice",
    "openvoice.api",
    "openvoice.se_extractor",
    "gradio",
    "tiktoken",
    "deep_translator",
    "numpy",
    "openai",
    "rarfile",
    "IPython",
    "IPython.utils",
    "IPython.utils.capture",
    "requests",
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Make tqdm transparent (pass-through) so for loops work in tests
class _FakeTqdm:
    """Minimal tqdm mock that passes through iterables and handles progress bars."""
    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable
    def __iter__(self):
        if self._iterable is not None:
            return iter(self._iterable)
        return iter([])
    def update(self, n=1):
        pass
    def close(self):
        pass

tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = _FakeTqdm
sys.modules["tqdm"] = tqdm_module
