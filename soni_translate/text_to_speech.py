from gtts import gTTS
import edge_tts
import asyncio
import nest_asyncio

def make_voice(tts_text, tts_voice, filename):
    try:
      nest_asyncio.apply()
      asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except 1:
      tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
      tts.save(filename)
      print('USE GTTS')
    except 2:
      tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
      tts.save(filename)
      print('REPLACE AUDIO GTTS')

def make_voice_gradio(tts_text, tts_voice, filename):
    print(tts_text, filename)
    try:
      asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except 1:
      tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
      tts.save(filename)
      print('USE GTTS')
    except 2:
      tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
      tts.save(filename)
      print('REPLACE AUDIO GTTS')
