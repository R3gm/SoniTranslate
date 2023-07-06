from gtts import gTTS
import edge_tts
import asyncio
import nest_asyncio

def make_voice(tts_text, tts_voice, filename,language):
    #print(tts_text, filename)
    try:
      nest_asyncio.apply()
      asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except:
      try:
          tts = gTTS(tts_text, lang=language)
          tts.save(filename)
          print(f'No audio was received. Please change the tts voice for {tts_voice}. USING gTTS.')
      except:
        tts = gTTS('a', lang=language)
        tts.save(filename)
        print('Error: Audio will be replaced.')

def make_voice_gradio(tts_text, tts_voice, filename, language):
    print(tts_text, filename)
    try:
      asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
    except:
      try:
        tts = gTTS(tts_text, lang=language)
        tts.save(filename)
        print(f'No audio was received. Please change the tts voice for {tts_voice}. USING gTTS.')
      except:
        tts = gTTS('a', lang=language)
        tts.save(filename)
        print('Error: Audio will be replaced.')
