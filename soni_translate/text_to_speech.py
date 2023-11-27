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
          print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
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
        print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
      except:
        tts = gTTS('a', lang=language)
        tts.save(filename)
        print('Error: Audio will be replaced.')





def audio_segmentation_to_voice(
    result_diarize, TRANSLATE_AUDIO_TO, max_accelerate_audio, 
    tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05
    ):
    
    audio_files = []
    speakers_list = []

    # Mapping speakers to voice variables
    speaker_to_voice = {
        'SPEAKER_00': tts_voice00,
        'SPEAKER_01': tts_voice01,
        'SPEAKER_02': tts_voice02,
        'SPEAKER_03': tts_voice03,
        'SPEAKER_04': tts_voice04,
        'SPEAKER_05': tts_voice05
    }

    for segment in tqdm(result_diarize['segments']):

        text = segment['text']
        start = segment['start']
        end = segment['end']

        try:
            speaker = segment['speaker']
        except KeyError:
            segment['speaker'] = "SPEAKER_99"
            speaker = segment['speaker']
            print(f"NO SPEAKER DETECT IN SEGMENT: TTS auxiliary will be used in the segment time {segment['start'], segment['text']}")

        # make the tts audio
        filename = f"audio/{start}.ogg"

        if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
            make_voice_gradio(text, speaker_to_voice[speaker], filename, TRANSLATE_AUDIO_TO)
        elif speaker == "SPEAKER_99":
            try:
                tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Using GTTS')
            except:
                tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Error: Audio will be replaced.')

        # duration
        duration_true = end - start
        duration_tts = librosa.get_duration(filename=filename)

        # porcentaje
        porcentaje = duration_tts / duration_true

        if porcentaje > max_accelerate_audio:
            porcentaje = max_accelerate_audio
        elif porcentaje <= 1.2 and porcentaje >= 0.8:
            porcentaje = 1.0
        elif porcentaje <= 0.79:
            porcentaje = 0.8

        # Smoth and round
        porcentaje = round(porcentaje+0.0, 1)

        # apply aceleration or opposite to the audio file in audio2 folder
        os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")

        duration_create = librosa.get_duration(filename=f"audio2/{filename}")
        audio_files.append(filename)
        speakers_list.append(speaker)

    return audio_files, speakers_list