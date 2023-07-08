from tqdm import tqdm
from deep_translator import GoogleTranslator

def translate_text(segments, TRANSLATE_AUDIO_TO):

    if TRANSLATE_AUDIO_TO == "zh":
        TRANSLATE_AUDIO_TO = "zh-CN"
    
    translator = GoogleTranslator(source='auto', target=TRANSLATE_AUDIO_TO)
    
    for line in tqdm(range(len(segments))):
        text = segments[line]['text']
        translated_line = translator.translate(text.strip())
        segments[line]['text'] = translated_line

    return segments
