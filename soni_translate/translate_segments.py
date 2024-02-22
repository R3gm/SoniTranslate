from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language
from .logging_setup import logger


def translate_iterative(segments, TRANSLATE_AUDIO_TO):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries, each containing 'text' as a key
        with the segment text to be translated.
    - TRANSLATE_AUDIO_TO (str): The language code to which the text should be
        translated.

    Returns:
    - list: A list of dictionaries with translated text segments in the specified language.

    Notes:
    - This function translates each text segment individually using the Google Translator.

    Example:
    segments = [{'text': 'This is the first segment.'}, {'text': 'And this is the second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    translator = GoogleTranslator(source="auto", target=TRANSLATE_AUDIO_TO)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line

    return segments_


def translate_batch(segments, TRANSLATE_AUDIO_TO, chunk_size=2000):
    """
    Translate a batch of text segments into the specified language in chunks respecting the character limit.

    Parameters:
    - segments (list): A list of dictionaries, each containing 'text' as a key with the segment text to be translated.
    - TRANSLATE_AUDIO_TO (str): The language code to which the text should be translated.
    - chunk_size (int, optional): The maximum character limit for each translation chunk (default is 2000); max 5000.

    Returns:
    - list: A list of dictionaries with translated text segments in the specified language.

    Notes:
    - This function splits the input segments into chunks respecting the character limit for translation.
    - It translates the chunks using the Google Translator.
    - If the chunked translation fails, it switches to iterative translation using `translate_iterative()`.

    Example:
    segments = [{'text': 'This is the first segment.'}, {'text': 'And this is the second segment.'}]
    translated_segments = translate_batch(segments, 'es', chunk_size=4000)
    """

    segments_copy = copy.deepcopy(segments)

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    for one_line in text_lines:
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
        else:
            text_merge.append(actual_chunk)
            one_line = " " if not one_line else one_line
            actual_chunk = one_line
    if actual_chunk:
        text_merge.append(actual_chunk)

    # translate chunks
    translator = GoogleTranslator(source="auto", target=TRANSLATE_AUDIO_TO)
    try:
        translated_lines = translator.translate_batch(text_merge)
    except Exception as error:
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative. Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, TRANSLATE_AUDIO_TO)

    # un chunk
    split_list = [sentence.split("|||||") for sentence in translated_lines]
    translated_lines = list(chain.from_iterable(split_list))

    # verify integrity ok
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> {translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[line].strip()
        return segments_copy
    else:
        logger.error(
            f"The translation in chunks failed, switching to iterative. {len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, TRANSLATE_AUDIO_TO)


def translate_text(
    segments,
    TRANSLATE_AUDIO_TO,
    translation_process="google_translator_batch",
    chunk_size=4500,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "google_translator_batch":
            return translate_batch(
                segments, fix_code_language(TRANSLATE_AUDIO_TO), chunk_size
            )
        case "google_translator_iterative":
            return translate_iterative(
                segments, fix_code_language(TRANSLATE_AUDIO_TO)
            )
        case "disable_translation":
            return segments
        case _:
            raise ValueError("No valid translation process")
