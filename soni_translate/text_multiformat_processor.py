from .logging_setup import logger
from whisperx.utils import get_writer
from .utils import remove_files, run_command, remove_directory_contents
from typing import List
import srt
import re
import os
import copy
import string
import soundfile as sf
from PIL import Image, ImageOps, ImageDraw, ImageFont

punctuation_list = list(
    string.punctuation + "¡¿«»„”“”‚‘’「」『』《》（）【】〈〉〔〕〖〗〘〙〚〛⸤⸥⸨⸩"
)
symbol_list = punctuation_list + ["", "..", "..."]


def extract_from_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        srt_content = file.read()

    subtitle_generator = srt.parse(srt_content)
    srt_content_list = list(subtitle_generator)

    return srt_content_list


def clean_text(text):

    # Remove content within square brackets
    text = re.sub(r'\[.*?\]', '', text)
    # Add pattern to remove content within <comment> tags
    text = re.sub(r'<comment>.*?</comment>', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove "♫" and "♪" content
    text = re.sub(r'♫.*?♫', '', text)
    text = re.sub(r'♪.*?♪', '', text)
    # Replace newline characters with an empty string
    text = text.replace("\n", ". ")
    # Remove double quotation marks
    text = text.replace('"', '')
    # Collapse multiple spaces and replace with a single space
    text = re.sub(r"\s+", " ", text)
    # Normalize spaces around periods
    text = re.sub(r"[\s\.]+(?=\s)", ". ", text)
    # Check if there are ♫ or ♪ symbols present
    if '♫' in text or '♪' in text:
        return ""

    text = text.strip()

    # Valid text
    return text if text not in symbol_list else ""


def srt_file_to_segments(file_path, speaker=False):
    try:
        srt_content_list = extract_from_srt(file_path)
    except Exception as error:
        logger.error(str(error))
        fixed_file = "fixed_sub.srt"
        remove_files(fixed_file)
        fix_sub = f'ffmpeg -i "{file_path}" "{fixed_file}" -y'
        run_command(fix_sub)
        srt_content_list = extract_from_srt(fixed_file)

    segments = []
    for segment in srt_content_list:

        text = clean_text(str(segment.content))

        if text:
            segments.append(
                {
                    "text": text,
                    "start": float(segment.start.total_seconds()),
                    "end": float(segment.end.total_seconds()),
                }
            )

    if not segments:
        raise Exception("No data found in srt subtitle file")

    if speaker:
        segments = [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    return {"segments": segments}


# documents


def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix):]
    return lines


def remove_hyphens(text: str) -> str:
    """

    This fails for:
    * Natural dashes: well-known, self-replication, use-cases, non-semantic,
                      Post-processing, Window-wise, viewpoint-dependent
    * Trailing math operands: 2 - 4
    * Names: Lopez-Ferreras, VGG-19, CIFAR-100
    """
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-"):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)


def pdf_to_txt(pdf_file, start_page, end_page):
    from pypdf import PdfReader

    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        logger.debug(f"Total pages: {reader.get_num_pages()}")
        text = ""

        start_page_idx = max((start_page-1), 0)
        end_page_inx = min((end_page), (reader.get_num_pages()))
        document_pages = reader.pages[start_page_idx:end_page_inx]
        logger.info(
            f"Selected pages from {start_page_idx} to {end_page_inx}: "
            f"{len(document_pages)}"
        )

        for page in document_pages:
            text += remove_hyphens(page.extract_text())
    return text


def docx_to_txt(docx_file):
    # https://github.com/AlJohri/docx2pdf update
    from docx import Document

    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def replace_multiple_elements(text, replacements):
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    replaced_text = pattern.sub(
        lambda match: replacements[match.group(0)], text
    )

    # Remove multiple spaces
    replaced_text = re.sub(r"\s+", " ", replaced_text)

    return replaced_text


def document_preprocessor(file_path, is_string, start_page, end_page):
    if not is_string:
        file_ext = os.path.splitext(file_path)[1].lower()

    if is_string:
        text = file_path
    elif file_ext == ".pdf":
        text = pdf_to_txt(file_path, start_page, end_page)
    elif file_ext == ".docx":
        text = docx_to_txt(file_path)
    elif file_ext == ".txt":
        with open(
            file_path, "r", encoding='utf-8', errors='replace'
        ) as file:
            text = file.read()
    else:
        raise Exception("Unsupported file format")

    # Add space to break segments more easily later
    replacements = {
        "、": "、 ",
        "。": "。 ",
        # "\n": " ",
    }
    text = replace_multiple_elements(text, replacements)

    # Save text to a .txt file
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = "./text_preprocessor.txt"

    with open(
        txt_file_path, "w", encoding='utf-8', errors='replace'
    ) as txt_file:
        txt_file.write(text)

    return txt_file_path, text


def split_text_into_chunks(text, chunk_size):
    words = re.findall(r"\b\w+\b", text)
    chunks = []
    current_chunk = ""
    for word in words:
        if (
            len(current_chunk) + len(word) + 1 <= chunk_size
        ):  # Adding 1 for the space between words
            if current_chunk:
                current_chunk += " "
            current_chunk += word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def determine_chunk_size(file_name):
    patterns = {
        re.compile(r".*-(Male|Female)$"): 1024,  # by character
        re.compile(r".* BARK$"): 100,  # t 64 256
        re.compile(r".* VITS$"): 500,
        re.compile(
            r".+\.(wav|mp3|ogg|m4a)$"
        ): 150,  # t 250 400 api automatic split
        re.compile(r".* VITS-onnx$"): 250,  # automatic sentence split
        re.compile(r".* OpenAI-TTS$"): 1024  # max charaters 4096
    }

    for pattern, chunk_size in patterns.items():
        if pattern.match(file_name):
            return chunk_size

    # Default chunk size if the file doesn't match any pattern; max 1800
    return 100


def plain_text_to_segments(result_text=None, chunk_size=None):
    if not chunk_size:
        chunk_size = 100
    text_chunks = split_text_into_chunks(result_text, chunk_size)

    segments_chunks = []
    for num, chunk in enumerate(text_chunks):
        chunk_dict = {
            "text": chunk,
            "start": (1.0 + num),
            "end": (2.0 + num),
            "speaker": "SPEAKER_00",
        }
        segments_chunks.append(chunk_dict)

    result_diarize = {"segments": segments_chunks}

    return result_diarize


def segments_to_plain_text(result_diarize):
    complete_text = ""
    for seg in result_diarize["segments"]:
        complete_text += seg["text"] + " "  # issue

    # Save text to a .txt file
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = "./text_translation.txt"

    with open(
        txt_file_path, "w", encoding='utf-8', errors='replace'
    ) as txt_file:
        txt_file.write(complete_text)

    return txt_file_path, complete_text


# doc to video

COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "light_gray": (200, 200, 200),
    "light_blue": (173, 216, 230),
    "light_green": (144, 238, 144),
    "light_yellow": (255, 255, 224),
    "light_pink": (255, 182, 193),
    "lavender": (230, 230, 250),
    "peach": (255, 218, 185),
    "light_cyan": (224, 255, 255),
    "light_salmon": (255, 160, 122),
    "light_green_yellow": (173, 255, 47),
}

BORDER_COLORS = ["dynamic"] + list(COLORS.keys())


def calculate_average_color(img):
    # Resize the image to a small size for faster processing
    img_small = img.resize((50, 50))
    # Calculate the average color
    average_color = img_small.convert("RGB").resize((1, 1)).getpixel((0, 0))
    return average_color


def add_border_to_image(
    image_path,
    target_width,
    target_height,
    border_color=None
):

    img = Image.open(image_path)

    # Calculate the width and height for the new image with borders
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Resize the image to fit the target resolution retaining aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        # Image is wider, calculate new height
        new_height = int(target_width / original_aspect_ratio)
        resized_img = img.resize((target_width, new_height))
    else:
        # Image is taller, calculate new width
        new_width = int(target_height * original_aspect_ratio)
        resized_img = img.resize((new_width, target_height))

    # Calculate padding for borders
    padding = (0, 0, 0, 0)
    if resized_img.size[0] != target_width or resized_img.size[1] != target_height:
        if original_aspect_ratio > target_aspect_ratio:
            # Add borders vertically
            padding = (0, (target_height - resized_img.size[1]) // 2, 0, (target_height - resized_img.size[1]) // 2)
        else:
            # Add borders horizontally
            padding = ((target_width - resized_img.size[0]) // 2, 0, (target_width - resized_img.size[0]) // 2, 0)

    # Add borders with specified color
    if not border_color or border_color == "dynamic":
        border_color = calculate_average_color(resized_img)
    else:
        border_color = COLORS.get(border_color, (0, 0, 0))

    bordered_img = ImageOps.expand(resized_img, padding, fill=border_color)

    bordered_img.save(image_path, format='PNG')

    return image_path


def resize_and_position_subimage(
    subimage,
    max_width,
    max_height,
    subimage_position,
    main_width,
    main_height
):
    subimage_width, subimage_height = subimage.size

    # Resize subimage if it exceeds maximum dimensions
    if subimage_width > max_width or subimage_height > max_height:
        # Calculate scaling factor
        width_scale = max_width / subimage_width
        height_scale = max_height / subimage_height
        scale = min(width_scale, height_scale)

        # Resize subimage
        subimage = subimage.resize(
            (int(subimage_width * scale), int(subimage_height * scale))
        )

    # Calculate position to place the subimage
    if subimage_position == "top-left":
        subimage_x = 0
        subimage_y = 0
    elif subimage_position == "top-right":
        subimage_x = main_width - subimage.width
        subimage_y = 0
    elif subimage_position == "bottom-left":
        subimage_x = 0
        subimage_y = main_height - subimage.height
    elif subimage_position == "bottom-right":
        subimage_x = main_width - subimage.width
        subimage_y = main_height - subimage.height
    else:
        raise ValueError(
            "Invalid subimage_position. Choose from 'top-left', 'top-right',"
            " 'bottom-left', or 'bottom-right'."
        )

    return subimage, subimage_x, subimage_y


def create_image_with_text_and_subimages(
    text,
    subimages,
    width,
    height,
    text_color,
    background_color,
    output_file
):
    # Create an image with the specified resolution and background color
    image = Image.new('RGB', (width, height), color=background_color)

    # Initialize ImageDraw object
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.load_default()  # You can specify your font file here

    # Calculate text size and position
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2

    # Draw text on the image
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    # Paste subimages onto the main image
    for subimage_path, subimage_position in subimages:
        # Open the subimage
        subimage = Image.open(subimage_path)

        # Convert subimage to RGBA mode if it doesn't have an alpha channel
        if subimage.mode != 'RGBA':
            subimage = subimage.convert('RGBA')

        # Resize and position the subimage
        subimage, subimage_x, subimage_y = resize_and_position_subimage(
            subimage, width / 4, height / 4, subimage_position, width, height
        )

        # Paste the subimage onto the main image
        image.paste(subimage, (int(subimage_x), int(subimage_y)), subimage)

    image.save(output_file)

    return output_file


def doc_to_txtximg_pages(
    document,
    width,
    height,
    start_page,
    end_page,
    bcolor
):
    from pypdf import PdfReader

    images_folder = "pdf_images/"
    os.makedirs(images_folder, exist_ok=True)
    remove_directory_contents(images_folder)

    # First image
    text_image = os.path.basename(document)[:-4]
    subimages = [("./assets/logo.jpeg", "top-left")]
    text_color = (255, 255, 255) if bcolor == "black" else (0, 0, 0)  # w|b
    background_color = COLORS.get(bcolor, (255, 255, 255))  # dynamic white
    first_image = "pdf_images/0000_00_aaa.png"

    create_image_with_text_and_subimages(
        text_image,
        subimages,
        width,
        height,
        text_color,
        background_color,
        first_image
    )

    reader = PdfReader(document)
    logger.debug(f"Total pages: {reader.get_num_pages()}")

    start_page_idx = max((start_page-1), 0)
    end_page_inx = min((end_page), (reader.get_num_pages()))
    document_pages = reader.pages[start_page_idx:end_page_inx]

    logger.info(
        f"Selected pages from {start_page_idx} to {end_page_inx}: "
        f"{len(document_pages)}"
    )

    data_doc = {}
    for i, page in enumerate(document_pages):

        count = 0
        images = []
        for image_file_object in page.images:
            img_name = f"{images_folder}{i:04d}_{count:02d}_{image_file_object.name}"
            if not img_name.lower().endswith('.png'):
                img_name = os.path.splitext(img_name)[0] + '.png'
            images.append(img_name)
            with open(img_name, "wb") as fp:
                fp.write(image_file_object.data)
                count += 1
            img_name = add_border_to_image(img_name, width, height, bcolor)

        data_doc[i] = {
            "text": remove_hyphens(page.extract_text()),
            "images": images
        }

    return data_doc


def page_data_to_segments(result_text=None, chunk_size=None):

    if not chunk_size:
        chunk_size = 100

    segments_chunks = []
    time_global = 0
    for page, result_data in result_text.items():
        # result_image = result_data["images"]
        result_text = result_data["text"]
        text_chunks = split_text_into_chunks(result_text, chunk_size)
        if not text_chunks:
            text_chunks = [" "]

        for chunk in text_chunks:
            chunk_dict = {
                "text": chunk,
                "start": (1.0 + time_global),
                "end": (2.0 + time_global),
                "speaker": "SPEAKER_00",
                "page": page,
            }
            segments_chunks.append(chunk_dict)
            time_global += 1

    result_diarize = {"segments": segments_chunks}

    return result_diarize


def update_page_data(result_diarize, doc_data):
    complete_text = ""
    current_page = result_diarize["segments"][0]["page"]
    text_page = ""

    for seg in result_diarize["segments"]:
        text = seg["text"] + " "  # issue
        complete_text += text

        page = seg["page"]

        if page == current_page:
            text_page += text
        else:
            doc_data[current_page]["text"] = text_page

            # Next
            text_page = text
            current_page = page

    if doc_data[current_page]["text"] != text_page:
        doc_data[current_page]["text"] = text_page

    return doc_data


def fix_timestamps_docs(result_diarize, audio_files):
    current_start = 0.0

    for seg, audio in zip(result_diarize["segments"], audio_files):
        duration = round(sf.info(audio).duration, 2)

        seg["start"] = current_start
        current_start += duration
        seg["end"] = current_start

    return result_diarize


def create_video_from_images(
    doc_data,
    result_diarize
):

    # First image path
    first_image = "pdf_images/0000_00_aaa.png"

    # Time segments and images
    max_pages_idx = len(doc_data) - 1
    current_page = result_diarize["segments"][0]["page"]
    duration_page = 0.0
    last_image = None

    for seg in result_diarize["segments"]:
        start = seg["start"]
        end = seg["end"]
        duration_seg = end - start

        page = seg["page"]

        if page == current_page:
            duration_page += duration_seg
        else:

            images = doc_data[current_page]["images"]

            if first_image:
                images = [first_image] + images
                first_image = None
            if not doc_data[min(max_pages_idx, (current_page+1))]["text"].strip():
                images = images + doc_data[min(max_pages_idx, (current_page+1))]["images"]
            if not images and last_image:
                images = [last_image]

            # Calculate images duration
            time_duration_per_image = round((duration_page / len(images)), 2)
            doc_data[current_page]["time_per_image"] = time_duration_per_image

            # Next values
            doc_data[current_page]["images"] = images
            last_image = images[-1]
            duration_page = duration_seg
            current_page = page

    if "time_per_image" not in doc_data[current_page].keys():
        images = doc_data[current_page]["images"]
        if first_image:
            images = [first_image] + images
        if not images:
            images = [last_image]
        time_duration_per_image = round((duration_page / len(images)), 2)
        doc_data[current_page]["time_per_image"] = time_duration_per_image

    # Timestamped image video.
    with open("list.txt", "w") as file:

        for i, page in enumerate(doc_data.values()):

            duration = page["time_per_image"]
            for img in page["images"]:
                if i == len(doc_data) - 1 and img == page["images"][-1]:  # Check if it's the last item
                    file.write(f"file {img}\n")
                    file.write(f"outpoint {duration}")
                else:
                    file.write(f"file {img}\n")
                    file.write(f"outpoint {duration}\n")

    out_video = "video_from_images.mp4"
    remove_files(out_video)

    cm = f"ffmpeg -y -f concat -i list.txt -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p {out_video}"
    cm_alt = f"ffmpeg -f concat -i list.txt -c:v libx264 -r 30 -pix_fmt yuv420p -y {out_video}"
    try:
        run_command(cm)
    except Exception as error:
        logger.error(str(error))
        remove_files(out_video)
        run_command(cm_alt)

    return out_video


def merge_video_and_audio(video_doc, final_wav_file):

    fixed_audio = "fixed_audio.mp3"
    remove_files(fixed_audio)
    cm = f"ffmpeg -i {final_wav_file} -c:a libmp3lame {fixed_audio}"
    run_command(cm)

    vid_out = "video_book.mp4"
    remove_files(vid_out)
    cm = f"ffmpeg -i {video_doc} -i {fixed_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {vid_out}"
    run_command(cm)

    return vid_out


# subtitles


def get_subtitle(
    language,
    segments_data,
    extension,
    filename=None,
    highlight_words=False,
):
    if not filename:
        filename = "task_subtitle"

    is_ass_extension = False
    if extension == "ass":
        is_ass_extension = True
        extension = "srt"

    sub_file = filename + "." + extension
    support_name = filename + ".mp3"
    remove_files(sub_file)

    writer = get_writer(extension, output_dir=".")
    word_options = {
        "highlight_words": highlight_words,
        "max_line_count": None,
        "max_line_width": None,
    }

    # Get data subs
    subtitle_data = copy.deepcopy(segments_data)
    subtitle_data["language"] = (
        "ja" if language in ["ja", "zh", "zh-TW"] else language
    )

    # Clean
    if not highlight_words:
        subtitle_data.pop("word_segments", None)
        for segment in subtitle_data["segments"]:
            for key in ["speaker", "chars", "words"]:
                segment.pop(key, None)

    writer(
        subtitle_data,
        support_name,
        word_options,
    )

    if is_ass_extension:
        temp_name = filename + ".ass"
        remove_files(temp_name)
        convert_sub = f'ffmpeg -i "{sub_file}" "{temp_name}" -y'
        run_command(convert_sub)
        sub_file = temp_name

    return sub_file


def process_subtitles(
    deep_copied_result,
    align_language,
    result_diarize,
    output_format_subtitle,
    TRANSLATE_AUDIO_TO,
):
    name_ori = "sub_ori."
    name_tra = "sub_tra."
    remove_files(
        [name_ori + output_format_subtitle, name_tra + output_format_subtitle]
    )

    writer = get_writer(output_format_subtitle, output_dir=".")
    word_options = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }

    # original lang
    subs_copy_result = copy.deepcopy(deep_copied_result)
    subs_copy_result["language"] = (
        "zh" if align_language == "zh-TW" else align_language
    )
    for segment in subs_copy_result["segments"]:
        segment.pop("speaker", None)

    try:
        writer(
            subs_copy_result,
            name_ori[:-1] + ".mp3",
            word_options,
        )
    except Exception as error:
        logger.error(str(error))
        if str(error) == "list indices must be integers or slices, not str":
            logger.error(
                "Related to poor word segmentation"
                " in segments after alignment."
            )
        subs_copy_result["segments"][0].pop("words")
        writer(
            subs_copy_result,
            name_ori[:-1] + ".mp3",
            word_options,
        )

    # translated lang
    subs_tra_copy_result = copy.deepcopy(result_diarize)
    subs_tra_copy_result["language"] = (
        "ja" if TRANSLATE_AUDIO_TO in ["ja", "zh", "zh-TW"] else align_language
    )
    subs_tra_copy_result.pop("word_segments", None)
    for segment in subs_tra_copy_result["segments"]:
        for key in ["speaker", "chars", "words"]:
            segment.pop(key, None)

    writer(
        subs_tra_copy_result,
        name_tra[:-1] + ".mp3",
        word_options,
    )

    return name_tra + output_format_subtitle


def linguistic_level_segments(
    result_base,
    linguistic_unit="word",  # word or char
):
    linguistic_unit = linguistic_unit[:4]
    linguistic_unit_key = linguistic_unit + "s"
    result = copy.deepcopy(result_base)

    if linguistic_unit_key not in result["segments"][0].keys():
        raise ValueError("No alignment detected, can't process")

    segments_by_unit = []
    for segment in result["segments"]:
        segment_units = segment[linguistic_unit_key]
        # segment_speaker = segment.get("speaker", "SPEAKER_00")

        for unit in segment_units:

            text = unit[linguistic_unit]

            if "start" in unit.keys():
                segments_by_unit.append(
                    {
                        "start": unit["start"],
                        "end": unit["end"],
                        "text": text,
                        # "speaker": segment_speaker,
                    }
                    )
            elif not segments_by_unit:
                pass
            else:
                segments_by_unit[-1]["text"] += text

    return {"segments": segments_by_unit}


def break_aling_segments(
    result: dict,
    break_characters: str = "",  # ":|,|.|"
):
    result_align = copy.deepcopy(result)

    break_characters_list = break_characters.split("|")
    break_characters_list = [i for i in break_characters_list if i != '']

    if not break_characters_list:
        logger.info("No valid break characters were specified.")
        return result

    logger.info(f"Redivide text segments by: {str(break_characters_list)}")

    # create new with filters
    normal = []

    def process_chars(chars, letter_new_start, num, text):
        start_key, end_key = "start", "end"
        start_value = end_value = None

        for char in chars:
            if start_key in char:
                start_value = char[start_key]
                break

        for char in reversed(chars):
            if end_key in char:
                end_value = char[end_key]
                break

        if not start_value or not end_value:
            raise Exception(
                f"Unable to obtain a valid timestamp for chars: {str(chars)}"
            )

        return {
            "start": start_value,
            "end": end_value,
            "text": text,
            "words": chars,
        }

    for i, segment in enumerate(result_align['segments']):

        logger.debug(f"- Process segment: {i}, text: {segment['text']}")
        # start = segment['start']
        letter_new_start = 0
        for num, char in enumerate(segment['chars']):

            if char["char"] is None:
                continue

            # if "start" in char:
            #     start = char["start"]

            # if "end" in char:
            #     end = char["end"]

            # Break by character
            if char['char'] in break_characters_list:

                text = segment['text'][letter_new_start:num+1]

                logger.debug(
                    f"Break in: {char['char']}, position: {num}, text: {text}"
                )

                chars = segment['chars'][letter_new_start:num+1]

                if not text:
                    logger.debug("No text")
                    continue

                if num == 0 and not text.strip():
                    logger.debug("blank space in start")
                    continue

                if len(text) == 1:
                    logger.debug(f"Short char append, num: {num}")
                    normal[-1]["text"] += text
                    normal[-1]["words"].append(chars)
                    continue

                # logger.debug(chars)
                normal_dict = process_chars(chars, letter_new_start, num, text)

                letter_new_start = num+1

                normal.append(normal_dict)

            # If we reach the end of the segment, add the last part of chars.
            if num == len(segment["chars"]) - 1:

                text = segment['text'][letter_new_start:num+1]

                # If remain text len is not default len text
                if num not in [len(text)-1, len(text)] and text:
                    logger.debug(f'Remaining text: {text}')

                if not text:
                    logger.debug("No remaining text.")
                    continue

                if len(text) == 1:
                    logger.debug(f"Short char append, num: {num}")
                    normal[-1]["text"] += text
                    normal[-1]["words"].append(chars)
                    continue

                chars = segment['chars'][letter_new_start:num+1]

                normal_dict = process_chars(chars, letter_new_start, num, text)

                letter_new_start = num+1

                normal.append(normal_dict)

    # Rename char to word
    for item in normal:
        words_list = item['words']
        for word_item in words_list:
            if 'char' in word_item:
                word_item['word'] = word_item.pop('char')

    # Convert to dict default
    break_segments = {"segments": normal}

    msg_count = (
        f"Segment count before: {len(result['segments'])}, "
        f"after: {len(break_segments['segments'])}."
    )
    logger.info(msg_count)

    return break_segments
