import re
import json
import os
from functools import lru_cache


def normalize_language_code(code):
    if not code:
        return ""
    return str(code).strip().replace("_", "-").lower()


def is_pt_br_language(code):
    return normalize_language_code(code) == "pt-br"


def _apply_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _replace_comboio(match):
    replacement = "trens" if match.group(1) else "trem"
    return _apply_case(match.group(0), replacement)


def _replace_autocarro(match):
    return _apply_case(match.group(0), "\u00f4nibus")


def _replace_telemovel(match):
    replacement = (
        "celulares"
        if match.group(0).lower().endswith("s")
        else "celular"
    )
    return _apply_case(match.group(0), replacement)


def _replace_facto(match):
    replacement = "fatos" if match.group(1) else "fato"
    return _apply_case(match.group(0), replacement)


_PT_BR_RULES = [
    (re.compile(r"\bcomboio(s)?\b", re.IGNORECASE), _replace_comboio),
    (re.compile(r"\bautocarro(s)?\b", re.IGNORECASE), _replace_autocarro),
    (
        re.compile(r"\btelem[o\u00f3]v(?:el|eis)\b", re.IGNORECASE),
        _replace_telemovel,
    ),
    (re.compile(r"\bfacto(s)?\b", re.IGNORECASE), _replace_facto),
]


def normalize_pt_br_text(text):
    if not text:
        return text

    normalized = text
    for pattern, replacer in _PT_BR_RULES:
        normalized = pattern.sub(replacer, normalized)

    normalized = _apply_pt_br_glossary(normalized)

    return normalized


def normalize_pt_br_segments(segments):
    for segment in segments:
        segment["text"] = normalize_pt_br_text(segment.get("text", ""))
    return segments


@lru_cache(maxsize=1)
def _load_pt_br_glossary():
    custom_path = os.getenv("SONITR_PT_BR_GLOSSARY", "").strip()
    candidates = [
        custom_path,
        "pt_br_glossary.json",
        os.path.join(os.path.dirname(__file__), "pt_br_glossary.json"),
    ]
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return {}


def _apply_pt_br_glossary(text):
    glossary = _load_pt_br_glossary()
    if not glossary:
        return text

    normalized = text
    for source, target in glossary.items():
        if not source:
            continue
        if source.startswith("re:"):
            pattern = source[3:]
            normalized = re.sub(pattern, target, normalized)
        else:
            pattern = re.compile(
                r"\b" + re.escape(source) + r"\b", re.IGNORECASE
            )

            def repl(match):
                return _apply_case(match.group(0), target)

            normalized = pattern.sub(repl, normalized)

    return normalized
