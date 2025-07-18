from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import os
import re
from typing import Any

# Path to the fastText language identification model
BASE_DIR = "/workspace/assignment4-data/cs336_data"
FASTTEXT_LANG_MODEL = os.path.join(BASE_DIR, "lid.176.bin")
FASTTEXT_NSFW_MODEL = os.path.join(BASE_DIR, "jigsaw_fasttext_bigrams_nsfw_final.bin")
FASTTEXT_HATE_SPEECH_MODEL = os.path.join(BASE_DIR, "jigsaw_fasttext_bigrams_hatespeech_final.bin")
FASTTEXT_QUALITY_MODEL = os.path.join(BASE_DIR, "fasttext_quality_model.bin")
_fasttext_lang_model = None
_fasttext_nsfw_model = None
_fasttext_hatespeech_model = None
_fasttext_quality_model = None

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html_str)

def identify_language(text: str) -> tuple[str, float]:
    global _fasttext_lang_model
    if _fasttext_lang_model is None:
        _fasttext_lang_model = fasttext.load_model(FASTTEXT_LANG_MODEL)
    predictions = _fasttext_lang_model.predict(text.replace("\n", " "))
    label = predictions[0][0]  # e.g., '__label__en'
    score = float(predictions[1][0])
    lang = label.replace("__label__", "")
    return lang, score

def mask_emails(text: str) -> tuple[str, int]:
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    matches = list(re.finditer(email_pattern, text))
    num_masked = len(matches)
    masked_text = re.sub(email_pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_masked

def mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    matches = list(re.finditer(phone_pattern, text))
    num_masked = len(matches)
    masked_text = re.sub(phone_pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, num_masked

def mask_ips(text: str) -> tuple[str, int]:
    ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    matches = list(re.finditer(ip_pattern, text))
    num_masked = len(matches)
    masked_text = re.sub(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, num_masked

def classify_nsfw(text: str) -> tuple[Any, float]:
    global _fasttext_nsfw_model
    if _fasttext_nsfw_model is None:
        _fasttext_nsfw_model = fasttext.load_model(FASTTEXT_NSFW_MODEL)
    predictions = _fasttext_nsfw_model.predict(text.replace("\n", " "))
    label = predictions[0][0]  # e.g., '__label__nsfw'
    label = label.replace("__label__", "")
    score = float(predictions[1][0])
    return label, score

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    global _fasttext_hatespeech_model
    if _fasttext_hatespeech_model is None:
        _fasttext_hatespeech_model = fasttext.load_model(FASTTEXT_HATE_SPEECH_MODEL)
    predictions = _fasttext_hatespeech_model.predict(text.replace("\n", " "))
    label = predictions[0][0]  # e.g., '__label__hatespeech'
    label = label.replace("__label__", "")
    score = float(predictions[1][0])
    return label, score

def classify_quality(text: str) -> tuple[Any, float]:
    global _fasttext_quality_model
    if _fasttext_quality_model is None:
        _fasttext_quality_model = fasttext.load_model(FASTTEXT_QUALITY_MODEL)
    predictions = _fasttext_quality_model.predict(text.replace("\n", " "))
    label = predictions[0][0]  # e.g., '__label__good'
    label = label.replace("__label__", "")
    score = float(predictions[1][0])
    return label, score

def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    lines = text.split("\n")
    # contain less than 50 words
    if len(words) < 50:
        return False
    # contain more than 100000 words
    if len(words) > 100000:
        return False
    # have a mean word length outside the range of 3 to 10 characters
    if len(words) > 0:
        mean_word_length = sum(len(word) for word in words) / len(words)
        if mean_word_length < 3 or mean_word_length > 10:
            return False
    # have more than 30% of the lines ending with an ellipsis
    if len(words) > 0:
        ellipsis_lines = [line for line in lines if line.endswith("...")]
        if len(ellipsis_lines) / len(lines) > 0.3:
            return False
    # have less than 80% of the lines with alphabetic characters
    if len(words) > 0:
        alphabetic_words = [word for word in words if word.isalpha()]
        if len(alphabetic_words) / len(words) < 0.8:
            return False
    return True


if __name__ == "__main__":
    warc_path = "cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    count = 0
    with open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.headers.get('WARC-Type') == 'response':
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                print("Text:", text[:200], "...\n")
                lang, lang_score = identify_language(text)
                if lang:
                    print("identify_language:", (lang, lang_score))
                masked_email_text, num_emails = mask_emails(text)
                if num_emails > 0:
                    print("mask_emails:", (masked_email_text[:200] + '...', num_emails))
                masked_phone_text, num_phones = mask_phone_numbers(text)
                if num_phones > 0:
                    print("mask_phone_numbers:", (masked_phone_text[:200] + '...', num_phones))
                masked_ip_text, num_ips = mask_ips(text)
                if num_ips > 0:
                    print("mask_ips:", (masked_ip_text[:200] + '...', num_ips))
                nsfw_label, nsfw_score = classify_nsfw(text)
                if nsfw_label != "nsfw":
                    print("classify_nsfw:", (nsfw_label, nsfw_score))
                toxic_label, toxic_score = classify_toxic_speech(text)
                if toxic_label != "non-toxic":
                    print("classify_toxic_speech:", (toxic_label, toxic_score))
                if gopher_quality_filter(text):
                    print("gopher_quality_filter: True")
                else:
                    print("gopher_quality_filter: False")
                count += 1
                if count > 20:
                    break
                
