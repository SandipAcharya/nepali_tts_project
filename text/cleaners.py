# text/cleaners.py
import unicodedata

def nepali_cleaners(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.strip().split())
    return text