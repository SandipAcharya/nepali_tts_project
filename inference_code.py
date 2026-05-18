# ==========================================================
# 0. Imports
# ==========================================================
import os
import torch
import IPython.display as ipd
import re
import unicodedata
from num_to_words import num_to_word
import numpy as np

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols

# ==========================================================
# 1. Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# 2. Paths & Checkpoints
# ==========================================================
project_path = "D:\pandata\inpanda\TTS\\nepali_tts_project\sahadev"
os.chdir(project_path)

log_dir = "logs"
config_path = "configs/nepali_base.json"
latest_checkpoint = os.path.join(log_dir, "G_172700.pth")

print(f"🔹 Loading Config: {config_path}")
print(f"🔹 Loading Checkpoint: {latest_checkpoint}")
print(f"🔹 Vocab Size: {len(symbols)}")

# ==========================================================
# 3. Load Config
# ==========================================================
hps = utils.get_hparams_from_file(config_path)

# ==========================================================
# 4. Build Model
# ==========================================================
net_g = SynthesizerTrn(
    len(symbols),
    80,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).to(device)

net_g.eval()

# ==========================================================
# 5. Load Checkpoint
# ==========================================================
checkpoint = torch.load(latest_checkpoint, map_location=device)

if "model" in checkpoint:
    net_g.load_state_dict(checkpoint["model"], strict=False)
else:
    net_g.load_state_dict(checkpoint, strict=False)

print("✅ Checkpoint loaded successfully")

# ==========================================================
# 6. NORMALIZATION (MATCH TRAINING EXACTLY)
# ==========================================================
ALLOWED_PUNCT = {",", "।", "?", "!", ":", "-", '"', "'"}

CHAR_MAP = {
    "ी": "ि", "ू": "ु", "श": "स", "ष": "स", "ञ": "न",
    "ई": "इ", "ऊ": "उ", "ऋ": "रि", "श्र": "स्र"
}

def normalize_numbers(text):
    def replace_number(match):
        number = match.group()
        try:
            return num_to_word(int(number), lang="ne")
        except:
            return number
    return re.sub(r"[0-9०-९]+", replace_number, text)

def remove_nukta(text):
    return text.replace("\u093c", "")

def normalize_candrabindu(text):
    return text.replace("ॅ", "ँ")

def normalize_punctuation_spacing(text):
    return re.sub(r"\s*([,।?!:\-])\s*", r" \1 ", text)

def attach_postpositions(text):
    pattern = re.compile(r"(\S+)\s+(मा|को|का|लाइ|ले|बाट|देखि|सम्म)(?=\s|$)")
    while True:
        new_text = pattern.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return text

def is_allowed_char(ch):
    cat = unicodedata.category(ch)
    return ch in ALLOWED_PUNCT or ch.isspace() or cat in ("Lo", "Mn", "Mc", "Nd")

def apply_character_mapping(text):
    text = text.replace("ज्ञ", "__GYA__")
    text = text.replace("क्ष", "__KSHA__")

    text = "".join(CHAR_MAP.get(ch, ch) for ch in text)

    text = text.replace("__GYA__", "ज्ञ")
    text = text.replace("__KSHA__", "क्ष")
    return text

def apply_orthographic_reduction(text):
    text = text.replace("\u200d", "").replace("\u200c", "")
    text = re.sub(r"([क-ह])ृ", r"\1्रि", text)
    return text

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = normalize_candrabindu(text)

    text = text.replace("\t", " ").replace("\n", " ")

    text = normalize_numbers(text)
    text = remove_nukta(text)
    text = apply_character_mapping(text)
    text = apply_orthographic_reduction(text)
    text = normalize_punctuation_spacing(text)

    text = attach_postpositions(text)

    text = "".join(ch for ch in text if is_allowed_char(ch))
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================================================
# 7. TEXT TO SEQUENCE (SAME AS TRAINING)
# ==========================================================
from text import get_text as _get_text

def get_text(text):
    return _get_text(text, hps)

# ==========================================================
# 8. SMART TEXT SPLITTING
# ==========================================================
def split_text(text, max_len=350):

    sentences = re.split(r'\n+', text)

    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        while len(sent) > max_len:
            split_idx = sent.rfind(" ", 0, max_len)
            if split_idx == -1:
                split_idx = max_len

            chunks.append(sent[:split_idx])
            sent = sent[split_idx:].strip()

        chunks.append(sent)

    return chunks

# ==========================================================
# 9. TTS INFERENCE
# ==========================================================
def tts_long(text, speaker_id=0, speed=1.0):

    chunks = split_text(text)

    full_audio = []
    pause = np.zeros(int(0.15 * hps.data.sampling_rate))  # 150ms pause

    for chunk in chunks:
        chunk = normalize_text(chunk)
        print(chunk)

        stn_tst = get_text(chunk)
        if stn_tst.size(0) == 0:
            continue

        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            sid = torch.LongTensor([speaker_id]).to(device)

            audio_chunk = net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.5,
                noise_scale_w=0.6,
                length_scale=speed
            )[0][0, 0].cpu().numpy()

            full_audio.append(audio_chunk)
            full_audio.append(pause)

    if len(full_audio) == 0:
        return None

    return np.concatenate(full_audio)

# ==========================================================
# 10. PLAY AUDIO
# ==========================================================
def play_audio(audio):
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

if __name__ == "__main__":

    text = """
    विस्फोटका कारण कार्यालयमा कार्यरत एक कर्मचारी घाइते भएका थिए । घाइते हुनेमा सिरहा नगरपालिका–१ निवासी सौरभ यादव छन् ।

     """
    text = """
    कथावाचन डिजिटल संलग्नता तथा इमर्सिभ ब्रान्ड एक्सपिरियन्सको उत्कृष्ट संयोजनमार्फत पिक्स अफ करेजले नेपालको हिमाली पहिचानलाई साहसी र जीवन्त रूपमा प्रस्तुत गरेको थियो। यस अभियानमा ‘लोकल एक्टिभेसन पार्टनर’का रूपमा जोडिन पाएकोमा डिजिटल इनले विशेष गौरव महसुस गरेको छ।
        """

    # text = """
    #       ललितपुरको ग्वार्कोमा जारी अभिमुखीकरणमा बुधबार सम्बोधन गर्दा सभापति लामिछाने हिजोको विद्रोही र आलोचक छविबाट माथि उठेर उनी एउटा जिम्मेवार नेता र संस्थागत नेतृत्वको रूपमा रूपान्तरण हुन खोजेको सन्देश दिए ।
    #   """

    text = """
         म्याडम क्युरी, फ्लोरेन्स नाइटिङ्गेल, जुनको ताबेई आदि यस्ता व्यक्तित्व हुन् जसले आफ्नो प्रतिभाको माध्यमबाट आफूलाई विश्वभर चिनाउन समर्थ भए र यिनी सबै नारी नै हुन् जसले शिक्षा पाएका थिए ।
      """

    # text="""
    # द आर्टिफिसियल इन्टेलिजेन्स मोडेल इज करेन्ट्ली ट्रेनिङ्ग अन अ ट्वान्टी-थ्री आवर डेटासेट विथ अ रिड्यूस्ड लर्नङ्ग रेट टु इम्प्रुभ द न्याचुरलनेस अफ द सिन्थेसाइज्ड भोइस।

    # """

    # text = """
    #
    # """मोरङ जिल्लाका टुट्पन्त, बिकाश पोखरेललाइ उनका साथि सहदेव चौलागाइले, सरिर सङ्गसँगै तिम्रो वालेट पनि मोटाउँदै जाओस् भन्दै भिन्न सैलीमा जन्मदिनको शुभकामना दिएका छन्।

#     text = """
#     माइ नेम इज राम एन्ड आइ एम फ्रम काठमाण्डू ।
# आइ एम अ स्टुडेन्ट एन्ड आइ स्टडी इन कलेज ।
# माइ फेवरेट सब्जेक्ट इज कम्प्युटर साइन्स एन्ड आइ लाइक प्रोग्रामिङ ।
# इन माइ फ्री टाइम, आइ लिसन म्युजिक एन्ड वाच भिडियोस ।
# माइ गोल इज टू बिकम अ सफ्टवेयर इन्जिनियर इन फ्युचर ।
#     """

    text="""
  सौरभका कारण लाइनखबरसँग ब्रान्ड सिन्थेसाइज्ड प्रवक्ता तथा सहायक रथी राजाराम बस्नेत फ्रम काठमाण्डू स्टडी वालेट यस्तो काम गर्दै आग्रह आएको बताउँछन् । सरकारले अध्यादेश ल्याएपछि विपक्षी प्रतिकूल दलहरु भए एक।
"""

    # text="""
    # हुन पनि विषम् परिस्थितिमा समयमै निर्वाचन गराएर वाहवाही कमाएकी प्रधानमन्त्री सुशीला कार्कीका समर्थकले समेत यो निर्णयको बचाउ गर्न सकेका छैनन् । किनभने यो निर्णय गलत हो भन्नका लागि बहुआयामिक कारणहरू छन् ।
    # """

    text="""
    लामिछानेको पूर्ववत् आक्रामकता अब नयाँ भूमिकासहित बदलिने संकेत देखिएको छ । उनले रास्वपाका कार्यकर्तालाई राम्रोसँग प्रशिक्षित गर्ने समय नै नपाएको स्वीकार गर्दै सामाजिक सञ्जालमा देखिने उत्तेजना र आक्रामकता बन्द गर्न निर्देशन दिएका छन् । उनले रास्वपा समर्थकहरुलाई विपक्षीमाथि तुच्छ शैलीमा जाइलाग्ने क्रम रोक्न पनि आह्वान गरेका छन् ।
    """

    text="""
    उच्च सरकारी सम्बद्ध स्रोतका अनुसार, यो अध्यादेशमार्फत सरकारले विभिन्न १ सय १० वटा कानुन संशोधन गर्दै १५ सय ३४ जनाको राजनीतिक नियुक्ति लिएका सार्वजनिक पदाधिकारीहरूको पदमुक्त हुने व्यवस्था गरिएको स्रोतले जानकारी दियो ।
    """
    print("\n--- Speaker 1 ---")
    audio1 = tts_long(text, speaker_id=0)
    play_audio(audio1)

    print("\n--- Speaker 2 ---")
    audio2 = tts_long(text, speaker_id=1)
    play_audio(audio2)

    print("\n--- Speaker 1 ---")
    audio1 = tts_long(text, speaker_id=2)
    play_audio(audio1)

    print("\n--- Speaker 2 ---")
    audio2 = tts_long(text, speaker_id=3)
    play_audio(audio2)

    print("\n--- Speaker 1 ---")
    audio1 = tts_long(text, speaker_id=4)
    play_audio(audio1)

    print("\n--- Speaker 2 ---")
    audio2 = tts_long(text, speaker_id=5)
    play_audio(audio2)

    print("\n--- Speaker 1 ---")
    audio1 = tts_long(text, speaker_id=6)
    play_audio(audio1)
