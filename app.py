"""
Nepali News Anchor TTS — Flask Backend
Pipeline: Text Normalization → VITS Inference → Audio Output
(G2P removed — normalized Devanagari text fed directly to VITS)
"""

import os
import io
import re
import unicodedata
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from urllib.parse import quote                          # ← NEW
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Edit these paths to match your setup
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_config":   "configs/nepali_base.json",
    "checkpoint_dir": "logs/",
    "static_dir":     "static",          # frontend files live here
}

SPEAKER_INFO = {
    3: {"name": "Male Normal News Anchor",    "role": "", "icon": "🎙️"},
    4: {"name": "Female Normal News Anchor",  "role": "", "icon": "🎙️"},
    5: {"name": "Male Breaking News Anchor",  "role": "", "icon": "⚡"},
    6: {"name": "Female Breaking News Anchor","role": "", "icon": "⚡"},
}

# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_PUNCT = {",", "।", "?", "!", ":", "-", '"', "'"}

CHAR_MAP = {
    "ी": "ि",
    "ू": "ु",
    "श": "स",
    "ष": "स",
    "ञ": "न",
    "ई": "इ",
    "ऊ": "उ",
    "ऋ": "रि",
    "श्र": "स्र",
}


def _normalize_numbers(text: str) -> str:
    try:
        from num_to_words import num_to_word

        def replace_number(match):
            raw = match.group()
            raw_ascii = raw.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
            try:
                return num_to_word(int(raw_ascii), lang="ne")
            except Exception:
                return raw

        return re.sub(r"[0-9०-९]+", replace_number, text)
    except ImportError:
        return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))


def _remove_nukta(text: str) -> str:
    return text.replace("\u093c", "")


def _normalize_candrabindu(text: str) -> str:
    return text.replace("ॅ", "ँ")


def _normalize_punctuation_spacing(text: str) -> str:
    return re.sub(r"\s*([,।?!:\-])\s*", r" \1 ", text)


def _attach_postpositions(text: str) -> str:
    pattern = re.compile(
        r"(\S+)\s+(मा|को|का|लाइ|ले|बाट|देखि|सम्म)(?=\s|$)"
    )
    while True:
        new_text = pattern.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return text


def _is_allowed_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return ch in ALLOWED_PUNCT or ch.isspace() or cat in ("Lo", "Mn", "Mc", "Nd")


def _apply_character_mapping(text: str) -> str:
    text = text.replace("ज्ञ", "__GYA__")
    text = text.replace("क्ष", "__KSHA__")
    text = "".join(CHAR_MAP.get(ch, ch) for ch in text)
    text = text.replace("__GYA__", "ज्ञ")
    text = text.replace("__KSHA__", "क्ष")
    return text


def _apply_orthographic_reduction(text: str) -> str:
    text = text.replace("\u200d", "").replace("\u200c", "")
    text = re.sub(r"([क-ह])ृ", r"\1्रि", text)
    return text


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _normalize_candrabindu(text)
    text = text.replace("\t", " ").replace("\n", " ")
    text = _normalize_numbers(text)
    text = _remove_nukta(text)
    text = _apply_character_mapping(text)
    text = _apply_orthographic_reduction(text)
    text = _normalize_punctuation_spacing(text)
    text = _attach_postpositions(text)
    text = "".join(ch for ch in text if _is_allowed_char(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# SMART TEXT SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def split_text(text: str, max_len: int = 350) -> list[str]:
    sentences = re.split(r"\n+", text)
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
        if sent:
            chunks.append(sent)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# VITS MODEL
# ─────────────────────────────────────────────────────────────────────────────
class VITSInference:

    def __init__(self, config_path: str, checkpoint_dir: str):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hps     = None
        self.net_g   = None
        self.loaded  = False
        self._utils  = None

        try:
            import utils
            import importlib
            importlib.reload(utils)
            from models import SynthesizerTrn
            from text.symbols import symbols

            self.hps = utils.get_hparams_from_file(config_path)

            ckpt_dir    = Path(checkpoint_dir)
            checkpoints = sorted(
                [f for f in ckpt_dir.iterdir()
                 if f.name.startswith("G_") and f.name.endswith(".pth")],
                key=lambda x: int(x.stem.split("_")[1])
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

            latest = checkpoints[-1]
            print(f"🔹 Loading checkpoint: {latest}")

            self.net_g = SynthesizerTrn(
                len(symbols),
                80,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            ).to(self.device)

            self.net_g.eval()

            checkpoint = torch.load(str(latest), map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint)
            self.net_g.load_state_dict(state_dict, strict=False)

            self._utils   = utils
            self._symbols = symbols
            self.loaded   = True
            print(f"✅ VITS model loaded on {self.device}")

        except Exception as exc:
            print(f"⚠️  Model not loaded: {exc}")
            print("   Running in TEXT-ONLY / DEMO mode")

    def _get_text(self, text: str) -> torch.LongTensor:
        from text import get_text as _get_text
        return _get_text(text, self.hps)

    def infer_long(
        self,
        text: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        noise_scale: float = 0.5,
        noise_scale_w: float = 0.6,
    ) -> tuple[np.ndarray | None, int | None]:
        if not self.loaded:
            return None, None

        chunks = split_text(text)
        full_audio: list[np.ndarray] = []
        pause = np.zeros(int(0.15 * self.hps.data.sampling_rate))

        for chunk in chunks:
            chunk = normalize_text(chunk)
            if not chunk:
                continue
            print(f"  chunk: {chunk[:80]}…" if len(chunk) > 80 else f"  chunk: {chunk}")

            stn = self._get_text(chunk)
            if stn.size(0) == 0:
                continue

            with torch.no_grad():
                x     = stn.unsqueeze(0).to(self.device)
                xlen  = torch.LongTensor([stn.size(0)]).to(self.device)
                sid   = torch.LongTensor([speaker_id]).to(self.device)

                audio_chunk = self.net_g.infer(
                    x, xlen, sid=sid,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=speed,
                )[0][0, 0].cpu().float().numpy()

            full_audio.append(audio_chunk)
            full_audio.append(pause)

        if not full_audio:
            return None, None

        return np.concatenate(full_audio), self.hps.data.sampling_rate


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=CONFIG["static_dir"])
CORS(app)

tts_model = VITSInference(CONFIG["model_config"], CONFIG["checkpoint_dir"])


@app.route("/")
def index():
    return send_from_directory(CONFIG["static_dir"], "index.html")


@app.route("/api/speakers", methods=["GET"])
def get_speakers():
    return jsonify(SPEAKER_INFO)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": tts_model.loaded,
        "device":       str(tts_model.device) if tts_model.loaded else "N/A",
        "speakers":     len(SPEAKER_INFO),
    })


@app.route("/api/normalize", methods=["POST"])
def api_normalize():
    """Preview: return normalized Devanagari text only."""
    data = request.get_json(force=True)
    text = data.get("text", "")
    return jsonify({"normalized": normalize_text(text)})


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    """Full pipeline: raw text → normalize → VITS → WAV bytes."""
    data         = request.get_json(force=True)
    raw_text     = data.get("text", "")
    speaker_id   = int(data.get("speaker_id", 0))
    speed        = float(data.get("speed", 1.0))
    noise        = float(data.get("noise_scale", 0.5))
    noise_w      = float(data.get("noise_scale_w", 0.6))

    if not raw_text.strip():
        return jsonify({"error": "Empty text"}), 400

    normalized = normalize_text(raw_text)

    if not tts_model.loaded:
        return jsonify({
            "demo_mode":  True,
            "original":   raw_text,
            "normalized": normalized,
            "message":    "Model not loaded — normalization pipeline is working correctly.",
        })

    audio, sr = tts_model.infer_long(
        raw_text, speaker_id, speed, noise, noise_w
    )

    if audio is None:
        return jsonify({"error": "Inference failed"}), 500

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    response = make_response(buf.read())
    response.headers["Content-Type"]        = "audio/wav"
    response.headers["Content-Disposition"] = (
        f'inline; filename="speaker{speaker_id}_output.wav"'
    )
    # ── FIX: URL-encode so Devanagari survives the HTTP header ──
    response.headers["X-Normalized-Text"] = quote(normalized)
    return response


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Nepali News Anchor TTS Server")
    print("=" * 60)
    print(f"  Open: http://localhost:5000")
    print(f"  Model loaded: {tts_model.loaded}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)