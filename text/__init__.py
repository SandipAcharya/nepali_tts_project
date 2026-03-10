import torch
from text.symbols import symbols, _symbol_to_id
from text.cleaners import nepali_cleaners


def text_to_sequence(text, add_blank=False):
    text = nepali_cleaners(text)
    sequence = []
    for ch in text:
        if ch in _symbol_to_id:
            sequence.append(_symbol_to_id[ch])
        else:
            sequence.append(_symbol_to_id["<unk>"])  # Fallback

    if add_blank:
        sequence = intersperse(sequence, 0)

    return sequence


def cleaned_text_to_sequence(cleaned_text, add_blank=False):
    sequence = [_symbol_to_id[c] for c in cleaned_text if c in _symbol_to_id]

    if add_blank:
        sequence = intersperse(sequence, 0)

    return sequence


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.add_blank)
    return torch.LongTensor(text_norm)