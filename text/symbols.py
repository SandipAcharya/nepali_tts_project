# text/symbols.py

_pad = "_"

# punctuation
_punctuation = "।,.!?- "

# Nepali letters
_letters = (
"अआइईउऊएऐओऔ"
"कखगघङ"
"चछजझञ"
"टठडढण"
"तथदधन"
"पफबभम"
"यरलव"
"शषसह"
)

# vowel signs and modifiers
_marks = "ािीुूेैोौंःँ्"

# full symbol list
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_marks)

# mappings
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}