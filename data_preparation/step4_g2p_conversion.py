"""
Nepali TTS Dataset Preprocessing Pipeline
Step 4: Corrected Grapheme-to-Phoneme (G2P) Conversion

Fixed Issues:
1. Correct IPA symbols for Nepali consonants
2. Proper schwa deletion at word boundaries
3. Better conjunct handling
4. Accurate Devanagari-to-phoneme mapping
"""

import re
import json
import unicodedata
from pathlib import Path
import pandas as pd
from collections import Counter


class NepaliG2P:
    """
    Corrected Rule-based Grapheme-to-Phoneme converter for Nepali
    """
    
    def __init__(self, config_path=None):
        """Initialize G2P converter"""
        
        # Auto-detect config path
        if config_path is None:
            config_paths = [
                "configs/config.json",
                "nepali_tts_project/configs/config.json"
            ]
            for path in config_paths:
                if Path(path).exists():
                    config_path = path
                    break
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.base_dir = Path(config_path).parent.parent
        
        # Initialize phoneme mappings
        self._initialize_phoneme_mappings()
        
        print("✅ Nepali G2P Converter initialized (CORRECTED VERSION)")
        print(f"   Vowels: {len(self.vowels)}")
        print(f"   Consonants: {len(self.consonants)}")
        print(f"   Matras: {len(self.matras)}")
    
    
    def _initialize_phoneme_mappings(self):
        """
        Initialize CORRECTED phoneme mappings for Nepali
        Based on actual Nepali phonology
        """
        
        # Independent vowels - VERIFIED
        self.vowels = {
            'अ': 'a',      # Short 'a' (not schwa in isolation)
            'आ': 'aː',     # Long 'aa'
            'इ': 'i',      # Short 'i'
            'ई': 'iː',     # Long 'ee'
            'उ': 'u',      # Short 'u'
            'ऊ': 'uː',     # Long 'oo'
            'ऋ': 'ri',     # Vocalic 'r'
            'ए': 'e',      # 'e' as in 'bed'
            'ऐ': 'ai',     # Diphthong 'ai'
            'ओ': 'o',      # 'o' as in 'go'
            'औ': 'au'      # Diphthong 'au'
        }
        
        # Vowel diacritics (matras) - VERIFIED
        self.matras = {
            'ा': 'aː',     # Long aa
            'ि': 'i',      # Short i
            'ी': 'iː',     # Long ee
            'ु': 'u',      # Short u
            'ू': 'uː',     # Long oo
            'ृ': 'ri',     # Vocalic r
            'े': 'e',      # e
            'ै': 'ai',     # ai
            'ो': 'o',      # o
            'ौ': 'au',     # au
            'ं': 'ŋ',      # Anusvara (nasal)
            'ः': 'h',      # Visarga
            'ँ': '̃',       # Chandrabindu (nasalization)
            '्': ''        # Halant (virama) - removes inherent vowel
        }
        
        # Consonants with inherent schwa 'ə' - CORRECTED IPA
        self.consonants = {
            # Velars (क-वर्ग)
            'क': 'kə',
            'ख': 'kʰə',    # Aspirated k
            'ग': 'ɡə',
            'घ': 'ɡʰə',    # Aspirated g
            'ङ': 'ŋə',     # Velar nasal
            
            # Palatals (च-वर्ग) - FIXED: च/छ are affricates
            'च': 'tʃə',    # 'ch' as in 'chat' (NOT ts)
            'छ': 'tʃʰə',   # Aspirated ch
            'ज': 'dʒə',    # 'j' as in 'jump' (NOT dz)
            'झ': 'dʒʰə',   # Aspirated j
            'ञ': 'ɲə',     # Palatal nasal
            
            # Retroflexes (ट-वर्ग)
            'ट': 'ʈə',
            'ठ': 'ʈʰə',
            'ड': 'ɖə',
            'ढ': 'ɖʰə',
            'ण': 'ɳə',     # Retroflex nasal
            
            # Dentals (त-वर्ग)
            'त': 'tə',
            'थ': 'tʰə',
            'द': 'də',
            'ध': 'dʰə',
            'न': 'nə',
            
            # Labials (प-वर्ग)
            'प': 'pə',
            'फ': 'pʰə',
            'ब': 'bə',
            'भ': 'bʰə',
            'म': 'mə',
            
            # Semivowels
            'य': 'jə',     # 'y' sound
            'र': 'rə',     # 'r' sound (alveolar tap)
            'ल': 'lə',     # 'l' sound
            'व': 'wə',     # 'w' or 'v' sound
            
            # Sibilants
            'श': 'ʃə',     # 'sh' sound
            'ष': 'ʃə',     # Also 'sh' (merged in Nepali)
            'स': 'sə',     # 's' sound
            
            # Glottal
            'ह': 'ɦə',     # Voiced 'h'
            
            # Special combinations
            'क्ष': 'kʃə',   # ksh
            'त्र': 'trə',   # tr
            'ज्ञ': 'ɡjə'    # gya
        }
        
        # Base consonants without inherent vowel
        self.consonant_base = {}
        for k, v in self.consonants.items():
            if v.endswith('ə'):
                self.consonant_base[k] = v[:-1]
            else:
                self.consonant_base[k] = v
        
        # Punctuation mappings
        self.punctuation = {
            '।': '.',      # Devanagari full stop
            '?': '?',
            '!': '!',
            ',': ',',
            ';': ';',
            ':': ':',
            '-': '-',
            ' ': ' ',
            '\n': ' ',
            '\t': ' '
        }
    
    
    def grapheme_to_phoneme(self, text):
        """
        Convert Nepali text to phonemes with CORRECTED rules
        """
        
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        phonemes = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Handle independent vowels
            if char in self.vowels:
                phonemes.append(self.vowels[char])
                i += 1
                continue
            
            # Handle consonants
            if char in self.consonants:
                # Look ahead for halant (virama)
                if i + 1 < len(text) and text[i + 1] == '्':
                    # Consonant + halant = just consonant sound (no inherent vowel)
                    phoneme = self.consonant_base[char]
                    
                    # Check if there's another consonant after halant
                    if i + 2 < len(text) and text[i + 2] in self.consonants:
                        # This is a conjunct - add first consonant
                        phonemes.append(phoneme)
                        i += 2  # Skip consonant and halant
                        continue
                    else:
                        # Halant at end or before non-consonant
                        phonemes.append(phoneme)
                        i += 2
                        continue
                
                # Look ahead for matra (vowel sign)
                elif i + 1 < len(text) and text[i + 1] in self.matras:
                    matra = text[i + 1]
                    
                    if matra == '्':  # Halant
                        phoneme = self.consonant_base[char]
                    else:
                        # Replace inherent vowel with matra
                        phoneme = self.consonant_base[char] + self.matras[matra]
                    
                    phonemes.append(phoneme)
                    i += 2  # Skip both consonant and matra
                    continue
                
                # Consonant without matra - has inherent schwa
                else:
                    phonemes.append(self.consonants[char])
                    i += 1
                    continue
            
            # Handle standalone matras (shouldn't occur but defensive)
            if char in self.matras and char != '्':
                phonemes.append(self.matras[char])
                i += 1
                continue
            
            # Handle punctuation
            if char in self.punctuation:
                phonemes.append(self.punctuation[char])
                i += 1
                continue
            
            # Unknown character - skip or keep
            if char.strip():  # If not whitespace
                phonemes.append(char)
            i += 1
        
        # Join phonemes
        result = ''.join(phonemes)
        
        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        # Apply schwa deletion rules (simplified)
        # In Nepali, word-final schwa is often deleted
        result = self._apply_schwa_deletion(result)
        
        return result
    
    
    def _apply_schwa_deletion(self, phoneme_text):
        """
        Apply simplified schwa deletion rules for Nepali
        Word-final schwa (ə) is typically deleted
        """
        
        # Split into words
        words = phoneme_text.split()
        processed_words = []
        
        for word in words:
            # Remove word-final schwa if preceded by consonant
            if word.endswith('ə') and len(word) > 2:
                # Check if there's a consonant before the schwa
                if word[-2] not in 'aeiouəː':
                    word = word[:-1]
            
            processed_words.append(word)
        
        return ' '.join(processed_words)
    
    
    def extract_phoneme_vocabulary(self, texts):
        """
        Extract unique phonemes from texts
        """
        
        all_phonemes = set()
        
        for text in texts:
            phoneme_text = self.grapheme_to_phoneme(text)
            # Add each character as a phoneme
            for char in phoneme_text:
                all_phonemes.add(char)
        
        # Sort for consistent ordering
        return sorted(all_phonemes)
    
    
    def test_conversion(self):
        """
        Test G2P conversion with common Nepali words
        """
        
        print("\n" + "=" * 70)
        print("TESTING G2P CONVERSION")
        print("=" * 70)
        
        test_words = [
            ('नमस्कार', 'namaskaːr'),         # Hello
            ('धन्यवाद', 'dʰanjəwaːd'),        # Thank you
            ('काठमाडौं', 'kaːʈʰəmaːɖau'),     # Kathmandu
            ('नेपाल', 'nepaːl'),              # Nepal
            ('पानी', 'paːniː'),               # Water
            ('खाना', 'kʰaːnaː'),              # Food
            ('किताब', 'kitaːb'),              # Book
            ('स्कूल', 'skuːl'),               # School
            ('मान्छे', 'maːntʃʰe'),           # Person
            ('बच्चा', 'bətʃtʃʰaː'),           # Child
        ]
        
        print("\nTest Conversions:")
        print("-" * 70)
        
        for nepali, expected in test_words:
            result = self.grapheme_to_phoneme(nepali)
            status = "✅" if result == expected else "⚠️"
            print(f"{status} {nepali:15} → {result:20} (expected: {expected})")
        
        print("=" * 70)
    
    
    def process_dataset(self):
        """
        Process the entire dataset and add phoneme representations
        """
        
        print("\n" + "=" * 70)
        print("NEPALI G2P CONVERSION - STEP 4 (CORRECTED)")
        print("=" * 70)
        
        # Load normalized inventory
        inventory_path = self.base_dir / 'data' / 'metadata' / 'data_inventory_normalized.csv'
        
        print(f"\n📖 Loading normalized inventory from: {inventory_path}")
        
        if not inventory_path.exists():
            raise FileNotFoundError(
                f"Normalized inventory not found: {inventory_path}\n"
                f"Please run Step 3 first to generate normalized texts."
            )
        
        df = pd.read_csv(inventory_path, encoding='utf-8')
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"\n🔄 Converting to phonemes...")
        
        # Convert all texts to phonemes
        phoneme_texts = []
        phoneme_lengths = []
        
        for idx, row in df.iterrows():
            if (idx + 1) % 100 == 0:
                print(f"   Progress: {idx+1}/{len(df)}")
            
            normalized_text = row['normalized_text']
            phoneme_text = self.grapheme_to_phoneme(normalized_text)
            
            phoneme_texts.append(phoneme_text)
            phoneme_lengths.append(len(phoneme_text))
        
        # Add to dataframe
        df['phoneme_text'] = phoneme_texts
        df['phoneme_length'] = phoneme_lengths
        
        # Extract phoneme vocabulary
        print(f"\n📊 Extracting phoneme vocabulary...")
        phoneme_vocab = self.extract_phoneme_vocabulary(phoneme_texts)
        
        print(f"✅ Found {len(phoneme_vocab)} unique phonemes")
        
        # Save phoneme vocabulary
        vocab_file = self.base_dir / 'tokenizers' / 'phoneme_vocab.json'
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            'phonemes': phoneme_vocab,
            'count': len(phoneme_vocab),
            'includes_punctuation': True,
            'note': 'Corrected IPA phonemes for Nepali'
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Phoneme vocabulary saved: {vocab_file}")
        
        # Save updated inventory
        output_file = self.base_dir / 'data' / 'metadata' / 'data_inventory_with_phonemes.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Inventory with phonemes saved: {output_file}")
        
        # Generate statistics
        print(f"\n📊 Phoneme Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Average phoneme length: {sum(phoneme_lengths) / len(phoneme_lengths):.1f}")
        print(f"   Min phoneme length: {min(phoneme_lengths)}")
        print(f"   Max phoneme length: {max(phoneme_lengths)}")
        print(f"   Unique phonemes: {len(phoneme_vocab)}")
        
        # Show phoneme distribution
        all_phonemes = ''.join(phoneme_texts)
        phoneme_dist = Counter(all_phonemes)
        
        print(f"\n📝 Top 15 Most Common Phonemes:")
        for phoneme, count in phoneme_dist.most_common(15):
            if phoneme != ' ':  # Skip space
                print(f"   '{phoneme}': {count:,}")
        
        # Sample conversions
        print(f"\n📝 Sample G2P Conversions (first 5):")
        for idx in range(min(5, len(df))):
            orig = df.iloc[idx]['normalized_text'][:60]
            phon = df.iloc[idx]['phoneme_text'][:60]
            print(f"\n   Text: {orig}")
            print(f"   Phonemes: {phon}")
        
        print("\n" + "=" * 70)
        print("✅ STEP 4 COMPLETE!")
        print("=" * 70)
        
        return df, phoneme_vocab


def main():
    """Main execution function"""
    
    # Initialize G2P converter
    g2p = NepaliG2P()
    
    # Run test conversions first
    g2p.test_conversion()
    
    # Process dataset
    df, vocab = g2p.process_dataset()
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Review phoneme conversions: data/metadata/data_inventory_with_phonemes.csv")
    print("   2. Check phoneme vocabulary: tokenizers/phoneme_vocab.json")
    print("   3. Verify test conversions above are accurate")
    print("   4. Ready for Step 5: Tokenizer Creation")


if __name__ == "__main__":
    main()