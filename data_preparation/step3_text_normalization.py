"""
Nepali TTS Dataset Preprocessing Pipeline
Step 3: Text Normalization for Nepali (Devanagari Script)

This script handles:
1. Devanagari number normalization (०-९ to words)
2. English number normalization (0-9 to Nepali words)
3. Punctuation handling
4. Special character removal
5. English word normalization
6. Date and time normalization
7. Currency and measurement normalization
"""

import re
import json
import unicodedata
from pathlib import Path
import pandas as pd


class NepaliTextNormalizer:
    """
    Comprehensive text normalizer for Nepali TTS
    """
    
    def __init__(self, config_path="configs/config.json"):
        """Initialize normalizer with configuration"""
        
        # Resolve absolute path
        if not Path(config_path).is_absolute():
            # If running from project root
            config_path = Path(config_path)
            if not config_path.exists():
                # Try from nepali_tts_project subfolder
                config_path = Path("nepali_tts_project") / config_path
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.text_config = self.config['text_processing']
        self.base_dir = Path(config_path).parent.parent
        
        # Devanagari digits to English digits mapping
        self.devanagari_to_english = str.maketrans('०१२३४५६७८९', '0123456789')
        
        # English digits to Devanagari mapping
        self.english_to_devanagari = str.maketrans('0123456789', '०१२३४५६७८९')
        
        # Number words in Nepali
        self.nepali_numbers = {
            '0': 'शून्य',
            '1': 'एक',
            '2': 'दुई',
            '3': 'तीन',
            '4': 'चार',
            '5': 'पाँच',
            '6': 'छ',
            '7': 'सात',
            '8': 'आठ',
            '9': 'नौ',
            '10': 'दस',
            '11': 'एघार',
            '12': 'बाह्र',
            '13': 'तेह्र',
            '14': 'चौध',
            '15': 'पन्ध्र',
            '16': 'सोह्र',
            '17': 'सत्र',
            '18': 'अठार',
            '19': 'उन्नाइस',
            '20': 'बीस',
            '21': 'एक्काइस',
            '22': 'बाइस',
            '23': 'तेइस',
            '24': 'चौबीस',
            '25': 'पच्चीस',
            '26': 'छब्बीस',
            '27': 'सत्ताइस',
            '28': 'अट्ठाइस',
            '29': 'उनन्तीस',
            '30': 'तीस',
            '40': 'चालीस',
            '50': 'पचास',
            '60': 'साठी',
            '70': 'सत्तरी',
            '80': 'असी',
            '90': 'नब्बे',
            '100': 'सय',
            '1000': 'हजार',
            '100000': 'लाख',
            '10000000': 'करोड'
        }
        
        # Common English words to Nepali
        self.english_to_nepali = {
            'january': 'जनवरी',
            'february': 'फेब्रुअरी',
            'march': 'मार्च',
            'april': 'अप्रिल',
            'may': 'मे',
            'june': 'जुन',
            'july': 'जुलाई',
            'august': 'अगस्ट',
            'september': 'सेप्टेम्बर',
            'october': 'अक्टोबर',
            'november': 'नोभेम्बर',
            'december': 'डिसेम्बर',
            'monday': 'सोमबार',
            'tuesday': 'मंगलबार',
            'wednesday': 'बुधबार',
            'thursday': 'बिहीबार',
            'friday': 'शुक्रबार',
            'saturday': 'शनिबार',
            'sunday': 'आइतबार',
            'am': 'बिहान',
            'pm': 'साँझ',
            'rs': 'रुपैयाँ',
            'rupees': 'रुपैयाँ',
            'km': 'किलोमिटर',
            'kg': 'किलोग्राम',
            'mr': 'श्री',
            'mrs': 'श्रीमती',
            'dr': 'डा'
        }
        
        print("✅ Nepali Text Normalizer initialized")
    
    
    def normalize_devanagari_digits(self, text):
        """Convert Devanagari digits (०-९) to words"""
        
        # Find all Devanagari numbers
        devanagari_pattern = re.compile(r'[०-९]+')
        
        def replace_devanagari_number(match):
            # Convert Devanagari to English digits first
            dev_num = match.group()
            eng_num = dev_num.translate(self.devanagari_to_english)
            # Then convert to words
            return self.number_to_nepali_words(eng_num)
        
        return devanagari_pattern.sub(replace_devanagari_number, text)
    
    
    def normalize_english_digits(self, text):
        """Convert English digits (0-9) to Nepali words"""
        
        # Find all English numbers
        english_pattern = re.compile(r'\b\d+\b')
        
        def replace_english_number(match):
            num = match.group()
            return self.number_to_nepali_words(num)
        
        return english_pattern.sub(replace_english_number, text)
    
    
    def number_to_nepali_words(self, num_str):
        """Convert a number string to Nepali words"""
        
        try:
            num = int(num_str)
        except ValueError:
            return num_str
        
        # Handle special cases
        if num == 0:
            return 'शून्य'
        
        if num < 0:
            return 'ऋण ' + self.number_to_nepali_words(str(abs(num)))
        
        # Direct lookup for small numbers
        if num_str in self.nepali_numbers:
            return self.nepali_numbers[num_str]
        
        # Handle compound numbers
        if num < 100:
            # Numbers like 31-39, 41-49, etc.
            tens = (num // 10) * 10
            ones = num % 10
            
            if tens == 30 and ones > 0:
                return ['एकतीस', 'बत्तीस', 'तेत्तीस', 'चौतीस', 'पैँतीस', 
                        'छत्तीस', 'सैँतीस', 'अठतीस', 'उनन्चालीस'][ones - 1]
            elif tens == 40 and ones > 0:
                return ['एकचालीस', 'बयालीस', 'त्रिचालीस', 'चवालीस', 'पैँतालीस',
                        'छयालीस', 'सच्चालीस', 'अठचालीस', 'उनन्पचास'][ones - 1]
            elif tens == 50 and ones > 0:
                return ['एकाउन्न', 'बाउन्न', 'त्रिपन्न', 'चउन्न', 'पच्पन्न',
                        'छपन्न', 'सन्ताउन्न', 'अन्ठाउन्न', 'उनन्साठी'][ones - 1]
            elif tens == 60 and ones > 0:
                return ['एकसट्ठी', 'बयसट्ठी', 'त्रिसट्ठी', 'चौंसट्ठी', 'पैंसट्ठी',
                        'छयसट्ठी', 'सतसट्ठी', 'अठसट्ठी', 'उनन्सत्तरी'][ones - 1]
            elif tens == 70 and ones > 0:
                return ['एकहत्तर', 'बहत्तर', 'त्रिहत्तर', 'चौहत्तर', 'पचहत्तर',
                        'छयहत्तर', 'सतहत्तर', 'अठहत्तर', 'उनासी'][ones - 1]
            elif tens == 80 and ones > 0:
                return ['एकासी', 'बयासी', 'त्रियासी', 'चौरासी', 'पचासी',
                        'छयासी', 'सतासी', 'अठासी', 'उनान्नब्बे'][ones - 1]
            elif tens == 90 and ones > 0:
                return ['एकान्नब्बे', 'बयान्नब्बे', 'त्रियान्नब्बे', 'चौरान्नब्बे', 'पन्चान्नब्बे',
                        'छयान्नब्बे', 'सन्तान्नब्बे', 'अन्ठान्नब्बे', 'उनान्सय'][ones - 1]
            else:
                tens_word = self.nepali_numbers.get(str(tens), '')
                ones_word = self.nepali_numbers.get(str(ones), '')
                return f"{tens_word} {ones_word}".strip()
        
        # Handle hundreds
        if num < 1000:
            hundreds = num // 100
            remainder = num % 100
            
            if hundreds == 1:
                result = 'एक सय'
            else:
                result = self.nepali_numbers.get(str(hundreds), str(hundreds)) + ' सय'
            
            if remainder > 0:
                result += ' ' + self.number_to_nepali_words(str(remainder))
            
            return result
        
        # Handle thousands (simplified for common cases)
        if num < 100000:
            thousands = num // 1000
            remainder = num % 1000
            
            result = self.number_to_nepali_words(str(thousands)) + ' हजार'
            
            if remainder > 0:
                result += ' ' + self.number_to_nepali_words(str(remainder))
            
            return result
        
        # For larger numbers, return as is or implement full logic
        return num_str
    
    
    def normalize_punctuation(self, text):
        """Normalize punctuation for better TTS"""
        
        # Keep important punctuation for prosody
        # Replace English punctuation with Nepali equivalents where applicable
        
        # Multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'\.{2,}', '।', text)  # Multiple periods to Devanagari full stop
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'!+', '!', text)
        
        # Normalize quotes
        text = text.replace('"', '').replace("'", '')
        text = text.replace('"', '').replace('"', '')
        text = text.replace(''', '').replace(''', '')
        
        return text.strip()
    
    
    def remove_special_characters(self, text):
        """Remove unwanted special characters while keeping Devanagari"""
        
        # Define allowed characters
        # Devanagari Unicode range: U+0900 to U+097F
        # Keep spaces and basic punctuation
        
        allowed_pattern = re.compile(r'[^\u0900-\u097F\s।?!,\-]')
        text = allowed_pattern.sub('', text)
        
        return text
    
    
    def normalize_english_words(self, text):
        """Normalize common English words to Nepali"""
        
        if not self.text_config.get('normalize_english', True):
            return text
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        for eng, nep in self.english_to_nepali.items():
            # Match whole words only
            pattern = r'\b' + eng + r'\b'
            text = re.sub(pattern, nep, text, flags=re.IGNORECASE)
        
        return text
    
    
    def normalize_dates(self, text):
        """Normalize date formats"""
        
        if not self.text_config.get('normalize_dates', True):
            return text
        
        # Pattern: DD/MM/YYYY or DD-MM-YYYY
        date_pattern = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})')
        
        def replace_date(match):
            day, month, year = match.groups()
            day_word = self.number_to_nepali_words(day)
            month_word = self.number_to_nepali_words(month)
            year_word = self.number_to_nepali_words(year)
            return f"{day_word} {month_word} {year_word}"
        
        return date_pattern.sub(replace_date, text)
    
    
    def clean_text(self, text):
        """Main cleaning pipeline"""
        
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Step 1: Normalize Unicode (decompose and recompose)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Normalize English words
        text = self.normalize_english_words(text)
        
        # Step 3: Normalize dates
        text = self.normalize_dates(text)
        
        # Step 4: Normalize numbers
        if self.text_config.get('normalize_numbers', True):
            # First Devanagari, then English
            text = self.normalize_devanagari_digits(text)
            text = self.normalize_english_digits(text)
        
        # Step 5: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 6: Remove unwanted special characters
        text = self.remove_special_characters(text)
        
        # Step 7: Final cleanup
        text = ' '.join(text.split())  # Remove extra whitespace
        text = text.strip()
        
        return text
    
    
    def process_dataset(self, inventory_file=None):
        """Process entire dataset and create normalized version"""
        
        print("\n" + "=" * 70)
        print("NEPALI TEXT NORMALIZATION - STEP 3")
        print("=" * 70)
        
        # Construct path properly
        if inventory_file is None:
            inventory_path = self.base_dir / 'data' / 'metadata' / 'data_inventory.csv'
        else:
            inventory_path = Path(inventory_file)
        
        print(f"\n📖 Loading inventory from: {inventory_path}")
        print(f"   Absolute path: {inventory_path.absolute()}")
        
        # Check if file exists
        if not inventory_path.exists():
            raise FileNotFoundError(
                f"Inventory file not found: {inventory_path}\n"
                f"Make sure you've run Step 2 first to generate the inventory file."
            )
        
        # Load inventory
        df = pd.read_csv(inventory_path, encoding='utf-8')
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"\n🔄 Normalizing text...")
        
        # Normalize all texts
        normalized_texts = []
        issues = []
        
        for idx, row in df.iterrows():
            if (idx + 1) % 100 == 0:
                print(f"   Progress: {idx+1}/{len(df)}")
            
            original_text = row['text']
            normalized_text = self.clean_text(original_text)
            
            # Check if normalization resulted in empty text
            if not normalized_text or len(normalized_text) < 3:
                issues.append({
                    'audio_id': row['audio_id'],
                    'original': original_text,
                    'normalized': normalized_text,
                    'issue': 'normalized_to_empty'
                })
            
            normalized_texts.append(normalized_text)
        
        # Add normalized text to dataframe
        df['normalized_text'] = normalized_texts
        
        # Save normalized inventory
        normalized_file = self.base_dir / 'data' / 'metadata' / 'data_inventory_normalized.csv'
        normalized_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(normalized_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Normalized inventory saved: {normalized_file}")
        
        # Report issues
        if issues:
            print(f"\n⚠️  Found {len(issues)} samples with normalization issues")
            issues_file = self.base_dir / 'data' / 'metadata' / 'normalization_issues.json'
            with open(issues_file, 'w', encoding='utf-8') as f:
                json.dump(issues, f, indent=4, ensure_ascii=False)
            print(f"   Issues saved to: {issues_file}")
        
        # Generate statistics
        print(f"\n📊 Normalization Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Successfully normalized: {len(df) - len(issues)}")
        print(f"   Issues found: {len(issues)}")
        
        # Sample comparisons
        print(f"\n📝 Sample Normalizations (first 5):")
        for idx in range(min(5, len(df))):
            orig = df.iloc[idx]['text'][:60]
            norm = df.iloc[idx]['normalized_text'][:60]
            print(f"\n   Original: {orig}...")
            print(f"   Normalized: {norm}...")
        
        print("\n" + "=" * 70)
        print("✅ STEP 3 COMPLETE!")
        print("=" * 70)
        
        return df


def main():
    """Main execution function"""
    
    # Detect if we're in the project root or need to adjust path
    config_paths = [
        "configs/config.json",                           # If in project root
        "nepali_tts_project/configs/config.json"         # If outside project
    ]
    
    config_path = None
    for path in config_paths:
        if Path(path).exists():
            config_path = path
            break
    
    if config_path is None:
        print("❌ Error: Cannot find config.json file!")
        print("   Make sure you're running from the project root directory.")
        print("   Or config.json exists in nepali_tts_project/configs/")
        return
    
    print(f"📁 Using config: {config_path}")
    
    # Initialize normalizer
    normalizer = NepaliTextNormalizer(config_path=config_path)
    
    # Process entire dataset
    df = normalizer.process_dataset()
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Review normalized texts in: data/metadata/data_inventory_normalized.csv")
    print("   2. Check any issues in: data/metadata/normalization_issues.json")
    print("   3. Ready for Step 4: Grapheme-to-Phoneme (G2P) Conversion")


if __name__ == "__main__":
    main()