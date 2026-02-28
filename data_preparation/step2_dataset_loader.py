"""
Nepali TTS Dataset Preprocessing Pipeline
Step 2: Dataset Loading and Verification

This script:
1. Loads the Excel file with transcriptions
2. Verifies all audio files exist
3. Checks audio quality (sample rate, duration, corruption)
4. Generates detailed statistics
5. Creates a data inventory
6. Flags problematic samples
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import audio libraries
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: librosa or soundfile not installed.")
    print("   Install with: pip install librosa soundfile")
    AUDIO_LIBS_AVAILABLE = False


class NepaliTTSDatasetLoader:
    """
    Loads and verifies Nepali TTS dataset
    """
    
    def __init__(self, config_path="nepali_tts_project/configs/config.json"):
        """Initialize loader with configuration"""
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Set paths
        self.base_dir = Path(config_path).parent.parent
        self.raw_data_path = self.base_dir / self.config['dataset']['raw_data_path']
        self.excel_file = self.raw_data_path / self.config['dataset']['excel_file']
        
        # Initialize data structures
        self.df = None
        self.audio_info = []
        self.problematic_files = []
        self.statistics = {}
        
        print("=" * 70)
        print("NEPALI TTS DATASET LOADER - STEP 2")
        print("=" * 70)
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"📁 Raw Data Path: {self.raw_data_path}")
        print(f"📊 Excel File: {self.excel_file}")
        print("=" * 70)
    
    
    def load_excel(self):
        """Load the Excel file with transcriptions"""
        
        print("\n📖 Loading Excel file...")
        
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
        
        # Load Excel
        self.df = pd.read_excel(self.excel_file)
        
        # Verify columns
        audio_col = self.config['dataset']['audio_column']
        text_col = self.config['dataset']['text_column']
        
        if audio_col not in self.df.columns or text_col not in self.df.columns:
            raise ValueError(f"Required columns not found! Expected: {audio_col}, {text_col}")
        
        print(f"✅ Loaded {len(self.df)} rows")
        print(f"   Columns: {list(self.df.columns)}")
        
        # Basic cleaning
        self.df[audio_col] = self.df[audio_col].astype(str).str.strip()
        self.df[text_col] = self.df[text_col].astype(str).str.strip()
        
        # Remove empty rows
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[audio_col, text_col])
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"   Removed {removed} empty rows")
        
        return self.df
    
    
    def verify_audio_files(self):
        """Verify all audio files exist and are valid"""
        
        print("\n🔍 Verifying audio files...")
        
        audio_col = self.config['dataset']['audio_column']
        missing_files = []
        found_files = []
        
        for idx, row in self.df.iterrows():
            audio_id = row[audio_col]
            
            # Try different extensions
            audio_file = None
            for ext in ['.wav', '.WAV', '.mp3', '.flac']:
                potential_file = self.raw_data_path / f"{audio_id}{ext}"
                if potential_file.exists():
                    audio_file = potential_file
                    break
            
            if audio_file is None:
                missing_files.append(audio_id)
            else:
                found_files.append({
                    'index': idx,
                    'audio_id': audio_id,
                    'file_path': audio_file,
                    'extension': audio_file.suffix
                })
        
        print(f"✅ Found: {len(found_files)} audio files")
        
        if missing_files:
            print(f"❌ Missing: {len(missing_files)} audio files")
            print(f"   First 5 missing: {missing_files[:5]}")
            self.problematic_files.extend(missing_files)
        
        # Store found files info
        self.audio_info = found_files
        
        return found_files, missing_files
    
    
    def analyze_audio_quality(self):
        """Analyze audio file quality (requires librosa)"""
        
        if not AUDIO_LIBS_AVAILABLE:
            print("\n⚠️  Skipping audio quality analysis (librosa not available)")
            return None
        
        print("\n🎵 Analyzing audio quality...")
        print("   This may take a few minutes...")
        
        durations = []
        sample_rates = []
        corrupted = []
        text_col = self.config['dataset']['text_column']
        
        for i, item in enumerate(self.audio_info):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{len(self.audio_info)}")
            
            try:
                # Load audio
                y, sr = librosa.load(item['file_path'], sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                durations.append(duration)
                sample_rates.append(sr)
                
                # Update audio info
                item['duration'] = duration
                item['sample_rate'] = sr
                item['num_samples'] = len(y)
                
                # Get corresponding text
                text = self.df.iloc[item['index']][text_col]
                item['text'] = text
                item['text_length'] = len(text)
                
            except Exception as e:
                corrupted.append({
                    'audio_id': item['audio_id'],
                    'error': str(e)
                })
                item['corrupted'] = True
        
        print(f"✅ Analyzed {len(durations)} audio files")
        
        if corrupted:
            print(f"❌ Found {len(corrupted)} corrupted files")
            self.problematic_files.extend([f['audio_id'] for f in corrupted])
        
        # Calculate statistics
        if durations:
            self.statistics['audio'] = {
                'total_files': len(durations),
                'total_duration_seconds': sum(durations),
                'total_duration_hours': sum(durations) / 3600,
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'sample_rates': dict(Counter(sample_rates)),
                'corrupted_count': len(corrupted)
            }
        
        return self.statistics['audio']
    
    
    def analyze_text_statistics(self):
        """Analyze text transcription statistics"""
        
        print("\n📝 Analyzing text statistics...")
        
        text_col = self.config['dataset']['text_column']
        
        # Get text lengths
        text_lengths = []
        word_counts = []
        char_distribution = Counter()
        
        for item in self.audio_info:
            if 'text' in item:
                text = item['text']
            else:
                text = self.df.iloc[item['index']][text_col]
                item['text'] = text
            
            # Character count
            text_lengths.append(len(text))
            
            # Word count (split by spaces)
            words = text.split()
            word_counts.append(len(words))
            
            # Character distribution
            for char in text:
                char_distribution[char] += 1
        
        # Calculate statistics
        self.statistics['text'] = {
            'total_samples': len(text_lengths),
            'total_characters': sum(text_lengths),
            'total_words': sum(word_counts),
            'mean_text_length': np.mean(text_lengths),
            'median_text_length': np.median(text_lengths),
            'min_text_length': np.min(text_lengths),
            'max_text_length': np.max(text_lengths),
            'mean_words': np.mean(word_counts),
            'unique_characters': len(char_distribution),
            'character_distribution': dict(char_distribution.most_common(50))
        }
        
        print(f"✅ Text statistics calculated")
        print(f"   Total words: {self.statistics['text']['total_words']:,}")
        print(f"   Unique characters: {self.statistics['text']['unique_characters']}")
        
        return self.statistics['text']
    
    
    def flag_problematic_samples(self):
        """Flag samples that don't meet quality criteria"""
        
        print("\n🚩 Flagging problematic samples...")
        
        min_duration = self.config['audio_processing']['min_duration']
        max_duration = self.config['audio_processing']['max_duration']
        min_text_length = self.config['text_processing']['min_text_length']
        max_text_length = self.config['text_processing']['max_text_length']
        
        flagged = []
        
        for item in self.audio_info:
            issues = []
            
            # Check duration
            if 'duration' in item:
                if item['duration'] < min_duration:
                    issues.append(f"too_short ({item['duration']:.2f}s)")
                if item['duration'] > max_duration:
                    issues.append(f"too_long ({item['duration']:.2f}s)")
            
            # Check text length
            if 'text_length' in item:
                if item['text_length'] < min_text_length:
                    issues.append(f"text_too_short ({item['text_length']} chars)")
                if item['text_length'] > max_text_length:
                    issues.append(f"text_too_long ({item['text_length']} chars)")
            
            # Check if corrupted
            if item.get('corrupted', False):
                issues.append("corrupted_file")
            
            if issues:
                flagged.append({
                    'audio_id': item['audio_id'],
                    'issues': issues
                })
                item['flagged'] = True
                item['issues'] = issues
        
        print(f"   Flagged: {len(flagged)} samples")
        
        if flagged:
            print(f"   Issue breakdown:")
            issue_types = Counter()
            for f in flagged:
                for issue in f['issues']:
                    issue_type = issue.split('(')[0].strip()
                    issue_types[issue_type] += 1
            
            for issue_type, count in issue_types.most_common():
                print(f"      • {issue_type}: {count}")
        
        return flagged
    
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    
    def save_inventory(self):
        """Save complete data inventory"""
        
        print("\n💾 Saving data inventory...")
        
        # Create inventory DataFrame
        inventory_data = []
        
        for item in self.audio_info:
            row = {
                'audio_id': item['audio_id'],
                'file_path': str(item['file_path']),
                'extension': item['extension'],
                'text': item.get('text', ''),
                'text_length': int(item.get('text_length', 0)),
                'duration': float(item.get('duration', 0)),
                'sample_rate': int(item.get('sample_rate', 0)),
                'flagged': bool(item.get('flagged', False)),
                'issues': '|'.join(item.get('issues', []))
            }
            inventory_data.append(row)
        
        inventory_df = pd.DataFrame(inventory_data)
        
        # Save to metadata folder
        metadata_path = self.base_dir / self.config['paths']['metadata']
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        inventory_file = metadata_path / 'data_inventory.csv'
        inventory_df.to_csv(inventory_file, index=False, encoding='utf-8')
        
        print(f"✅ Inventory saved: {inventory_file}")
        
        # Save statistics as JSON (convert numpy types first)
        stats_file = metadata_path / 'dataset_statistics.json'
        serializable_stats = self.convert_to_json_serializable(self.statistics)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Statistics saved: {stats_file}")
        
        return inventory_df
    
    
    def generate_report(self):
        """Generate a comprehensive report"""
        
        print("\n" + "=" * 70)
        print("📊 DATASET VERIFICATION REPORT")
        print("=" * 70)
        
        # Dataset overview
        print("\n📁 DATASET OVERVIEW:")
        print(f"   Total samples in Excel: {len(self.df)}")
        print(f"   Audio files found: {len(self.audio_info)}")
        print(f"   Problematic files: {len(self.problematic_files)}")
        
        # Audio statistics
        if 'audio' in self.statistics:
            audio_stats = self.statistics['audio']
            print(f"\n🎵 AUDIO STATISTICS:")
            print(f"   Total duration: {audio_stats['total_duration_hours']:.2f} hours")
            print(f"   Average duration: {audio_stats['mean_duration']:.2f} seconds")
            print(f"   Duration range: {audio_stats['min_duration']:.2f}s - {audio_stats['max_duration']:.2f}s")
            print(f"   Sample rates: {audio_stats['sample_rates']}")
        
        # Text statistics
        if 'text' in self.statistics:
            text_stats = self.statistics['text']
            print(f"\n📝 TEXT STATISTICS:")
            print(f"   Total words: {text_stats['total_words']:,}")
            print(f"   Average words per sentence: {text_stats['mean_words']:.1f}")
            print(f"   Unique characters: {text_stats['unique_characters']}")
            print(f"   Text length range: {text_stats['min_text_length']} - {text_stats['max_text_length']} chars")
        
        # Quality flags
        flagged_count = sum(1 for item in self.audio_info if item.get('flagged', False))
        usable_count = len(self.audio_info) - flagged_count
        
        print(f"\n✅ QUALITY ASSESSMENT:")
        print(f"   Usable samples: {usable_count}")
        print(f"   Flagged samples: {flagged_count}")
        print(f"   Usability rate: {(usable_count/len(self.audio_info)*100):.1f}%")
        
        print("\n" + "=" * 70)
        print("✅ STEP 2 COMPLETE!")
        print("=" * 70)
    
    
    def run_complete_verification(self):
        """Run the complete verification pipeline"""
        
        # Step 1: Load Excel
        self.load_excel()
        
        # Step 2: Verify audio files
        self.verify_audio_files()
        
        # Step 3: Analyze audio quality
        self.analyze_audio_quality()
        
        # Step 4: Analyze text
        self.analyze_text_statistics()
        
        # Step 5: Flag problematic samples
        self.flag_problematic_samples()
        
        # Step 6: Save inventory
        self.save_inventory()
        
        # Step 7: Generate report
        self.generate_report()
        
        return self.audio_info, self.statistics


def main():
    """Main execution function"""
    
    # Initialize loader
    loader = NepaliTTSDatasetLoader()
    
    # Run complete verification
    audio_info, statistics = loader.run_complete_verification()
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Review the data inventory: data/metadata/data_inventory.csv")
    print("   2. Check flagged samples if any")
    print("   3. Ready for Step 3: Text Normalization")


if __name__ == "__main__":
    main()