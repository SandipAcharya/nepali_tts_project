"""
Nepali TTS Dataset Preprocessing Pipeline
Step 6: INTEGRATED Audio Preprocessing (Enhanced + Pipeline Integration)

This combines:
- Friend's advanced audio processing functions (noise reduction, SNR filtering)
- Complete dataset processing pipeline
- Backward compatibility with Steps 1-5
- Comprehensive reporting and statistics
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    print("❌ Error: Required audio libraries not installed!")
    print("   Install with: pip install librosa soundfile scipy")
    AUDIO_LIBS_AVAILABLE = False
    exit(1)


class EnhancedNepaliAudioProcessor:
    """
    Professional audio preprocessing combining advanced features with pipeline integration
    
    Features:
    - High-quality resampling (Kaiser windowed)
    - Advanced silence removal with gap reduction
    - Spectral noise reduction
    - SNR-based quality filtering
    - RMS loudness normalization
    - Comprehensive quality checks
    - Full dataset processing pipeline
    - Backward compatible with Steps 1-5
    """
    
    def __init__(self, config_path=None):
        """Initialize enhanced audio processor"""
        
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
        
        if not config_path or not Path(config_path).exists():
            raise FileNotFoundError(
                "Config file not found!\n"
                "Please run: python update_config_step6.py first"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.base_dir = Path(config_path).parent.parent
        self.audio_config = self.config['audio_processing']
        
        # Core audio parameters
        self.target_sr = self.audio_config['sample_rate']
        self.min_duration = self.audio_config['min_duration']
        self.max_duration = self.audio_config['max_duration']
        self.target_db = self.audio_config['target_db']
        
        # Advanced parameters
        self.top_db = self.audio_config.get('vad_silence_threshold', 40)
        self.min_snr = self.audio_config.get('min_snr', 10)
        self.noise_reduce = self.audio_config.get('noise_reduce', True)
        self.max_silence_duration = self.audio_config.get('max_silence_gap', 0.3)
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'total_processed': 0,
            'resampled': 0,
            'trimmed': 0,
            'noise_reduced': 0,
            'normalized': 0,
            'rejected': 0,
            'duration_too_short': 0,
            'duration_too_long': 0,
            'low_amplitude': 0,
            'low_snr': 0
        }
        
        print("✅ Enhanced Nepali Audio Preprocessor initialized")
        print(f"   Target sample rate: {self.target_sr} Hz")
        print(f"   Duration range: {self.min_duration}s - {self.max_duration}s")
        print(f"   Target loudness: {self.target_db} dB")
        print(f"   Minimum SNR: {self.min_snr} dB")
        print(f"   Noise reduction: {'Enabled' if self.noise_reduce else 'Disabled'}")
        print(f"   Advanced trimming: {'Enabled' if self.audio_config.get('use_advanced_trimming', True) else 'Disabled'}")
    
    
    # ========================================================================
    # FRIEND'S ADVANCED PROCESSING FUNCTIONS
    # ========================================================================
    
    def load_audio(self, audio_path):
        """Load audio file with error handling"""
        try:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            return audio, sr
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None, None
    
    
    def convert_to_mono(self, audio):
        """Convert stereo to mono by averaging channels"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        return audio
    
    
    def resample_audio(self, audio, orig_sr):
        """
        Resample audio to target sample rate using high-quality Kaiser windowed method
        """
        if orig_sr == self.target_sr:
            return audio
        
        audio_resampled = librosa.resample(
            audio, 
            orig_sr=orig_sr, 
            target_sr=self.target_sr,
            res_type='kaiser_best'
        )
        
        self.stats['resampled'] += 1
        return audio_resampled
    
    
    def remove_silence_advanced(self, audio, sr):
        """
        Advanced silence removal:
        - Removes leading/trailing silence
        - Reduces long gaps in middle to max 0.3s
        - Preserves natural pauses
        """
        if not self.audio_config.get('use_advanced_trimming', True):
            # Fallback to basic trimming
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
            self.stats['trimmed'] += 1
            return audio_trimmed
        
        # Step 1: Remove leading and trailing silence
        audio_trimmed, _ = librosa.effects.trim(
            audio,
            top_db=self.top_db,
            frame_length=2048,
            hop_length=512
        )
        
        # Check if trimming was too aggressive
        duration_ratio = len(audio_trimmed) / len(audio)
        if duration_ratio < 0.3:
            return audio  # Keep original
        
        # Step 2: Detect non-silent intervals
        intervals = librosa.effects.split(audio_trimmed, top_db=self.top_db)
        
        # Step 3: Reconstruct with limited silence gaps
        max_silence_samples = int(self.max_silence_duration * sr)
        audio_segments = []
        
        for i, (start, end) in enumerate(intervals):
            audio_segments.append(audio_trimmed[start:end])
            
            # Add limited silence between segments (except after last)
            if i < len(intervals) - 1:
                next_start = intervals[i + 1][0]
                silence_duration = next_start - end
                
                if silence_duration > max_silence_samples:
                    # Add only max allowed silence
                    audio_segments.append(np.zeros(max_silence_samples))
                else:
                    # Keep original silence (natural pauses)
                    audio_segments.append(audio_trimmed[end:next_start])
        
        audio_cleaned = np.concatenate(audio_segments) if audio_segments else audio_trimmed
        
        self.stats['trimmed'] += 1
        return audio_cleaned
    
    
    def reduce_noise_spectral(self, audio, sr):
        """
        Spectral noise reduction using frequency-domain filtering
        Removes background noise while preserving speech quality
        """
        if not self.noise_reduce:
            return audio
        
        try:
            # Compute spectrogram
            hop_length = self.audio_config.get('hop_length', 512)
            frame_length = self.audio_config.get('win_length', 2048)
            
            D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
            magnitude, phase = librosa.magphase(D)
            
            # Estimate noise profile from quietest 10%
            noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
            
            # Apply spectral gate (1.5x threshold)
            mask = magnitude > (noise_profile * 1.5)
            magnitude_cleaned = magnitude * mask
            
            # Reconstruct audio (preserve phase)
            D_cleaned = magnitude_cleaned * phase
            audio_cleaned = librosa.istft(D_cleaned, hop_length=hop_length)
            
            self.stats['noise_reduced'] += 1
            return audio_cleaned
            
        except Exception as e:
            print(f"⚠️  Noise reduction failed: {e}, using original audio")
            return audio
    
    
    def normalize_loudness(self, audio):
        """
        Normalize audio to target loudness (-20 dBFS standard)
        """
        if not self.audio_config.get('normalize_audio', True):
            return audio
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        if rms == 0:
            return audio
        
        # Current dBFS
        current_dbfs = 20 * np.log10(rms)
        
        # Calculate gain needed
        gain_db = self.target_db - current_dbfs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        audio_normalized = audio * gain_linear
        
        # Peak limiting (prevent clipping)
        max_val = np.abs(audio_normalized).max()
        if max_val > 0.95:
            audio_normalized = audio_normalized * (0.95 / max_val)
        
        self.stats['normalized'] += 1
        return audio_normalized
    
    
    def calculate_snr(self, audio):
        """
        Calculate Signal-to-Noise Ratio in decibels
        Higher SNR = cleaner recording
        """
        # Calculate energy
        energy = audio ** 2
        threshold = np.percentile(energy, 50)
        
        signal_indices = energy > threshold
        noise_indices = energy <= threshold
        
        # Handle edge cases
        if np.sum(noise_indices) == 0:
            return 100
        
        # Calculate power
        signal_power = np.mean(energy[signal_indices])
        noise_power = np.mean(energy[noise_indices])
        
        if noise_power == 0:
            return 100
        
        # SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    
    def check_audio_quality(self, audio, sr):
        """
        Comprehensive quality checks with SNR filtering
        """
        quality_metrics = {}
        
        # 1. Duration check
        duration = len(audio) / sr
        quality_metrics['duration'] = duration
        
        if duration < self.min_duration:
            self.stats['duration_too_short'] += 1
            return False, f"duration_too_short ({duration:.2f}s)", quality_metrics
        
        if duration > self.max_duration:
            self.stats['duration_too_long'] += 1
            return False, f"duration_too_long ({duration:.2f}s)", quality_metrics
        
        # 2. Amplitude check
        max_amplitude = np.abs(audio).max()
        quality_metrics['max_amplitude'] = float(max_amplitude)
        
        if max_amplitude < 0.001:
            self.stats['low_amplitude'] += 1
            return False, "low_amplitude", quality_metrics
        
        # 3. NaN/Inf check
        if np.isnan(audio).any() or np.isinf(audio).any():
            return False, "invalid_values", quality_metrics
        
        # 4. SNR check (CRITICAL for quality)
        snr = self.calculate_snr(audio)
        quality_metrics['snr'] = float(snr)
        
        if snr < self.min_snr:
            self.stats['low_snr'] += 1
            return False, f"low_snr ({snr:.2f}dB)", quality_metrics
        
        # 5. RMS calculation
        rms = np.sqrt(np.mean(audio**2))
        quality_metrics['rms'] = float(rms)
        quality_metrics['rms_db'] = float(20 * np.log10(rms)) if rms > 0 else -np.inf
        
        return True, "valid", quality_metrics
    
    
    def process_single_audio(self, audio_path, output_path):
        """
        Complete preprocessing pipeline for single audio file
        
        Pipeline:
        1. Load → 2. Mono → 3. Resample → 4. Silence removal
        5. Noise reduction → 6. Normalization → 7. Quality check → 8. Save
        """
        # Load audio
        audio, orig_sr = self.load_audio(audio_path)
        
        if audio is None:
            return False, {"error": "load_failed", "duration": 0, "snr": 0, "rms": 0, "max_amplitude": 0}
        
        orig_duration = len(audio) / orig_sr
        
        try:
            # Step 1: Convert to mono
            audio = self.convert_to_mono(audio)
            
            # Step 2: Resample
            audio = self.resample_audio(audio, orig_sr)
            
            # Step 3: Advanced silence trimming
            audio = self.remove_silence_advanced(audio, self.target_sr)
            
            # Step 4: Spectral noise reduction
            audio = self.reduce_noise_spectral(audio, self.target_sr)
            
            # Step 5: Loudness normalization
            audio = self.normalize_loudness(audio)
            
            # Step 6: Quality check
            is_valid, reason, quality_metrics = self.check_audio_quality(
                audio, self.target_sr
            )
            
            if not is_valid:
                self.stats['rejected'] += 1
                # Return metrics even for rejected files
                return False, {"error": reason, **quality_metrics}
            
            # Step 7: Save processed audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
            
            # Compile info
            final_duration = len(audio) / self.target_sr
            info = {
                "original_sr": int(orig_sr),
                "original_duration": float(orig_duration),
                "final_duration": float(final_duration),
                "sample_rate": int(self.target_sr),
                "n_samples": int(len(audio)),
                "silence_removed": float(orig_duration - final_duration),
                **quality_metrics,
                "valid": True
            }
            
            self.stats['total_processed'] += 1
            return True, info
            
        except Exception as e:
            # Include default metrics for failed processing
            return False, {
                "error": f"processing_failed: {str(e)}", 
                "duration": 0, 
                "snr": 0, 
                "rms": 0, 
                "max_amplitude": 0
            }
    
    
    # ========================================================================
    # DATASET PROCESSING PIPELINE (Integration with Steps 1-5)
    # ========================================================================
    
    def process_dataset(self):
        """
        Process entire dataset with full pipeline integration
        """
        
        print("\n" + "=" * 70)
        print("ENHANCED AUDIO PREPROCESSING - STEP 6")
        print("=" * 70)
        
        # Load inventory from Step 4 (with phonemes)
        inventory_path = self.base_dir / 'data' / 'metadata' / 'data_inventory_with_phonemes.csv'
        
        print(f"\n📖 Loading inventory: {inventory_path}")
        
        if not inventory_path.exists():
            print("❌ Error: Inventory file not found!")
            print("   Please run Steps 1-4 first.")
            return None
        
        df = pd.read_csv(inventory_path, encoding='utf-8')
        
        print(f"✅ Loaded {len(df)} samples")
        
        # Setup directories
        raw_audio_dir = self.base_dir / self.config['dataset']['raw_data_path']
        processed_audio_dir = self.base_dir / self.config['paths']['processed_audio']
        processed_audio_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎵 Processing audio files...")
        print(f"   Input: {raw_audio_dir}")
        print(f"   Output: {processed_audio_dir}")
        print(f"   Features: Noise reduction, SNR filtering, Advanced trimming")
        
        # Process each file
        processing_results = []
        self.stats['total_files'] = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            audio_id = row['audio_id']
            input_path = Path(row['file_path'])
            output_path = processed_audio_dir / f"{audio_id}.wav"
            
            # Process
            success, info = self.process_single_audio(input_path, output_path)
            
            # Store results
            result = {
                'audio_id': audio_id,
                'processed': success,
                'processed_path': str(output_path) if success else None,
                **info
            }
            processing_results.append(result)
        
        # Merge results with inventory
        results_df = pd.DataFrame(processing_results)
        df_merged = df.merge(results_df, on='audio_id', how='left')
        
        # Save updated inventory
        output_file = self.base_dir / 'data' / 'metadata' / 'data_inventory_processed.csv'
        df_merged.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Processing complete!")
        print(f"   Updated inventory: {output_file}")
        
        # Generate comprehensive report
        self.generate_report(df_merged)
        
        return df_merged
    
    
    def generate_report(self, df):
        """
        Generate comprehensive processing report
        """
        
        print("\n" + "=" * 70)
        print("📊 ENHANCED AUDIO PREPROCESSING REPORT")
        print("=" * 70)
        
        # Overall statistics
        total = len(df)
        processed = df['processed'].sum()
        rejected = total - processed
        
        print(f"\n📁 Overall Statistics:")
        print(f"   Total files: {total}")
        print(f"   Successfully processed: {processed} ({processed/total*100:.1f}%)")
        print(f"   Rejected: {rejected} ({rejected/total*100:.1f}%)")
        
        # Processing operations
        print(f"\n🔧 Processing Operations Applied:")
        print(f"   Files resampled: {self.stats['resampled']}")
        print(f"   Silence trimmed: {self.stats['trimmed']}")
        print(f"   Noise reduced: {self.stats['noise_reduced']}")
        print(f"   Normalized: {self.stats['normalized']}")
        
        # Rejection breakdown
        if rejected > 0:
            print(f"\n⚠️  Rejection Breakdown:")
            print(f"   Too short (< {self.min_duration}s): {self.stats['duration_too_short']}")
            print(f"   Too long (> {self.max_duration}s): {self.stats['duration_too_long']}")
            print(f"   Low amplitude: {self.stats['low_amplitude']}")
            print(f"   Low SNR (< {self.min_snr} dB): {self.stats['low_snr']}")
            
            # Analyze actual error types from DataFrame
            failed_df = df[df['processed'] == False]
            if len(failed_df) > 0:
                error_types = failed_df['error'].value_counts()
                print(f"\n   Error Types (from {len(failed_df)} failed files):")
                for error, count in error_types.items():
                    print(f"      • {error}: {count}")
        
        
        # Valid files statistics
        valid_df = df[df['processed'] == True]
        
        if len(valid_df) > 0:
            print(f"\n⏱️  Duration Statistics (Processed Files):")
            print(f"   Mean: {valid_df['final_duration'].mean():.2f}s")
            print(f"   Median: {valid_df['final_duration'].median():.2f}s")
            print(f"   Min: {valid_df['final_duration'].min():.2f}s")
            print(f"   Max: {valid_df['final_duration'].max():.2f}s")
            print(f"   Total: {valid_df['final_duration'].sum()/3600:.2f} hours")
            print(f"   Silence removed: {valid_df['silence_removed'].sum()/60:.2f} minutes")
            
            print(f"\n🔊 Quality Metrics:")
            print(f"   Mean SNR: {valid_df['snr'].mean():.2f} dB")
            print(f"   Min SNR: {valid_df['snr'].min():.2f} dB")
            print(f"   Mean RMS: {valid_df['rms'].mean():.4f}")
            print(f"   Mean peak: {valid_df['max_amplitude'].mean():.4f}")
        
        # Save detailed statistics
        stats_file = self.base_dir / 'data' / 'metadata' / 'audio_preprocessing_stats.json'
        stats_data = {
            'total_files': int(total),
            'processed': int(processed),
            'rejected': int(rejected),
            'success_rate': float(processed / total * 100),
            'processing_operations': self.stats,
            'duration_stats': {
                'mean': float(valid_df['final_duration'].mean()) if len(valid_df) > 0 else 0,
                'median': float(valid_df['final_duration'].median()) if len(valid_df) > 0 else 0,
                'min': float(valid_df['final_duration'].min()) if len(valid_df) > 0 else 0,
                'max': float(valid_df['final_duration'].max()) if len(valid_df) > 0 else 0,
                'total_hours': float(valid_df['final_duration'].sum()/3600) if len(valid_df) > 0 else 0,
                'silence_removed_minutes': float(valid_df['silence_removed'].sum()/60) if len(valid_df) > 0 else 0
            },
            'quality_metrics': {
                'mean_snr': float(valid_df['snr'].mean()) if len(valid_df) > 0 else 0,
                'min_snr': float(valid_df['snr'].min()) if len(valid_df) > 0 else 0,
                'mean_rms': float(valid_df['rms'].mean()) if len(valid_df) > 0 else 0,
                'mean_peak': float(valid_df['max_amplitude'].mean()) if len(valid_df) > 0 else 0
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Detailed statistics saved: {stats_file}")
        
        print("\n" + "=" * 70)
        print("✅ STEP 6 COMPLETE - ENHANCED AUDIO PROCESSING!")
        print("=" * 70)


def main():
    """Main execution function"""
    
    if not AUDIO_LIBS_AVAILABLE:
        print("❌ Cannot proceed without audio libraries!")
        print("   Install: pip install librosa soundfile scipy")
        return
    
    print("\n" + "=" * 70)
    print("NEPALI TTS - ENHANCED AUDIO PREPROCESSING")
    print("Combining Advanced Features with Full Pipeline Integration")
    print("=" * 70)
    
    try:
        # Initialize processor
        processor = EnhancedNepaliAudioProcessor()
        
        # Process dataset
        df = processor.process_dataset()
        
        if df is not None:
            print("\n🎯 NEXT STEPS:")
            print("   1. Review processed audio: data/processed/")
            print("   2. Check stats: data/metadata/audio_preprocessing_stats.json")
            print("   3. Listen to samples to verify quality")
            print("   4. Ready for Step 7: Dataset Splitting (Train/Valid/Test)")
            
            print("\n📊 Processing Summary:")
            print(f"   ✅ Processed: {processor.stats['total_processed']} files")
            print(f"   ❌ Rejected: {processor.stats['rejected']} files")
            print(f"   🔇 Noise reduced: {processor.stats['noise_reduced']} files")
            print(f"   ✂️  Trimmed: {processor.stats['trimmed']} files")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Solution:")
        print("   Run: python update_config_step6.py")
        print("   Then run this script again")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()