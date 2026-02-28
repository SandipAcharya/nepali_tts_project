"""
Nepali TTS Dataset Preprocessing Pipeline
Step 1: Project Structure and Configuration Setup

This script creates the complete folder structure and configuration file
for the Nepali TTS preprocessing pipeline.
"""

import os
import json
from pathlib import Path

def create_project_structure(base_dir="nepali_tts_project"):
    """
    Creates the complete folder structure for TTS preprocessing
    
    Structure:
    nepali_tts_project/
    ├── data/
    │   ├── raw/              # Original audio + Excel file
    │   ├── processed/        # Processed audio files
    │   ├── metadata/         # Generated metadata files
    │   └── splits/           # Train/valid/test splits
    ├── preprocessing/
    │   ├── text/             # Text normalization modules
    │   ├── audio/            # Audio processing modules
    │   └── g2p/              # Grapheme-to-phoneme converter
    ├── tokenizers/           # Saved tokenizer files
    ├── configs/              # Configuration files
    ├── checkpoints/          # Model checkpoints (for later)
    ├── logs/                 # Training logs
    └── outputs/              # Generated audio samples
    """
    
    # Define all directories
    directories = [
        "data/raw",
        "data/processed",
        "data/metadata",
        "data/splits",
        "preprocessing/text",
        "preprocessing/audio",
        "preprocessing/g2p",
        "tokenizers",
        "configs",
        "checkpoints",
        "logs",
        "outputs"
    ]
    
    # Create base directory
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create all subdirectories
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith("preprocessing/"):
            init_file = dir_path / "__init__.py"
            init_file.touch()
    
    print(f" Project structure created at: {base_path.absolute()}")
    return base_path

"""
def create_config_file(base_dir="nepali_tts_project"):
    
    #Creates a comprehensive configuration file for the pipeline
    
    
    config = {
        "dataset": {
            "name": "nepali_tts_dataset",
            "raw_data_path": "data/raw",
            "excel_file": "nepali_data.xlsx",  # Update with your actual filename
            "audio_column": "audio_id",
            "text_column": "sentence",
            "sample_rate": 22050,  # Standard for TTS
            "total_samples": 2740
        },
        
        "text_processing": {
            "language": "nepali",
            "remove_punctuation": False,  # Keep some for prosody
            "normalize_numbers": True,
            "convert_to_phonemes": True,
            "lowercase": False,  # Nepali doesn't have case
            "max_text_length": 200,
            "min_text_length": 5
        },
        
        "audio_processing": {
            "sample_rate": 22050,
            "trim_silence": True,
            "silence_threshold_db": -40,
            "normalize_audio": True,
            "target_db": -20,
            "min_duration": 1.0,  # seconds
            "max_duration": 15.0,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
            "fmin": 0,
            "fmax": 8000
        },
        
        "dataset_split": {
            "train_ratio": 0.85,
            "valid_ratio": 0.10,
            "test_ratio": 0.05,
            "random_seed": 42
        },
        
        "tokenizer": {
            "type": "character",  # or "phoneme"
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_blank": True  # For VITS
        },
        
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0002,
            "num_epochs": 1000,
            "save_every": 50,
            "eval_every": 10
        },
        
        "paths": {
            "processed_audio": "data/processed",
            "metadata": "data/metadata",
            "train_file": "data/splits/train.txt",
            "valid_file": "data/splits/valid.txt",
            "test_file": "data/splits/test.txt",
            "tokenizer_path": "tokenizers/tokenizer.json",
            "checkpoint_dir": "checkpoints"
        }
    }
    
    # Save configuration
    base_path = Path(base_dir)
    config_path = base_path / "configs" / "config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f" Configuration file created at: {config_path}")
    return config
"""
def create_config_file(base_dir="nepali_tts_project"):
    """
    Creates comprehensive configuration for Nepali TTS preprocessing and training
    """
    
    config = {
        "dataset": {
            "name": "nepali_tts_dataset",
            "raw_data_path": "data/raw",
            "excel_file": "nepali_data.xlsx",
            "audio_column": "audio_id",
            "text_column": "sentence",
            "sample_rate": 22050,
            "total_samples": 2740,
            "n_speakers": 1,
            "speaker_id": 0
        },
        
        "text_processing": {
            "language": "nepali",
            "cleaners": ["nepali_cleaners"],  # Will implement in Step 3
            "remove_punctuation": False,
            "normalize_numbers": True,
            "normalize_dates": True,
            "normalize_english": True,
            "convert_to_phonemes": True,
            "phoneme_probability": 0.5,  # Mix of graphemes and phonemes
            "lowercase": False,
            "max_text_length": 200,
            "min_text_length": 5,
            # Nepali number rules (Bangla-inspired)
            "number_system": {
                "use_devanagari_digits": True,
                "expand_numbers": True,
                "expand_dates": True
            }
        },
        
        "audio_processing": {
            "sample_rate": 22050,
            "use_webrtc_vad": True,
            "vad_silence_threshold": -40,
            "trim_silence": True,
            "loudness_normalization": True,
            "target_lufs": -16.0,   # Professional loudness standard
            "normalize_audio": True,
            "target_db": -20,
            "min_duration": 1.0,
            "max_duration": 15.0,
            # Temporal parameters (Bangla paper: 50ms window, 12.5ms hop)
            "hop_length": 256,      # 256/22050 ≈ 11.6ms
            "win_length": 1024,     # 1024/22050 ≈ 46.4ms
            "n_fft": 1024,
            "n_mels": 80,
            "fmin": 0,
            "fmax": 8000,
            "resolution_bits": 16
        },
        
        "tokenizer": {
            "type": "character",  # Fixed: singular form
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "blank_token": "<blank>",  # For VITS CTC loss
            "add_blank": True,
            # Character set configuration
            "character_config": {
                "nepali_set": "characters/nepali_chars.json"
            }
        },
        
        "dataset_split": {
            "train_ratio": 0.85,
            "valid_ratio": 0.10,
            "test_ratio": 0.05,
            "random_seed": 42,
            "shuffle": True
        },
        
        "dataset_formatting": {
            "use_formatter": True,
            "metadata_file": "data/metadata/metadata.csv",
            "formatter_type": "vits",
            "include_speaker_name": True,
            "format": "wav|text|speaker_id"  # VITS format
        },
        
        "training": {
            # GPU-optimized settings (adjust based on your hardware)
            "batch_size": 16,              # Safe for 8GB GPU
            "learning_rate": 0.0002,
            "num_epochs": 1000,            # Start conservatively
            "total_steps": 100000,         # ~685 epochs worth
            "save_every": 1000,            # Checkpoint every 1000 steps
            "eval_every": 500,             # Evaluate every 500 steps
            "warmup_steps": 4000,          # Learning rate warmup
            "gradient_clip": 1.0,          # Gradient clipping
            # Advanced settings
            "fp16_run": False,             # Set True if GPU supports mixed precision
            "grad_accum_steps": 1,         # Increase if batch_size too large
            "log_every": 100,              # Log frequency
            # Scaling guide for your reference:
            # 8GB GPU:  batch_size=16
            # 12GB GPU: batch_size=24-32
            # 16GB GPU: batch_size=48-64
            # 24GB GPU: batch_size=64+ (like Bangla paper)
        },
        
        "paths": {
            "processed_audio": "data/processed",
            "metadata": "data/metadata",
            "train_file": "data/splits/train.txt",
            "valid_file": "data/splits/valid.txt",
            "test_file": "data/splits/test.txt",
            "tokenizer_path": "tokenizers/tokenizer.json",
            "character_set": "tokenizers/characters.json",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs"
        },
        
        # Additional metadata for tracking
        "metadata": {
            "version": "1.0",
            "created_for": "Nepali VITS TTS",
            "based_on": "Bangla TTS methodology",
            "notes": "Initial configuration - adjust training params based on GPU"
        }
    }
    
    # Save configuration
    base_path = Path(base_dir)
    config_path = base_path / "configs" / "config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f" Nepali TTS Configuration created at: {config_path}")
    print(f"\n Configuration Summary:")
    print(f"   • Dataset: {config['dataset']['total_samples']} samples")
    print(f"   • Speakers: {config['dataset']['n_speakers']}")
    print(f"   • Sample Rate: {config['audio_processing']['sample_rate']} Hz")
    print(f"   • Batch Size: {config['training']['batch_size']}")
    print(f"   • Training Steps: {config['training']['total_steps']}")
    
    return config
def create_readme(base_dir="nepali_tts_project"):
    """
    Creates a README file with project information
    """
    
    readme_content = """# Nepali TTS Dataset Preprocessing Pipeline

## Overview
This project implements a complete preprocessing pipeline for Nepali Text-to-Speech (TTS) 
dataset preparation, following VITS architecture requirements.

## Project Structure
```
nepali_tts_project/
├── data/                 # Dataset files
├── preprocessing/        # Processing modules
├── tokenizers/          # Tokenizer files
├── configs/             # Configuration files
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── outputs/             # Generated samples
```

## Dataset Information
- Total Samples: 2740
- Language: Nepali (नेपाली)
- Format: WAV audio + Excel transcriptions

## Processing Steps
1. ✅ Project structure setup
2. ⏳ Dataset loading and verification
3. ⏳ Text normalization
4. ⏳ Grapheme-to-phoneme conversion
5. ⏳ Audio preprocessing
6. ⏳ Tokenizer creation
7. ⏳ Dataset splitting
8. ⏳ Metadata generation

## Usage
Follow the step-by-step preprocessing scripts in order.

## Requirements
- Python 3.8+
- pandas
- numpy
- librosa
- scipy
- pydub

## Author
Nepali TTS Project Team
"""
    
    base_path = Path(base_dir)
    readme_path = base_path / "README.md"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ README created at: {readme_path}")


def main():
    """
    Main function to setup the project
    """
    print("=" * 60)
    print("NEPALI TTS PREPROCESSING PIPELINE - STEP 1")
    print("Setting up project structure and configuration")
    print("=" * 60)
    print()
    
    # Create project structure
    base_path = create_project_structure()
    print()
    
    # Create configuration file
    config = create_config_file()
    print()
    
    # Create README
    create_readme()
    print()
    
    print("=" * 60)
    print("✅ STEP 1 COMPLETE!")
    print("=" * 60)
    print()
    print("NEXT STEPS:")
    print("1. Copy your TTS/ folder contents to: data/raw/")
    print("2. Update 'excel_file' in configs/config.json with your actual filename")
    print("3. Run Step 2: Dataset loading and verification")
    print()
    print("📁 Your project is ready at:", base_path.absolute())
    

if __name__ == "__main__":
    main()
    print("\n✅ Configuration file ready for Step 2!")