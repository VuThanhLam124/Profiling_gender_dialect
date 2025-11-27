"""
Data Preparation Script for Speaker Profiling

This script prepares audio data for training:
1. Validates audio files
2. Generates metadata CSV
3. Computes statistics
4. Optionally splits data

Usage:
    python prepare_data.py --audio_dir path/to/audio --output_dir path/to/output
    python prepare_data.py --audio_dir path/to/audio --output_dir path/to/output --split
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils import setup_logging, get_logger


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse audio filename to extract metadata
    
    Expected format: {speaker}_{gender}_{dialect}_{utterance_id}.wav
    Example: spk001_male_north_0001.wav
    
    Args:
        filename: Audio filename
    
    Returns:
        Dictionary with speaker, gender, dialect info or None if invalid
    """
    try:
        name = Path(filename).stem
        parts = name.split('_')
        
        if len(parts) >= 4:
            speaker = parts[0]
            gender = parts[1].capitalize()
            dialect = parts[2].capitalize()
            
            if gender not in ['Male', 'Female']:
                return None
            if dialect not in ['North', 'Central', 'South']:
                return None
            
            return {
                'speaker': speaker,
                'gender': gender,
                'dialect': dialect
            }
    except Exception:
        pass
    return None


def validate_audio(
    audio_path: Path,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    sampling_rate: int = 16000
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate audio file
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        sampling_rate: Expected sampling rate
    
    Returns:
        Tuple of (is_valid, duration, error_message)
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
        duration = len(audio) / sr
        
        if duration < min_duration:
            return False, duration, f"Too short ({duration:.2f}s < {min_duration}s)"
        if duration > max_duration:
            return False, duration, f"Too long ({duration:.2f}s > {max_duration}s)"
        
        if np.max(np.abs(audio)) < 1e-6:
            return False, duration, "Silent audio"
        
        return True, duration, None
        
    except Exception as e:
        return False, None, str(e)


def scan_audio_directory(
    audio_dir: Path,
    extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
) -> List[Path]:
    """
    Scan directory for audio files
    
    Args:
        audio_dir: Directory to scan
        extensions: List of valid audio extensions
    
    Returns:
        List of audio file paths
    """
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
        audio_files.extend(audio_dir.glob(f'*{ext.upper()}'))
    return sorted(audio_files)


def prepare_metadata(
    audio_dir: str,
    output_dir: str,
    metadata_file: Optional[str] = None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    sampling_rate: int = 16000,
    validate: bool = True
) -> pd.DataFrame:
    """
    Prepare metadata from audio directory
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Output directory for metadata
        metadata_file: Optional existing metadata file
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        sampling_rate: Audio sampling rate
        validate: Whether to validate audio files
    
    Returns:
        DataFrame with metadata
    """
    logger = get_logger()
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if metadata_file and os.path.exists(metadata_file):
        logger.info(f"Loading existing metadata from {metadata_file}")
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    logger.info(f"Scanning audio directory: {audio_dir}")
    audio_files = scan_audio_directory(audio_dir)
    logger.info(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    records = []
    invalid_files = []
    parse_errors = []
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        filename = audio_path.name
        
        metadata = parse_filename(filename)
        if metadata is None:
            parse_errors.append(filename)
            continue
        
        if validate:
            is_valid, duration, error = validate_audio(
                audio_path, min_duration, max_duration, sampling_rate
            )
            if not is_valid:
                invalid_files.append((filename, error))
                continue
            metadata['duration'] = duration
        
        metadata['audio_name'] = filename
        records.append(metadata)
    
    if parse_errors:
        logger.warning(f"Could not parse {len(parse_errors)} filenames")
        error_file = output_dir / 'parse_errors.txt'
        with open(error_file, 'w') as f:
            f.write('\n'.join(parse_errors))
        logger.info(f"Parse errors saved to {error_file}")
    
    if invalid_files:
        logger.warning(f"Found {len(invalid_files)} invalid audio files")
        error_file = output_dir / 'invalid_audio.txt'
        with open(error_file, 'w') as f:
            for filename, error in invalid_files:
                f.write(f"{filename}: {error}\n")
        logger.info(f"Invalid files saved to {error_file}")
    
    df = pd.DataFrame(records)
    logger.info(f"Created metadata with {len(df)} valid samples")
    
    return df


def compute_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute dataset statistics
    
    Args:
        df: Metadata DataFrame
    
    Returns:
        Dictionary with statistics
    """
    logger = get_logger()
    
    stats = {
        'total_samples': len(df),
        'total_speakers': df['speaker'].nunique(),
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'dialect_distribution': df['dialect'].value_counts().to_dict(),
    }
    
    if 'duration' in df.columns:
        stats['duration'] = {
            'total_hours': df['duration'].sum() / 3600,
            'mean_seconds': df['duration'].mean(),
            'min_seconds': df['duration'].min(),
            'max_seconds': df['duration'].max()
        }
    
    gender_dialect = df.groupby(['gender', 'dialect']).size().unstack(fill_value=0)
    stats['gender_dialect_distribution'] = gender_dialect.to_dict()
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Total speakers: {stats['total_speakers']}")
    logger.info(f"  Gender distribution: {stats['gender_distribution']}")
    logger.info(f"  Dialect distribution: {stats['dialect_distribution']}")
    
    if 'duration' in stats:
        logger.info(f"  Total duration: {stats['duration']['total_hours']:.2f} hours")
        logger.info(f"  Mean duration: {stats['duration']['mean_seconds']:.2f} seconds")
    
    return stats


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by: str = 'speaker'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets using speaker-based split
    
    Args:
        df: Metadata DataFrame
        test_size: Test set ratio
        val_size: Validation set ratio
        random_state: Random seed
        stratify_by: Column to stratify by ('speaker' for speaker-based split)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger = get_logger()
    
    if stratify_by == 'speaker':
        speakers = df['speaker'].unique()
        
        train_speakers, test_speakers = train_test_split(
            speakers,
            test_size=test_size,
            random_state=random_state
        )
        
        train_speakers, val_speakers = train_test_split(
            train_speakers,
            test_size=val_size / (1 - test_size),
            random_state=random_state
        )
        
        train_df = df[df['speaker'].isin(train_speakers)].reset_index(drop=True)
        val_df = df[df['speaker'].isin(val_speakers)].reset_index(drop=True)
        test_df = df[df['speaker'].isin(test_speakers)].reset_index(drop=True)
        
        assert len(set(train_speakers) & set(val_speakers)) == 0
        assert len(set(train_speakers) & set(test_speakers)) == 0
        assert len(set(val_speakers) & set(test_speakers)) == 0
        
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size / (1 - test_size), random_state=random_state
        )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prepare data for Speaker Profiling")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for metadata"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Existing metadata CSV file (optional)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split data into train/val/test"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Audio sampling rate (default: 16000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip audio validation"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'prepare_data.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("=" * 50)
    logger.info("DATA PREPARATION")
    logger.info("=" * 50)
    logger.info(f"Audio directory: {args.audio_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    df = prepare_metadata(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        sampling_rate=args.sampling_rate,
        validate=not args.skip_validation
    )
    
    stats = compute_statistics(df)
    
    import json
    stats_file = output_dir / 'statistics.json'
    
    stats_serializable = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            stats_serializable[k] = {
                str(k2): (float(v2) if isinstance(v2, (np.floating, np.integer)) else v2)
                for k2, v2 in v.items()
            }
        else:
            stats_serializable[k] = int(v) if isinstance(v, (np.integer,)) else v
    
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    logger.info(f"Statistics saved to {stats_file}")
    
    if args.split:
        train_df, val_df, test_df = split_data(
            df,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed
        )
        
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        logger.info(f"Train set saved to {output_dir / 'train.csv'}")
        logger.info(f"Val set saved to {output_dir / 'val.csv'}")
        logger.info(f"Test set saved to {output_dir / 'test.csv'}")
    else:
        df.to_csv(output_dir / 'metadata.csv', index=False)
        logger.info(f"Metadata saved to {output_dir / 'metadata.csv'}")
    
    logger.info("=" * 50)
    logger.info("Data preparation completed!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
