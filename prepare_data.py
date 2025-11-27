"""
Feature Extraction Script for Speaker Profiling

Extract and cache WavLM features from audio datasets.
Run once, reuse for multiple training experiments.

Supported datasets:
- ViSpeech: Vietnamese speech dataset
- (Extend by adding new dataset classes)

Usage:
    python prepare_data.py --dataset vispeech --config configs/finetune.yaml --output_dir features/vispeech
    python prepare_data.py --dataset vispeech --config configs/finetune.yaml --output_dir features/vispeech --split train
"""

import os
import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from src.utils import setup_logging, get_logger, load_config, set_seed


class BaseDataset(ABC):
    """
    Base class for dataset feature extraction.
    Extend this class to add support for new datasets.
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """
        Args:
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test', 'clean_test', 'noisy_test')
        """
        self.config = config
        self.split = split
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_duration = config['audio']['max_duration']
        self.logger = get_logger()
        
        # Label mappings
        self.gender_map = config['labels']['gender']
        self.dialect_map = config['labels']['dialect']
    
    @abstractmethod
    def get_metadata(self) -> pd.DataFrame:
        """
        Load and return metadata DataFrame.
        Must contain columns: audio_name, speaker, gender, dialect
        
        Returns:
            DataFrame with metadata
        """
        pass
    
    @abstractmethod
    def get_audio_path(self, audio_name: str) -> Path:
        """
        Get full path to audio file.
        
        Args:
            audio_name: Audio filename from metadata
            
        Returns:
            Full path to audio file
        """
        pass
    
    def load_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array, or None if failed
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            
            # Truncate/pad to max_duration
            max_length = int(self.sampling_rate * self.max_duration)
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading {audio_path}: {e}")
            return None
    
    def get_labels(self, row: pd.Series) -> Tuple[int, int]:
        """
        Get numeric labels from metadata row.
        
        Args:
            row: Metadata row
            
        Returns:
            Tuple of (gender_label, dialect_label)
        """
        gender_label = self.gender_map[row['gender']]
        dialect_label = self.dialect_map[row['dialect']]
        return gender_label, dialect_label


class ViSpeechDataset(BaseDataset):
    """
    ViSpeech dataset handler.
    
    Expected structure:
        ViSpeech/
        ├── trainset/
        ├── clean_testset/
        ├── noisy_testset/
        └── metadata/
            ├── trainset.csv
            ├── clean_testset.csv
            └── noisy_testset.csv
    """
    
    SPLIT_MAPPING = {
        'train': ('train_meta', 'train_audio'),
        'clean_test': ('clean_test_meta', 'clean_test_audio'),
        'noisy_test': ('noisy_test_meta', 'noisy_test_audio'),
    }
    
    def __init__(self, config: Dict, split: str = 'train'):
        super().__init__(config, split)
        
        if split not in self.SPLIT_MAPPING and split not in ['val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.SPLIT_MAPPING.keys())}")
        
        # For train split, we use train_meta and train_audio
        # val/test can be derived from train with speaker-based split
        if split in ['train', 'val', 'test']:
            self.meta_key, self.audio_key = 'train_meta', 'train_audio'
        else:
            self.meta_key, self.audio_key = self.SPLIT_MAPPING[split]
        
        self.metadata_path = Path(config['data'][self.meta_key])
        self.audio_dir = Path(config['data'][self.audio_key])
        
        self.logger.info(f"ViSpeech dataset - Split: {split}")
        self.logger.info(f"  Metadata: {self.metadata_path}")
        self.logger.info(f"  Audio dir: {self.audio_dir}")
    
    def get_metadata(self) -> pd.DataFrame:
        """Load ViSpeech metadata"""
        df = pd.read_csv(self.metadata_path)
        self.logger.info(f"Loaded {len(df)} samples from {self.metadata_path}")
        return df
    
    def get_audio_path(self, audio_name: str) -> Path:
        """Get audio path for ViSpeech"""
        return self.audio_dir / audio_name


# Registry for dataset classes
DATASET_REGISTRY = {
    'vispeech': ViSpeechDataset,
    # Add more datasets here:
    # 'common_voice': CommonVoiceDataset,
    # 'huggingface': HuggingFaceDataset,
}


class FeatureExtractor:
    """
    Extract and cache WavLM features from audio.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: str = None
    ):
        self.logger = get_logger()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Loading WavLM model: {model_name}")
        self.logger.info(f"Device: {self.device}")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("WavLM model loaded successfully")
    
    @torch.no_grad()
    def extract(self, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """
        Extract WavLM features from audio.
        
        Args:
            audio: Audio waveform
            sampling_rate: Audio sampling rate
            
        Returns:
            Feature array of shape [T, hidden_size]
        """
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)
        
        # Get last hidden state [1, T, hidden_size] -> [T, hidden_size]
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        return features


def extract_and_save_features(
    dataset: BaseDataset,
    extractor: FeatureExtractor,
    output_dir: Path,
    batch_size: int = 1
) -> Dict:
    """
    Extract features from dataset and save to disk.
    
    Args:
        dataset: Dataset instance
        extractor: Feature extractor
        output_dir: Output directory
        batch_size: Batch size (currently only 1 supported)
        
    Returns:
        Statistics dictionary
    """
    logger = get_logger()
    
    # Create output directories
    features_dir = output_dir / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = dataset.get_metadata()
    
    # Prepare records for new metadata
    records = []
    stats = {
        'total': len(df),
        'success': 0,
        'failed': 0,
        'failed_files': []
    }
    
    logger.info(f"Extracting features for {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_name = row['audio_name']
        audio_path = dataset.get_audio_path(audio_name)
        
        # Load audio
        audio = dataset.load_audio(audio_path)
        if audio is None:
            stats['failed'] += 1
            stats['failed_files'].append(audio_name)
            continue
        
        # Extract features
        try:
            features = extractor.extract(audio, dataset.sampling_rate)
            
            # Save features
            feature_name = Path(audio_name).stem + '.npy'
            feature_path = features_dir / feature_name
            np.save(feature_path, features)
            
            # Get labels
            gender_label, dialect_label = dataset.get_labels(row)
            
            # Record metadata
            records.append({
                'audio_name': audio_name,
                'feature_name': feature_name,
                'speaker': row['speaker'],
                'gender': row['gender'],
                'dialect': row['dialect'],
                'gender_label': gender_label,
                'dialect_label': dialect_label,
                'feature_shape': list(features.shape)
            })
            
            stats['success'] += 1
            
        except Exception as e:
            logger.error(f"Error extracting features for {audio_name}: {e}")
            stats['failed'] += 1
            stats['failed_files'].append(audio_name)
    
    # Save metadata
    metadata_df = pd.DataFrame(records)
    metadata_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Save statistics
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")
    
    # Log summary
    logger.info(f"Feature extraction completed:")
    logger.info(f"  Success: {stats['success']}/{stats['total']}")
    logger.info(f"  Failed: {stats['failed']}/{stats['total']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache WavLM features from audio datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"Dataset name. Available: {list(DATASET_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for cached features"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to process (train, clean_test, noisy_test)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'extraction.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(args.config)
    set_seed(args.seed)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info("-" * 60)
    
    # Create dataset
    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(config, split=args.split)
    
    # Create feature extractor
    extractor = FeatureExtractor(
        model_name=config['model']['name'],
        device=args.device
    )
    
    # Extract features
    stats = extract_and_save_features(
        dataset=dataset,
        extractor=extractor,
        output_dir=output_dir
    )
    
    logger.info("=" * 60)
    logger.info("Feature extraction completed!")
    logger.info("=" * 60)
    logger.info(f"\nTo use cached features in training, update config:")
    logger.info(f"  data:")
    logger.info(f"    use_cached_features: true")
    logger.info(f"    feature_dir: \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
