"""
Finetune Script for Speaker Profiling Model

Train full model (encoder + heads) from raw audio with data augmentation.

Usage:
    python finetune.py --config configs/finetune.yaml
"""

import os
import io
import random
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False

from src.models import MultiTaskSpeakerModelFromConfig
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    set_seed,
    count_parameters,
    format_number
)


class AudioAugmentation:
    """Audio augmentation using audiomentations library"""
    
    def __init__(self, sampling_rate=16000, augment_prob=0.8):
        self.sampling_rate = sampling_rate
        self.augment_prob = augment_prob
        
        if AUGMENTATION_AVAILABLE:
            self.augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.3),
                Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
            ])
        else:
            self.augment = None
    
    def __call__(self, audio):
        if self.augment is not None and random.random() < self.augment_prob:
            return self.augment(samples=audio, sample_rate=self.sampling_rate)
        return audio


class ViSpeechDataset(Dataset):
    """
    Dataset class for speaker profiling - loads raw audio.
    Supports data augmentation for training.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        audio_dir: str, 
        feature_extractor,
        config: dict,
        is_training: bool = True
    ):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_length = int(self.sampling_rate * config['audio']['max_duration'])
        self.is_training = is_training
        self.logger = get_logger()
        
        # Data augmentation (only for training)
        augment_prob = config.get('augmentation', {}).get('prob', 0.8)
        if is_training and AUGMENTATION_AVAILABLE:
            self.augmentation = AudioAugmentation(self.sampling_rate, augment_prob)
            self.logger.info(f"Augmentation ENABLED (prob={augment_prob})")
        else:
            self.augmentation = None
            if is_training and not AUGMENTATION_AVAILABLE:
                self.logger.warning("audiomentations not installed. Augmentation DISABLED.")
            else:
                self.logger.info("Augmentation DISABLED (validation/test)")
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, audio_name):
        audio_path = self.audio_dir / audio_name
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Apply augmentation (training only)
            if self.is_training and self.augmentation is not None:
                audio = self.augmentation(audio)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Pad or truncate
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            else:
                if self.is_training:
                    # Random crop for training
                    start = np.random.randint(0, len(audio) - self.max_length + 1)
                else:
                    # Center crop for validation/test
                    start = (len(audio) - self.max_length) // 2
                audio = audio[start:start + self.max_length]
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading {audio_path}: {e}")
            return np.zeros(self.max_length)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = self.load_audio(row['audio_name'])
        
        # Extract features using feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'gender_labels': torch.tensor(row['gender_label'], dtype=torch.long),
            'dialect_labels': torch.tensor(row['dialect_label'], dtype=torch.long)
        }


class ViMDDataset(Dataset):
    """
    Dataset class for ViMD (HuggingFace format).
    Loads audio from HuggingFace datasets with path/bytes/array support.
    """
    
    def __init__(
        self, 
        hf_dataset,
        feature_extractor,
        config: dict,
        is_training: bool = True
    ):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_length = int(self.sampling_rate * config['audio']['max_duration'])
        self.is_training = is_training
        self.logger = get_logger()
        
        # Label mappings
        self.region_to_dialect = config['labels'].get('region_to_dialect', {
            'North': 0, 'Central': 1, 'South': 2
        })
        self.gender_map = config['labels'].get('gender', {
            'Male': 0, 'Female': 1, 0: 0, 1: 1
        })
        
        # Data augmentation (only for training)
        augment_prob = config.get('augmentation', {}).get('prob', 0.8)
        if is_training and AUGMENTATION_AVAILABLE:
            self.augmentation = AudioAugmentation(self.sampling_rate, augment_prob)
            self.logger.info(f"Augmentation ENABLED (prob={augment_prob})")
        else:
            self.augmentation = None
            if is_training and not AUGMENTATION_AVAILABLE:
                self.logger.warning("audiomentations not installed. Augmentation DISABLED.")
    
    def __len__(self):
        return len(self.dataset)
    
    def load_audio_from_hf(self, audio_data):
        """Load audio from HuggingFace dataset format (path/bytes/array)"""
        try:
            audio = None
            
            # Case 1: audio_data is dict with 'path', 'bytes', or 'array'
            if isinstance(audio_data, dict):
                # Try loading from path first
                if 'path' in audio_data and audio_data['path']:
                    try:
                        audio, sr = librosa.load(audio_data['path'], sr=self.sampling_rate, mono=True)
                    except Exception:
                        pass
                
                # Try loading from bytes
                if audio is None and 'bytes' in audio_data and audio_data['bytes']:
                    try:
                        audio_bytes = io.BytesIO(audio_data['bytes'])
                        audio, sr = sf.read(audio_bytes)
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)  # Convert to mono
                        if sr != self.sampling_rate:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                    except Exception:
                        pass
                
                # Try loading from pre-decoded array
                if audio is None and 'array' in audio_data:
                    try:
                        audio = np.array(audio_data['array'], dtype=np.float32)
                        sr = audio_data.get('sampling_rate', self.sampling_rate)
                        if sr != self.sampling_rate:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                    except Exception:
                        pass
            
            # Case 2: audio_data is string (path)
            elif isinstance(audio_data, str):
                audio, sr = librosa.load(audio_data, sr=self.sampling_rate, mono=True)
            
            # Validation
            if audio is None or len(audio) == 0:
                return np.zeros(self.max_length, dtype=np.float32)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Trim silence
            if len(audio) > 100:
                try:
                    audio, _ = librosa.effects.trim(audio, top_db=20)
                except Exception:
                    pass
            
            # Apply augmentation (training only)
            if self.is_training and self.augmentation is not None and len(audio) > 100:
                try:
                    audio = self.augmentation(audio)
                except Exception:
                    pass
            
            # Normalize
            if len(audio) > 0:
                max_val = np.max(np.abs(audio))
                if max_val > 1e-8:
                    audio = audio / max_val
                else:
                    return np.zeros(self.max_length, dtype=np.float32)
            
            # Pad or truncate
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            else:
                if self.is_training:
                    start = np.random.randint(0, max(1, len(audio) - self.max_length))
                else:
                    start = max(0, (len(audio) - self.max_length) // 2)
                audio = audio[start:start + self.max_length]
            
            return audio.astype(np.float32)
            
        except Exception as e:
            return np.zeros(self.max_length, dtype=np.float32)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Load audio
            audio = self.load_audio_from_hf(item.get('audio'))
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Map labels - ViMD uses 'gender' (int) and 'region' (string)
            gender_raw = item.get('gender', 0)
            region = item.get('region', 'North')
            
            # Gender: support both int and string
            if isinstance(gender_raw, int):
                gender_label = gender_raw
            else:
                gender_label = self.gender_map.get(gender_raw, 0)
            
            # Dialect: map region to dialect label
            dialect_label = self.region_to_dialect.get(region, 0)
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'gender_labels': torch.tensor(gender_label, dtype=torch.long),
                'dialect_labels': torch.tensor(dialect_label, dtype=torch.long)
            }
            
        except Exception as e:
            # Return dummy data to prevent crash
            dummy_audio = np.zeros(self.max_length, dtype=np.float32)
            inputs = self.feature_extractor(
                dummy_audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            return {
                'input_values': inputs.input_values.squeeze(0),
                'gender_labels': torch.tensor(0, dtype=torch.long),
                'dialect_labels': torch.tensor(0, dtype=torch.long)
            }


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gender_labels = inputs.pop("gender_labels")
        dialect_labels = inputs.pop("dialect_labels")
        
        outputs = model(
            input_values=inputs["input_values"],
            gender_labels=gender_labels,
            dialect_labels=dialect_labels
        )
        
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        gender_labels = inputs.pop("gender_labels")
        dialect_labels = inputs.pop("dialect_labels")
        
        with torch.no_grad():
            outputs = model(
                input_values=inputs["input_values"],
                gender_labels=gender_labels,
                dialect_labels=dialect_labels
            )
            loss = outputs["loss"]
        
        return (
            loss,
            (outputs["gender_logits"], outputs["dialect_logits"]),
            (gender_labels, dialect_labels)
        )


class WandbCallback(TrainerCallback):
    """Callback for logging metrics to WandB"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or wandb.run is None:
            return
        
        step = state.global_step
        metrics_to_log = {}
        
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                metrics_to_log[key] = value
        
        if metrics_to_log:
            wandb.log(metrics_to_log, step=step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or wandb.run is None:
            return
        
        step = state.global_step
        eval_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_metrics[f"eval_{key}" if not key.startswith("eval_") else key] = value
        
        if eval_metrics:
            wandb.log(eval_metrics, step=step)


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    gender_logits, dialect_logits = pred.predictions
    gender_labels, dialect_labels = pred.label_ids
    
    gender_preds = np.argmax(gender_logits, axis=-1)
    dialect_preds = np.argmax(dialect_logits, axis=-1)
    
    gender_acc = accuracy_score(gender_labels, gender_preds)
    gender_f1 = f1_score(gender_labels, gender_preds, average='weighted')
    dialect_acc = accuracy_score(dialect_labels, dialect_preds)
    dialect_f1 = f1_score(dialect_labels, dialect_preds, average='weighted')
    
    return {
        'gender_acc': gender_acc,
        'gender_f1': gender_f1,
        'dialect_acc': dialect_acc,
        'dialect_f1': dialect_f1,
        'combined_f1': (gender_f1 + dialect_f1) / 2
    }


def load_and_split_data(config, logger):
    """Load metadata and split by speaker (for ViSpeech CSV format)"""
    
    # Load metadata
    train_meta_path = config['data']['train_meta']
    logger.info(f"Loading metadata from {train_meta_path}...")
    
    train_df = pd.read_csv(train_meta_path)
    logger.info(f"Columns found: {list(train_df.columns)}")
    
    # Auto-detect column names (support different naming conventions)
    col_mapping = {
        'audio_name': ['audio_name', 'filename', 'file', 'path', 'audio_path'],
        'gender': ['gender', 'sex'],
        'dialect': ['dialect', 'accent', 'region'],
        'speaker': ['speaker', 'speaker_id', 'spk_id', 'spk']
    }
    
    detected_cols = {}
    for target, candidates in col_mapping.items():
        for col in candidates:
            if col in train_df.columns:
                detected_cols[target] = col
                break
        if target not in detected_cols:
            raise ValueError(f"Could not find column for '{target}'. Expected one of: {candidates}")
    
    logger.info(f"Column mapping: {detected_cols}")
    
    # Rename columns to standard names
    rename_map = {v: k for k, v in detected_cols.items() if k != v}
    if rename_map:
        train_df = train_df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    
    # Map labels
    gender_map = config['labels']['gender']
    dialect_map = config['labels']['dialect']
    
    train_df['gender_label'] = train_df['gender'].map(gender_map)
    train_df['dialect_label'] = train_df['dialect'].map(dialect_map)
    
    # Split by speaker to avoid data leakage
    val_split = config['data'].get('val_split', 0.15)
    unique_speakers = train_df['speaker'].unique()
    
    train_speakers, val_speakers = train_test_split(
        unique_speakers,
        test_size=val_split,
        random_state=config['seed'],
        shuffle=True
    )
    
    train_data = train_df[train_df['speaker'].isin(train_speakers)].reset_index(drop=True)
    val_data = train_df[train_df['speaker'].isin(val_speakers)].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_data):,} samples ({len(train_speakers)} speakers)")
    logger.info(f"Validation: {len(val_data):,} samples ({len(val_speakers)} speakers)")
    
    # Verify no speaker leakage
    assert len(set(train_speakers) & set(val_speakers)) == 0, "Speaker leakage detected!"
    logger.info("No speaker leakage between train/val")
    
    return train_data, val_data


def load_vimd_data(config, logger):
    """Load ViMD dataset from HuggingFace format"""
    
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library required for ViMD. Install: pip install datasets")
    
    vimd_path = config['data']['vimd_path']
    logger.info(f"Loading ViMD dataset from {vimd_path}...")
    
    ds = load_dataset(vimd_path, keep_in_memory=False)
    
    # Check available splits
    available_splits = list(ds.keys())
    logger.info(f"Available splits: {available_splits}")
    
    # Handle both 'valid' and 'validation' keys
    val_key = None
    if 'validation' in available_splits:
        val_key = 'validation'
    elif 'valid' in available_splits:
        val_key = 'valid'
    
    logger.info(f"ViMD Train: {len(ds['train']):,} samples")
    if val_key:
        logger.info(f"ViMD Validation: {len(ds[val_key]):,} samples")
    if 'test' in available_splits:
        logger.info(f"ViMD Test: {len(ds['test']):,} samples")
    
    # Normalize split name to 'valid'
    if val_key == 'validation':
        ds['valid'] = ds['validation']
    
    logger.info(f"Available columns: {ds['train'].column_names}")
    
    return ds, val_key or 'valid'


def main(config_path):
    """Main training function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("SPEAKER PROFILING TRAINING (Full Finetune)")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(config_path)
    set_seed(config['seed'])
    
    # Determine data source: vispeech (CSV) or vimd (HuggingFace)
    data_source = config['data'].get('source', 'vispeech')
    logger.info(f"Data source: {data_source}")
    
    # Setup WandB
    wandb_config = config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', True)
    
    if wandb_enabled:
        wandb_api_key = wandb_config.get('api_key', 'f05e29c3466ec288e97041e0e3d541c4087096a6')
        wandb.login(key=wandb_api_key)
        
        project_name = wandb_config.get('project', 'speaker-profiling')
        run_name = wandb_config.get('run_name', None)
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_name": config['model']['name'],
                "batch_size": config['training']['batch_size'],
                "learning_rate": config['training']['learning_rate'],
                "num_epochs": config['training']['num_epochs'],
                "dropout": config['model']['dropout'],
                "dialect_loss_weight": config['loss']['dialect_weight'],
                "seed": config['seed'],
            }
        )
        logger.info(f"WandB project: {project_name}")
    
    # Log config
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Dialect Loss Weight: {config['loss']['dialect_weight']}x")
    logger.info("-" * 60)
    
    # Load feature extractor (auto-detect based on model type)
    model_name = config['model']['name']
    logger.info(f"Loading feature extractor for {model_name}...")
    
    # Check if model is ECAPA-TDNN (SpeechBrain) - no HuggingFace feature extractor
    is_ecapa = 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower()
    
    if is_ecapa:
        # ECAPA-TDNN uses simple audio processing, use Wav2Vec2 feature extractor
        # which just normalizes audio to 16kHz
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        logger.info("Using Wav2Vec2 feature extractor for ECAPA-TDNN (audio normalization only)")
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    # Load data based on source
    if data_source == 'vimd':
        # Load ViMD (HuggingFace format)
        vimd_ds, val_key = load_vimd_data(config, logger)
        
        logger.info("Creating ViMD datasets...")
        train_dataset = ViMDDataset(
            hf_dataset=vimd_ds['train'],
            feature_extractor=feature_extractor,
            config=config,
            is_training=True
        )
        
        val_dataset = ViMDDataset(
            hf_dataset=vimd_ds[val_key],
            feature_extractor=feature_extractor,
            config=config,
            is_training=False
        )
    else:
        # Load ViSpeech (CSV format)
        train_df, val_df = load_and_split_data(config, logger)
        
        logger.info("Creating ViSpeech datasets...")
        train_dataset = ViSpeechDataset(
            dataframe=train_df,
            audio_dir=config['data']['train_audio'],
            feature_extractor=feature_extractor,
            config=config,
            is_training=True
        )
        
        val_dataset = ViSpeechDataset(
            dataframe=val_df,
            audio_dir=config['data']['train_audio'],
            feature_extractor=feature_extractor,
            config=config,
            is_training=False
        )
    
    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    
    # Load full model (encoder + heads)
    logger.info("Loading model...")
    model = MultiTaskSpeakerModelFromConfig(config)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {format_number(total_params)}")
    logger.info(f"Trainable parameters: {format_number(trainable_params)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output']['dir'],
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        max_grad_norm=config['training']['gradient_clip'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        load_best_model_at_end=True,
        metric_for_best_model=config['output']['metric_for_best_model'],
        greater_is_better=True,
        save_total_limit=config['output']['save_total_limit'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        logging_steps=50,
        logging_first_step=True,
        report_to='none',
        remove_unused_columns=False,
        seed=config['seed'],
    )
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['early_stopping']['patience'],
        early_stopping_threshold=config['early_stopping']['threshold']
    )
    
    # Callbacks
    callbacks = [early_stopping]
    if wandb_enabled:
        callbacks.append(WandbCallback())
    
    try:
        # Trainer
        trainer = MultiTaskTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Steps per epoch: ~{len(train_dataset) // config['training']['batch_size']}")
        trainer.train()
        
        # Save best model
        output_dir = os.path.join(config['output']['dir'], 'best_model')
        logger.info(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        
        # Log final metrics to WandB
        if wandb_enabled and wandb.run is not None:
            final_metrics = trainer.evaluate()
            wandb.log({f"final_{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float))})
            wandb.save(os.path.join(output_dir, "*"))
            logger.info(f"WandB run: {wandb.run.url}")
        
        logger.info("Training completed!")
        
    finally:
        if wandb_enabled and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Speaker Profiling Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/finetune.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    main(args.config)
