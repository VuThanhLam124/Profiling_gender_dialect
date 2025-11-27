"""
Finetune Script for Speaker Profiling Model
Architecture: WavLM + Attentive Pooling + LayerNorm + Deeper Heads

Usage:
    python finetune.py --config configs/finetune.yaml
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from yaml file using OmegaConf"""
    config = OmegaConf.load(config_path)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioAugmentation:
    """
    Audio augmentation for training
    - Gaussian noise injection
    - Speed perturbation (time stretch)
    - Pitch shifting
    - Gain adjustment
    """
    
    def __init__(self, config):
        aug_config = config['augmentation']
        self.sampling_rate = config['audio']['sampling_rate']
        self.augment_prob = aug_config['probability']
        
        self.augment = Compose([
            AddGaussianNoise(
                min_amplitude=aug_config['gaussian_noise']['min_amplitude'],
                max_amplitude=aug_config['gaussian_noise']['max_amplitude'],
                p=aug_config['gaussian_noise']['probability']
            ),
            TimeStretch(
                min_rate=aug_config['time_stretch']['min_rate'],
                max_rate=aug_config['time_stretch']['max_rate'],
                leave_length_unchanged=False,
                p=aug_config['time_stretch']['probability']
            ),
            PitchShift(
                min_semitones=aug_config['pitch_shift']['min_semitones'],
                max_semitones=aug_config['pitch_shift']['max_semitones'],
                p=aug_config['pitch_shift']['probability']
            ),
            Shift(
                min_shift=aug_config['shift']['min_shift'],
                max_shift=aug_config['shift']['max_shift'],
                p=aug_config['shift']['probability']
            ),
            Gain(
                min_gain_db=aug_config['gain']['min_gain_db'],
                max_gain_db=aug_config['gain']['max_gain_db'],
                p=aug_config['gain']['probability']
            ),
        ])
    
    def __call__(self, audio):
        if random.random() < self.augment_prob:
            return self.augment(samples=audio, sample_rate=self.sampling_rate)
        return audio


class ViSpeechDataset(Dataset):
    """Dataset class for ViSpeech data"""
    
    def __init__(self, dataframe, audio_dir, feature_extractor, config, is_training=True):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_length = int(self.sampling_rate * config['audio']['max_duration'])
        self.is_training = is_training
        
        if is_training and config['augmentation']['enabled']:
            self.augmentation = AudioAugmentation(config)
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, audio_name):
        audio_path = self.audio_dir / audio_name
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            if self.is_training and self.augmentation is not None:
                audio = self.augmentation(audio)
            
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            else:
                if self.is_training:
                    start = np.random.randint(0, len(audio) - self.max_length + 1)
                else:
                    start = (len(audio) - self.max_length) // 2
                audio = audio[start:start + self.max_length]
            
            return audio
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.max_length)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = self.load_audio(row['audio_name'])
        
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


class AttentivePooling(nn.Module):
    """Attention-based pooling for temporal aggregation"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def forward(self, x, mask=None):
        attn_weights = self.attention(x)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return pooled, attn_weights.squeeze(-1)


class MultiTaskSpeakerModel(torch.nn.Module):
    """
    Multi-task model for gender and dialect classification
    
    Architecture:
        Audio -> WavLM -> Last Hidden [B,T,768]
                              |
                     Attentive Pooling [B,768]
                              |
                     Layer Normalization
                              |
                         Dropout(0.1)
                              |
              +---------------+---------------+
              |                               |
        Gender Head (2 layers)     Dialect Head (3 layers)
              |                               |
            [B,2]                           [B,3]
    """
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['model']
        self.dialect_loss_weight = config['loss']['dialect_weight']
        
        self.wavlm = WavLMModel.from_pretrained(model_config['name'])
        
        if model_config['freeze_encoder']:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            print("Encoder FROZEN")
        
        hidden_size = self.wavlm.config.hidden_size
        head_hidden_dim = model_config.get('head_hidden_dim', 256)
        dropout = model_config['dropout']
        
        self.attentive_pooling = AttentivePooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, model_config['num_genders'])
        )
        
        self.dialect_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, model_config['num_dialects'])
        )
        
        print(f"Architecture: WavLM + Attentive Pooling + LayerNorm")
        print(f"Hidden size: {hidden_size}")
        print(f"Head hidden dim: {head_hidden_dim}")
        print(f"Dropout: {dropout}")
        
    def forward(self, input_values, attention_mask=None, 
                gender_labels=None, dialect_labels=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        pooled, attn_weights = self.attentive_pooling(hidden_states, attention_mask)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        gender_logits = self.gender_head(pooled)
        dialect_logits = self.dialect_head(pooled)
        
        loss = None
        if gender_labels is not None and dialect_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            gender_loss = loss_fct(gender_logits, gender_labels)
            dialect_loss = loss_fct(dialect_logits, dialect_labels)
            loss = gender_loss + self.dialect_loss_weight * dialect_loss
        
        return {
            'loss': loss,
            'gender_logits': gender_logits,
            'dialect_logits': dialect_logits,
            'attention_weights': attn_weights
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


def load_and_prepare_data(config):
    """Load and prepare training data with speaker-based split"""
    
    print("Loading metadata...")
    train_df = pd.read_csv(config['data']['train_meta'])
    
    gender_map = config['labels']['gender']
    dialect_map = config['labels']['dialect']
    
    train_df['gender_label'] = train_df['gender'].map(gender_map)
    train_df['dialect_label'] = train_df['dialect'].map(dialect_map)
    
    unique_speakers = train_df['speaker'].unique()
    train_speakers, val_speakers = train_test_split(
        unique_speakers,
        test_size=config['data']['val_split'],
        random_state=config['seed'],
        shuffle=True
    )
    
    train_data = train_df[train_df['speaker'].isin(train_speakers)].reset_index(drop=True)
    val_data = train_df[train_df['speaker'].isin(val_speakers)].reset_index(drop=True)
    
    print(f"Train: {len(train_data):,} samples ({len(train_speakers)} speakers)")
    print(f"Validation: {len(val_data):,} samples ({len(val_speakers)} speakers)")
    
    assert len(set(train_speakers) & set(val_speakers)) == 0, "Speaker leakage detected!"
    
    return train_data, val_data


def main(config_path):
    """Main training function"""
    
    print("SPEAKER PROFILING TRAINING")
    print("-" * 50)
    
    # Load config
    config = load_config(config_path)
    set_seed(config['seed'])
    
    # Print config
    print(f"Model: {config['model']['name']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print("-" * 50)
    
    # Load data
    train_df, val_df = load_and_prepare_data(config)
    
    # Feature extractor
    print("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['model']['name'])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ViSpeechDataset(
        train_df, 
        config['data']['train_audio'], 
        feature_extractor, 
        config,
        is_training=True
    )
    
    val_dataset = ViSpeechDataset(
        val_df, 
        config['data']['train_audio'], 
        feature_extractor, 
        config,
        is_training=False
    )
    
    # Model
    print("Loading model...")
    model = MultiTaskSpeakerModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save best model
    output_dir = os.path.join(config['output']['dir'], 'best_model')
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
    
    print("\nTraining completed!")


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
