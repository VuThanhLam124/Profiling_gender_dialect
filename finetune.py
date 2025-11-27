"""
Finetune Script for Speaker Profiling Model

Usage:
    python finetune.py --config configs/finetune.yaml
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain

from src.models import MultiTaskSpeakerModelFromConfig
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    set_seed,
    preprocess_audio,
    count_parameters,
    format_number
)


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
    """Dataset class for ViSpeech data - supports both raw audio and cached features"""
    
    def __init__(self, dataframe, audio_dir, feature_extractor, config, is_training=True):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_duration = config['audio']['max_duration']
        self.is_training = is_training
        self.logger = get_logger()
        
        # Check if using cached features
        self.use_cached_features = config['data'].get('use_cached_features', False)
        if self.use_cached_features:
            self.feature_dir = Path(config['data']['feature_dir']) / 'features'
            self.logger.info(f"Using cached features from: {self.feature_dir}")
        
        if is_training and config['augmentation']['enabled'] and not self.use_cached_features:
            self.augmentation = AudioAugmentation(config)
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, audio_name):
        audio_path = self.audio_dir / audio_name
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            
            if self.is_training and self.augmentation is not None:
                audio = self.augmentation(audio)
            
            audio = preprocess_audio(
                audio,
                sampling_rate=self.sampling_rate,
                max_duration=self.max_duration,
                center_crop=not self.is_training
            )
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading {audio_path}: {e}")
            max_length = int(self.sampling_rate * self.max_duration)
            return np.zeros(max_length)
    
    def load_cached_features(self, audio_name):
        """Load pre-extracted features from cache"""
        feature_name = Path(audio_name).stem + '.npy'
        feature_path = self.feature_dir / feature_name
        
        try:
            features = np.load(feature_path)
            return torch.from_numpy(features).float()
        except Exception as e:
            self.logger.error(f"Error loading features {feature_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.use_cached_features:
            # Load pre-extracted features
            features = self.load_cached_features(row['audio_name'])
            if features is None:
                # Fallback to zeros
                features = torch.zeros(249, 768)  # Default WavLM output shape for 5s audio
            
            return {
                'input_features': features,
                'gender_labels': torch.tensor(row['gender_label'], dtype=torch.long),
                'dialect_labels': torch.tensor(row['dialect_label'], dtype=torch.long)
            }
        else:
            # Load raw audio and extract features on-the-fly
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


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning - supports both raw audio and cached features"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gender_labels = inputs.pop("gender_labels")
        dialect_labels = inputs.pop("dialect_labels")
        
        # Support both input_values (raw audio) and input_features (cached)
        if "input_features" in inputs:
            outputs = model(
                input_features=inputs["input_features"],
                gender_labels=gender_labels,
                dialect_labels=dialect_labels
            )
        else:
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
            # Support both input_values (raw audio) and input_features (cached)
            if "input_features" in inputs:
                outputs = model(
                    input_features=inputs["input_features"],
                    gender_labels=gender_labels,
                    dialect_labels=dialect_labels
                )
            else:
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


class MLflowCallback(TrainerCallback):
    """Callback for logging metrics to MLflow"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Filter and log metrics
        step = state.global_step
        metrics_to_log = {}
        
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                metrics_to_log[key] = value
        
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log, step=step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        # Log evaluation metrics
        step = state.global_step
        eval_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_metrics[f"eval_{key}" if not key.startswith("eval_") else key] = value
        
        if eval_metrics:
            mlflow.log_metrics(eval_metrics, step=step)


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
    logger = get_logger()
    
    logger.info("Loading metadata...")
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
    
    logger.info(f"Train: {len(train_data):,} samples ({len(train_speakers)} speakers)")
    logger.info(f"Validation: {len(val_data):,} samples ({len(val_speakers)} speakers)")
    
    assert len(set(train_speakers) & set(val_speakers)) == 0, "Speaker leakage detected!"
    
    return train_data, val_data


def main(config_path):
    """Main training function"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("SPEAKER PROFILING TRAINING")
    logger.info("=" * 50)
    
    # Load config
    config = load_config(config_path)
    set_seed(config['seed'])
    
    # Setup MLflow
    mlflow_config = config.get('mlflow', {})
    mlflow_enabled = mlflow_config.get('enabled', True)
    
    if mlflow_enabled:
        # Set tracking URI if provided
        tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment name
        experiment_name = mlflow_config.get('experiment_name', 'speaker-profiling')
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
    # Log config
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info("-" * 50)
    
    # Load data
    train_df, val_df = load_and_prepare_data(config)
    
    # Feature extractor
    logger.info("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['model']['name'])
    
    # Create datasets
    logger.info("Creating datasets...")
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
    if mlflow_enabled:
        callbacks.append(MLflowCallback())
    
    # Start MLflow run
    if mlflow_enabled:
        run_name = mlflow_config.get('run_name', None)
        mlflow.start_run(run_name=run_name)
        
        # Log parameters
        mlflow.log_params({
            "model_name": config['model']['name'],
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "num_epochs": config['training']['num_epochs'],
            "weight_decay": config['training']['weight_decay'],
            "warmup_ratio": config['training']['warmup_ratio'],
            "dropout": config['model']['dropout'],
            "freeze_encoder": config['model']['freeze_encoder'],
            "dialect_loss_weight": config['loss']['dialect_weight'],
            "max_audio_duration": config['audio']['max_duration'],
            "sampling_rate": config['audio']['sampling_rate'],
            "augmentation_enabled": config['augmentation']['enabled'],
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "seed": config['seed'],
        })
        
        # Log config file as artifact
        mlflow.log_artifact(config_path, artifact_path="config")
    
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
        trainer.train()
        
        # Save best model
        output_dir = os.path.join(config['output']['dir'], 'best_model')
        logger.info(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        
        # Log model to MLflow
        if mlflow_enabled:
            # Log final metrics
            final_metrics = trainer.evaluate()
            mlflow.log_metrics({
                f"final_{k}": v for k, v in final_metrics.items() 
                if isinstance(v, (int, float))
            })
            
            # Log model artifacts
            mlflow.log_artifacts(output_dir, artifact_path="model")
            
            # Log best model with MLflow PyTorch
            mlflow.pytorch.log_model(
                model, 
                artifact_path="pytorch_model",
                registered_model_name=mlflow_config.get('registered_model_name', None)
            )
            
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        logger.info("Training completed!")
        
    finally:
        if mlflow_enabled:
            mlflow.end_run()


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
