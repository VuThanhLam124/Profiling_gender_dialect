"""
Finetune Script for Speaker Profiling Model

Train model using pre-extracted features from dataset folders.

Usage:
    python finetune.py --config configs/finetune.yaml
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from torch.utils.data import Dataset

from src.models import ClassificationHeadModelFromConfig
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    set_seed,
    count_parameters,
    format_number
)


class SpeakerDataset(Dataset):
    """
    Dataset class for speaker profiling - loads pre-extracted features from folder.
    
    Expected folder structure:
        datasets/ViSpeech/train/
        ├── features/
        │   ├── audio001.npy
        │   └── ...
        └── metadata.csv
    """
    
    def __init__(self, data_dir: str, config: dict, is_training: bool = True):
        """
        Args:
            data_dir: Path to dataset folder (e.g., 'datasets/ViSpeech/train')
            config: Configuration dictionary
            is_training: Whether this is training set
        """
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / 'features'
        self.is_training = is_training
        self.logger = get_logger()
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.df = pd.read_csv(metadata_path)
        self.logger.info(f"Loaded {len(self.df)} samples from {metadata_path}")
        
        # Verify feature directory exists
        if not self.feature_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.feature_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load pre-extracted features
        feature_name = row['feature_name']
        feature_path = self.feature_dir / feature_name
        
        try:
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()
        except Exception as e:
            self.logger.error(f"Error loading {feature_path}: {e}")
            features = torch.zeros(249, 768)  # Default shape for 5s audio
        
        return {
            'input_features': features,
            'gender_labels': torch.tensor(row['gender_label'], dtype=torch.long),
            'dialect_labels': torch.tensor(row['dialect_label'], dtype=torch.long)
        }


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning with pre-extracted features"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gender_labels = inputs.pop("gender_labels")
        dialect_labels = inputs.pop("dialect_labels")
        
        outputs = model(
            input_features=inputs["input_features"],
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
                input_features=inputs["input_features"],
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
    logger.info(f"Dataset: {config['data']['train_dir']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info("-" * 50)
    
    # Create datasets from pre-extracted features
    logger.info("Loading datasets...")
    train_dataset = SpeakerDataset(
        data_dir=config['data']['train_dir'],
        config=config,
        is_training=True
    )
    
    val_dataset = SpeakerDataset(
        data_dir=config['data']['val_dir'],
        config=config,
        is_training=False
    )
    
    logger.info(f"Train: {len(train_dataset):,} samples")
    logger.info(f"Validation: {len(val_dataset):,} samples")
    
    # Model (classification heads only - WavLM not needed for pre-extracted features)
    logger.info("Loading model...")
    model = ClassificationHeadModelFromConfig(config)
    
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
            "train_dir": config['data']['train_dir'],
            "val_dir": config['data']['val_dir'],
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "num_epochs": config['training']['num_epochs'],
            "weight_decay": config['training']['weight_decay'],
            "warmup_ratio": config['training']['warmup_ratio'],
            "dropout": config['model']['dropout'],
            "dialect_loss_weight": config['loss']['dialect_weight'],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
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
