"""
Evaluation Script for Speaker Profiling Model

Usage:
    python eval.py --config configs/eval.yaml
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import Wav2Vec2FeatureExtractor, Trainer, TrainingArguments
from torch.utils.data import Dataset

from src.models import MultiTaskSpeakerModel
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    load_model_checkpoint,
    preprocess_audio
)


class ViSpeechDataset(Dataset):
    """Dataset class for evaluation"""
    
    def __init__(self, dataframe, audio_dir, feature_extractor, config):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_duration = config['audio']['max_duration']
        self.logger = get_logger()
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, audio_name):
        audio_path = self.audio_dir / audio_name
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            audio = preprocess_audio(
                audio,
                sampling_rate=self.sampling_rate,
                max_duration=self.max_duration
            )
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading {audio_path}: {e}")
            max_length = int(self.sampling_rate * self.max_duration)
            return np.zeros(max_length)
    
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


def evaluate_dataset(trainer, dataset, dataset_name, config):
    """Evaluate model on a dataset and log detailed report"""
    logger = get_logger()
    
    logger.info(f"EVALUATING ON {dataset_name.upper()}")
    logger.info("-" * 50)
    
    results = trainer.predict(dataset)
    gender_logits, dialect_logits = results.predictions
    gender_labels, dialect_labels = results.label_ids
    
    gender_preds = np.argmax(gender_logits, axis=-1)
    dialect_preds = np.argmax(dialect_logits, axis=-1)
    
    gender_acc = accuracy_score(gender_labels, gender_preds) * 100
    dialect_acc = accuracy_score(dialect_labels, dialect_preds) * 100
    gender_f1 = f1_score(gender_labels, gender_preds, average='weighted') * 100
    dialect_f1 = f1_score(dialect_labels, dialect_preds, average='weighted') * 100
    
    logger.info("Overall Metrics:")
    logger.info(f"  Gender  - Accuracy: {gender_acc:.2f}%  |  F1: {gender_f1:.2f}%")
    logger.info(f"  Dialect - Accuracy: {dialect_acc:.2f}%  |  F1: {dialect_f1:.2f}%")
    
    logger.info("Gender Classification Report:")
    logger.info("\n" + classification_report(gender_labels, gender_preds,
                               target_names=['Male', 'Female'],
                               digits=4))
    
    logger.info("Dialect Classification Report:")
    logger.info("\n" + classification_report(dialect_labels, dialect_preds,
                               target_names=['North', 'Central', 'South'],
                               digits=4))
    
    logger.info("Gender Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(gender_labels, gender_preds)}")
    
    logger.info("Dialect Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(dialect_labels, dialect_preds)}")
    
    return {
        'gender_acc': gender_acc,
        'gender_f1': gender_f1,
        'dialect_acc': dialect_acc,
        'dialect_f1': dialect_f1
    }


def compare_with_baseline(clean_results, noisy_results, config):
    """Compare results with baseline from PACLIC 2024"""
    logger = get_logger()
    baseline = config['baseline']
    
    logger.info("COMPARISON WITH BASELINE (PACLIC 2024 - ResNet34)")
    logger.info("-" * 70)
    header = f"{'Task':<15} {'Test Set':<12} {'Baseline':<12} {'Our Model':<12} {'Delta':<12}"
    logger.info(header)
    logger.info("-" * 70)
    
    our_results = {
        'gender': {'clean': clean_results['gender_acc'], 'noisy': noisy_results['gender_acc']},
        'dialect': {'clean': clean_results['dialect_acc'], 'noisy': noisy_results['dialect_acc']}
    }
    
    for task in ['gender', 'dialect']:
        for test_set in ['clean', 'noisy']:
            baseline_val = baseline[task][test_set]
            our_val = our_results[task][test_set]
            delta = our_val - baseline_val
            delta_str = f"{delta:+.2f}%"
            
            logger.info(f"{task.capitalize():<15} {test_set.capitalize():<12} "
                  f"{baseline_val:<12.2f} {our_val:<12.2f} {delta_str:<12}")


def main(config_path):
    """Main evaluation function"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("SPEAKER PROFILING EVALUATION")
    logger.info("=" * 50)
    
    config = load_config(config_path)
    
    logger.info(f"Model checkpoint: {config['model']['checkpoint']}")
    logger.info("-" * 50)
    
    # Load feature extractor
    logger.info("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['model']['checkpoint'])
    
    # Load model
    logger.info("Loading model...")
    model = MultiTaskSpeakerModel(config['model']['name'])
    model = load_model_checkpoint(model, config['model']['checkpoint'])
    logger.info("Model loaded successfully")
    
    # Load test data
    logger.info("Loading test data...")
    
    clean_test_df = pd.read_csv(config['data']['clean_test_meta'])
    noisy_test_df = pd.read_csv(config['data']['noisy_test_meta'])
    
    gender_map = config['labels']['gender']
    dialect_map = config['labels']['dialect']
    
    for df in [clean_test_df, noisy_test_df]:
        df['gender_label'] = df['gender'].map(gender_map)
        df['dialect_label'] = df['dialect'].map(dialect_map)
    
    logger.info(f"Clean test: {len(clean_test_df):,} samples")
    logger.info(f"Noisy test: {len(noisy_test_df):,} samples")
    
    # Create datasets
    clean_test_dataset = ViSpeechDataset(
        clean_test_df, 
        config['data']['clean_test_audio'], 
        feature_extractor, 
        config
    )
    
    noisy_test_dataset = ViSpeechDataset(
        noisy_test_df, 
        config['data']['noisy_test_audio'], 
        feature_extractor, 
        config
    )
    
    # Create trainer for evaluation
    eval_args = TrainingArguments(
        output_dir=config['output']['dir'],
        per_device_eval_batch_size=config['evaluation']['batch_size'],
        dataloader_num_workers=config['evaluation']['dataloader_num_workers'],
        remove_unused_columns=False,
        report_to='none',
    )
    
    trainer = MultiTaskTrainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate
    clean_results = evaluate_dataset(trainer, clean_test_dataset, "Clean Test Set", config)
    noisy_results = evaluate_dataset(trainer, noisy_test_dataset, "Noisy Test Set", config)
    
    # Compare with baseline
    compare_with_baseline(clean_results, noisy_results, config)
    
    # Save results
    if config['output']['save_predictions']:
        os.makedirs(config['output']['dir'], exist_ok=True)
        results = {
            'clean_test': clean_results,
            'noisy_test': noisy_results
        }
        with open(os.path.join(config['output']['dir'], 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {config['output']['dir']}/results.json")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Speaker Profiling Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/eval.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    main(args.config)
