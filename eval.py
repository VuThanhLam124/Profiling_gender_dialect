"""
Evaluation Script for Speaker Profiling Model
Architecture: WavLM + Attentive Pooling + LayerNorm + Deeper Heads

Usage:
    python eval.py --config configs/eval.yaml
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import librosa
from omegaconf import OmegaConf
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import Wav2Vec2FeatureExtractor, Trainer, TrainingArguments
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

from src.models import MultiTaskSpeakerModel


def load_config(config_path):
    """Load configuration from yaml file using OmegaConf"""
    config = OmegaConf.load(config_path)
    return config


class ViSpeechDataset(Dataset):
    """Dataset class for evaluation"""
    
    def __init__(self, dataframe, audio_dir, feature_extractor, config):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.feature_extractor = feature_extractor
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_length = int(self.sampling_rate * config['audio']['max_duration'])
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, audio_name):
        audio_path = self.audio_dir / audio_name
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
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
    """Evaluate model on a dataset and print detailed report"""
    
    print(f"\nEVALUATING ON {dataset_name.upper()}")
    print("-" * 50)
    
    results = trainer.predict(dataset)
    gender_logits, dialect_logits = results.predictions
    gender_labels, dialect_labels = results.label_ids
    
    gender_preds = np.argmax(gender_logits, axis=-1)
    dialect_preds = np.argmax(dialect_logits, axis=-1)
    
    gender_acc = accuracy_score(gender_labels, gender_preds) * 100
    dialect_acc = accuracy_score(dialect_labels, dialect_preds) * 100
    gender_f1 = f1_score(gender_labels, gender_preds, average='weighted') * 100
    dialect_f1 = f1_score(dialect_labels, dialect_preds, average='weighted') * 100
    
    print(f"\nOverall Metrics:")
    print(f"  Gender  - Accuracy: {gender_acc:.2f}%  |  F1: {gender_f1:.2f}%")
    print(f"  Dialect - Accuracy: {dialect_acc:.2f}%  |  F1: {dialect_f1:.2f}%")
    
    print(f"\nGender Classification Report:")
    print(classification_report(gender_labels, gender_preds,
                               target_names=['Male', 'Female'],
                               digits=4))
    
    print(f"\nDialect Classification Report:")
    print(classification_report(dialect_labels, dialect_preds,
                               target_names=['North', 'Central', 'South'],
                               digits=4))
    
    print(f"\nGender Confusion Matrix:")
    print(confusion_matrix(gender_labels, gender_preds))
    
    print(f"\nDialect Confusion Matrix:")
    print(confusion_matrix(dialect_labels, dialect_preds))
    
    return {
        'gender_acc': gender_acc,
        'gender_f1': gender_f1,
        'dialect_acc': dialect_acc,
        'dialect_f1': dialect_f1
    }


def compare_with_baseline(clean_results, noisy_results, config):
    """Compare results with baseline from PACLIC 2024"""
    
    baseline = config['baseline']
    
    print("\nCOMPARISON WITH BASELINE (PACLIC 2024 - ResNet34)")
    print("-" * 70)
    print(f"{'Task':<15} {'Test Set':<12} {'Baseline':<12} {'Our Model':<12} {'Delta':<12}")
    print("-" * 70)
    
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
            
            print(f"{task.capitalize():<15} {test_set.capitalize():<12} "
                  f"{baseline_val:<12.2f} {our_val:<12.2f} {delta_str:<12}")


def main(config_path):
    """Main evaluation function"""
    
    print("SPEAKER PROFILING EVALUATION")
    print("-" * 50)
    
    config = load_config(config_path)
    
    print(f"Model checkpoint: {config['model']['checkpoint']}")
    print("-" * 50)
    
    # Load feature extractor
    print("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['model']['checkpoint'])
    
    # Load model
    print("Loading model...")
    model = MultiTaskSpeakerModel(config['model']['name'])
    
    # Load checkpoint weights
    checkpoint_path = os.path.join(config['model']['checkpoint'], 'model.safetensors')
    if os.path.exists(checkpoint_path):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        checkpoint_path = os.path.join(config['model']['checkpoint'], 'pytorch_model.bin')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    print("Model loaded successfully")
    
    # Load test data
    print("\nLoading test data...")
    
    clean_test_df = pd.read_csv(config['data']['clean_test_meta'])
    noisy_test_df = pd.read_csv(config['data']['noisy_test_meta'])
    
    gender_map = config['labels']['gender']
    dialect_map = config['labels']['dialect']
    
    for df in [clean_test_df, noisy_test_df]:
        df['gender_label'] = df['gender'].map(gender_map)
        df['dialect_label'] = df['dialect'].map(dialect_map)
    
    print(f"Clean test: {len(clean_test_df):,} samples")
    print(f"Noisy test: {len(noisy_test_df):,} samples")
    
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
        print(f"\nResults saved to {config['output']['dir']}/results.json")
    
    print("\nEvaluation completed!")


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
