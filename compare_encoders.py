"""
Compare Different Encoders for Speaker Profiling

This script trains and evaluates models with different encoders:
- WavLM (microsoft/wavlm-base-plus)
- HuBERT (facebook/hubert-base-ls960)
- Wav2Vec2 (facebook/wav2vec2-base-960h)
- Whisper (openai/whisper-small)
- ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)

Usage:
    python compare_encoders.py --config configs/finetune.yaml --output_dir results/comparison

Output:
    - Training logs for each encoder
    - Comparison table (CSV and Markdown)
    - Best model checkpoint for each encoder
"""

import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.utils import setup_logging, get_logger, set_seed


# Encoders to compare
ENCODERS = {
    'wavlm-base': {
        'name': 'microsoft/wavlm-base-plus',
        'hidden_size': 768,
        'description': 'WavLM Base Plus'
    },
    'hubert-base': {
        'name': 'facebook/hubert-base-ls960',
        'hidden_size': 768,
        'description': 'HuBERT Base'
    },
    'wav2vec2-base': {
        'name': 'facebook/wav2vec2-base-960h',
        'hidden_size': 768,
        'description': 'Wav2Vec2 Base'
    },
    'whisper-small': {
        'name': 'openai/whisper-small',
        'hidden_size': 768,
        'description': 'Whisper Small'
    },
    'ecapa-tdnn': {
        'name': 'speechbrain/spkrec-ecapa-voxceleb',
        'hidden_size': 192,
        'description': 'ECAPA-TDNN (SpeechBrain)',
        'is_ecapa': True
    },
}


def extract_features_for_encoder(
    encoder_key: str,
    config: dict,
    output_base_dir: Path,
    logger
) -> Path:
    """Extract features using specified encoder"""
    from prepare_data import ViSpeechDataset, FeatureExtractor, extract_and_save_features
    
    encoder_info = ENCODERS[encoder_key]
    model_name = encoder_info['name']
    
    # Output directory for this encoder's features
    feature_dir = output_base_dir / 'features' / encoder_key
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting features with: {encoder_info['description']}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {feature_dir}")
    logger.info(f"{'='*60}")
    
    # Create feature extractor
    extractor = FeatureExtractor(model_name=model_name)
    
    # Extract train features
    train_dir = feature_dir / 'train'
    if not (train_dir / 'metadata.csv').exists():
        train_dataset = ViSpeechDataset(config, split='train')
        extract_and_save_features(train_dataset, extractor, train_dir)
    else:
        logger.info(f"Train features already exist at {train_dir}")
    
    # Extract val features
    val_dir = feature_dir / 'val'
    if not (val_dir / 'metadata.csv').exists():
        val_dataset = ViSpeechDataset(config, split='val')
        extract_and_save_features(val_dataset, extractor, val_dir)
    else:
        logger.info(f"Val features already exist at {val_dir}")
    
    return feature_dir


def train_model_for_encoder(
    encoder_key: str,
    feature_dir: Path,
    config: dict,
    output_base_dir: Path,
    logger
) -> dict:
    """Train model with pre-extracted features"""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import Dataset, DataLoader
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    
    from src.models import ClassificationHeadModel
    
    encoder_info = ENCODERS[encoder_key]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with: {encoder_info['description']}")
    logger.info(f"Features from: {feature_dir}")
    logger.info(f"{'='*60}")
    
    # Dataset class
    class FeatureDataset(Dataset):
        def __init__(self, data_dir: Path):
            self.data_dir = Path(data_dir)
            self.feature_dir = self.data_dir / 'features'
            self.df = pd.read_csv(self.data_dir / 'metadata.csv')
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            features = np.load(self.feature_dir / row['feature_name'])
            return {
                'input_features': torch.from_numpy(features).float(),
                'gender_labels': torch.tensor(row['gender_label'], dtype=torch.long),
                'dialect_labels': torch.tensor(row['dialect_label'], dtype=torch.long)
            }
    
    # Load datasets
    train_dataset = FeatureDataset(feature_dir / 'train')
    val_dataset = FeatureDataset(feature_dir / 'val')
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = ClassificationHeadModel(
        hidden_size=encoder_info['hidden_size'],
        num_genders=config['model'].get('num_genders', 2),
        num_dialects=config['model'].get('num_dialects', 3),
        dropout=config['model'].get('dropout', 0.1),
        head_hidden_dim=config['model'].get('head_hidden_dim', 256),
        dialect_loss_weight=config.get('loss', {}).get('dialect_weight', 3.0)
    )
    
    # Output directory
    output_dir = output_base_dir / 'checkpoints' / encoder_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        metric_for_best_model='dialect_acc',
        greater_is_better=True,
        save_total_limit=2,
        fp16=config['training'].get('fp16', True),
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 2),
        logging_steps=50,
        report_to='none',
        remove_unused_columns=False,
        seed=config.get('seed', 42),
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        gender_logits, dialect_logits = predictions[0], predictions[1]
        gender_labels, dialect_labels = labels[0], labels[1]
        
        gender_preds = np.argmax(gender_logits, axis=1)
        dialect_preds = np.argmax(dialect_logits, axis=1)
        
        return {
            'gender_acc': accuracy_score(gender_labels, gender_preds),
            'gender_f1': f1_score(gender_labels, gender_preds, average='weighted'),
            'dialect_acc': accuracy_score(dialect_labels, dialect_preds),
            'dialect_f1': f1_score(dialect_labels, dialect_preds, average='weighted'),
        }
    
    # Custom trainer
    class MultiTaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(
                input_features=inputs['input_features'],
                gender_labels=inputs['gender_labels'],
                dialect_labels=inputs['dialect_labels']
            )
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(
                    input_features=inputs['input_features'],
                    gender_labels=inputs['gender_labels'],
                    dialect_labels=inputs['dialect_labels']
                )
            
            loss = outputs['loss']
            logits = (outputs['gender_logits'], outputs['dialect_logits'])
            labels = (inputs['gender_labels'], inputs['dialect_labels'])
            
            return (loss, logits, labels)
    
    # Train
    start_time = time.time()
    
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_threshold=config['early_stopping']['threshold']
        )],
    )
    
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Save best model
    best_model_dir = output_dir / 'best_model'
    trainer.save_model(str(best_model_dir))
    
    # Results
    results = {
        'encoder': encoder_key,
        'model_name': encoder_info['name'],
        'description': encoder_info['description'],
        'hidden_size': encoder_info['hidden_size'],
        'gender_acc': eval_results['eval_gender_acc'],
        'gender_f1': eval_results['eval_gender_f1'],
        'dialect_acc': eval_results['eval_dialect_acc'],
        'dialect_f1': eval_results['eval_dialect_f1'],
        'training_time_min': training_time / 60,
        'best_model_path': str(best_model_dir),
    }
    
    logger.info(f"\nResults for {encoder_info['description']}:")
    logger.info(f"  Gender Acc: {results['gender_acc']:.4f}")
    logger.info(f"  Gender F1:  {results['gender_f1']:.4f}")
    logger.info(f"  Dialect Acc: {results['dialect_acc']:.4f}")
    logger.info(f"  Dialect F1:  {results['dialect_f1']:.4f}")
    logger.info(f"  Training Time: {results['training_time_min']:.2f} min")
    
    return results


def create_comparison_table(results: list, output_dir: Path, logger):
    """Create comparison table in CSV and Markdown format"""
    df = pd.DataFrame(results)
    
    # Sort by dialect_acc (primary metric)
    df = df.sort_values('dialect_acc', ascending=False)
    
    # Save CSV
    csv_path = output_dir / 'comparison_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Create Markdown table
    md_content = f"""# Encoder Comparison Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Encoder | Gender Acc | Gender F1 | Dialect Acc | Dialect F1 | Time (min) |
|---------|------------|-----------|-------------|------------|------------|
"""
    
    for _, row in df.iterrows():
        md_content += f"| {row['description']} | {row['gender_acc']:.4f} | {row['gender_f1']:.4f} | {row['dialect_acc']:.4f} | {row['dialect_f1']:.4f} | {row['training_time_min']:.1f} |\n"
    
    # Best encoder
    best = df.iloc[0]
    md_content += f"""
## Best Encoder

**{best['description']}** achieved the highest dialect accuracy of **{best['dialect_acc']:.4f}**.

### Model Details
- Model Name: `{best['model_name']}`
- Hidden Size: {best['hidden_size']}
- Gender Accuracy: {best['gender_acc']:.4f}
- Dialect Accuracy: {best['dialect_acc']:.4f}
- Best Model Path: `{best['best_model_path']}`

## Notes

- All models use the same architecture: **Attentive Pooling + LayerNorm + Classification Heads**
- Only the encoder backbone is different
- Dialect classification is weighted 3x in the loss function
"""
    
    md_path = output_dir / 'comparison_results.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"Saved Markdown: {md_path}")
    
    # Print table
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    print(df[['description', 'gender_acc', 'dialect_acc', 'training_time_min']].to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare Different Encoders")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/encoder_comparison",
        help="Output directory for results"
    )
    parser.add_argument(
        "--encoders",
        type=str,
        nargs='+',
        default=list(ENCODERS.keys()),
        choices=list(ENCODERS.keys()),
        help="Encoders to compare"
    )
    parser.add_argument(
        "--skip_extraction",
        action='store_true',
        help="Skip feature extraction (use existing features)"
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'comparison.log'
    logger = setup_logging(log_file=str(log_file))
    
    logger.info("="*60)
    logger.info("ENCODER COMPARISON FOR SPEAKER PROFILING")
    logger.info("="*60)
    
    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    set_seed(config.get('seed', 42))
    
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Encoders: {args.encoders}")
    
    results = []
    
    for encoder_key in args.encoders:
        try:
            # Extract features
            if not args.skip_extraction:
                feature_dir = extract_features_for_encoder(
                    encoder_key, config, output_dir, logger
                )
            else:
                feature_dir = output_dir / 'features' / encoder_key
            
            # Train and evaluate
            result = train_model_for_encoder(
                encoder_key, feature_dir, config, output_dir, logger
            )
            results.append(result)
            
            # Save intermediate results
            with open(output_dir / 'results_intermediate.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error with encoder {encoder_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison table
    if results:
        create_comparison_table(results, output_dir, logger)
        
        # Save final results
        with open(output_dir / 'results_final.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    logger.info("\nComparison completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
