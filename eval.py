"""
Evaluation Script for Speaker Profiling Model

Usage:
    # Evaluate on pre-extracted features (recommended - fast)
    python eval.py --checkpoint output/best_model --test_dir datasets/clean_test
    
    # Evaluate on both clean and noisy test sets
    python eval.py --checkpoint output/best_model \\
        --test_dir datasets/clean_test \\
        --test_dir2 datasets/noisy_test
    
    # With custom settings
    python eval.py --checkpoint output/best_model \\
        --test_dir datasets/clean_test \\
        --batch_size 64 \\
        --output_dir results/
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.models import ClassificationHeadModel
from src.utils import setup_logging, get_logger


# ============================================================
# Dataset Class
# ============================================================

class FeatureDataset(Dataset):
    """Dataset for pre-extracted features"""
    
    def __init__(self, data_dir):
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


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(model, dataloader, device):
    """Run evaluation and return predictions"""
    model.eval()
    all_gender_preds, all_dialect_preds = [], []
    all_gender_labels, all_dialect_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['input_features'].to(device)
            outputs = model(input_features=features)
            
            all_gender_preds.extend(outputs['gender_logits'].argmax(dim=-1).cpu().numpy())
            all_dialect_preds.extend(outputs['dialect_logits'].argmax(dim=-1).cpu().numpy())
            all_gender_labels.extend(batch['gender_labels'].numpy())
            all_dialect_labels.extend(batch['dialect_labels'].numpy())
    
    return {
        'gender_preds': np.array(all_gender_preds),
        'dialect_preds': np.array(all_dialect_preds),
        'gender_labels': np.array(all_gender_labels),
        'dialect_labels': np.array(all_dialect_labels)
    }


def print_results(results, dataset_name, logger):
    """Print detailed evaluation results"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RESULTS ON {dataset_name.upper()}")
    logger.info("=" * 60)
    
    gender_acc = accuracy_score(results['gender_labels'], results['gender_preds']) * 100
    gender_f1 = f1_score(results['gender_labels'], results['gender_preds'], average='weighted') * 100
    dialect_acc = accuracy_score(results['dialect_labels'], results['dialect_preds']) * 100
    dialect_f1 = f1_score(results['dialect_labels'], results['dialect_preds'], average='weighted') * 100
    
    logger.info(f"Gender  - Accuracy: {gender_acc:.2f}%  |  F1: {gender_f1:.2f}%")
    logger.info(f"Dialect - Accuracy: {dialect_acc:.2f}%  |  F1: {dialect_f1:.2f}%")
    
    logger.info("")
    logger.info("--- Gender Classification Report ---")
    report = classification_report(results['gender_labels'], results['gender_preds'],
                                   target_names=['Male', 'Female'], digits=4)
    for line in report.split('\n'):
        logger.info(line)
    
    logger.info("")
    logger.info("--- Dialect Classification Report ---")
    report = classification_report(results['dialect_labels'], results['dialect_preds'],
                                   target_names=['North', 'Central', 'South'], digits=4)
    for line in report.split('\n'):
        logger.info(line)
    
    logger.info("")
    logger.info("Gender Confusion Matrix:")
    cm = confusion_matrix(results['gender_labels'], results['gender_preds'])
    logger.info(f"           Male  Female")
    logger.info(f"Male     {cm[0][0]:6d}  {cm[0][1]:6d}")
    logger.info(f"Female   {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    logger.info("")
    logger.info("Dialect Confusion Matrix:")
    cm = confusion_matrix(results['dialect_labels'], results['dialect_preds'])
    logger.info(f"           North  Central  South")
    logger.info(f"North    {cm[0][0]:6d}  {cm[0][1]:7d}  {cm[0][2]:5d}")
    logger.info(f"Central  {cm[1][0]:6d}  {cm[1][1]:7d}  {cm[1][2]:5d}")
    logger.info(f"South    {cm[2][0]:6d}  {cm[2][1]:7d}  {cm[2][2]:5d}")
    
    return {
        'dataset': dataset_name,
        'gender_acc': gender_acc,
        'gender_f1': gender_f1,
        'dialect_acc': dialect_acc,
        'dialect_f1': dialect_f1
    }


def compare_with_baseline(results_list, logger):
    """Compare results with PACLIC 2024 baseline"""
    # Baseline from PACLIC 2024 (ResNet34)
    baseline = {
        'gender': {'clean': 95.35, 'noisy': 88.71},
        'dialect': {'clean': 59.49, 'noisy': 45.67}
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON WITH BASELINE (PACLIC 2024 - ResNet34)")
    logger.info("=" * 70)
    logger.info(f"{'Task':<10} {'Test Set':<12} {'Baseline':<15} {'Our Model':<15} {'Delta':<10}")
    logger.info("-" * 70)
    
    for r in results_list:
        dataset_name = r['dataset'].lower()
        test_type = 'clean' if 'clean' in dataset_name else 'noisy'
        
        for task in ['gender', 'dialect']:
            baseline_val = baseline[task][test_type]
            our_val = r[f'{task}_acc']
            delta = our_val - baseline_val
            delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
            
            logger.info(f"{task.capitalize():<10} {test_type.capitalize():<12} "
                       f"{baseline_val:.2f}%{'':<8} {our_val:.2f}%{'':<8} {delta_str}")


def load_checkpoint(checkpoint_dir, device):
    """Load model from checkpoint directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Try different checkpoint formats
    if (checkpoint_dir / 'pytorch_model.bin').exists():
        state_dict = torch.load(checkpoint_dir / 'pytorch_model.bin', map_location=device)
    elif (checkpoint_dir / 'model.safetensors').exists():
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_dir / 'model.safetensors')
    else:
        # Find any .bin or .pt file
        model_files = list(checkpoint_dir.glob('*.bin')) + list(checkpoint_dir.glob('*.pt'))
        if model_files:
            state_dict = torch.load(model_files[0], map_location=device)
        else:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")
    
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speaker Profiling Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test features directory (with features/ and metadata.csv)")
    parser.add_argument("--test_dir2", type=str, default=None,
                        help="Path to second test features directory (optional, e.g., noisy test)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size of encoder (768 for base, 1024 for large)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results JSON")
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("SPEAKER PROFILING EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test dir: {args.test_dir}")
    if args.test_dir2:
        logger.info(f"Test dir 2: {args.test_dir2}")
    
    # Load model
    logger.info("")
    logger.info("Loading model...")
    model = ClassificationHeadModel(
        hidden_size=args.hidden_size,
        num_genders=2,
        num_dialects=3,
        dropout=0.1,
        head_hidden_dim=256,
        dialect_loss_weight=3.0
    )
    
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
    
    # Evaluate test sets
    results_list = []
    
    # Test set 1
    logger.info("")
    logger.info(f"Loading test data from {args.test_dir}...")
    test_dataset = FeatureDataset(args.test_dir)
    logger.info(f"Loaded {len(test_dataset)} samples")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Determine dataset name from path
    test_name = Path(args.test_dir).name
    if 'clean' in test_name.lower():
        test_name = "Clean Test Set"
    elif 'noisy' in test_name.lower():
        test_name = "Noisy Test Set"
    else:
        test_name = "Test Set"
    
    results = evaluate_model(model, test_loader, device)
    metrics = print_results(results, test_name, logger)
    results_list.append(metrics)
    
    # Test set 2 (optional)
    if args.test_dir2:
        logger.info("")
        logger.info(f"Loading test data from {args.test_dir2}...")
        test_dataset2 = FeatureDataset(args.test_dir2)
        logger.info(f"Loaded {len(test_dataset2)} samples")
        
        test_loader2 = DataLoader(
            test_dataset2, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        
        test_name2 = Path(args.test_dir2).name
        if 'clean' in test_name2.lower():
            test_name2 = "Clean Test Set"
        elif 'noisy' in test_name2.lower():
            test_name2 = "Noisy Test Set"
        else:
            test_name2 = "Test Set 2"
        
        results2 = evaluate_model(model, test_loader2, device)
        metrics2 = print_results(results2, test_name2, logger)
        results_list.append(metrics2)
    
    # Compare with baseline
    compare_with_baseline(results_list, logger)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'results.json')
        with open(output_file, 'w') as f:
            json.dump(results_list, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
