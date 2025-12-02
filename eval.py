"""
Evaluation Script for Speaker Profiling Model

Usage:
    # Evaluate with raw audio (default - using config file)
    python eval.py --checkpoint output/best_model --config configs/finetune.yaml \\
        --test_name clean_test --test_name2 noisy_test
    
    # With custom output directory
    python eval.py --checkpoint output/best_model --config configs/finetune.yaml \\
        --test_name clean_test --output_dir results/
"""

import os
import io
import argparse
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoFeatureExtractor

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from src.models import MultiTaskSpeakerModel
from src.utils import setup_logging, get_logger


# ============================================================
# Dataset Class
# ============================================================

class RawAudioTestDataset(Dataset):
    """Dataset for raw audio evaluation"""
    
    def __init__(self, metadata_path, audio_dir, config, feature_extractor):
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_csv(metadata_path)
        self.config = config
        self.feature_extractor = feature_extractor
        self.sr = config.get('audio', {}).get('sampling_rate', 16000)
        self.max_duration = config.get('audio', {}).get('max_duration', 5)
        
        # Auto-detect column names
        col_mapping = {
            'audio_name': ['audio_name', 'filename', 'file', 'path', 'audio_path'],
            'gender': ['gender', 'sex'],
            'dialect': ['dialect', 'accent', 'region'],
        }
        
        self.col_names = {}
        for target, candidates in col_mapping.items():
            for col in candidates:
                if col in self.df.columns:
                    self.col_names[target] = col
                    break
        
        # Label mappings
        labels_config = config.get('labels', {})
        self.gender_map = labels_config.get('gender', {'Male': 0, 'Female': 1})
        self.dialect_map = labels_config.get('dialect', {'North': 0, 'Central': 1, 'South': 2})
    
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, path):
        """Load and preprocess audio"""
        waveform, _ = librosa.load(path, sr=self.sr)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        
        # Normalize
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        
        # Truncate/pad
        max_len = int(self.sr * self.max_duration)
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        
        return waveform
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get audio path (auto-detected column)
        audio_col = self.col_names.get('audio_name', 'filename')
        audio_path = self.audio_dir / row[audio_col]
        waveform = self._load_audio(str(audio_path))
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=self.sr, 
            return_tensors="pt",
            padding=True
        )
        
        # Get labels (auto-detected columns)
        gender_col = self.col_names.get('gender', 'gender')
        dialect_col = self.col_names.get('dialect', 'accent')
        
        gender_label = self.gender_map.get(row[gender_col], 0)
        dialect_label = self.dialect_map.get(row[dialect_col], 0)
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'gender_labels': torch.tensor(gender_label, dtype=torch.long),
            'dialect_labels': torch.tensor(dialect_label, dtype=torch.long)
        }


class ViMDTestDataset(Dataset):
    """Dataset for ViMD evaluation (HuggingFace format)"""
    
    def __init__(self, hf_dataset, config, feature_extractor):
        self.dataset = hf_dataset
        self.config = config
        self.feature_extractor = feature_extractor
        self.sr = config.get('audio', {}).get('sampling_rate', 16000)
        self.max_duration = config.get('audio', {}).get('max_duration', 5)
        self.max_length = int(self.sr * self.max_duration)
        
        # Label mappings
        labels_config = config.get('labels', {})
        self.gender_map = labels_config.get('gender', {'Male': 0, 'Female': 1, 0: 0, 1: 1})
        self.region_to_dialect = labels_config.get('region_to_dialect', {
            'North': 0, 'Central': 1, 'South': 2
        })
    
    def __len__(self):
        return len(self.dataset)
    
    def _load_audio_from_hf(self, audio_data):
        """Load audio from HuggingFace dataset format"""
        try:
            audio = None
            
            if isinstance(audio_data, dict):
                if 'path' in audio_data and audio_data['path']:
                    try:
                        audio, sr = librosa.load(audio_data['path'], sr=self.sr, mono=True)
                    except Exception:
                        pass
                
                if audio is None and 'bytes' in audio_data and audio_data['bytes']:
                    try:
                        audio_bytes = io.BytesIO(audio_data['bytes'])
                        audio, sr = sf.read(audio_bytes)
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)
                        if sr != self.sr:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                    except Exception:
                        pass
                
                if audio is None and 'array' in audio_data:
                    try:
                        audio = np.array(audio_data['array'], dtype=np.float32)
                        sr = audio_data.get('sampling_rate', self.sr)
                        if sr != self.sr:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                    except Exception:
                        pass
            
            elif isinstance(audio_data, str):
                audio, sr = librosa.load(audio_data, sr=self.sr, mono=True)
            
            if audio is None or len(audio) == 0:
                return np.zeros(self.max_length, dtype=np.float32)
            
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if len(audio) > 100:
                try:
                    audio, _ = librosa.effects.trim(audio, top_db=20)
                except Exception:
                    pass
            
            max_val = np.max(np.abs(audio))
            if max_val > 1e-8:
                audio = audio / max_val
            
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            
            return audio.astype(np.float32)
            
        except Exception:
            return np.zeros(self.max_length, dtype=np.float32)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            audio = self._load_audio_from_hf(item.get('audio'))
            
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True
            )
            
            gender_raw = item.get('gender', 0)
            region = item.get('region', 'North')
            
            if isinstance(gender_raw, int):
                gender_label = gender_raw
            else:
                gender_label = self.gender_map.get(gender_raw, 0)
            
            dialect_label = self.region_to_dialect.get(region, 0)
            
            return {
                'input_values': inputs.input_values.squeeze(0),
                'gender_labels': torch.tensor(gender_label, dtype=torch.long),
                'dialect_labels': torch.tensor(dialect_label, dtype=torch.long)
            }
            
        except Exception:
            dummy_audio = np.zeros(self.max_length, dtype=np.float32)
            inputs = self.feature_extractor(
                dummy_audio,
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=True
            )
            return {
                'input_values': inputs.input_values.squeeze(0),
                'gender_labels': torch.tensor(0, dtype=torch.long),
                'dialect_labels': torch.tensor(0, dtype=torch.long)
            }


def collate_fn(batch):
    """Custom collate function for variable length audio"""
    max_len = max(item['input_values'].shape[0] for item in batch)
    
    input_values = []
    attention_mask = []
    
    for item in batch:
        seq_len = item['input_values'].shape[0]
        # Pad
        padded = torch.zeros(max_len)
        padded[:seq_len] = item['input_values']
        input_values.append(padded)
        
        # Attention mask
        mask = torch.zeros(max_len)
        mask[:seq_len] = 1.0
        attention_mask.append(mask)
    
    return {
        'input_values': torch.stack(input_values),
        'attention_mask': torch.stack(attention_mask),
        'gender_labels': torch.stack([item['gender_labels'] for item in batch]),
        'dialect_labels': torch.stack([item['dialect_labels'] for item in batch])
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
            input_values = batch['input_values'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            
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


def load_vimd_test_data(config):
    """Load ViMD test dataset from HuggingFace format"""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library required for ViMD. Install: pip install datasets")
    
    vimd_path = config['data']['vimd_path']
    ds = load_dataset(vimd_path, keep_in_memory=False)
    
    return ds.get('test')


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
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (for dataset paths)")
    parser.add_argument("--test_name", type=str, default="clean_test",
                        help="Test set name: 'clean_test', 'noisy_test', or 'vimd_test'")
    parser.add_argument("--test_name2", type=str, default=None,
                        help="Second test set name (optional)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results JSON")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("SPEAKER PROFILING EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test set: {args.test_name}")
    if args.test_name2:
        logger.info(f"Test set 2: {args.test_name2}")
    
    # Load feature extractor
    model_name = config.get('model', {}).get('name', 'microsoft/wavlm-base-plus')
    
    # Check if model is ECAPA-TDNN (SpeechBrain)
    is_ecapa = 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower()
    
    if is_ecapa:
        # ECAPA-TDNN: use Wav2Vec2 feature extractor for audio normalization
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        logger.info("Using Wav2Vec2 feature extractor for ECAPA-TDNN")
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    # Load model
    logger.info("")
    logger.info("Loading model...")
    model_config = config.get('model', {})
    model = MultiTaskSpeakerModel(
        model_name=model_name,
        num_genders=model_config.get('num_genders', 2),
        num_dialects=model_config.get('num_dialects', 3),
        dropout=model_config.get('dropout', 0.15),
        head_hidden_dim=model_config.get('head_hidden_dim', 256),
        freeze_encoder=False
    )
    
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
    
    # Helper to create dataset
    def create_test_dataset(test_name):
        if test_name == 'vimd_test':
            vimd_test = load_vimd_test_data(config)
            logger.info(f"Loaded ViMD test: {len(vimd_test)} samples")
            return ViMDTestDataset(vimd_test, config, feature_extractor), "ViMD Test Set"
        else:
            data_config = config.get('data', {})
            meta_key = f"{test_name}_meta"
            audio_key = f"{test_name}_audio"
            test_meta = data_config.get(meta_key)
            test_audio = data_config.get(audio_key)
            logger.info(f"  Metadata: {test_meta}")
            logger.info(f"  Audio: {test_audio}")
            dataset = RawAudioTestDataset(test_meta, test_audio, config, feature_extractor)
            display_name = "Clean Test Set" if 'clean' in test_name else "Noisy Test Set"
            return dataset, display_name
    
    # Evaluate test sets
    results_list = []
    
    # Test set 1
    logger.info("")
    logger.info(f"Loading test data: {args.test_name}")
    test_dataset, test_display = create_test_dataset(args.test_name)
    logger.info(f"Loaded {len(test_dataset)} samples")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    results = evaluate_model(model, test_loader, device)
    metrics = print_results(results, test_display, logger)
    results_list.append(metrics)
    
    # Test set 2 (optional)
    if args.test_name2:
        logger.info("")
        logger.info(f"Loading test data: {args.test_name2}")
        test_dataset2, test_display2 = create_test_dataset(args.test_name2)
        logger.info(f"Loaded {len(test_dataset2)} samples")
        
        test_loader2 = DataLoader(
            test_dataset2, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        results2 = evaluate_model(model, test_loader2, device)
        metrics2 = print_results(results2, test_display2, logger)
        results_list.append(metrics2)
    
    # Compare with baseline (only for ViSpeech)
    if 'vimd' not in args.test_name:
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
