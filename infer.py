"""
Inference Script for Speaker Profiling Model
Architecture: WavLM + Attentive Pooling + LayerNorm + Deeper Heads

Usage:
    python infer.py --config configs/infer.yaml --audio path/to/audio.wav
    python infer.py --config configs/infer.yaml --audio_dir path/to/audio_folder
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from omegaconf import OmegaConf
import json
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from yaml file using OmegaConf"""
    config = OmegaConf.load(config_path)
    return config


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
    
    def __init__(self, model_name, num_genders=2, num_dialects=3, 
                 dropout=0.1, head_hidden_dim=256):
        super().__init__()
        
        self.wavlm = WavLMModel.from_pretrained(model_name)
        hidden_size = self.wavlm.config.hidden_size
        
        self.attentive_pooling = AttentivePooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_genders)
        )
        
        self.dialect_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, num_dialects)
        )
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        pooled, attn_weights = self.attentive_pooling(hidden_states, attention_mask)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        gender_logits = self.gender_head(pooled)
        dialect_logits = self.dialect_head(pooled)
        
        return gender_logits, dialect_logits


class SpeakerProfiler:
    """Speaker Profiler for inference"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['inference']['device'] if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_length = int(self.sampling_rate * config['audio']['max_duration'])
        
        self.gender_labels = config['labels']['gender']
        self.dialect_labels = config['labels']['dialect']
        
        self._load_model()
    
    def _load_model(self):
        """Load model and feature extractor"""
        print("Loading model...")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config['model']['checkpoint']
        )
        
        self.model = MultiTaskSpeakerModel(self.config['model']['name'])
        
        checkpoint_path = os.path.join(self.config['model']['checkpoint'], 'model.safetensors')
        if os.path.exists(checkpoint_path):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            self.model.load_state_dict(state_dict)
        else:
            checkpoint_path = os.path.join(self.config['model']['checkpoint'], 'pytorch_model.bin')
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values
    
    def predict(self, audio_path):
        """Predict gender and dialect from audio file"""
        input_values = self.preprocess_audio(audio_path)
        input_values = input_values.to(self.device)
        
        with torch.no_grad():
            gender_logits, dialect_logits = self.model(input_values)
        
        gender_probs = torch.softmax(gender_logits, dim=-1).cpu().numpy()[0]
        dialect_probs = torch.softmax(dialect_logits, dim=-1).cpu().numpy()[0]
        
        gender_pred = int(np.argmax(gender_probs))
        dialect_pred = int(np.argmax(dialect_probs))
        
        result = {
            'audio_path': str(audio_path),
            'gender': {
                'prediction': self.gender_labels[gender_pred],
                'code': gender_pred,
                'confidence': float(gender_probs[gender_pred]),
                'probabilities': {
                    self.gender_labels[0]: float(gender_probs[0]),
                    self.gender_labels[1]: float(gender_probs[1])
                }
            },
            'dialect': {
                'prediction': self.dialect_labels[dialect_pred],
                'code': dialect_pred,
                'confidence': float(dialect_probs[dialect_pred]),
                'probabilities': {
                    self.dialect_labels[0]: float(dialect_probs[0]),
                    self.dialect_labels[1]: float(dialect_probs[1]),
                    self.dialect_labels[2]: float(dialect_probs[2])
                }
            }
        }
        
        return result
    
    def predict_batch(self, audio_paths):
        """Predict gender and dialect for multiple audio files"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': str(audio_path),
                    'error': str(e)
                })
        return results


def print_result(result):
    """Print prediction result"""
    print(f"\nAudio: {result['audio_path']}")
    print("-" * 40)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    gender = result['gender']
    dialect = result['dialect']
    
    print(f"Gender:  {gender['prediction']} (code: {gender['code']}, confidence: {gender['confidence']:.2%})")
    print(f"Dialect: {dialect['prediction']} (code: {dialect['code']}, confidence: {dialect['confidence']:.2%})")


def main(config_path, audio_path=None, audio_dir=None):
    """Main inference function"""
    
    print("SPEAKER PROFILING INFERENCE")
    print("-" * 50)
    
    config = load_config(config_path)
    
    if audio_path:
        config['input']['audio_path'] = audio_path
    if audio_dir:
        config['input']['audio_dir'] = audio_dir
    
    profiler = SpeakerProfiler(config)
    
    audio_files = []
    
    if config['input']['audio_path']:
        audio_files.append(config['input']['audio_path'])
    
    if config['input']['audio_dir']:
        audio_dir = Path(config['input']['audio_dir'])
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
            audio_files.extend(audio_dir.glob(ext))
    
    if not audio_files:
        print("No audio files found. Please specify --audio or --audio_dir")
        return
    
    print(f"\nProcessing {len(audio_files)} audio file(s)...")
    
    results = profiler.predict_batch(audio_files)
    
    for result in results:
        print_result(result)
    
    if config['output']['save_results']:
        os.makedirs(config['output']['dir'], exist_ok=True)
        output_path = os.path.join(config['output']['dir'], 'predictions.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    
    print("\nInference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Profiling Inference")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/infer.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        default=None,
        help="Path to single audio file"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default=None,
        help="Path to directory containing audio files"
    )
    args = parser.parse_args()
    
    main(args.config, args.audio, args.audio_dir)
