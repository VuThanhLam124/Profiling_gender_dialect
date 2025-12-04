"""
Inference Script for Speaker Profiling Model

Usage:
    python infer.py --config configs/infer.yaml --audio path/to/audio.wav
    python infer.py --config configs/infer.yaml --audio_dir path/to/audio_folder
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor

from src.models import MultiTaskSpeakerModel
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    get_device,
    load_model_checkpoint,
    load_and_preprocess_audio
)


class SpeakerProfiler:
    """Speaker Profiler for inference"""
    
    def __init__(self, config):
        self.logger = get_logger()
        self.config = config
        self.device = get_device(config['inference']['device'])
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_duration = config['audio']['max_duration']
        
        self.gender_labels = config['labels']['gender']
        self.dialect_labels = config['labels']['dialect']
        
        # Check if this is a Whisper/PhoWhisper model
        model_name = config['model']['name'].lower()
        self.is_whisper = 'whisper' in model_name or 'phowhisper' in model_name
        
        self._load_model()
    
    def _load_model(self):
        """Load model and feature extractor"""
        self.logger.info("Loading model...")
        
        model_name = self.config['model']['name']
        is_ecapa = 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower()
        
        if is_ecapa:
            # ECAPA-TDNN: use Wav2Vec2 feature extractor for audio normalization
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            )
        elif self.is_whisper:
            # Whisper/PhoWhisper: use WhisperFeatureExtractor
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_name
            )
        else:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.config['model']['checkpoint']
            )
        
        self.model = MultiTaskSpeakerModel(model_name)
        self.model = load_model_checkpoint(
            self.model,
            self.config['model']['checkpoint'],
            str(self.device)
        )
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model loaded on {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Whisper requires 30 seconds of audio
        if self.is_whisper:
            max_duration = 30
        else:
            max_duration = self.max_duration
        
        audio = load_and_preprocess_audio(
            audio_path,
            sampling_rate=self.sampling_rate,
            max_duration=max_duration
        )
        
        # Whisper needs exactly 30 seconds - pad if necessary
        if self.is_whisper:
            target_len = self.sampling_rate * 30
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Whisper uses 'input_features', WavLM/HuBERT/Wav2Vec2 use 'input_values'
        if self.is_whisper:
            return inputs.input_features
        else:
            return inputs.input_values
    
    def predict(self, audio_path):
        """Predict gender and dialect from audio file"""
        input_values = self.preprocess_audio(audio_path)
        input_values = input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            gender_logits = outputs['gender_logits']
            dialect_logits = outputs['dialect_logits']
        
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
                self.logger.error(f"Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': str(audio_path),
                    'error': str(e)
                })
        return results


def log_result(result, logger):
    """Log prediction result"""
    logger.info(f"Audio: {result['audio_path']}")
    
    if 'error' in result:
        logger.error(f"Error: {result['error']}")
        return
    
    gender = result['gender']
    dialect = result['dialect']
    
    logger.info(
        f"  Gender: {gender['prediction']} "
        f"(code: {gender['code']}, confidence: {gender['confidence']:.2%})"
    )
    logger.info(
        f"  Dialect: {dialect['prediction']} "
        f"(code: {dialect['code']}, confidence: {dialect['confidence']:.2%})"
    )


def main(config_path, audio_path=None, audio_dir=None):
    """Main inference function"""
    logger = setup_logging()
    
    logger.info("=" * 50)
    logger.info("SPEAKER PROFILING INFERENCE")
    logger.info("=" * 50)
    
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
        logger.warning("No audio files found. Please specify --audio or --audio_dir")
        return
    
    logger.info(f"Processing {len(audio_files)} audio file(s)...")
    
    results = profiler.predict_batch(audio_files)
    
    logger.info("-" * 50)
    for result in results:
        log_result(result, logger)
    logger.info("-" * 50)
    
    if config['output']['save_results']:
        os.makedirs(config['output']['dir'], exist_ok=True)
        output_path = os.path.join(config['output']['dir'], 'predictions.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    logger.info("Inference completed!")


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
