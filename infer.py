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
import librosa
from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor

from src.models import MultiTaskSpeakerModel
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    get_device,
    load_model_checkpoint,
    resolve_checkpoint_dir,
    load_and_preprocess_audio
)

def _detect_head_hidden_dim(checkpoint_dir: str) -> int:
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import safe_open
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                if "gender_head.0.weight" in f.keys():
                    return int(f.get_tensor("gender_head.0.weight").shape[0])
        except Exception:
            pass
    return 256


class SpeakerProfiler:
    """Speaker Profiler for inference"""
    
    def __init__(self, config):
        self.logger = get_logger()
        self.config = config
        self.device = get_device(config['inference']['device'])
        self.sampling_rate = config['audio']['sampling_rate']
        self.max_duration = config['audio']['max_duration']
        self.checkpoint_dir = resolve_checkpoint_dir(config['model']['checkpoint'])
        self.preprocess_mode = (config.get("preprocess", {}) or {}).get("mode", "space_v2")
        
        self.gender_labels = config['labels']['gender']
        self.dialect_labels = config['labels']['dialect']
        
        # Check if this is a Whisper/PhoWhisper model
        model_name = config['model']['name'].lower()
        self.is_whisper = 'whisper' in model_name or 'phowhisper' in model_name
        
        self._load_model()

    def _load_audio_array(self, audio_path: str) -> np.ndarray:
        if self.preprocess_mode == "default":
            max_duration = 30 if self.is_whisper else self.max_duration
            audio = load_and_preprocess_audio(
                audio_path,
                sampling_rate=self.sampling_rate,
                max_duration=max_duration
            )

            if self.is_whisper:
                target_len = self.sampling_rate * 30
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))

            return audio

        # space_v2: match HF Space v2 behavior
        max_duration = 30 if self.is_whisper else self.max_duration
        waveform, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        max_samples = int(max_duration * self.sampling_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        if self.is_whisper:
            whisper_length = self.sampling_rate * 30
            if len(waveform) < whisper_length:
                waveform = np.pad(waveform, (0, whisper_length - len(waveform)))

        return waveform

    def _extract_batch_inputs(self, audio_batch):
        inputs = self.feature_extractor(
            audio_batch,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )

        if self.is_whisper:
            return inputs.input_features, None

        attention_mask = getattr(inputs, "attention_mask", None)
        return inputs.input_values, attention_mask
    
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
                model_name
            )

        head_hidden_dim = _detect_head_hidden_dim(self.checkpoint_dir)
        self.model = MultiTaskSpeakerModel(model_name, head_hidden_dim=head_hidden_dim, freeze_encoder=True)
        self.model = load_model_checkpoint(
            self.model,
            self.checkpoint_dir,
            str(self.device)
        )
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Model loaded on {self.device}")
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        audio = self._load_audio_array(audio_path)
        input_tensor, _ = self._extract_batch_inputs(audio)
        return input_tensor
    
    def predict(self, audio_path):
        """Predict gender and dialect from one audio file."""
        return self.predict_batch([audio_path])[0]
    
    def predict_batch(self, audio_paths):
        """Predict gender and dialect for multiple audio files (supports batching)."""
        batch_size = int(self.config.get("inference", {}).get("batch_size", 1) or 1)
        batch_size = max(1, batch_size)

        def build_result(path, gender_probs, dialect_probs):
            gender_pred = int(np.argmax(gender_probs))
            dialect_pred = int(np.argmax(dialect_probs))
            return {
                'audio_path': str(path),
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

        results = []

        for start in range(0, len(audio_paths), batch_size):
            chunk = audio_paths[start:start + batch_size]
            chunk_results = [None] * len(chunk)

            ok_audio = []
            ok_indices = []
            for i, audio_path in enumerate(chunk):
                try:
                    ok_audio.append(self._load_audio_array(str(audio_path)))
                    ok_indices.append(i)
                except Exception as e:
                    self.logger.error(f"Error processing {audio_path}: {e}")
                    chunk_results[i] = {'audio_path': str(audio_path), 'error': str(e)}

            if ok_indices:
                try:
                    input_tensor, attention_mask = self._extract_batch_inputs(ok_audio)
                    input_tensor = input_tensor.to(self.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(input_values=input_tensor, attention_mask=attention_mask)
                        gender_logits = outputs['gender_logits']
                        dialect_logits = outputs['dialect_logits']

                    gender_probs_batch = torch.softmax(gender_logits, dim=-1).cpu().numpy()
                    dialect_probs_batch = torch.softmax(dialect_logits, dim=-1).cpu().numpy()

                    for j, idx in enumerate(ok_indices):
                        chunk_results[idx] = build_result(chunk[idx], gender_probs_batch[j], dialect_probs_batch[j])
                except RuntimeError as e:
                    # Graceful fallback for CUDA OOM: process one-by-one for this chunk
                    msg = str(e).lower()
                    if "out of memory" in msg and batch_size > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.logger.warning("CUDA OOM during batched inference; retrying with batch_size=1 for this chunk.")
                        for idx in ok_indices:
                            try:
                                one = self.predict_batch([chunk[idx]])
                                chunk_results[idx] = one[0]
                            except Exception as ex:
                                chunk_results[idx] = {'audio_path': str(chunk[idx]), 'error': str(ex)}
                    else:
                        raise

            results.extend(chunk_results)

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
