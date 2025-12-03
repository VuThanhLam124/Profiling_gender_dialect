"""
Gradio Web Interface for Speaker Profiling

Usage:
    python app.py
    python app.py --config configs/infer.yaml --share
"""

import os
import argparse
import tempfile
import numpy as np
import torch
import librosa
import gradio as gr
from pathlib import Path

from src.models import MultiTaskSpeakerModel
from src.utils import (
    setup_logging,
    get_logger,
    load_config,
    get_device,
    load_model_checkpoint,
    preprocess_audio
)


class SpeakerProfilerApp:
    """Gradio application for speaker profiling"""
    
    def __init__(self, config_path: str):
        self.logger = setup_logging(name="gradio_app")
        self.config = load_config(config_path)
        self.device = get_device(self.config['inference']['device'])
        
        self.sampling_rate = self.config['audio']['sampling_rate']
        self.max_duration = self.config['audio']['max_duration']
        
        self.gender_labels = self.config['labels']['gender']
        self.dialect_labels = self.config['labels']['dialect']
        
        self._load_model()
    
    def _load_model(self):
        """Load model and feature extractor"""
        from transformers import Wav2Vec2FeatureExtractor
        
        self.logger.info("Loading model...")
        
        model_name = self.config['model']['name']
        is_ecapa = 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower()
        
        if is_ecapa:
            # ECAPA-TDNN: use Wav2Vec2 feature extractor for audio normalization
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
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
    
    def predict(self, audio_input):
        """
        Predict gender and dialect from audio
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array) from Gradio
        
        Returns:
            Tuple of (gender_result, dialect_result, details)
        """
        if audio_input is None:
            return "No audio", "No audio", "Please upload or record audio"
        
        try:
            sr, audio = audio_input
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
            
            if sr != self.sampling_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            
            audio = preprocess_audio(
                audio,
                sampling_rate=self.sampling_rate,
                max_duration=self.max_duration
            )
            
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_values)
                gender_logits = outputs['gender_logits']
                dialect_logits = outputs['dialect_logits']
            
            gender_probs = torch.softmax(gender_logits, dim=-1).cpu().numpy()[0]
            dialect_probs = torch.softmax(dialect_logits, dim=-1).cpu().numpy()[0]
            
            gender_pred = int(np.argmax(gender_probs))
            dialect_pred = int(np.argmax(dialect_probs))
            
            gender_name = self.gender_labels[gender_pred]
            dialect_name = self.dialect_labels[dialect_pred]
            
            gender_conf = gender_probs[gender_pred] * 100
            dialect_conf = dialect_probs[dialect_pred] * 100
            
            gender_result = f"{gender_name} ({gender_conf:.1f}%)"
            dialect_result = f"{dialect_name} ({dialect_conf:.1f}%)"
            
            details = self._format_details(gender_probs, dialect_probs)
            
            self.logger.info(f"Prediction: Gender={gender_name}, Dialect={dialect_name}")
            
            return gender_result, dialect_result, details
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "Error", "Error", f"Error: {str(e)}"
    
    def _format_details(self, gender_probs: np.ndarray, dialect_probs: np.ndarray) -> str:
        """Format detailed prediction results"""
        lines = []
        lines.append("Gender Probabilities:")
        for i, label in enumerate(self.gender_labels):
            lines.append(f"  {label}: {gender_probs[i]*100:.2f}%")
        
        lines.append("")
        lines.append("Dialect Probabilities:")
        for i, label in enumerate(self.dialect_labels):
            lines.append(f"  {label}: {dialect_probs[i]*100:.2f}%")
        
        return "\n".join(lines)
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        # Gradio < 4.0 doesn't support theme in Blocks
        with gr.Blocks(title="Vietnamese Speaker Profiling") as demo:
            
            gr.Markdown(
                """
                # Vietnamese Speaker Profiling
                
                Identify gender and dialect from Vietnamese speech audio.
                
                **Model:** Encoder + Attentive Pooling + LayerNorm + MultiHead Classifier
                
                **Supported dialects:** North, Central, South
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Input Audio",
                        type="numpy",
                        sources=["upload", "microphone"]
                    )
                    
                    submit_btn = gr.Button("Analyze", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    gender_output = gr.Textbox(
                        label="Gender",
                        interactive=False
                    )
                    dialect_output = gr.Textbox(
                        label="Dialect",
                        interactive=False
                    )
                    details_output = gr.Textbox(
                        label="Details",
                        lines=8,
                        interactive=False
                    )
            
            gr.Markdown(
                """
                ---
                **Notes:**
                - Supported formats: WAV, MP3
                - Recommended duration: 3-10 seconds
                """
            )
            
            submit_btn.click(
                fn=self.predict,
                inputs=[audio_input],
                outputs=[gender_output, dialect_output, details_output]
            )
            
            clear_btn.click(
                fn=lambda: (None, "", "", ""),
                inputs=[],
                outputs=[audio_input, gender_output, dialect_output, details_output]
            )
        
        return demo


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Speaker Profiling Web Interface")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number (default: 7860)"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    app = SpeakerProfilerApp(args.config)
    demo = app.create_interface()
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
