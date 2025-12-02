"""
Model Architecture for Speaker Profiling
Supports multiple encoders: WavLM, HuBERT, Wav2Vec2, Whisper, ECAPA-TDNN
Architecture: Encoder + Attentive Pooling + LayerNorm + Classification Heads
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WavLMModel,
    HubertModel,
    Wav2Vec2Model,
    WhisperModel,
    AutoConfig
)

# SpeechBrain ECAPA-TDNN support - lazy import to avoid torchaudio issues
SPEECHBRAIN_AVAILABLE = None  # Will be set on first use
EncoderClassifier = None  # Will be imported lazily

def _check_speechbrain():
    """Lazily check and import SpeechBrain"""
    global SPEECHBRAIN_AVAILABLE, EncoderClassifier
    if SPEECHBRAIN_AVAILABLE is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier as _EncoderClassifier
            EncoderClassifier = _EncoderClassifier
            SPEECHBRAIN_AVAILABLE = True
        except (ImportError, AttributeError) as e:
            SPEECHBRAIN_AVAILABLE = False
            logger.warning(f"SpeechBrain not available: {e}")
    return SPEECHBRAIN_AVAILABLE

logger = logging.getLogger("speaker_profiling")


# ECAPA-TDNN wrapper class for consistent interface
class ECAPATDNNEncoder(nn.Module):
    """
    Wrapper for SpeechBrain ECAPA-TDNN encoder.
    
    ECAPA-TDNN outputs fixed-size embeddings (192 or 512 dim) instead of 
    frame-level features like WavLM/HuBERT. This wrapper handles the difference.
    
    Supported models:
        - speechbrain/spkrec-ecapa-voxceleb: 192-dim embeddings
        - speechbrain/spkrec-xvect-voxceleb: 512-dim embeddings (x-vector)
    """
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        super().__init__()
        
        # Lazy import SpeechBrain
        if not _check_speechbrain():
            raise ImportError(
                "SpeechBrain is required for ECAPA-TDNN. "
                "Install with: pip install speechbrain"
            )
        
        self.model_name = model_name
        
        # Detect if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.encoder = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.split('/')[-1]}",
            run_opts={"device": device}
        )
        
        # Force float32 for all encoder parameters
        self.encoder.mods.float()
        
        # Determine embedding size
        if "ecapa" in model_name.lower():
            self.embedding_size = 192
        elif "xvect" in model_name.lower():
            self.embedding_size = 512
        else:
            self.embedding_size = 192  # default
        
        # Config-like object for compatibility
        class Config:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size
        
        self.config = Config(self.embedding_size)
        
        # Track current device
        self._current_device = device
    
    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Extract embeddings from audio.
        
        Args:
            input_values: Audio waveform [B, T]
            attention_mask: Not used for ECAPA-TDNN
        
        Returns:
            Object with last_hidden_state attribute [B, 1, H]
        """
        # Get device from input
        device = input_values.device
        
        # Move encoder to same device as input if needed
        if str(device) != str(self._current_device):
            self.encoder.to(device)
            self.encoder.mods.float()  # Ensure float32 after move
            self._current_device = device
        
        # Ensure input is float32 and on correct device
        input_values = input_values.float().to(device)
        
        # SpeechBrain expects [B, T] audio at 16kHz
        # encode_batch handles feature extraction internally
        with torch.no_grad():
            # Set encoder to eval mode to handle BatchNorm properly
            self.encoder.eval()
            embeddings = self.encoder.encode_batch(input_values)  # [B, 1, H]
        
        # Ensure output is float32
        embeddings = embeddings.float()
        
        # Return object compatible with HuggingFace models
        class Output:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state
        
        return Output(embeddings)


# Encoder registry - maps model type to class and hidden size
ENCODER_REGISTRY = {
    # WavLM variants
    "microsoft/wavlm-base": {"class": WavLMModel, "hidden_size": 768},
    "microsoft/wavlm-base-plus": {"class": WavLMModel, "hidden_size": 768},
    "microsoft/wavlm-large": {"class": WavLMModel, "hidden_size": 1024},
    
    # HuBERT variants
    "facebook/hubert-base-ls960": {"class": HubertModel, "hidden_size": 768},
    "facebook/hubert-large-ls960-ft": {"class": HubertModel, "hidden_size": 1024},
    "facebook/hubert-xlarge-ls960-ft": {"class": HubertModel, "hidden_size": 1280},
    
    # Wav2Vec2 variants
    "facebook/wav2vec2-base": {"class": Wav2Vec2Model, "hidden_size": 768},
    "facebook/wav2vec2-base-960h": {"class": Wav2Vec2Model, "hidden_size": 768},
    "facebook/wav2vec2-large": {"class": Wav2Vec2Model, "hidden_size": 1024},
    "facebook/wav2vec2-large-960h": {"class": Wav2Vec2Model, "hidden_size": 1024},
    "facebook/wav2vec2-xls-r-300m": {"class": Wav2Vec2Model, "hidden_size": 1024},
    
    # Whisper variants (encoder only)
    "openai/whisper-tiny": {"class": WhisperModel, "hidden_size": 384, "is_whisper": True},
    "openai/whisper-base": {"class": WhisperModel, "hidden_size": 512, "is_whisper": True},
    "openai/whisper-small": {"class": WhisperModel, "hidden_size": 768, "is_whisper": True},
    "openai/whisper-medium": {"class": WhisperModel, "hidden_size": 1024, "is_whisper": True},
    "openai/whisper-large": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    "openai/whisper-large-v2": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    "openai/whisper-large-v3": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    
    # ECAPA-TDNN (SpeechBrain)
    "speechbrain/spkrec-ecapa-voxceleb": {
        "class": ECAPATDNNEncoder, 
        "hidden_size": 192, 
        "is_ecapa": True
    },
    "speechbrain/spkrec-xvect-voxceleb": {
        "class": ECAPATDNNEncoder, 
        "hidden_size": 512, 
        "is_ecapa": True
    },
}


def get_encoder_info(model_name: str) -> dict:
    """Get encoder class and hidden size for a model name"""
    if model_name in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[model_name]
    
    # Check for ECAPA-TDNN / SpeechBrain models
    # Note: We don't check SPEECHBRAIN_AVAILABLE here - the actual import
    # will happen lazily in ECAPATDNNEncoder.__init__() when the model is used
    if 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower():
        hidden_size = 512 if 'xvect' in model_name.lower() else 192
        return {"class": ECAPATDNNEncoder, "hidden_size": hidden_size, "is_ecapa": True}
    
    # Try to auto-detect from config
    try:
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = getattr(config, 'hidden_size', 768)
        
        if 'wavlm' in model_name.lower():
            return {"class": WavLMModel, "hidden_size": hidden_size}
        elif 'hubert' in model_name.lower():
            return {"class": HubertModel, "hidden_size": hidden_size}
        elif 'wav2vec2' in model_name.lower():
            return {"class": Wav2Vec2Model, "hidden_size": hidden_size}
        elif 'whisper' in model_name.lower():
            return {"class": WhisperModel, "hidden_size": hidden_size, "is_whisper": True}
        else:
            # Default to Wav2Vec2 architecture
            return {"class": Wav2Vec2Model, "hidden_size": hidden_size}
    except Exception as e:
        logger.warning(f"Could not auto-detect encoder for {model_name}: {e}")
        return {"class": WavLMModel, "hidden_size": 768}


class AttentivePooling(nn.Module):
    """
    Attention-based pooling for temporal aggregation
    
    Takes sequence of hidden states and produces a single vector
    by computing attention weights and performing weighted sum.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Hidden states [B, T, H]
            mask: Attention mask [B, T]
        
        Returns:
            pooled: Pooled representation [B, H]
            attn_weights: Attention weights [B, T]
        """
        attn_weights = self.attention(x)  # [B, T, 1]
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return pooled, attn_weights.squeeze(-1)


class MultiTaskSpeakerModel(nn.Module):
    """
    Multi-task model for gender and dialect classification
    
    Architecture:
        Audio -> Encoder (WavLM/HuBERT/Wav2Vec2/Whisper/ECAPA-TDNN) -> Last Hidden [B,T,H]
                              |
                     Attentive Pooling [B,H] (skipped for ECAPA-TDNN)
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
    
    Supported encoders:
        - WavLM: microsoft/wavlm-base-plus, microsoft/wavlm-large
        - HuBERT: facebook/hubert-base-ls960, facebook/hubert-large-ls960-ft
        - Wav2Vec2: facebook/wav2vec2-base, facebook/wav2vec2-large-960h
        - Whisper: openai/whisper-base, openai/whisper-small, openai/whisper-medium
        - ECAPA-TDNN: speechbrain/spkrec-ecapa-voxceleb (192-dim embeddings)
    
    Args:
        model_name: Pretrained encoder model name or path
        num_genders: Number of gender classes (default: 2)
        num_dialects: Number of dialect classes (default: 3)
        dropout: Dropout probability (default: 0.1)
        head_hidden_dim: Hidden dimension for classification heads (default: 256)
        freeze_encoder: Whether to freeze encoder (default: False)
        dialect_loss_weight: Weight for dialect loss in multi-task learning (default: 3.0)
    """
    
    def __init__(
        self, 
        model_name: str,
        num_genders: int = 2, 
        num_dialects: int = 3, 
        dropout: float = 0.1, 
        head_hidden_dim: int = 256,
        freeze_encoder: bool = False,
        dialect_loss_weight: float = 3.0
    ):
        super().__init__()
        
        self.model_name = model_name
        self.dialect_loss_weight = dialect_loss_weight
        
        # Get encoder info and load model
        encoder_info = get_encoder_info(model_name)
        encoder_class = encoder_info["class"]
        self.is_whisper = encoder_info.get("is_whisper", False)
        self.is_ecapa = encoder_info.get("is_ecapa", False)
        
        logger.info(f"Loading encoder: {model_name}")
        logger.info(f"Encoder class: {encoder_class.__name__}")
        
        # Load pretrained encoder
        if self.is_ecapa:
            # ECAPA-TDNN uses different loading mechanism
            self.encoder = encoder_class(model_name)
        else:
            self.encoder = encoder_class.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        
        logger.info(f"Hidden size: {hidden_size}")
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        
        # Pooling and normalization (ECAPA-TDNN already outputs pooled embeddings)
        self.attentive_pooling = AttentivePooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Gender classification head (2 layers)
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_genders)
        )
        
        # Dialect classification head (3 layers - deeper for harder task)
        self.dialect_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, num_dialects)
        )
    
    def forward(
        self, 
        input_values: torch.Tensor = None,
        input_features: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        gender_labels: torch.Tensor = None, 
        dialect_labels: torch.Tensor = None
    ):
        """
        Forward pass - supports both raw audio and pre-extracted features
        
        Args:
            input_values: Audio waveform [B, T] (for raw audio mode)
            input_features: Pre-extracted features [B, T, H] or [B, 1, H] for ECAPA
            attention_mask: Attention mask [B, T]
            gender_labels: Gender labels [B] (optional, for training)
            dialect_labels: Dialect labels [B] (optional, for training)
        
        Returns:
            dict with keys:
                - loss: Combined loss (if labels provided)
                - gender_logits: Gender predictions [B, num_genders]
                - dialect_logits: Dialect predictions [B, num_dialects]
                - attention_weights: Attention weights from pooling [B, T] (None for ECAPA)
        """
        # Get hidden states from either raw audio or pre-extracted features
        if input_features is not None:
            # Use pre-extracted features directly
            hidden_states = input_features
        elif input_values is not None:
            # Extract features from encoder
            hidden_states = self._encode(input_values, attention_mask)
        else:
            raise ValueError("Either input_values or input_features must be provided")
        
        # Handle ECAPA-TDNN (outputs [B, 1, H] - already pooled embeddings)
        if self.is_ecapa or hidden_states.shape[1] == 1:
            # ECAPA-TDNN outputs already pooled embeddings
            pooled = hidden_states.squeeze(1)  # [B, H]
            attn_weights = None
        else:
            # Create proper attention mask for hidden states (encoder downsamples audio)
            # Hidden states have different sequence length than input audio
            if attention_mask is not None and hidden_states.shape[1] != attention_mask.shape[1]:
                # Create new mask based on hidden states length
                batch_size, seq_len, _ = hidden_states.shape
                pooled_mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
            else:
                pooled_mask = attention_mask
            
            # Attentive pooling
            pooled, attn_weights = self.attentive_pooling(hidden_states, pooled_mask)
        
        # Normalization and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Classification heads
        gender_logits = self.gender_head(pooled)
        dialect_logits = self.dialect_head(pooled)
        
        # Compute loss if labels provided
        loss = None
        if gender_labels is not None and dialect_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            gender_loss = loss_fct(gender_logits, gender_labels)
            dialect_loss = loss_fct(dialect_logits, dialect_labels)
            loss = gender_loss + self.dialect_loss_weight * dialect_loss
        
        return {
            'loss': loss,
            'gender_logits': gender_logits,
            'dialect_logits': dialect_logits,
            'attention_weights': attn_weights
        }
    
    def _encode(
        self, 
        input_values: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract hidden states from encoder
        
        Args:
            input_values: Audio waveform [B, T]
            attention_mask: Attention mask [B, T]
        
        Returns:
            hidden_states: Hidden states [B, T, H] or [B, 1, H] for ECAPA-TDNN
        """
        if self.is_ecapa:
            # ECAPA-TDNN outputs fixed-size embeddings [B, 1, H]
            outputs = self.encoder(input_values, attention_mask)
            hidden_states = outputs.last_hidden_state
        elif self.is_whisper:
            # Whisper uses encoder-decoder, we only use encoder
            outputs = self.encoder.encoder(input_values)
            hidden_states = outputs.last_hidden_state
        else:
            # WavLM, HuBERT, Wav2Vec2
            outputs = self.encoder(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    def get_embeddings(
        self, 
        input_values: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract speaker embeddings (pooled representations)
        
        Args:
            input_values: Audio waveform [B, T]
            attention_mask: Attention mask [B, T]
        
        Returns:
            embeddings: Speaker embeddings [B, H]
        """
        hidden_states = self._encode(input_values, attention_mask)
        
        if self.is_ecapa or hidden_states.shape[1] == 1:
            # ECAPA-TDNN already outputs pooled embeddings
            pooled = hidden_states.squeeze(1)
        else:
            pooled, _ = self.attentive_pooling(hidden_states, attention_mask)
        
        pooled = self.layer_norm(pooled)
        return pooled


class MultiTaskSpeakerModelFromConfig(MultiTaskSpeakerModel):
    """
    Multi-task model initialized from OmegaConf config
    
    Supports multiple encoders: WavLM, HuBERT, Wav2Vec2, Whisper
    Use this for inference with raw audio input.
    
    Usage:
        config = OmegaConf.load('configs/finetune.yaml')
        model = MultiTaskSpeakerModelFromConfig(config)
    """
    
    def __init__(self, config):
        model_config = config['model']
        
        super().__init__(
            model_name=model_config['name'],
            num_genders=model_config.get('num_genders', 2),
            num_dialects=model_config.get('num_dialects', 3),
            dropout=model_config.get('dropout', 0.1),
            head_hidden_dim=model_config.get('head_hidden_dim', 256),
            freeze_encoder=model_config.get('freeze_encoder', False),
            dialect_loss_weight=config.get('loss', {}).get('dialect_weight', 3.0)
        )
        
        logger.info(f"Architecture: {model_config['name']} + Attentive Pooling + LayerNorm")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Head hidden dim: {model_config.get('head_hidden_dim', 256)}")
        logger.info(f"Dropout: {model_config.get('dropout', 0.1)}")


class ClassificationHeadModel(nn.Module):
    """
    Lightweight model with only classification heads (no encoder).
    
    Use this for training with pre-extracted features to save memory.
    Hidden_size depends on encoder: WavLM-base=768, WavLM-large=1024, etc.
    
    Usage:
        model = ClassificationHeadModel(config)
        output = model(input_features=features, gender_labels=y_gender, dialect_labels=y_dialect)
    """
    
    def __init__(
        self, 
        hidden_size: int = 768,
        num_genders: int = 2, 
        num_dialects: int = 3, 
        dropout: float = 0.1, 
        head_hidden_dim: int = 256,
        dialect_loss_weight: float = 3.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dialect_loss_weight = dialect_loss_weight
        
        # Pooling and normalization
        self.attentive_pooling = AttentivePooling(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Gender classification head (2 layers)
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_genders)
        )
        
        # Dialect classification head (3 layers - deeper for harder task)
        self.dialect_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, num_dialects)
        )
        
        logger.info(f"ClassificationHeadModel initialized (hidden_size={hidden_size})")
    
    def forward(
        self, 
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        gender_labels: torch.Tensor = None, 
        dialect_labels: torch.Tensor = None
    ):
        """
        Forward pass for pre-extracted features
        
        Args:
            input_features: Pre-extracted WavLM features [B, T, H]
            attention_mask: Attention mask [B, T]
            gender_labels: Gender labels [B] (optional, for training)
            dialect_labels: Dialect labels [B] (optional, for training)
        
        Returns:
            dict with keys:
                - loss: Combined loss (if labels provided)
                - gender_logits: Gender predictions [B, num_genders]
                - dialect_logits: Dialect predictions [B, num_dialects]
                - attention_weights: Attention weights from pooling [B, T]
        """
        # Attentive pooling
        pooled, attn_weights = self.attentive_pooling(input_features, attention_mask)
        
        # Normalization and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Classification heads
        gender_logits = self.gender_head(pooled)
        dialect_logits = self.dialect_head(pooled)
        
        # Compute loss if labels provided
        loss = None
        if gender_labels is not None and dialect_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            gender_loss = loss_fct(gender_logits, gender_labels)
            dialect_loss = loss_fct(dialect_logits, dialect_labels)
            loss = gender_loss + self.dialect_loss_weight * dialect_loss
        
        return {
            'loss': loss,
            'gender_logits': gender_logits,
            'dialect_logits': dialect_logits,
            'attention_weights': attn_weights
        }


class ClassificationHeadModelFromConfig(ClassificationHeadModel):
    """
    Lightweight classification model initialized from OmegaConf config.
    
    Use this for training with pre-extracted features.
    """
    
    def __init__(self, config):
        model_config = config['model']
        
        super().__init__(
            hidden_size=model_config.get('hidden_size', 768),  # WavLM base hidden size
            num_genders=model_config.get('num_genders', 2),
            num_dialects=model_config.get('num_dialects', 3),
            dropout=model_config.get('dropout', 0.1),
            head_hidden_dim=model_config.get('head_hidden_dim', 256),
            dialect_loss_weight=config.get('loss', {}).get('dialect_weight', 3.0)
        )
        
        logger.info("Architecture: Attentive Pooling + LayerNorm + Classification Heads")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Head hidden dim: {model_config.get('head_hidden_dim', 256)}")
        logger.info(f"Dropout: {model_config.get('dropout', 0.1)}")
