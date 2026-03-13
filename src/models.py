"""
Model Architecture for Speaker Profiling
Supports multiple encoders: WavLM, HuBERT, Wav2Vec2, Whisper, ECAPA-TDNN
Architecture: Encoder + Attentive Pooling + LayerNorm + Classification Heads
https://github.com/VuThanhLam124/Profiling_gender_dialect/blob/main/ARCHITECTURE.md
"""

import logging
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional

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

try:
    from peft import LoraConfig, inject_adapter_in_model
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    inject_adapter_in_model = None
    PEFT_AVAILABLE = False

SPEECHBRAIN_AVAILABLE = None  # Will be set on first use
EncoderClassifier = None  # Will be imported lazily
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]

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


def normalize_lora_target_modules(target_modules: Optional[Iterable[str]]) -> List[str]:
    """Normalize LoRA target modules from config into a clean string list."""
    if target_modules is None:
        return list(DEFAULT_LORA_TARGET_MODULES)

    if isinstance(target_modules, str):
        modules = [part.strip() for part in target_modules.split(",")]
    else:
        modules = [str(part).strip() for part in target_modules]

    modules = [module for module in modules if module]
    return modules or list(DEFAULT_LORA_TARGET_MODULES)


def get_model_init_kwargs_from_config(
    config: Dict[str, Any],
    include_loss_weight: bool = True
) -> Dict[str, Any]:
    """Build MultiTaskSpeakerModel kwargs from a config object."""
    model_config = config.get('model', {})

    kwargs = {
        'model_name': model_config['name'],
        'num_genders': model_config.get('num_genders', 2),
        'num_dialects': model_config.get('num_dialects', 3),
        'dropout': model_config.get('dropout', 0.1),
        'head_hidden_dim': model_config.get('head_hidden_dim', 256),
        'dialect_head_hidden_dims': model_config.get('dialect_head_hidden_dims'),
        'freeze_encoder': model_config.get('freeze_encoder', False),
        'use_lora': model_config.get('use_lora', False),
        'lora_r': model_config.get('lora_r', 16),
        'lora_alpha': model_config.get('lora_alpha', 32),
        'lora_dropout': model_config.get('lora_dropout', 0.1),
        'lora_bias': model_config.get('lora_bias', 'none'),
        'lora_target_modules': model_config.get('lora_target_modules'),
    }

    if include_loss_weight:
        kwargs['dialect_loss_weight'] = config.get('loss', {}).get('dialect_weight', 3.0)
        kwargs['dialect_class_weights'] = _resolve_class_weights_from_config(
            config=config,
            label_key='dialect',
            weights_key='dialect_class_weights',
        )

    return kwargs


def _resolve_class_weights_from_config(
    config: Dict[str, Any],
    label_key: str,
    weights_key: str,
) -> Optional[List[float]]:
    """Resolve class weights from config into class-index order."""
    labels_cfg = (config.get('labels') or {}).get(label_key) or {}
    loss_cfg = config.get('loss') or {}
    weights_cfg = loss_cfg.get(weights_key)
    if weights_cfg is None:
        return None

    if isinstance(weights_cfg, (list, tuple)) or (
        not isinstance(weights_cfg, (str, bytes)) and hasattr(weights_cfg, "__iter__") and not hasattr(weights_cfg, "items")
    ):
        return [float(value) for value in weights_cfg]

    if not isinstance(weights_cfg, Mapping) and not hasattr(weights_cfg, "items"):
        raise ValueError(
            f"loss.{weights_key} must be a list or mapping. Got: {type(weights_cfg).__name__}"
        )

    ordered_labels = sorted(
        ((label_name, int(label_id)) for label_name, label_id in labels_cfg.items() if isinstance(label_name, str)),
        key=lambda item: item[1],
    )
    if not ordered_labels:
        raise ValueError(f"labels.{label_key} must contain string label names mapped to ids.")

    class_weights = [1.0] * len(ordered_labels)
    for raw_key, raw_value in weights_cfg.items():
        if isinstance(raw_key, int) or (isinstance(raw_key, str) and str(raw_key).isdigit()):
            class_index = int(raw_key)
            if not 0 <= class_index < len(class_weights):
                raise ValueError(
                    f"loss.{weights_key} contains out-of-range class index {class_index}."
                )
        else:
            class_index = labels_cfg.get(str(raw_key))
            if class_index is None:
                raise ValueError(
                    f"loss.{weights_key} contains unknown label '{raw_key}'. "
                    f"Expected one of {[name for name, _ in ordered_labels]}."
                )
            class_index = int(class_index)
        class_weights[class_index] = float(raw_value)

    return class_weights


def _normalize_hidden_dims(hidden_dims: Optional[Iterable[int]], fallback_hidden_dim: int) -> List[int]:
    """Resolve hidden dimensions for configurable MLP heads."""
    if hidden_dims is None:
        normalized = [int(fallback_hidden_dim), max(1, int(fallback_hidden_dim) // 2)]
    elif isinstance(hidden_dims, int):
        normalized = [int(hidden_dims)]
    else:
        normalized = [int(value) for value in hidden_dims]

    normalized = [value for value in normalized if value > 0]
    if not normalized:
        raise ValueError("dialect_head_hidden_dims must contain at least one positive integer.")
    return normalized


def _build_mlp_head(input_dim: int, hidden_dims: Iterable[int], output_dim: int, dropout: float) -> nn.Sequential:
    """Build a simple MLP classifier head."""
    layers: List[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


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
        
        self.encoder.mods.float()
        
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
        device = input_values.device
        
        if str(device) != str(self._current_device):
            self.encoder.to(device)
            self.encoder.mods.float()
            self._current_device = device
        
        input_values = input_values.float().to(device)
        
        with torch.no_grad():
            self.encoder.eval()
            embeddings = self.encoder.encode_batch(input_values)
        
        embeddings = embeddings.float()
        
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
    
    # Vietnamese Wav2Vec2 (VLSP2020)
    "nguyenvulebinh/wav2vec2-base-vi-vlsp2020": {"class": Wav2Vec2Model, "hidden_size": 768},
    "phongdtd/wavLM-VLSP-vi-base": {"class": WavLMModel, "hidden_size": 768},
    
    # Whisper variants (encoder only)
    "openai/whisper-tiny": {"class": WhisperModel, "hidden_size": 384, "is_whisper": True},
    "openai/whisper-base": {"class": WhisperModel, "hidden_size": 512, "is_whisper": True},
    "openai/whisper-small": {"class": WhisperModel, "hidden_size": 768, "is_whisper": True},
    "openai/whisper-medium": {"class": WhisperModel, "hidden_size": 1024, "is_whisper": True},
    "openai/whisper-large": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    "openai/whisper-large-v2": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    "openai/whisper-large-v3": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    
    # PhoWhisper - Vietnamese fine-tuned Whisper (VinAI)
    "vinai/PhoWhisper-tiny": {"class": WhisperModel, "hidden_size": 384, "is_whisper": True},
    "vinai/PhoWhisper-base": {"class": WhisperModel, "hidden_size": 512, "is_whisper": True},
    "vinai/PhoWhisper-small": {"class": WhisperModel, "hidden_size": 768, "is_whisper": True},
    "vinai/PhoWhisper-medium": {"class": WhisperModel, "hidden_size": 1024, "is_whisper": True},
    "vinai/PhoWhisper-large": {"class": WhisperModel, "hidden_size": 1280, "is_whisper": True},
    
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
    if 'ecapa' in model_name.lower() or 'speechbrain' in model_name.lower():
        hidden_size = 512 if 'xvect' in model_name.lower() else 192
        return {"class": ECAPATDNNEncoder, "hidden_size": hidden_size, "is_ecapa": True}
    try:
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = getattr(config, 'hidden_size', 768)
        
        if 'wavlm' in model_name.lower():
            return {"class": WavLMModel, "hidden_size": hidden_size}
        elif 'hubert' in model_name.lower():
            return {"class": HubertModel, "hidden_size": hidden_size}
        elif 'wav2vec2' in model_name.lower():
            return {"class": Wav2Vec2Model, "hidden_size": hidden_size}
        elif 'whisper' in model_name.lower() or 'phowhisper' in model_name.lower():
            return {"class": WhisperModel, "hidden_size": hidden_size, "is_whisper": True}
        else:
            return {"class": Wav2Vec2Model, "hidden_size": hidden_size}
    except Exception as e:
        logger.warning(f"Could not auto-detect encoder for {model_name}: {e}")
        return {"class": WavLMModel, "hidden_size": 768}


class AttentivePooling(nn.Module):
    """
    Attention-based pooling for temporal aggregation, the most advanced technique in this codebase.
    
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

        # đoạn này code nâng cao: đảm bảo mô hình sẽ lờ đi các phần đệm
        # thường là khoàng lặng, những khoảng đó sẽ bị gán 1 số âm rất lớn
        if mask is not None: 
            mask = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        # chuyển về xác suất (softmax apply trên chiều thời gian dim=1 => biến tất cả điểm số thô thành 1 phân phối xác suất
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
        Gender Head (2 layers)     Dialect Head (4 layers)
              |                               |
            [B,2]                           [B,3]
    
    Supported encoders:
        - WavLM: microsoft/wavlm-base-plus, microsoft/wavlm-large
        - HuBERT: facebook/hubert-base-ls960, facebook/hubert-large-ls960-ft
        - Wav2Vec2: facebook/wav2vec2-base, facebook/wav2vec2-large-960h
        - Whisper: openai/whisper-base, openai/whisper-small, openai/whisper-medium, vinai/PhoWhisper-small, etc.
        - Vietnamese Wav2Vec2: nguyenvulebinh/wav2vec2-base-vi-vlsp2020
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
        dialect_head_hidden_dims: Optional[Iterable[int]] = None,
        freeze_encoder: bool = False,
        dialect_loss_weight: float = 3.0,
        dialect_class_weights: Optional[Iterable[float]] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_bias: str = "none",
        lora_target_modules: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.dialect_loss_weight = dialect_loss_weight
        self.dialect_head_hidden_dims = _normalize_hidden_dims(dialect_head_hidden_dims, head_hidden_dim)
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_bias = str(lora_bias)
        self.lora_target_modules = normalize_lora_target_modules(lora_target_modules)
        if dialect_class_weights is not None:
            dialect_class_weights = [float(weight) for weight in dialect_class_weights]
            if len(dialect_class_weights) != int(num_dialects):
                raise ValueError(
                    f"dialect_class_weights must have {num_dialects} values, got {len(dialect_class_weights)}"
                )
            self.register_buffer(
                "dialect_class_weights",
                torch.tensor(dialect_class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("dialect_class_weights", None, persistent=False)
        
        # Get encoder info and load model
        encoder_info = get_encoder_info(model_name)
        encoder_class = encoder_info["class"]
        self.is_whisper = encoder_info.get("is_whisper", False)
        self.is_ecapa = encoder_info.get("is_ecapa", False)
        
        logger.info(f"Loading encoder: {model_name}")
        logger.info(f"Encoder class: {encoder_class.__name__}")
        
        # Load pretrained encoder
        if self.is_ecapa:
            self.encoder = encoder_class(model_name)
        else:
            self.encoder = encoder_class.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        
        logger.info(f"Hidden size: {hidden_size}")
        
        if self.use_lora:
            self._apply_lora()
        elif freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        
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
        
        # Dialect classification head is configurable so we can deepen it for harder accent boundaries.
        self.dialect_head = _build_mlp_head(
            input_dim=hidden_size,
            hidden_dims=self.dialect_head_hidden_dims,
            output_dim=num_dialects,
            dropout=dropout,
        )

    def _apply_lora(self):
        """Freeze the base encoder and inject LoRA adapters into attention projections."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required when model.use_lora=true. Install with: pip install peft"
            )
        if self.is_ecapa:
            raise ValueError("LoRA is not supported for ECAPA-TDNN in this repo.")
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be > 0, got {self.lora_r}")

        for param in self.encoder.parameters():
            param.requires_grad = False

        target_model = self.encoder.encoder if self.is_whisper else self.encoder
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            target_modules=self.lora_target_modules,
        )
        inject_adapter_in_model(lora_config, target_model)

        encoder_trainable = sum(
            param.numel() for param in self.encoder.parameters() if param.requires_grad
        )
        scope = "Whisper encoder only" if self.is_whisper else "encoder"
        logger.info(
            "LoRA enabled on %s: r=%s, alpha=%s, dropout=%s, targets=%s",
            scope,
            self.lora_r,
            self.lora_alpha,
            self.lora_dropout,
            self.lora_target_modules,
        )
        logger.info(f"Trainable encoder parameters after LoRA injection: {encoder_trainable:,}")
    
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
        if input_features is not None:
            hidden_states = input_features
        elif input_values is not None:
            hidden_states = self._encode(input_values, attention_mask)
        else:
            raise ValueError("Either input_values or input_features must be provided")

        if self.is_ecapa or hidden_states.shape[1] == 1:
            pooled = hidden_states.squeeze(1)
            attn_weights = None
        else:
            if attention_mask is not None and hidden_states.shape[1] != attention_mask.shape[1]:
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
            gender_loss = nn.CrossEntropyLoss()(gender_logits, gender_labels)
            dialect_loss = nn.CrossEntropyLoss(weight=self.dialect_class_weights)(
                dialect_logits,
                dialect_labels,
            )
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

        super().__init__(**get_model_init_kwargs_from_config(config, include_loss_weight=True))
        
        logger.info(f"Architecture: {model_config['name']} + Attentive Pooling + LayerNorm")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Head hidden dim: {model_config.get('head_hidden_dim', 256)}")
        logger.info(f"Dialect head hidden dims: {self.dialect_head_hidden_dims}")
        logger.info(f"Dropout: {model_config.get('dropout', 0.1)}")
        logger.info(f"LoRA enabled: {model_config.get('use_lora', False)}")
        if model_config.get('use_lora', False):
            logger.info(
                "LoRA config: r=%s, alpha=%s, dropout=%s, targets=%s",
                model_config.get('lora_r', 16),
                model_config.get('lora_alpha', 32),
                model_config.get('lora_dropout', 0.1),
                normalize_lora_target_modules(model_config.get('lora_target_modules')),
            )
        if self.dialect_class_weights is not None:
            logger.info(f"Dialect class weights: {self.dialect_class_weights.tolist()}")


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
        
        # Dialect classification head (3 layers)
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
