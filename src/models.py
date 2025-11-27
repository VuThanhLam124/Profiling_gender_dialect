"""
Model Architecture for Speaker Profiling
WavLM + Attentive Pooling + LayerNorm
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

logger = logging.getLogger("speaker_profiling")


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
    
    Args:
        model_name: Pretrained WavLM model name or path
        num_genders: Number of gender classes (default: 2)
        num_dialects: Number of dialect classes (default: 3)
        dropout: Dropout probability (default: 0.1)
        head_hidden_dim: Hidden dimension for classification heads (default: 256)
        freeze_encoder: Whether to freeze WavLM encoder (default: False)
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
        
        self.dialect_loss_weight = dialect_loss_weight
        
        # Load pretrained WavLM
        self.wavlm = WavLMModel.from_pretrained(model_name)
        hidden_size = self.wavlm.config.hidden_size
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
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
            input_features: Pre-extracted WavLM features [B, T, H] (for cached mode)
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
        # Get hidden states from either raw audio or pre-extracted features
        if input_features is not None:
            # Use pre-extracted features directly
            hidden_states = input_features
        elif input_values is not None:
            # Extract features from WavLM
            outputs = self.wavlm(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, T, H]
        else:
            raise ValueError("Either input_values or input_features must be provided")
        
        # Attentive pooling
        pooled, attn_weights = self.attentive_pooling(hidden_states, attention_mask)
        
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
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled, _ = self.attentive_pooling(hidden_states, attention_mask)
        pooled = self.layer_norm(pooled)
        return pooled


class MultiTaskSpeakerModelFromConfig(MultiTaskSpeakerModel):
    """
    Multi-task model initialized from OmegaConf config
    
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
        
        logger.info("Architecture: WavLM + Attentive Pooling + LayerNorm")
        logger.info(f"Hidden size: {self.wavlm.config.hidden_size}")
        logger.info(f"Head hidden dim: {model_config.get('head_hidden_dim', 256)}")
        logger.info(f"Dropout: {model_config.get('dropout', 0.1)}")
