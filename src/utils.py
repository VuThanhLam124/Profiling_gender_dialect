"""
Utility functions for Speaker Profiling
"""

import os
import logging
import random
import numpy as np
import torch
import librosa
from pathlib import Path
from omegaconf import OmegaConf
from typing import Union, Optional, Tuple


def resolve_checkpoint_dir(checkpoint: Union[str, Path]) -> str:
    """
    Resolve a model checkpoint directory.

    Supports:
    - Local directory paths
    - Hugging Face Hub model repos via:
        - "hf:owner/repo" or "hf://owner/repo"
        - "owner/repo" (only if local path doesn't exist)
      Optional revision: "hf:owner/repo@rev"
    - Hugging Face Hub Spaces via:
        - "hf-space:owner/space#subdir" (optional "@rev")
    """
    checkpoint_str = str(checkpoint)

    # Local path
    if os.path.exists(checkpoint_str):
        return checkpoint_str

    # HF Hub reference
    repo_id = None
    revision = None
    repo_type = "model"
    subdir = None

    if checkpoint_str.startswith("hf-space://"):
        repo_id = checkpoint_str[len("hf-space://") :]
        repo_type = "space"
    elif checkpoint_str.startswith("hf-space:"):
        repo_id = checkpoint_str[len("hf-space:") :]
        repo_type = "space"
    elif checkpoint_str.startswith("hf://"):
        repo_id = checkpoint_str[len("hf://") :]
        repo_type = "model"
    elif checkpoint_str.startswith("hf:"):
        repo_id = checkpoint_str[len("hf:") :]
        repo_type = "model"
    else:
        # Fallback: treat as repo id if it looks like "owner/repo"
        parts = checkpoint_str.split("/")
        if len(parts) == 2 and all(parts) and " " not in checkpoint_str:
            repo_id = checkpoint_str

    if repo_id is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_str}. "
            "Provide an existing directory path or an HF repo like 'hf:owner/repo'."
        )

    if "#" in repo_id:
        repo_id, subdir = repo_id.split("#", 1)

    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download checkpoints from the Hub. "
            "Install it with: pip install huggingface_hub"
        ) from e

    local_dir = snapshot_download(repo_id, repo_type=repo_type, revision=revision)
    if subdir:
        local_dir = os.path.join(local_dir, subdir)
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Resolved checkpoint directory does not exist: {local_dir}")
    return local_dir


def setup_logging(
    name: str = "speaker_profiling",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "speaker_profiling") -> logging.Logger:
    """Get existing logger or create new one"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


def load_config(config_path: str) -> OmegaConf:
    """
    Load configuration from yaml file
    
    Args:
        config_path: Path to yaml config file
    
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_audio(
    audio_path: Union[str, Path],
    sampling_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    
    Args:
        audio_path: Path to audio file
        sampling_rate: Target sampling rate
        mono: Whether to convert to mono
    
    Returns:
        Tuple of (audio array, sampling rate)
    """
    audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=mono)
    return audio, sr


def preprocess_audio(
    audio: np.ndarray,
    sampling_rate: int = 16000,
    max_duration: float = 10.0,
    trim_db: int = 20,
    normalize: bool = True,
    center_crop: bool = True
) -> np.ndarray:
    """
    Preprocess audio for model input
    
    Args:
        audio: Raw audio array
        sampling_rate: Audio sampling rate
        max_duration: Maximum duration in seconds
        trim_db: Threshold for silence trimming
        normalize: Whether to normalize audio
        center_crop: If True, center crop; else random crop (for training)
    
    Returns:
        Preprocessed audio array
    """
    max_length = int(sampling_rate * max_duration)
    
    audio, _ = librosa.effects.trim(audio, top_db=trim_db)
    
    if normalize:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    elif len(audio) > max_length:
        if center_crop:
            start = (len(audio) - max_length) // 2
        else:
            start = np.random.randint(0, len(audio) - max_length + 1)
        audio = audio[start:start + max_length]
    
    return audio


def load_and_preprocess_audio(
    audio_path: Union[str, Path],
    sampling_rate: int = 16000,
    max_duration: float = 10.0,
    trim_db: int = 20,
    normalize: bool = True,
    center_crop: bool = True
) -> np.ndarray:
    """
    Load and preprocess audio file in one step
    
    Args:
        audio_path: Path to audio file
        sampling_rate: Target sampling rate
        max_duration: Maximum duration in seconds
        trim_db: Threshold for silence trimming
        normalize: Whether to normalize audio
        center_crop: If True, center crop; else random crop
    
    Returns:
        Preprocessed audio array
    """
    audio, _ = load_audio(audio_path, sampling_rate)
    return preprocess_audio(
        audio, 
        sampling_rate, 
        max_duration, 
        trim_db, 
        normalize, 
        center_crop
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load model from checkpoint
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
    
    Returns:
        Model with loaded weights
    """
    logger = get_logger()

    checkpoint_path = resolve_checkpoint_dir(checkpoint_path)
    
    safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
    pytorch_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        logger.info(f"Loading checkpoint from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
        logger.info(f"Loading checkpoint from {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_path}. "
            f"Expected 'model.safetensors' or 'pytorch_model.bin'"
        )
    
    model.load_state_dict(state_dict)
    return model


def get_device(device_str: str = 'cuda') -> torch.device:
    """
    Get torch device, fallback to CPU if CUDA not available
    
    Args:
        device_str: Desired device string ('cuda' or 'cpu')
    
    Returns:
        torch.device instance
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"
