"""
Unified dataset loading utilities for multi-source speaker profiling.

This module merges multiple Hugging Face audio datasets into a single
canonical schema that can be consumed by the existing training pipeline.

Canonical columns after normalization:
    - audio: Hugging Face Audio feature or dict-like audio payload
    - gender: integer class id following config.labels.gender
    - region: one of {"North", "Central", "South"}
    - source_speaker_id: string speaker identifier when available
    - source: dataset source name
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

try:
    from datasets import Audio, concatenate_datasets, load_dataset
except ImportError:  # pragma: no cover - handled at runtime by caller
    Audio = None
    concatenate_datasets = None
    load_dataset = None


SUPPORTED_UNIFIED_SOURCES = ("vimd", "lsvsc", "visec")

DEFAULT_LSVSC_DIALECT_MAP = {
    "northern dialect": "North",
    "central dialect": "Central",
    "southern dialect": "South",
}

DEFAULT_VISEC_ACCENT_MAP = {
    "north": "North",
    "mid": "Central",
    "middle": "Central",
    "central": "Central",
    "south": "South",
}

DEFAULT_REGION_ALIASES = {
    "north": "North",
    "northern": "North",
    "north dialect": "North",
    "central": "Central",
    "middle": "Central",
    "mid": "Central",
    "south": "South",
    "southern": "South",
}


def load_unified_data(config, logger):
    """
    Load, normalize, and merge ViMD + LSVSC + ViSEC into train/validation splits.

    The merged datasets use the same canonical schema expected by ViMDDataset:
        - audio
        - gender (int)
        - region (North/Central/South)
    """
    if load_dataset is None or concatenate_datasets is None:
        raise ImportError("datasets library required for unified data. Install: pip install datasets")

    unified_cfg = config["data"].get("unified", {})
    seed = int(config["seed"])
    enabled_sources = _get_enabled_sources(unified_cfg)
    if not enabled_sources:
        raise ValueError("data.unified.enabled_sources resolved to an empty set.")

    logger.info("Loading unified multi-source dataset...")
    logger.info(f"Enabled sources: {', '.join(enabled_sources)}")

    source_splits = {}
    for source_name in enabled_sources:
        source_splits[source_name] = _load_single_source(
            source_name=source_name,
            config=config,
            unified_cfg=unified_cfg,
            logger=logger,
        )

    train_parts = []
    val_parts = []
    for source_name in enabled_sources:
        train_ds = source_splits[source_name]["train"]
        val_ds = source_splits[source_name]["validation"]
        train_parts.append(train_ds)
        val_parts.append(val_ds)

        logger.info(
            f"Unified source '{source_name}': "
            f"train={len(train_ds):,}, val={len(val_ds):,}"
        )
        _log_label_stats(train_ds, logger, f"{source_name}/train")
        _log_label_stats(val_ds, logger, f"{source_name}/val")

    merge_strategy = str(unified_cfg.get("train_merge_strategy", "balanced_concat")).lower()
    sampling_probs = _resolve_sampling_probs(unified_cfg, enabled_sources)
    logger.info(f"Unified train merge strategy: {merge_strategy}")
    logger.info(f"Unified sampling probabilities: {sampling_probs}")

    if merge_strategy == "balanced_concat":
        merged_train = _build_balanced_concat(
            train_parts_by_source={src: source_splits[src]["train"] for src in enabled_sources},
            sampling_probs=sampling_probs,
            seed=seed,
            logger=logger,
        )
    elif merge_strategy == "concat":
        merged_train = concatenate_datasets(train_parts).shuffle(seed=seed)
    else:
        raise ValueError(
            f"Unsupported data.unified.train_merge_strategy: {merge_strategy}. "
            "Use 'balanced_concat' or 'concat'."
        )

    merged_val = concatenate_datasets(val_parts)

    logger.info(f"Unified merged train: {len(merged_train):,} samples")
    logger.info(f"Unified merged val: {len(merged_val):,} samples")
    _log_source_mix(merged_train, logger, "merged/train")
    _log_source_mix(merged_val, logger, "merged/val")

    return merged_train, merged_val


def _get_enabled_sources(unified_cfg) -> List[str]:
    explicit = unified_cfg.get("enabled_sources")
    if explicit:
        enabled_sources = [str(source).strip().lower() for source in explicit if str(source).strip()]
    else:
        enabled_sources = []
        for source_name in SUPPORTED_UNIFIED_SOURCES:
            source_cfg = unified_cfg.get(source_name, {})
            if bool(source_cfg.get("enabled", True)):
                enabled_sources.append(source_name)

    enabled_sources = list(dict.fromkeys(enabled_sources))
    unknown_sources = [source for source in enabled_sources if source not in SUPPORTED_UNIFIED_SOURCES]
    if unknown_sources:
        raise ValueError(
            f"Unsupported unified sources: {unknown_sources}. "
            f"Supported: {list(SUPPORTED_UNIFIED_SOURCES)}"
        )
    return enabled_sources


def _load_single_source(source_name: str, config, unified_cfg, logger):
    if source_name == "vimd":
        return _load_vimd_source(config, unified_cfg, logger)
    if source_name == "lsvsc":
        return _load_lsvsc_source(config, unified_cfg, logger)
    if source_name == "visec":
        return _load_visec_source(config, unified_cfg, logger)
    raise ValueError(f"Unsupported unified source: {source_name}")


def _resolve_cache_dir(config, unified_cfg, source_name: str) -> str | None:
    source_cfg = unified_cfg.get(source_name, {})
    cache_dir = (
        source_cfg.get("cache_dir")
        or unified_cfg.get("cache_dir")
        or config["data"].get("hf_cache_dir")
    )
    if cache_dir is None:
        return None
    cache_dir = str(cache_dir).strip()
    return cache_dir or None


def _load_source_split(dataset_path: str, split_name: str, logger, source_name: str, cache_dir: str | None = None):
    logger.info(f"Loading source '{source_name}' split='{split_name}' from: {dataset_path}")
    load_kwargs = {
        "split": split_name,
        "keep_in_memory": False,
    }
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        load_kwargs["cache_dir"] = cache_dir
        logger.info(f"Using Hugging Face cache dir for '{source_name}': {cache_dir}")

    dataset = load_dataset(dataset_path, **load_kwargs)
    return _disable_audio_decoding(dataset, logger, source_name, split_name)


def _disable_audio_decoding(dataset_obj, logger, source_name: str, split_name: str | None = None):
    """Disable automatic Audio decoding so item access does not require torchcodec."""
    if Audio is None:
        return dataset_obj

    if hasattr(dataset_obj, "items"):
        split_items = list(dataset_obj.items())
    else:
        split_items = [(split_name or "unknown", dataset_obj)]

    for current_split_name, split in split_items:
        features = getattr(split, "features", None) or {}
        for column_name, feature in features.items():
            if isinstance(feature, Audio) and getattr(feature, "decode", False):
                updated_split = split.cast_column(
                    column_name,
                    Audio(sampling_rate=getattr(feature, "sampling_rate", None), decode=False),
                )
                if hasattr(dataset_obj, "items"):
                    dataset_obj[current_split_name] = updated_split
                else:
                    dataset_obj = updated_split
                logger.info(
                    f"Disabled HF audio decoding for source='{source_name}', split='{current_split_name}', column='{column_name}'"
                )
                split = updated_split

    return dataset_obj


def _load_vimd_source(config, unified_cfg, logger):
    dataset_path = str(unified_cfg.get("vimd", {}).get("path", config["data"].get("vimd_path", "nguyendv02/ViMD_Dataset")))
    cache_dir = _resolve_cache_dir(config, unified_cfg, "vimd")

    train_ds = _load_source_split(dataset_path, "train", logger, "vimd", cache_dir=cache_dir)
    try:
        val_ds = _load_source_split(dataset_path, "validation", logger, "vimd", cache_dir=cache_dir)
    except Exception:
        val_ds = _load_source_split(dataset_path, "valid", logger, "vimd", cache_dir=cache_dir)

    train_ds = _normalize_split(train_ds, source_name="vimd", config=config, unified_cfg=unified_cfg)
    val_ds = _normalize_split(
        val_ds,
        source_name="vimd",
        config=config,
        unified_cfg=unified_cfg,
    )
    return {"train": train_ds, "validation": val_ds}


def _load_lsvsc_source(config, unified_cfg, logger):
    dataset_path = str(unified_cfg.get("lsvsc", {}).get("path", "doof-ferb/LSVSC"))
    cache_dir = _resolve_cache_dir(config, unified_cfg, "lsvsc")

    raw_train = _load_source_split(dataset_path, "train", logger, "lsvsc", cache_dir=cache_dir)
    raw_val = _load_source_split(dataset_path, "validation", logger, "lsvsc", cache_dir=cache_dir)
    logger.info(f"LSVSC raw train={len(raw_train):,}, val={len(raw_val):,}")

    train_ds = _normalize_split(
        raw_train,
        source_name="lsvsc",
        config=config,
        unified_cfg=unified_cfg,
    )
    val_ds = _normalize_split(
        raw_val,
        source_name="lsvsc",
        config=config,
        unified_cfg=unified_cfg,
    )
    return {"train": train_ds, "validation": val_ds}


def _load_visec_source(config, unified_cfg, logger):
    dataset_path = str(unified_cfg.get("visec", {}).get("path", "hustep-lab/ViSEC"))
    cache_dir = _resolve_cache_dir(config, unified_cfg, "visec")
    base = _load_source_split(dataset_path, "train", logger, "visec", cache_dir=cache_dir)
    if "path" in base.column_names and "audio" not in base.column_names:
        base = base.rename_column("path", "audio")

    visec_cfg = unified_cfg.get("visec", {})
    val_split = float(visec_cfg.get("val_split", 0.1))
    test_split = float(visec_cfg.get("test_split", 0.1))
    split_seed = int(visec_cfg.get("split_seed", config["seed"]))

    train_raw, val_raw, _ = _split_dataset_by_speaker(
        dataset=base,
        speaker_column="speaker_id",
        val_split=val_split,
        test_split=test_split,
        seed=split_seed,
    )

    train_ds = _normalize_split(
        train_raw,
        source_name="visec",
        config=config,
        unified_cfg=unified_cfg,
    )
    val_ds = _normalize_split(
        val_raw,
        source_name="visec",
        config=config,
        unified_cfg=unified_cfg,
    )
    return {"train": train_ds, "validation": val_ds}


def _split_dataset_by_speaker(dataset, speaker_column: str, val_split: float, test_split: float, seed: int):
    total_holdout = float(val_split) + float(test_split)
    if not (0.0 < total_holdout < 1.0):
        raise ValueError(
            f"Speaker split requires val_split + test_split in (0,1). "
            f"Got val={val_split}, test={test_split}"
        )

    speaker_ids = np.array(dataset[speaker_column])
    unique_speakers = np.unique(speaker_ids)
    if len(unique_speakers) < 3:
        raise ValueError("Need at least 3 unique speakers to create train/val/test splits.")

    train_speakers, holdout_speakers = train_test_split(
        unique_speakers,
        test_size=total_holdout,
        random_state=seed,
        shuffle=True,
    )

    if test_split <= 0:
        val_speakers = holdout_speakers
        test_speakers = np.array([], dtype=holdout_speakers.dtype)
    elif val_split <= 0:
        val_speakers = np.array([], dtype=holdout_speakers.dtype)
        test_speakers = holdout_speakers
    else:
        relative_test = test_split / total_holdout
        val_speakers, test_speakers = train_test_split(
            holdout_speakers,
            test_size=relative_test,
            random_state=seed,
            shuffle=True,
        )

    train_idx = np.flatnonzero(np.isin(speaker_ids, train_speakers)).tolist()
    val_idx = np.flatnonzero(np.isin(speaker_ids, val_speakers)).tolist()
    test_idx = np.flatnonzero(np.isin(speaker_ids, test_speakers)).tolist()

    return dataset.select(train_idx), dataset.select(val_idx), dataset.select(test_idx)


def _normalize_split(dataset, source_name: str, config, unified_cfg):
    if source_name == "visec" and "path" in dataset.column_names and "audio" not in dataset.column_names:
        dataset = dataset.rename_column("path", "audio")

    if "audio" not in dataset.column_names:
        raise ValueError(f"Source '{source_name}' is missing required 'audio' column after normalization.")

    labels_gender = config["labels"]["gender"]
    lsvsc_map = _get_lsvsc_dialect_map(unified_cfg)
    num_rows = len(dataset)
    raw_gender = dataset["gender"] if "gender" in dataset.column_names else [None] * num_rows
    raw_region = dataset["region"] if "region" in dataset.column_names else [None] * num_rows
    raw_dialect = dataset["dialect"] if "dialect" in dataset.column_names else [None] * num_rows
    raw_accent = dataset["accent"] if "accent" in dataset.column_names else [None] * num_rows
    raw_speaker_id = dataset["speaker_id"] if "speaker_id" in dataset.column_names else [None] * num_rows
    raw_speaker_id_vimd = dataset["speakerID"] if "speakerID" in dataset.column_names else [None] * num_rows

    valid_indices = []
    gender_ids = []
    region_values = []
    speaker_ids = []
    source_values = []

    for idx in range(num_rows):
        example = {
            "gender": raw_gender[idx],
            "region": raw_region[idx],
            "dialect": raw_dialect[idx],
            "accent": raw_accent[idx],
            "speaker_id": raw_speaker_id[idx],
            "speakerID": raw_speaker_id_vimd[idx],
        }
        gender_id = _map_gender_to_label(example.get("gender"), source_name, labels_gender)
        region = _map_region_to_coarse(example, source_name, lsvsc_map)
        if gender_id is None or region is None:
            continue

        valid_indices.append(idx)
        gender_ids.append(gender_id)
        region_values.append(region)
        speaker_ids.append(_extract_speaker_id(example, source_name))
        source_values.append(source_name)

    if not valid_indices:
        raise ValueError(f"Source '{source_name}' has no valid samples after label normalization.")

    dataset = dataset.select(valid_indices)
    remove_columns = [column for column in dataset.column_names if column != "audio"]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    dataset = dataset.add_column("gender", gender_ids)
    dataset = dataset.add_column("region", region_values)
    dataset = dataset.add_column("source_speaker_id", speaker_ids)
    dataset = dataset.add_column("source", source_values)

    return dataset


def _extract_speaker_id(example, source_name: str) -> str:
    if source_name == "vimd":
        raw = example.get("speakerID", "")
    elif source_name == "visec":
        raw = example.get("speaker_id", "")
    else:
        raw = ""
    return "" if raw is None else str(raw)


def _map_gender_to_label(raw_gender, source_name: str, label_map) -> int | None:
    canonical = _canonical_gender(raw_gender, source_name)
    if canonical is None:
        return None
    if canonical not in label_map:
        raise ValueError(
            f"Canonical gender '{canonical}' is missing from labels.gender in config."
        )
    return int(label_map[canonical])


def _canonical_gender(raw_gender, source_name: str) -> str | None:
    if raw_gender is None:
        return None

    if source_name == "vimd":
        if raw_gender in {0, "0"}:
            return "Male"
        if raw_gender in {1, "1"}:
            return "Female"

    value = str(raw_gender).strip().lower()
    if value in {"male", "m"}:
        return "Male"
    if value in {"female", "f"}:
        return "Female"
    return None


def _map_region_to_coarse(example, source_name: str, lsvsc_map: Dict[str, str]) -> str | None:
    if source_name == "vimd":
        raw_region = example.get("region")
        return _canonical_region(raw_region)
    if source_name == "visec":
        raw_accent = example.get("accent")
        if raw_accent is None:
            return None
        key = str(raw_accent).strip().lower()
        mapped = DEFAULT_VISEC_ACCENT_MAP.get(key)
        return _canonical_region(mapped)
    if source_name == "lsvsc":
        raw_dialect = example.get("dialect")
        if raw_dialect is None:
            return None
        key = str(raw_dialect).strip().lower()
        mapped = lsvsc_map.get(key)
        return _canonical_region(mapped)
    return None


def _canonical_region(raw_value) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None

    if value in {"North", "Central", "South"}:
        return value

    lowered = value.lower()
    mapped = DEFAULT_REGION_ALIASES.get(lowered)
    if mapped is not None:
        return mapped
    return None


def _get_lsvsc_dialect_map(unified_cfg) -> Dict[str, str]:
    custom_map = unified_cfg.get("lsvsc", {}).get("dialect_map")
    if not custom_map:
        return DEFAULT_LSVSC_DIALECT_MAP

    normalized = {}
    for raw_label, canonical in custom_map.items():
        normalized[str(raw_label).strip().lower()] = str(canonical).strip()
    return normalized


def _resolve_sampling_probs(unified_cfg, enabled_sources: List[str]) -> Dict[str, float]:
    probs_cfg = unified_cfg.get("sampling_probs")
    if not probs_cfg:
        uniform = 1.0 / len(enabled_sources)
        return {source: uniform for source in enabled_sources}

    raw_probs = {}
    for source in enabled_sources:
        value = float(probs_cfg.get(source, 0.0))
        if value <= 0:
            raise ValueError(
                f"data.unified.sampling_probs.{source} must be > 0 for enabled source '{source}'. "
                "Set a positive probability or disable the source."
            )
        raw_probs[source] = value

    total = sum(raw_probs.values())
    if total <= 0:
        raise ValueError("data.unified.sampling_probs must sum to a positive value.")
    return {source: value / total for source, value in raw_probs.items()}


def _build_balanced_concat(train_parts_by_source: Dict[str, object], sampling_probs: Dict[str, float], seed: int, logger):
    source_sizes = {source: len(dataset) for source, dataset in train_parts_by_source.items()}
    total_size = max(
        math.ceil(source_sizes[source] / max(sampling_probs[source], 1e-12))
        for source in train_parts_by_source
    )

    repeated_parts = []
    final_counts = {}
    for idx, (source, dataset) in enumerate(train_parts_by_source.items()):
        target_count = max(source_sizes[source], int(round(total_size * sampling_probs[source])))
        target_count = min(max(target_count, 1), max(target_count, source_sizes[source]))
        repeated_parts.append(_repeat_dataset(dataset, target_count, seed + (idx + 1) * 997))
        final_counts[source] = target_count

    logger.info(f"Balanced concat target epoch size: {sum(final_counts.values()):,}")
    logger.info(f"Balanced concat per-source targets: {final_counts}")

    return concatenate_datasets(repeated_parts).shuffle(seed=seed)


def _repeat_dataset(dataset, target_count: int, seed: int):
    num_rows = len(dataset)
    if num_rows == 0:
        raise ValueError("Cannot repeat an empty dataset.")

    rng = np.random.default_rng(seed)
    if target_count <= num_rows:
        indices = rng.permutation(num_rows)[:target_count]
    else:
        full_repeats = target_count // num_rows
        remainder = target_count % num_rows
        chunks = [rng.permutation(num_rows) for _ in range(full_repeats)]
        if remainder:
            chunks.append(rng.permutation(num_rows)[:remainder])
        indices = np.concatenate(chunks, axis=0)

    return dataset.select(indices.tolist())


def _log_label_stats(dataset, logger, name: str):
    if len(dataset) == 0:
        logger.warning(f"{name}: dataset is empty after normalization/filtering.")
        return

    region_counts = Counter(dataset["region"])
    gender_counts = Counter(dataset["gender"])
    logger.info(f"{name} region distribution: {dict(region_counts)}")
    logger.info(f"{name} gender distribution: {dict(gender_counts)}")


def _log_source_mix(dataset, logger, name: str):
    if "source" not in dataset.column_names:
        return
    source_counts = Counter(dataset["source"])
    logger.info(f"{name} source mix: {dict(source_counts)}")
