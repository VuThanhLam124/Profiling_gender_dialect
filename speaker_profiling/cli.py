import argparse
import json
import os
from pathlib import Path

from infer import SpeakerProfiler


def _default_max_duration(encoder_name: str) -> int:
    name = (encoder_name or "").lower()
    if "whisper" in name or "phowhisper" in name:
        return 30
    return 5


def infer_cli():
    parser = argparse.ArgumentParser(description="Vietnamese Speaker Profiling (Inference)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hf:Thanh-Lam/profiling-gender-dialect-pho",
        help="Checkpoint directory or HF Hub repo (e.g. hf:owner/repo).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vinai/PhoWhisper-base",
        help="Encoder name used to build the architecture (e.g. vinai/PhoWhisper-base).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="space_v2",
        choices=["space_v2", "default"],
        help="Preprocessing mode (space_v2 matches HF Space v2).",
    )
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--max_duration", type=int, default=None, help="Seconds.")
    parser.add_argument("--audio", type=str, default=None, help="Path to a single audio file.")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audio files.")
    parser.add_argument("--out_dir", type=str, default="output/predictions")
    parser.add_argument("--gender0_label", type=str, default="Female", help="Label for class 0.")
    parser.add_argument("--gender1_label", type=str, default="Male", help="Label for class 1.")
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        parser.error("Provide --audio or --audio_dir")

    max_duration = args.max_duration or _default_max_duration(args.encoder)

    config = {
        "model": {
            "checkpoint": args.checkpoint,
            "name": args.encoder,
            "head_hidden_dim": 256,
        },
        "audio": {
            "sampling_rate": args.sampling_rate,
            "max_duration": max_duration,
        },
        "inference": {
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "preprocess": {
            "mode": args.preprocess,
        },
        "input": {
            "audio_path": args.audio,
            "audio_dir": args.audio_dir,
        },
        "output": {
            "dir": args.out_dir,
            "save_results": True,
            "format": "json",
        },
        "labels": {
            "gender": {0: args.gender0_label, 1: args.gender1_label},
            "dialect": {0: "North", 1: "Central", 2: "South"},
        },
    }

    profiler = SpeakerProfiler(config)

    audio_files = []
    if args.audio:
        audio_files.append(args.audio)
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
            audio_files.extend(str(p) for p in audio_dir.glob(ext))

    results = profiler.predict_batch(audio_files)

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(output_path)
