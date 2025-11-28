# Vietnamese Speaker Profiling

Identify gender and dialect (region) from Vietnamese speech using deep learning.

**Model Architecture:** WavLM + Attentive Pooling + LayerNorm

## Features

- Gender classification: Male / Female
- Dialect classification: North / Central / South (Vietnamese regions)
- Web interface with Gradio
- Support multiple audio formats: WAV, MP3, FLAC, OGG, M4A

## Installation

```bash
# Clone repository
git clone https://github.com/VuThanhLam124/Profiling_gender_dialect.git
cd Profiling_gender_dialect

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Profiling_gender_dialect/
├── src/
│   ├── __init__.py
│   ├── models.py          # Model architecture
│   └── utils.py           # Utility functions
├── configs/
│   ├── finetune.yaml      # Training config
│   ├── eval.yaml          # Evaluation config
│   └── infer.yaml         # Inference config
├── app.py                 # Gradio web interface
├── finetune.py            # Training script
├── eval.py                # Evaluation script
├── infer.py               # Inference script
├── prepare_data.py        # Data preparation script
├── requirements.txt
└── README.md
```

## Dataset

Download ViSpeech dataset:
- Link: https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing

Reference: https://github.com/TranNguyenNB/ViSpeech

### Data Structure

```
ViSpeech/
├── trainset/              # Training audio files
├── clean_testset/         # Clean test audio files
├── noisy_testset/         # Noisy test audio files
└── metadata/
    ├── trainset.csv
    ├── clean_testset.csv
    └── noisy_testset.csv
```

Metadata CSV columns: `audio_name`, `speaker`, `gender` (Male/Female), `dialect` (North/Central/South)

## Usage

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Feature Extraction (run once per dataset)                       │
│     python prepare_data.py --dataset vispeech --split train         │
│                  ↓                                                   │
│     datasets/ViSpeech/train/                                        │
│     ├── features/*.npy                                              │
│     └── metadata.csv                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. Training (run many times with different configs)                │
│     python finetune.py --config configs/finetune.yaml               │
│                  ↓                                                   │
│     output/speaker-profiling/best_model/                            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. Evaluation / Inference                                          │
│     python eval.py --config configs/eval.yaml                       │
│     python infer.py --audio path/to/audio.wav                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Feature Extraction (Required)

Extract WavLM features and save to dataset-specific folders. **Run once, reuse many times** for experiments with different hyperparameters.

```bash
# Extract features for training set
python prepare_data.py --dataset vispeech \
    --split train \
    --output_dir datasets/ViSpeech/train

# Extract features for validation set
python prepare_data.py --dataset vispeech \
    --split val \
    --output_dir datasets/ViSpeech/val

# Extract features for test sets
python prepare_data.py --dataset vispeech \
    --split clean_test \
    --output_dir datasets/ViSpeech/clean_test

python prepare_data.py --dataset vispeech \
    --split noisy_test \
    --output_dir datasets/ViSpeech/noisy_test
```

**Output structure:**
```
datasets/ViSpeech/train/
├── features/           # Pre-extracted WavLM hidden states
│   ├── audio001.npy    # Shape: [T, 768]
│   ├── audio002.npy
│   └── ...
└── metadata.csv        # Labels: audio_name, gender, dialect, gender_label, dialect_label, feature_name
```

**Benefits:**
- Skip WavLM forward pass during training → faster experiments
- Easily switch datasets by changing paths in config
- Support multiple datasets: `datasets/ViSpeech/`, `datasets/ViMD/`, etc.

### 2. Training

Copy config template and edit paths:

```bash
cp configs/finetune.yaml.example configs/finetune.yaml
```

Edit `configs/finetune.yaml`:

```yaml
data:
  train_dir: "datasets/ViSpeech/train"  # Contains features/ and metadata.csv
  val_dir: "datasets/ViSpeech/val"
```

Run training with MLflow tracking:

```bash
python finetune.py --config configs/finetune.yaml
```

View experiments in MLflow UI:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 3. Evaluation

Edit data paths in `configs/eval.yaml`:

```yaml
model:
  checkpoint: "outputs/best_model"

data:
  clean_test_meta: "ViSpeech/metadata/clean_testset.csv"
  clean_test_audio: "ViSpeech/clean_testset"
  noisy_test_meta: "ViSpeech/metadata/noisy_testset.csv"
  noisy_test_audio: "ViSpeech/noisy_testset"
```

Run evaluation:

```bash
python eval.py --config configs/eval.yaml
```

### 4. Inference

Command line:

```bash
# Single audio file
python infer.py --config configs/infer.yaml --audio path/to/audio.wav

# Directory of audio files
python infer.py --config configs/infer.yaml --audio_dir path/to/folder
```

### 5. Web Interface (Gradio)

```bash
# Start local server
python app.py --config configs/infer.yaml

# Create public link
python app.py --config configs/infer.yaml --share

# Custom port
python app.py --config configs/infer.yaml --port 8080
```

Open browser at `http://localhost:7860`

## Model Architecture

```
      Audio Input
          |
          v
WavLM Encoder (pretrained)
          |
          v
Hidden States [B, T, 768]
          |
          v
Attentive Pooling [B, 768]
          |
          v
  Layer Normalization
          |
          v
    Dropout (0.1)
          |
    +---------------+
    |               |
    v               v
Gender Head    Dialect Head
(2 layers)      (3 layers)
    |               |
    v               v
  [B, 2]          [B, 3]
```

## Labels

| Gender | Code | Dialect | Code |
|--------|------|---------|------|
| Male   | 0    | North   | 0    |
| Female | 1    | Central | 1    |
|        |      | South   | 2    |

## Configuration

Key parameters in config files:

```yaml
# Model (classification heads only for training)
model:
  name: "microsoft/wavlm-base-plus"  # For reference
  hidden_size: 768                   # WavLM hidden dimension
  dropout: 0.1
  head_hidden_dim: 256

# Training
training:
  batch_size: 32
  learning_rate: 5e-5
  num_epochs: 15
  weight_decay: 0.0125

# Dataset paths (point to extracted features)
data:
  train_dir: "datasets/ViSpeech/train"
  val_dir: "datasets/ViSpeech/val"

# MLflow tracking
mlflow:
  enabled: true
  tracking_uri: "mlruns"
  experiment_name: "speaker-profiling"

# Loss
loss:
  dialect_weight: 3.0  # Higher weight for dialect task
```

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@misc{speaker_profiling_vietnamese,
  author = {Vu Thanh Lam},
  title = {Vietnamese Speaker Profiling},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/VuThanhLam124/Profiling_gender_dialect}
}
```

## Acknowledgments

- ViSpeech dataset: https://github.com/TranNguyenNB/ViSpeech
- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
