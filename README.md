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

### 1. Data Preparation

```bash
# Prepare metadata from audio directory
python prepare_data.py --audio_dir path/to/audio --output_dir path/to/output

# With train/val/test split
python prepare_data.py --audio_dir path/to/audio --output_dir path/to/output --split

# Skip audio validation (faster)
python prepare_data.py --audio_dir path/to/audio --output_dir path/to/output --skip_validation
```

### 2. Training

Edit data paths in `configs/finetune.yaml`:

```yaml
data:
  train_meta: "ViSpeech/metadata/trainset.csv"
  train_audio: "ViSpeech/trainset"
  val_split: 0.15
```

Run training:

```bash
python finetune.py --config configs/finetune.yaml
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
    +------------------+
    |                  |
    v                  v
Gender Head       Dialect Head
(2 layers)        (3 layers)
    |                  |
    v                  v
[B, 2]             [B, 3]
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
# Model
model:
  name: "microsoft/wavlm-base-plus"
  dropout: 0.1
  head_hidden_dim: 256
  freeze_encoder: false

# Audio
audio:
  sampling_rate: 16000
  max_duration: 10.0

# Training
training:
  batch_size: 8
  learning_rate: 1e-5
  num_epochs: 10
  weight_decay: 0.01

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
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/VuThanhLam124/Profiling_gender_dialect}
}
```

## Acknowledgments

- ViSpeech dataset: https://github.com/TranNguyenNB/ViSpeech
- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
