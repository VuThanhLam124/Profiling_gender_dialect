# Vietnamese Speaker Profiling# Vietnamese Speaker Profiling# Vietnamese Speaker Profiling



Identify gender and dialect (region) from Vietnamese speech using deep learning.



**Model Architecture:** Encoder + Attentive Pooling + LayerNorm + Classification HeadsIdentify gender and dialect (region) from Vietnamese speech using deep learning.Identify gender and dialect (region) from Vietnamese speech using deep learning.



**Supported Encoders:** WavLM, HuBERT, Wav2Vec2, Whisper



**Supported Datasets:** ViSpeech (CSV), ViMD (HuggingFace)**Model Architecture:** Encoder + Attentive Pooling + LayerNorm + Classification Heads**Model Architecture:** Encoder + Attentive Pooling + LayerNorm + Classification Heads



## Features



- Gender classification: Male / Female**Supported Encoders:** WavLM, HuBERT, Wav2Vec2, Whisper**Supported Encoders:** WavLM, HuBERT, Wav2Vec2, Whisper

- Dialect classification: North / Central / South (Vietnamese regions)

- Full model finetuning from raw audio

- Data augmentation for improved generalization

- Multiple encoder support for comparison experiments## Features## Features

- Multiple dataset support (ViSpeech, ViMD)

- Web interface with Gradio

- MLflow experiment tracking

- Support multiple audio formats: WAV, MP3, FLAC, OGG, M4A- Gender classification: Male / Female- Gender classification: Male / Female



## Installation- Dialect classification: North / Central / South (Vietnamese regions)- Dialect classification: North / Central / South (Vietnamese regions)



```bash- Full model finetuning from raw audio- Multiple encoder support for comparison experiments

# Clone repository

git clone https://github.com/VuThanhLam124/Profiling_gender_dialect.git- Data augmentation for improved generalization- Web interface with Gradio

cd Profiling_gender_dialect

- Multiple encoder support for comparison experiments- MLflow experiment tracking

# Install dependencies

pip install -r requirements.txt- Web interface with Gradio- Support multiple audio formats: WAV, MP3, FLAC, OGG, M4A

```

- MLflow experiment tracking

## Project Structure

- Support multiple audio formats: WAV, MP3, FLAC, OGG, M4A## Installation

```

Profiling_gender_dialect/

├── src/

│   ├── __init__.py## Installation```bash

│   ├── models.py              # Model architecture (multi-encoder support)

│   └── utils.py               # Utility functions# Clone repository

├── configs/

│   ├── finetune.yaml          # Training config```bashgit clone https://github.com/VuThanhLam124/Profiling_gender_dialect.git

│   ├── eval.yaml              # Evaluation config

│   └── infer.yaml             # Inference config# Clone repositorycd Profiling_gender_dialect

├── notebooks/

│   └── speaker-profiling.ipynb  # Kaggle notebookgit clone https://github.com/VuThanhLam124/Profiling_gender_dialect.git

├── app.py                     # Gradio web interface

├── finetune.py                # Training script (raw audio)cd Profiling_gender_dialect# Install dependencies

├── eval.py                    # Evaluation script

├── infer.py                   # Inference scriptpip install -r requirements.txt

├── requirements.txt

└── README.md# Install dependencies```

```

pip install -r requirements.txt

## Datasets

```## Project Structure

### ViSpeech (CSV format)

Download: https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing



Reference: https://github.com/TranNguyenNB/ViSpeech## Project Structure```



```Profiling_gender_dialect/

ViSpeech/

├── trainset/              # Training audio files```├── src/

├── clean_testset/         # Clean test audio files

├── noisy_testset/         # Noisy test audio filesProfiling_gender_dialect/│   ├── __init__.py

└── metadata/

    ├── trainset.csv├── src/│   ├── models.py              # Model architecture (multi-encoder support)

    ├── clean_testset.csv

    └── noisy_testset.csv│   ├── __init__.py│   └── utils.py               # Utility functions

```

│   ├── models.py              # Model architecture (multi-encoder support)├── configs/

### ViMD (HuggingFace format)

Path on Kaggle: `/kaggle/input/vimd-dataset`│   └── utils.py               # Utility functions│   ├── finetune.yaml          # Training config



```├── configs/│   ├── finetune.yaml.example  # Config template

vimd-dataset/

├── train/│   ├── finetune.yaml          # Training config│   ├── eval.yaml              # Evaluation config

├── valid/

└── test/│   ├── eval.yaml              # Evaluation config│   └── infer.yaml             # Inference config

```

│   └── infer.yaml             # Inference config├── notebooks/

Columns: `audio` (dict), `gender` (int: 0/1), `region` (str: North/Central/South)

├── notebooks/│   └── speaker-profiling.ipynb  # Kaggle notebook

## Usage

│   └── speaker-profiling.ipynb  # Kaggle notebook├── app.py                     # Gradio web interface

### Workflow Overview

├── app.py                     # Gradio web interface├── finetune.py                # Training script

```

┌─────────────────────────────────────────────────────────────────────┐├── finetune.py                # Training script (raw audio)├── eval.py                    # Evaluation script

│  1. Training (full model finetuning from raw audio)                 │

│     python finetune.py --config configs/finetune.yaml               │├── eval.py                    # Evaluation script├── infer.py                   # Inference script

│                  |                                                   │

│     outputs/speaker-profiling/best_model/                           │├── infer.py                   # Inference script├── prepare_data.py            # Feature extraction script

└─────────────────────────────────────────────────────────────────────┘

                                    |├── requirements.txt├── compare_encoders.py        # Encoder comparison script

┌─────────────────────────────────────────────────────────────────────┐

│  2. Evaluation / Inference                                          │└── README.md├── requirements.txt

│     python eval.py --checkpoint outputs/best_model --test clean     │

│     python infer.py --audio path/to/audio.wav                       │```└── README.md

└─────────────────────────────────────────────────────────────────────┘

``````



### 1. Training## Dataset



Edit `configs/finetune.yaml`:## Dataset



```yamlDownload ViSpeech dataset:

model:

  name: "microsoft/wavlm-base-plus"- Link: https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharingDownload ViSpeech dataset:

  freeze_encoder: false

  dropout: 0.1- Link: https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing

  head_hidden_dim: 256

Reference: https://github.com/TranNguyenNB/ViSpeech

data:

  source: "vispeech"  # Options: vispeech, vimdReference: https://github.com/TranNguyenNB/ViSpeech

  

  # ViSpeech paths### Data Structure

  train_meta: "/path/to/ViSpeech/metadata/trainset.csv"

  train_audio: "/path/to/ViSpeech/trainset"### Data Structure

  

  # ViMD path (HuggingFace format)```

  vimd_path: "/kaggle/input/vimd-dataset"

ViSpeech/```

training:

  batch_size: 32├── trainset/              # Training audio filesViSpeech/

  learning_rate: 5e-5

  num_epochs: 15├── clean_testset/         # Clean test audio files├── trainset/              # Training audio files



augmentation:├── noisy_testset/         # Noisy test audio files├── clean_testset/         # Clean test audio files

  enabled: true

  prob: 0.8└── metadata/├── noisy_testset/         # Noisy test audio files

```

    ├── trainset.csv└── metadata/

Run training:

    ├── clean_testset.csv    ├── trainset.csv

```bash

# Train with ViSpeech    └── noisy_testset.csv    ├── clean_testset.csv

python finetune.py --config configs/finetune.yaml

```    └── noisy_testset.csv

# Train with ViMD (change data.source to "vimd" in config)

python finetune.py --config configs/finetune.yaml```

```

Metadata CSV columns: `audio_name`, `speaker`, `gender` (Male/Female), `dialect` (North/Central/South)

View experiments in MLflow UI:

Metadata CSV columns: `audio_name`, `speaker`, `gender` (Male/Female), `dialect` (North/Central/South)

```bash

mlflow ui --port 5000## Usage

# Open http://localhost:5000

```## Usage



### 2. Evaluation### Workflow Overview



```bash### Workflow Overview

# Evaluate on ViSpeech clean test

python eval.py \```

    --checkpoint outputs/speaker-profiling/best_model \

    --config configs/eval.yaml \┌─────────────────────────────────────────────────────────────────────┐```

    --test_name clean_test

│  1. Training (full model finetuning from raw audio)                 │┌─────────────────────────────────────────────────────────────────────┐

# Evaluate on ViSpeech noisy test

python eval.py \│     python finetune.py --config configs/finetune.yaml               ││  1. Feature Extraction (run once per dataset)                       │

    --checkpoint outputs/speaker-profiling/best_model \

    --config configs/eval.yaml \│                  |                                                   ││     python prepare_data.py --dataset vispeech --split train         │

    --test_name noisy_test

│     outputs/speaker-profiling/best_model/                           ││                  ↓                                                   │

# Evaluate on ViMD test

python eval.py \└─────────────────────────────────────────────────────────────────────┘│     datasets/ViSpeech/train/                                        │

    --checkpoint outputs/speaker-profiling/best_model \

    --config configs/eval.yaml \                                    |│     ├── features/*.npy                                              │

    --test_name vimd_test

```┌─────────────────────────────────────────────────────────────────────┐│     └── metadata.csv                                                │



### 3. Inference│  2. Evaluation / Inference                                          │└─────────────────────────────────────────────────────────────────────┘



```bash│     python eval.py --checkpoint outputs/best_model --test clean     │                                    ↓

# Single audio file

python infer.py --config configs/infer.yaml --audio path/to/audio.wav│     python infer.py --audio path/to/audio.wav                       │┌─────────────────────────────────────────────────────────────────────┐



# Directory of audio files└─────────────────────────────────────────────────────────────────────┘│  2. Training (run many times with different configs)                │

python infer.py --config configs/infer.yaml --audio_dir path/to/folder

``````│     python finetune.py --config configs/finetune.yaml               │



### 4. Web Interface (Gradio)│                  ↓                                                   │



```bash### 1. Training│     output/speaker-profiling/best_model/                            │

# Start local server

python app.py --config configs/infer.yaml└─────────────────────────────────────────────────────────────────────┘



# Create public linkEdit `configs/finetune.yaml`:                                    ↓

python app.py --config configs/infer.yaml --share

┌─────────────────────────────────────────────────────────────────────┐

# Custom port

python app.py --config configs/infer.yaml --port 8080```yaml│  3. Evaluation / Inference                                          │

```

model:│     python eval.py --config configs/eval.yaml                       │

Open browser at `http://localhost:7860`

  encoder_name: "microsoft/wavlm-base-plus"  # or hubert, wav2vec2│     python infer.py --audio path/to/audio.wav                       │

## Model Architecture

  freeze_encoder: false  # Full model finetuning└─────────────────────────────────────────────────────────────────────┘

```

      Audio Input (raw waveform)  dropout: 0.1```

          |

          v  head_hidden_dim: 256

    Feature Extractor

          |### 1. Feature Extraction (Required)

          v

Encoder (WavLM/HuBERT/Wav2Vec2/Whisper)data:

          |

          v  data_dir: "/path/to/ViSpeech"  # Root directory containing trainset/, metadata/Extract encoder features and save to dataset-specific folders. **Run once per encoder, reuse many times** for experiments.

Hidden States [B, T, H]

          |  max_audio_len: 10.0            # Max audio length in seconds

          v

Attentive Pooling [B, H]```bash

          |

          vtraining:# Extract features for training set (default: WavLM)

  Layer Normalization

          |  batch_size: 8python prepare_data.py --dataset vispeech \

          v

    Dropout (0.1)  gradient_accumulation_steps: 4  # Effective batch size = 32    --config configs/finetune.yaml \

          |

    +---------------+  learning_rate: 5e-5    --split train \

    |               |

    v               v  num_epochs: 15    --output_dir datasets/ViSpeech/train

Gender Head    Dialect Head

(2 layers)      (3 layers)  warmup_ratio: 0.125

    |               |

    v               v  weight_decay: 0.0125# Extract features for validation set

  [B, 2]          [B, 3]

```python prepare_data.py --dataset vispeech \



### Supported Encodersaugmentation:    --config configs/finetune.yaml \



| Encoder | Model Name | Hidden Size |  enabled: true    --split val \

|---------|------------|-------------|

| WavLM Base | `microsoft/wavlm-base-plus` | 768 |  prob: 0.8    --output_dir datasets/ViSpeech/val

| WavLM Large | `microsoft/wavlm-large` | 1024 |

| HuBERT Base | `facebook/hubert-base-ls960` | 768 |```

| HuBERT Large | `facebook/hubert-large-ls960-ft` | 1024 |

| Wav2Vec2 Base | `facebook/wav2vec2-base-960h` | 768 |# Extract features for test sets

| Wav2Vec2 Large | `facebook/wav2vec2-large-960h` | 1024 |

| Whisper Small | `openai/whisper-small` | 768 |Run training:python prepare_data.py --dataset vispeech \

| Whisper Medium | `openai/whisper-medium` | 1024 |

    --config configs/finetune.yaml \

## Data Augmentation

```bash    --split clean_test \

When `augmentation.enabled: true`, the following augmentations are applied with probability `augmentation.prob`:

python finetune.py --config configs/finetune.yaml    --output_dir datasets/ViSpeech/clean_test

- AddGaussianNoise: Simulates background noise

- TimeStretch: Speed up/slow down without pitch change``````

- PitchShift: Change pitch without speed change

- Shift: Shift audio in time

- Gain: Adjust volume

View experiments in MLflow UI:**Output structure:**

## Column Auto-Detection

```

The training script automatically detects column names in metadata CSV:

```bashdatasets/ViSpeech/train/

| Purpose | Accepted Column Names |

|---------|----------------------|mlflow ui --port 5000├── features/           # Pre-extracted hidden states

| Audio filename | `audio_name`, `filename`, `file`, `path`, `audio` |

| Gender | `gender`, `sex` |# Open http://localhost:5000│   ├── audio001.npy    # Shape: [T, hidden_size]

| Dialect | `dialect`, `accent`, `region` |

| Speaker ID | `speaker`, `speaker_id`, `spk`, `spk_id` |```│   ├── audio002.npy



## Labels│   └── ...



| Gender | Code | Dialect | Code |### 2. Evaluation└── metadata.csv        # Labels: audio_name, gender, dialect, gender_label, dialect_label, feature_name

|--------|------|---------|------|

| Male   | 0    | North   | 0    |```

| Female | 1    | Central | 1    |

|        |      | South   | 2    |```bash



## Kaggle Training# Evaluate on clean test set**Benefits:**



Use the provided notebook for training on Kaggle with free GPU:python eval.py \- Skip encoder forward pass during training - faster experiments



1. Upload `notebooks/speaker-profiling.ipynb` to Kaggle    --checkpoint outputs/speaker-profiling/best_model \- Easily switch datasets by changing paths in config

2. Add ViSpeech/ViMD dataset to the notebook

3. Enable GPU accelerator (T4 or P100)    --config configs/finetune.yaml \- Support multiple datasets: `datasets/ViSpeech/`, `datasets/ViMD/`, etc.

4. Run all cells

    --test_name clean_test

The notebook tests multiple encoders (WavLM, HuBERT, Wav2Vec2) with 5 epochs each for comparison.

### 2. Training

## License

# Evaluate on noisy test set

MIT License

python eval.py \Copy config template and edit paths:

## Citation

    --checkpoint outputs/speaker-profiling/best_model \

If you use this code, please cite:

    --config configs/finetune.yaml \```bash

```bibtex

@misc{speaker_profiling_vietnamese,    --test_name noisy_testcp configs/finetune.yaml.example configs/finetune.yaml

  author = {Vu Thanh Lam},

  title = {Vietnamese Speaker Profiling},``````

  year = {2025},

  publisher = {GitHub},

  url = {https://github.com/VuThanhLam124/Profiling_gender_dialect}

}### 3. InferenceEdit `configs/finetune.yaml`:

```



## Acknowledgments

```bash```yaml

- ViSpeech dataset: https://github.com/TranNguyenNB/ViSpeech

- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm# Single audio filedata:

- HuBERT: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

- Wav2Vec2: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vecpython infer.py --config configs/infer.yaml --audio path/to/audio.wav  train_dir: "datasets/ViSpeech/train"  # Contains features/ and metadata.csv

- Whisper: https://github.com/openai/whisper

  val_dir: "datasets/ViSpeech/val"

# Directory of audio files```

python infer.py --config configs/infer.yaml --audio_dir path/to/folder

```Run training with MLflow tracking:



### 4. Web Interface (Gradio)```bash

python finetune.py --config configs/finetune.yaml

```bash```

# Start local server

python app.py --config configs/infer.yamlView experiments in MLflow UI:



# Create public link```bash

python app.py --config configs/infer.yaml --sharemlflow ui --port 5000

# Open http://localhost:5000

# Custom port```

python app.py --config configs/infer.yaml --port 8080

```### 3. Evaluation



Open browser at `http://localhost:7860`Edit data paths in `configs/eval.yaml`:



## Model Architecture```yaml

model:

```  checkpoint: "outputs/best_model"

      Audio Input (raw waveform)

          |data:

          v  clean_test_meta: "ViSpeech/metadata/clean_testset.csv"

    Feature Extractor  clean_test_audio: "ViSpeech/clean_testset"

          |  noisy_test_meta: "ViSpeech/metadata/noisy_testset.csv"

          v  noisy_test_audio: "ViSpeech/noisy_testset"

Encoder (WavLM/HuBERT/Wav2Vec2/Whisper)```

          |

          vRun evaluation:

Hidden States [B, T, H]

          |```bash

          vpython eval.py --config configs/eval.yaml

Attentive Pooling [B, H]```

          |

          v### 4. Inference

  Layer Normalization

          |Command line:

          v

    Dropout (0.1)```bash

          |# Single audio file

    +---------------+python infer.py --config configs/infer.yaml --audio path/to/audio.wav

    |               |

    v               v# Directory of audio files

Gender Head    Dialect Headpython infer.py --config configs/infer.yaml --audio_dir path/to/folder

(2 layers)      (3 layers)```

    |               |

    v               v### 5. Web Interface (Gradio)

  [B, 2]          [B, 3]

``````bash

# Start local server

### Supported Encoderspython app.py --config configs/infer.yaml



| Encoder | Model Name | Hidden Size |# Create public link

|---------|------------|-------------|python app.py --config configs/infer.yaml --share

| WavLM Base | `microsoft/wavlm-base-plus` | 768 |

| WavLM Large | `microsoft/wavlm-large` | 1024 |# Custom port

| HuBERT Base | `facebook/hubert-base-ls960` | 768 |python app.py --config configs/infer.yaml --port 8080

| HuBERT Large | `facebook/hubert-large-ls960-ft` | 1024 |```

| Wav2Vec2 Base | `facebook/wav2vec2-base-960h` | 768 |

| Wav2Vec2 Large | `facebook/wav2vec2-large-960h` | 1024 |Open browser at `http://localhost:7860`

| Whisper Small | `openai/whisper-small` | 768 |

| Whisper Medium | `openai/whisper-medium` | 1024 |## Model Architecture



## Data Augmentation```

      Audio Input

When `augmentation.enabled: true`, the following augmentations are applied with probability `augmentation.prob`:          |

          v

- AddGaussianNoise: Simulates background noiseEncoder (WavLM/HuBERT/Wav2Vec2/Whisper)

- TimeStretch: Speed up/slow down without pitch change          |

- PitchShift: Change pitch without speed change          v

- Shift: Shift audio in timeHidden States [B, T, H]

- Gain: Adjust volume          |

          v

## Column Auto-DetectionAttentive Pooling [B, H]

          |

The training script automatically detects column names in metadata CSV:          v

  Layer Normalization

| Purpose | Accepted Column Names |          |

|---------|----------------------|          v

| Audio filename | `audio_name`, `filename`, `file`, `path`, `audio` |    Dropout (0.1)

| Gender | `gender`, `sex` |          |

| Dialect | `dialect`, `accent`, `region` |    +---------------+

| Speaker ID | `speaker`, `speaker_id`, `spk`, `spk_id` |    |               |

    v               v

## LabelsGender Head    Dialect Head

(2 layers)      (3 layers)

| Gender | Code | Dialect | Code |    |               |

|--------|------|---------|------|    v               v

| Male   | 0    | North   | 0    |  [B, 2]          [B, 3]

| Female | 1    | Central | 1    |```

|        |      | South   | 2    |

### Supported Encoders

## Kaggle Training

| Encoder | Model Name | Hidden Size |

Use the provided notebook for training on Kaggle with free GPU:|---------|------------|-------------|

| WavLM Base | `microsoft/wavlm-base-plus` | 768 |

1. Upload `notebooks/speaker-profiling.ipynb` to Kaggle| WavLM Large | `microsoft/wavlm-large` | 1024 |

2. Add ViSpeech dataset to the notebook| HuBERT Base | `facebook/hubert-base-ls960` | 768 |

3. Enable GPU accelerator (T4 or P100)| HuBERT Large | `facebook/hubert-large-ls960-ft` | 1024 |

4. Run all cells| Wav2Vec2 Base | `facebook/wav2vec2-base-960h` | 768 |

| Wav2Vec2 Large | `facebook/wav2vec2-large-960h` | 1024 |

The notebook tests multiple encoders (WavLM, HuBERT, Wav2Vec2) with 5 epochs each for comparison.| Whisper Small | `openai/whisper-small` | 768 |

| Whisper Medium | `openai/whisper-medium` | 1024 |

## License

## Encoder Comparison

MIT License

Compare different encoders with the same architecture:

## Citation

```bash

If you use this code, please cite:# Compare all encoders

python compare_encoders.py \

```bibtex    --config configs/finetune.yaml \

@misc{speaker_profiling_vietnamese,    --output_dir results/encoder_comparison

  author = {Vu Thanh Lam},

  title = {Vietnamese Speaker Profiling},# Compare specific encoders

  year = {2025},python compare_encoders.py \

  publisher = {GitHub},    --config configs/finetune.yaml \

  url = {https://github.com/VuThanhLam124/Profiling_gender_dialect}    --output_dir results/encoder_comparison \

}    --encoders wavlm-base hubert-base wav2vec2-base whisper-small

``````



## Acknowledgments**Output:**

```

- ViSpeech dataset: https://github.com/TranNguyenNB/ViSpeechresults/encoder_comparison/

- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm├── features/                    # Extracted features per encoder

- HuBERT: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert├── checkpoints/                 # Trained models per encoder

- Wav2Vec2: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec├── comparison_results.csv       # Results table

- Whisper: https://github.com/openai/whisper├── comparison_results.md        # Markdown report

└── comparison.log
```

**Example Results:**

| Encoder | Gender Acc | Dialect Acc | Time (min) |
|---------|------------|-------------|------------|
| WavLM Base Plus | 0.9512 | 0.8234 | 25.3 |
| HuBERT Base | 0.9456 | 0.8156 | 24.1 |
| Wav2Vec2 Base | 0.9389 | 0.7989 | 23.8 |
| Whisper Small | 0.9234 | 0.7756 | 28.5 |

## Labels

| Gender | Code | Dialect | Code |
|--------|------|---------|------|
| Male   | 0    | North   | 0    |
| Female | 1    | Central | 1    |
|        |      | South   | 2    |

## Configuration

Key parameters in config files:

```yaml
# Model - supports multiple encoders
model:
  name: "microsoft/wavlm-base-plus"  # or hubert, wav2vec2, whisper
  hidden_size: 768                   # Depends on encoder
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

## Kaggle Training

Use the provided notebook for training on Kaggle with free GPU:

1. Upload `notebooks/speaker-profiling.ipynb` to Kaggle
2. Add ViSpeech dataset to the notebook
3. Enable GPU accelerator (T4 or P100)
4. Run all cells

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
- HuBERT: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
- Wav2Vec2: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
- Whisper: https://github.com/openai/whisper
