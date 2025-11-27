# Speaker Profiling

Nhận diện giới tính và vùng miền từ giọng nói tiếng Việt.

**Kiến trúc:** WavLM + Attentive Pooling + LayerNorm

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu

```
ViSpeech/
├── trainset/           # Audio train
├── clean_testset/      # Audio test sạch
├── noisy_testset/      # Audio test nhiễu
└── metadata/
    ├── trainset.csv
    ├── clean_testset.csv
    └── noisy_testset.csv
```

File metadata cần có các cột: `audio_name`, `speaker`, `gender` (Male/Female), `dialect` (North/Central/South)

## Huấn luyện

### 1. Chỉnh path dữ liệu trong `configs/finetune.yaml`:

```yaml
data:
  train_meta: "ViSpeech/metadata/trainset.csv"   # Path đến file metadata
  train_audio: "ViSpeech/trainset"               # Path đến folder audio
  val_split: 0.15
```

### 2. Chạy training:

```bash
python finetune.py --config configs/finetune.yaml
```

## Đánh giá

Chỉnh path trong `configs/eval.yaml`:

```yaml
data:
  clean_test_meta: "ViSpeech/metadata/clean_testset.csv" # Path đến file metadata
  clean_test_audio: "ViSpeech/clean_testset"             # Path đến folder audio
  noisy_test_meta: "ViSpeech/metadata/noisy_testset.csv" # Path đến file metadata
  noisy_test_audio: "ViSpeech/noisy_testset"             # Path đến folder audio
```

```bash
python eval.py --config configs/eval.yaml
```

## Suy luận

```bash
# Single file
python infer.py --config configs/infer.yaml --audio path/to/audio.wav

# Folder
python infer.py --config configs/infer.yaml --audio_dir path/to/folder
```

## Label

| Gender | Code | Dialect | Code |
|--------|------|---------|------|
| Male   | 0    | North   | 0    |
| Female | 1    | Central | 1    |
|        |      | South   | 2    |
