# Vietnamese Speaker Profiling

### Finetune

```bash
python finetune.py --config configs/finetune.yaml
```
### Eval:

```bash
python eval.py --checkpoint output/best_model --config configs/eval.yaml --test_name clean_test
```

### Infer:

```bash
python infer.py --config configs/infer.yaml --audio path/to/audio.wav
```

### Infer (Pho model on Hugging Face Hub)

```bash
python infer.py --config configs/infer_pho_hf.yaml --audio path/to/audio.wav
```

### Install from PyPI (inference)

```bash
pip install vn-speaker-profiling
vn-speaker-profiling-infer --audio path/to/audio.wav
vn-speaker-profiling-infer --audio_dir path/to/folder --batch_size 8
```

### Publish to PyPI (maintainers)

```bash
python -m pip install -U build twine
python -m build
python -m twine upload dist/*
```

### Datasets:

- [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)

- [ViSpeech Dataset](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing)

### Pretrained Models:
- [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)
- [Wav2Vec2 Base](https://huggingface.co/facebook/wav2vec2-base)
- [HuBERT Base](https://huggingface.co/facebook/hubert-base-ls960)
- [Whisper Base](https://huggingface.co/openai/whisper-base)
- [PhoWhisper-Base](https://huggingface.co/vinai/PhoWhisper-base)


### In Kaggle:
- https://www.kaggle.com/datasets/thanhlamdev/vimd-dataset
- https://www.kaggle.com/datasets/thanhlamdev/vispeech

### Architecture:
        Audio -> Encoder (WavLM/HuBERT/Wav2Vec2/Whisper) -> Last Hidden [B,T,H]
                              |
                     Attentive Pooling [B,H]
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

### Result:
| Mô hình | Kích thước tham số | Nhiệm vụ phân loại | Acc. ViSpeech (Clean) | Acc. ViSpeech (Noisy) | Acc. ViMD (Baseline) |
|---------|-------------------|-------------------|----------------------|----------------------|----------------------|
| wavlm-base-plus | ~94 triệu | Gender | 96.53% | 97.35% | 98.66% |
| | | Dialect | 88.33% | 84.41% | 88.49% |
| wav2vec2-base | ~95 triệu | Gender | 93.13% | 95.59% | 98.52% |
| | | Dialect | 87.13% | 83.63% | 88.65% |
| hubert-base-ls960 | ~96 triệu | Gender | 96.93% | 96.67% | 98.62% |
| | | Dialect | 87.40% | 82.55% | 87.52% |
| spkrec-ecapa-voxceleb | ~22 triệu | Gender | 96.80% | 98.43% | N/A |
| | | Dialect | 65.33% | 65.10% | N/A |
| PhoWhisper-base | ~73 triệu | Gender | 95.53% | N/A | 98.57% |
| | | Dialect | 93.28% | N/A | 90.67% |
| wav2vec2-base-vi-vlsp2020 | ~95 triệu | Gender | N/A | N/A | 98.72% |
| | | Dialect | N/A | N/A | 90.61% |

### Model
https://drive.google.com/drive/folders/1UCOVh9ut8jHmCFfMKgwM2_Mi_3rGhd94?usp=sharing

### Citation:
https://github.com/TranNguyenNB/ViSpeech

https://huggingface.co/datasets/nguyendv02/ViMD_Dataset
