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

### Datasets:

- [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)

- [ViSpeech Dataset](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing)

### Pretrained Models:
- [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)
- [Wav2Vec2 Base](https://huggingface.co/facebook/wav2vec2-base)
- [HuBERT Base](https://huggingface.co/facebook/hubert-base-ls960)


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