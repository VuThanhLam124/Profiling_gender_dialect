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
