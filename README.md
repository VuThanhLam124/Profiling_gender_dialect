# Vietnamese Speaker Profiling# Vietnamese Speaker Profiling

 

Finetune:Finetune:

```bash

python finetune.py --config configs/finetune.yamlpython ...

```

Eval:

Eval:

```bashpython ...

python eval.py --checkpoint output/best_model --config configs/eval.yaml --test_name clean_test

```Infer:

python ...

Infer:

```bash### Datasets:

python infer.py --config configs/infer.yaml --audio path/to/audio.wav- [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)

```- [ViSpeech Dataset](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing)



### Datasets:### Pretrained Models:

- [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)- [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)

- [ViSpeech Dataset](https://drive.google.com/file/d/1-BbOHf42o6eBje2WqQiiRKMtNxmZiRf9/view?usp=sharing)

### Pretrained Models:
- [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)
