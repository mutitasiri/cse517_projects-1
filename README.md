# cse517_projects

## Prerequisite

1. `git clone https://github.com/karansikka1/documentIntent_emnlp19`
2. Install pytorch
3. Start development


## Run on CPU

```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json
```

## Run on GPU

```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --device cuda:0
```