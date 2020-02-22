# cse517_projects

## Prerequisite

1. `git clone https://github.com/karansikka1/documentIntent_emnlp19`
2. Install anaconda
3. `conda env create -f environment.yml`.
4. `conda activate cse517_projects`.
5. Start development

## Run Tensorboard

```
tensorboard --logdir ./logs --host 0.0.0.0
```
Then, you can access the site at `[host].cs.washington.edu:[port_number]`.

## Run on CPU

```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --classification intent \
    --tensorboard
```

## Run on GPU

```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --device cuda:0 \
    --classification intent \
    --tensorboard
```

## Run using the provided script

```
./run_all.sh
```
