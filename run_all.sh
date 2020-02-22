python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split0 \
    --log_dir ./logs \
    --tensorboard

python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_1.json \
    ./documentIntent_emnlp19/splits/val_split_1.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split1 \
    --log_dir ./logs \
    --tensorboard

python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_2.json \
    ./documentIntent_emnlp19/splits/val_split_2.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split2 \
    --log_dir ./logs \
    --tensorboard

python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_3.json \
    ./documentIntent_emnlp19/splits/val_split_3.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split3 \
    --log_dir ./logs \
    --tensorboard

python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_4.json \
    ./documentIntent_emnlp19/splits/val_split_4.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split4 \
    --log_dir ./logs \
    --tensorboard