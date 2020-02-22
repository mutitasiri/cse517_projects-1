device="cuda:1"
log_dir="./logs"

for type in image_text text_only image_only
do
    for classification in intent semiotic contextual
    do
        for split in 0 1 2 3 4 
        do
            echo "Running ${type}-${classification} split no. ${split}"
            python main.py --${type} \
                ./documentIntent_emnlp19/splits/train_split_${split}.json \
                ./documentIntent_emnlp19/splits/val_split_${split}.json \
                ./documentIntent_emnlp19/labels/intent_labels.json \
                ./documentIntent_emnlp19/labels/semiotic_labels.json \
                ./documentIntent_emnlp19/labels/contextual_labels.json \
                --classification ${classification} \
                --name ${type}_${classification}_split${split} \
                --log_dir ${log_dir} \
                --device ${device} \
                --tensorboard
        done
    done
done