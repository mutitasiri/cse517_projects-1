import argparse
import json
from typing import Dict, Optional, List, Any
import multiprocessing
import tqdm
import torch
import torch.utils.tensorboard as tensorboard

from data import Dictionary, ImageOnlyDataset, ImageTextDataset, TextOnlyDataset
from network import Model

def collate_fn_pad(batch):
    """
    Padds batch of variable length
    """
    output = {
        'id': [],
        'label': {
            'intent': [],
            'semiotic': [],
            'contextual': [],
        },
        'caption': [],
        'image': [],
    }

    max_caption_length = 0

    for sample in batch:
        output['id'].append(sample['id'])
        output['label']['intent'].append(sample['label']['intent'])
        output['label']['semiotic'].append(sample['label']['semiotic'])
        output['label']['contextual'].append(sample['label']['contextual'])
        output['caption'].append(sample['caption'])
        output['image'].append(sample['image'])

    output['label']['intent'] = torch.LongTensor(output['label']['intent'])
    output['label']['semiotic'] = torch.LongTensor(output['label']['semiotic'])
    output['label']['contextual'] = torch.LongTensor(output['label']['contextual'])
    output['caption'] = torch.nn.utils.rnn.pad_sequence(output['caption']).t() # (batch_size, sequence_length)
    output['image'] = torch.stack(output['image'], dim=0)
    return output

def main(args: argparse.Namespace):
    # Load input data
    with open(args.train_metadata, 'r') as f:
        train_posts = json.load(f)

    with open(args.train_metadata, 'r') as f:
        val_posts = json.load(f)

    # Load labels
    labels = {}
    with open(args.label_intent, 'r') as f:
        intent_labels = json.load(f)
        labels['intent'] = {}
        for label in intent_labels:
            labels['intent'][label] = len(labels['intent'])

    with open(args.label_semiotic, 'r') as f:
        semiotic_labels = json.load(f)
        labels['semiotic'] = {}
        for label in semiotic_labels:
            labels['semiotic'][label] = len(labels['semiotic'])

    with open(args.label_contextual, 'r') as f:
        contextual_labels = json.load(f)
        labels['contextual'] = {}
        for label in contextual_labels:
            labels['contextual'][label] = len(labels['contextual'])

    # Build dictionary from training set
    train_captions = []
    for post in train_posts:
        train_captions.append(post['orig_caption'])
    dictionary = Dictionary(tokenizer_method="TreebankWordTokenizer")
    dictionary.build_dictionary_from_captions(train_captions)

    # Set up torch device
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
        kwargs = {'pin_memory': True}
    else:
        device = torch.device('cpu')
        kwargs = {}

    # Set up number of workers
    num_workers = min(multiprocessing.cpu_count(), args.num_workers)

    # Set up data loaders differently based on the task
    # TODO: Extend to ELMo + word2vec etc.
    if args.type == 'image_only':
        train_dataset = ImageOnlyDataset() # TODO: fix
        val_dataset = ImageOnlyDataset()
    elif args.type == 'image_text':
        train_dataset = ImageTextDataset(train_posts, labels, dictionary)
        val_dataset = ImageTextDataset(val_posts, labels, dictionary)
    elif args.type == 'text_only':
        train_dataset = TextOnlyDataset() # TODO: fix
        val_dataset = TextOnlyDataset()
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.shuffle,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn_pad,
                                                    **kwargs)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn_pad,
                                                  **kwargs)

    # Set up the model
    model = Model(vocab_size=dictionary.size()).to(device)

    # Set up an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup tensorboard
    if args.tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=args.log_dir, flush_secs=1)
    else:
        writer = None

    # Training loop
    for epoch in range(args.epochs):
        for mode in ["train", "eval"]:
            # Set up a progress bar
            if mode == "train":
                pbar = tqdm.tqdm(enumerate(train_data_loader), total=len(train_data_loader))
                model.train()
            else:
                pbar = tqdm.tqdm(enumerate(val_data_loader), total=len(val_data_loader))
                model.eval()

            total_loss = 0
            for _, batch in pbar:
                caption_data = batch['caption'].to(device)
                image_data = batch['image'].to(device)
                label_intent = batch['label']['intent'].to(device)
                label_semiotic = batch['label']['semiotic'].to(device)
                label_contextual = batch['label']['contextual'].to(device)

                if mode == "train":
                    model.zero_grad()

                pred = model(image_data, caption_data)

                intent_loss = loss_fn(pred['intent'], label_intent)
                semiotic_loss = loss_fn(pred['semiotic'], label_semiotic)
                contextual_loss = loss_fn(pred['contextual'], label_contextual)
                loss = intent_loss + semiotic_loss + contextual_loss

                total_loss += loss.item()

                if mode == "train":
                    loss.backward()
                    optimizer.step()

            # Terminate the progress bar
            pbar.close()

            print("[{}] Epoch {}: Loss = {}".format(mode, epoch, total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Instagram Intent Classifier")
    parser.add_argument('train_metadata', type=str, help="Path to training metadata")
    parser.add_argument('val_metadata', type=str, help="Path to validation metadata")
    parser.add_argument('label_intent', type=str, help="Path to label for intent")
    parser.add_argument('label_semiotic', type=str, help="Path to label for semiotic")
    parser.add_argument('label_contextual', type=str, help="Path to label for contextual")
    group = parser.add_mutually_exclusive_group(required=True) # TODO: add help
    group.add_argument('--image_only', dest='type', action='store_const', const='image_only')
    group.add_argument('--image_text', dest='type', action='store_const', const='image_text')
    group.add_argument('--text_only', dest='type', action='store_const', const='text_only')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Whether to shuffle data loader")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--device', type=str, default='cpu', help="Pytorch device")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs', help="Log directory")
    print(parser.parse_args())
    main(parser.parse_args())

"""
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json
""" # pylint: disable=pointless-string-statement
