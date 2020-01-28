import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np
import tqdm
import torch
import torch.utils.tensorboard as tensorboard

from data import ImageOnlyDataset, ImageTextDataset, TextOnlyDataset
from network import Model

def main(args: argparse.Namespace):
    # Load input data
    with open(args.input_metadata, 'r') as f:
        posts = json.load(f)

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

    # Sample output
    # output = parse_post(posts[0])
    # import pdb; pdb.set_trace()

    # Set up data loaders differently based on the task
    # TODO: Extend to ELMo + word2vec etc.
    dataset = None
    if args.type == 'image_only':
        dataset = ImageOnlyDataset() # TODO: fix
    elif args.type == 'image_text':
        dataset = ImageTextDataset(posts, labels)
    elif args.type == 'text_only':
        dataset = TextOnlyDataset() # TODO: fix
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=args.shuffle,
                                              num_workers=args.num_workers)

    # Set up the model
    model = Model(vocab_size=10)

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
        # Set up a progress bar
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

        total_loss = 0
        for _, batch in pbar:
            caption_data = batch['caption']
            image_data = batch['image']
            label = batch['label']

            model.zero_grad()

            pred = model(image_data, caption_data)

            intent_loss = loss_fn(pred['intent'], label['intent'])
            semiotic_loss = loss_fn(pred['semiotic'], label['semiotic'])
            contextual_loss = loss_fn(pred['contextual'], label['contextual'])
            loss = intent_loss + semiotic_loss + contextual_loss

            total_loss += loss.item()
            loss.backward()

            optimizer.step()

        # Terminate the progress bar
        pbar.close()

        print("Epoch {}: Loss = {}".format(epoch, total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Instagram Intent Classifier")
    parser.add_argument('input_metadata', type=str, help="")
    parser.add_argument('label_intent', type=str, help="")
    parser.add_argument('label_semiotic', type=str, help="")
    parser.add_argument('label_contextual', type=str, help="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_only', dest='type', action='store_const', const='image_only')
    group.add_argument('--image_text', dest='type', action='store_const', const='image_text')
    group.add_argument('--text_only', dest='type', action='store_const', const='text_only')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Whether to shuffle data loader")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs', help="Log directory")
    print(parser.parse_args())
    main(parser.parse_args())

"""
python main.py --image_text \
    ./documentIntent_emnlp19/master_metadata.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json
""" # pylint: disable=pointless-string-statement
