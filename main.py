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

    # TODO: fix
    # Load labels
    # with open(args.label_intent, 'r') as f:
    #     intent_labels = json.load(f)
    # with open(args.label_semiotic, 'r') as f:
    #     semiotic_labels = json.load(f)
    # with open(args.label_contextual, 'r') as f:
    #     contextual_labels = json.load(f)

    # Sample output
    # output = parse_post(posts[0])
    # import pdb; pdb.set_trace()
    
    # Set up data loaders differently based on the task
    # TODO: Extend to ELMo + word2vec etc.
    data_loader = None
    if args.type == 'image_only':
        data_loader = ImageOnlyDataset(posts)
    elif args.type == 'image_text':
        data_loader = ImageTextDataset() # TODO: fix
    elif args.type == 'text_only':
        data_loader = TextOnlyDataset() # TODO: fix
    
    # Set up the model
    model = Model(vocab_size = 1)
    
    # Set up an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    # TODO: Set up loss function
    loss = None
    
    # Setup tensorboard
    if args.tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=args.log_dir, flush_secs=1)
    else:
        writer = None
    
    # Set up a progress bar
    pbar = tqdm.tqdm(range(data_loader))
    
    # Training loop
    for batch_id, batch in enumerate(pbar):
        # TODO: training loop
        pass
    
    # Terminate the progress bar
    pbar.close()

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
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs', help="Log directory")
    print(parser.parse_args())
    main(parser.parse_args())

"""
python main.py \
    ./documentIntent_emnlp19/master_metadata.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json
""" # pylint: disable=pointless-string-statement
