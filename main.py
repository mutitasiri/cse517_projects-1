import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import multiprocessing
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard

from data import Dictionary, ImageOnlyDataset, ImageTextDataset, TextOnlyDataset, \
                 collate_fn_pad_image_text, collate_fn_pad_image_only, collate_fn_pad_text_only
from network import Model

def main(args: argparse.Namespace):
    # Load input data
    with open(args.train_metadata, 'r') as f:
        train_posts = json.load(f)

    with open(args.val_metadata, 'r') as f:
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
        train_dataset = ImageOnlyDataset(train_posts, labels)
        val_dataset = ImageOnlyDataset(val_posts, labels)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=args.shuffle,
                                                        num_workers=num_workers,
                                                        collate_fn=collate_fn_pad_image_only,
                                                        **kwargs)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn_pad_image_only,
                                                    **kwargs)
    elif args.type == 'image_text':
        train_dataset = ImageTextDataset(train_posts, labels, dictionary)
        val_dataset = ImageTextDataset(val_posts, labels, dictionary)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=args.shuffle,
                                                        num_workers=num_workers,
                                                        collate_fn=collate_fn_pad_image_text,
                                                        **kwargs)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn_pad_image_text,
                                                    **kwargs)
    elif args.type == 'text_only':
        train_dataset = TextOnlyDataset(train_posts, labels, dictionary)
        val_dataset = TextOnlyDataset(val_posts, labels, dictionary)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=args.shuffle,
                                                        num_workers=num_workers,
                                                        collate_fn=collate_fn_pad_text_only,
                                                        **kwargs)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    num_workers=num_workers,
                                                    collate_fn=collate_fn_pad_text_only,
                                                    **kwargs)

    # Set up the model
    model = Model(vocab_size=dictionary.size()).to(device)

    # Set up an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma) # decay by 0.1 every 15 epochs

    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup tensorboard
    if args.tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=args.log_dir + "/" + args.name, flush_secs=1)
    else:
        writer = None

    # Training loop
    if args.classification == 'intent':
        keys = ['intent']
    elif args.classification == 'semiotic':
        keys = ['semiotic']
    elif args.classification == 'contextual':
        keys = ['contextual']
    elif args.classification == 'all':
        keys = ['intent', 'semiotic', 'contextual']
    else:
        raise ValueError("args.classification doesn't exist.")
    best_auc_ovr = 0.0
    best_auc_ovo = 0.0
    best_acc = 0.0
    best_model = None
    best_optimizer = None
    best_scheduler = None
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
            label = dict.fromkeys(keys, np.array([], dtype=np.int))
            pred = dict.fromkeys(keys, None)
            for _, batch in pbar:
                if 'caption' not in batch:
                    caption_data = None
                else:
                    caption_data = batch['caption'].to(device)
                if 'image' not in batch:
                    image_data = None
                else:
                    image_data = batch['image'].to(device)
                label_batch = {}
                for key in keys:
                    label_batch[key] = batch['label'][key].to(device)
                    
                if mode == "train":
                    model.zero_grad()

                pred_batch = model(image_data, caption_data)
                
                for key in keys:
                    label[key] = np.concatenate((label[key], batch['label'][key].cpu().numpy()))
                    x = pred_batch[key].detach().cpu().numpy()
                    x_max = np.max(x, axis=1).reshape(-1, 1)
                    z = np.exp(x - x_max)
                    prediction_scores = z / np.sum(z, axis=1).reshape(-1, 1)
                    if pred[key] is not None:
                        pred[key] = np.vstack((pred[key], prediction_scores))
                    else:
                        pred[key] = prediction_scores
                       
                loss_batch = {}
                loss = None
                for key in keys:
                    loss_batch[key] = loss_fn(pred_batch[key], label_batch[key])
                    if loss is None:
                        loss = loss_batch[key]
                    else:
                        loss += loss_bath[key] 

                total_loss += loss.item()

                if mode == "train":
                    loss.backward()
                    optimizer.step()

            # Terminate the progress bar
            pbar.close()
            
            # Update lr scheduler
            if mode == "train":
                scheduler.step()

            for key in keys:
                auc_score_ovr = roc_auc_score(label[key], pred[key], multi_class='ovr') # pylint: disable-all
                auc_score_ovo = roc_auc_score(label[key], pred[key], multi_class='ovo') # pylint: disable-all
                accuracy = accuracy_score(label[key], np.argmax(pred[key], axis=1))
                print("[{} - {}] [AUC-OVR={:.3f}, AUC-OVO={:.3f}, ACC={:.3f}]".format(mode, key, auc_score_ovr, auc_score_ovo, accuracy))
                
                if mode == "eval":
                    best_auc_ovr = max(best_auc_ovr, auc_score_ovr)
                    best_auc_ovo = max(best_auc_ovo, auc_score_ovo)
                    best_acc = max(best_acc, accuracy)
                    best_model = model
                    best_optimizer = optimizer
                    best_scheduler = scheduler
                
                if writer:
                    writer.add_scalar('AUC-OVR/{}-{}'.format(mode, key), auc_score_ovr, epoch)
                    writer.add_scalar('AUC-OVO/{}-{}'.format(mode, key), auc_score_ovo, epoch)
                    writer.add_scalar('ACC/{}-{}'.format(mode, key), accuracy, epoch)
                    writer.flush()

            if writer:
                writer.add_scalar('Loss/{}'.format(mode), total_loss, epoch)
                writer.flush()

            print("[{}] Epoch {}: Loss = {}".format(mode, epoch, total_loss))

    hparam_dict = {
        'train_split': args.train_metadata,
        'val_split': args.val_metadata,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': args.shuffle,
        'lr_scheduler_gamma': args.lr_scheduler_gamma,
        'lr_scheduler_step_size': args.lr_scheduler_step_size,
    }
    metric_dict = {
        'AUC-OVR': best_auc_ovr,
        'AUC-OVO': best_auc_ovo,
        'ACC': best_acc
    }

    if writer:
        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        writer.flush()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    torch.save({
        'hparam_dict': hparam_dict,
        'metric_dict': metric_dict,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, Path(args.output_dir) / '{}.pt'.format(args.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Instagram Intent Classifier")
    parser.add_argument('train_metadata', type=str, help="Path to training metadata")
    parser.add_argument('val_metadata', type=str, help="Path to validation metadata")
    parser.add_argument('label_intent', type=str, help="Path to label for intent")
    parser.add_argument('label_semiotic', type=str, help="Path to label for semiotic")
    parser.add_argument('label_contextual', type=str, help="Path to label for contextual")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_only', dest='type', action='store_const', const='image_only')
    group.add_argument('--image_text', dest='type', action='store_const', const='image_text')
    group.add_argument('--text_only', dest='type', action='store_const', const='text_only')
    parser.add_argument('--classification', type=str, required=True, help="Type of loss function to optimize for")
    parser.add_argument('--name', type=str, default='reproduce_experiment', help="Name of the experiment")
    parser.add_argument('--epochs', type=int, default=70, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate (default is 5e-4 according to the paper)")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Whether to shuffle data loader")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--device', type=str, default='cpu', help="Pytorch device")
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.9, help="Decay factor for learning rate scheduler")
    parser.add_argument('--lr_scheduler_step_size', type=int, default=15, help="Step size for learning rate scheduler")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs', help="Log directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output directory")
    print(parser.parse_args())
    main(parser.parse_args())

"""
python main.py --image_only \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --name intent_split0 \
    --log_dir ./logs \
    --device cuda:1 \
    --tensorboard
""" # pylint: disable=pointless-string-statement
