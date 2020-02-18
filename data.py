from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer

UNK_TOKEN = '<UNK>'

import tensorflow_hub as hub
import tensorflow as tf

#print("Loading elmo")
#elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

# def elmo_vectors(x):
#     embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.tables_initializer())
#         # return average of ELMo features
#     return sess.run(tf.reduce_mean(embeddings,1))

class Dictionary:
    def __init__(self, tokenizer_method: str = "TreebankWordTokenizer"):
        self.token2idx = {}
        self.tokenizer = None

        if tokenizer_method == "TreebankWordTokenizer":
            self.tokenizer = TreebankWordTokenizer()
            #self.tokenizer = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        else:
            raise NotImplementedError("tokenizer_method {} doesn't exist".format(tokenizer_method))

        self.add_token(UNK_TOKEN) # Add UNK token

    def build_dictionary_from_captions(self, captions: List[str]):
        for caption in captions:
            tokens = self.tokenizer.tokenize(caption)
            #tokens = tokenizer(caption, signature="default", as_dict=True)["elmo"]
            for token in tokens:
                self.add_token(token)

    def size(self) -> int:
        return len(self.token2idx)

    def add_token(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)

    def lookup_token(self, token: str) -> int:
        if token in self.token2idx:
            return self.token2idx[token]
        return self.token2idx[UNK_TOKEN]

def parse_post(post: Dict[str, Any],
               image_retriever: str = "pretrained",
               image_basedir: Optional[str] = "documentIntent_emnlp19/resnet18_feat") -> Dict[str, Any]:
    """
    Parse an input post. This function will read an image and store it in `numpy.ndarray` format.

    Args:
        post (Dict[str, Any]) - post
        image_retriever (str) - method for obtaining an image
        image_basedir (Optional[str]) - base directory for obtaining an image (used when `image_retriever = 'pretrained'`)
    """
    id = post['id']
    label_intent = post['intent']
    label_semiotic = post['semiotic']
    label_contextual = post['contextual']
    caption = post['caption']

    if image_retriever == "url":
        image = post['url']
        raise NotImplementedError("Currently cannot download an image from {}".format(image))
    elif image_retriever == "pretrained":
        # image_path = Path(image_basedir) / "{}.npy".format(id)
        image_path = 'resnet18_feat/{}.npy'.format(id)
        image = np.load(image_path)
    elif image_retriever == "ignored":
        image = None
    else:
        raise NotImplementedError("image_retriever method doesn't exist")

    output_dict = {
        'id': id,
        'label': {
            'intent': label_intent,
            'semiotic': label_semiotic,
            'contextual': label_contextual,
        },
        'caption': caption,
        'image': image,
    }

    return output_dict

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            #print("raw caption: ?", self.posts[post_id]['caption']) #this is still raw text
            #tokenized_captions = self.dictionary.tokenizer.tokenize(self.posts[post_id]['caption'])
            
            #self.posts[post_id]['caption'] = list(map(self.dictionary.lookup_token, tokenized_captions))
            # try:
            #     if len([self.posts[post_id]['caption']]) == 0:
            #         self.posts[post_id]['caption'] = "na"
            #     self.posts[post_id]['caption'] = elmo([self.posts[post_id]['caption']])
            # except:
            #     print ("Exceptions: ",self.posts[post_id]['caption'])
            #print("raw caption after the list map", self.posts[post_id]['caption']) #this is the vectorized word
    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        # output['caption'] = torch.LongTensor(output['caption'])#.flatten())
        output['image'] = torch.from_numpy(output['image']) # pylint: disable=undefined-variable, no-member
        return output

class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]]):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]
                
    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['image'] = torch.from_numpy(output['image']) # pylint: disable=undefined-variable, no-member
        return output

class TextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            tokenized_captions = self.dictionary.tokenizer.tokenize(self.posts[post_id]['caption'])
            self.posts[post_id]['caption'] = list(map(self.dictionary.lookup_token, tokenized_captions))

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['caption'] = torch.LongTensor(output['caption'])
        return output
