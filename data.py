from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np
import torch

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
    caption = post['orig_caption']

    if image_retriever == "url":
        image = post['url']
        raise NotImplementedError("Currently cannot download an image from {}".format(image))
    elif image_retriever == "pretrained":
        image_path = Path(image_basedir) / "{}.npy".format(id)
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
    def __init__(self, posts: List[Dict[str, Any]]):
        self.posts = list(lambda post: parse_post(post, image_retriever="pretrained"), posts)
    
    def __len__(self) -> int:
        return len(self.posts)
    
    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['image'] = torch.from_numpy(output['image']) # pylint: disable=undefined-variable, no-member
        return output

class ImageOnlyDataset(torch.utils.data.Dataset):
    pass

class TextOnlyDataset(torch.utils.data.Dataset):
    pass