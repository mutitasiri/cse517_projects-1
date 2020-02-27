from typing import Optional, Tuple
import torch
import torchvision.models as models

class word2vec(torch.nn.Module):
    """
    word2vec class
    """
    def __init__(self, vocab_size: int, embedding_size: int = 300):
        """
        Constructor

        Args:
            vocab_size (int) - Size of the dictionary
            embedding_size (int) - Dimension of the token embedding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Forward pass

        Args:
            x (torch.Tensor) - Input tensor
        """
        x = self.embedding_layer(x)
        return x

class CBOW(torch.nn.Module):
    """
    CBOW class
    """
    def __init__(self, vocab_size: int, embedding_size: int = 300, context_size: int = 2):
        """
        Constructor

        Args:
            vocab_size (int) - Size of the dictionary
            embedding_size (int) - Dimension of the token embedding
            context_size (int) - Size of surrounding contexts of the word
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Forward pass

        Args:
            x (torch.Tensor) - Input tensor
        """
        context_size = self.context_size
        nbatch = x.shape[0]
        lseq = x.shape[1]
        
        out = torch.LongTensor().to(x.device)
        nbatch = x.shape[0]
        lseq = x.shape[1]
        for i in range(nbatch):
            temp_tensor = x[i]
            context = torch.LongTensor().to(x.device)
            for j in range(lseq):
                context_before = torch.LongTensor([temp_tensor[j-k] if j-k >= 0 else torch.LongTensor([0]) for k in range(1, context_size+1)]).to(x.device)
                context_after = torch.LongTensor([temp_tensor[j+k] if j+k < lseq else torch.LongTensor([0]) for k in range(1, context_size+1)]).to(x.device)
                temp = torch.cat((context_before, context_after))
                context = torch.cat((context, temp), dim=0)
            context = context.view(lseq, context_size*2)
            out = torch.cat((out, context))
        out = out.view(nbatch, lseq, context_size*2)
        
        out = torch.sum(self.embedding_layer(out), dim=1)
        return out    
    
class Model(torch.nn.Module):
    """
    Model class
    """
    def __init__(self,
                 vocab_size: int,
                 token_embedding_size: int = 300,
                 context_size: int = 2,
                 image_network_type: str = "identity",
                 caption_network_type: str = "word2vec",
                 joint_embedding_size: int = 128,
                 intent_dims: int = 7,
                 semiotic_dims: int = 3,
                 contextual_dims: int = 3):
        """
        Constructor

        Args:
            vocab_size (int) - Size of the dictionary
            token_embedding_size (int) - Dimension of the token embedding
            context_size (int) - Size of surrounding contexts of the word
            image_network_type
            caption_network_type
            joint_embedding_size
            intent_dims
            semiotic_dims
            contextual_dims
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_size = token_embedding_size
        self.context_size = context_size
        self.image_network_type = image_network_type
        self.caption_network_type = caption_network_type
        self.joint_embedding_size = joint_embedding_size
        self.intent_dims = intent_dims
        self.semiotic_dims = semiotic_dims
        self.contextual_dims = contextual_dims

        if self.image_network_type == "resnet18":
            self.image_network = models.resnet18(pretrained=True)
        elif self.image_network_type == "identity":
            self.image_network = torch.nn.Identity()
        else:
            raise NotImplementedError("image_network_type = {} not available".format(self.image_network_type))

        if self.caption_network_type == "word2vec":
            self.caption_network = word2vec(self.vocab_size, self.token_embedding_size)
        elif self.caption_network_type == "CBOW":
            self.caption_network = CBOW(self.vocab_size, self.token_embedding_size, self.context_size)
        elif self.caption_network_type == "ELMo":
            raise NotImplementedError
        else:
            raise NotImplementedError("caption_network_type = {} not available".format(self.caption_network_type))

        self.image_joint_embedding_layer = torch.nn.Linear(512, self.joint_embedding_size)
        self.caption_hidden_layer = torch.nn.Linear(2*self.token_embedding_size, self.token_embedding_size)
        self.caption_GRU = torch.nn.GRU(self.token_embedding_size, self.token_embedding_size, 1)

        self.caption_joint_embedding_layer = torch.nn.Linear(self.token_embedding_size, self.joint_embedding_size)

        self.intent_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.intent_dims)
        self.semiotic_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.semiotic_dims)
        self.contextual_prediction_layer = torch.nn.Linear(self.joint_embedding_size, self.contextual_dims)

    def forward(self, x_img: Optional[torch.Tensor], x_caption: Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # pylint: disable=arguments-differ
        """
        Forward pass

        Args:
            x_img (Optional[torch.Tensor]) - Input image tensor
            x_caption (Optional[torch.Tensor]) - Input caption tensor
        """
        if x_img is not None:
            # Get an image embedding
            x_img = self.image_network(x_img) # (batch_size, img_embedding_dim)
            
            # Get an image-text joint embedding
            x_img_joint_embedding = self.image_joint_embedding_layer(x_img)
        else:
            x_img_joint_embedding = torch.zeros((x_caption.shape[0], self.joint_embedding_size), device=x_caption.device)

        if x_caption is not None:
            # Get a text embedding
            x_caption = self.caption_network(x_caption) # (batch_size, caption_length, word_embedding_dim)

            # Iterate over the caption
            caption_length = x_caption.size(1)
            hidden = torch.zeros(1, x_caption.size(0), self.token_embedding_size).to(x_caption.device)
            out = torch.zeros(1, x_caption.size(0), self.token_embedding_size).to(x_caption.device)
            for t in range(caption_length):
                # x_caption_combined = torch.cat((hidden, x_caption[:, t, :]), dim=1)
                _, hidden = self.caption_GRU(torch.unsqueeze(x_caption[:, t, :], 0), hidden)
            hidden = torch.squeeze(hidden)

            x_caption_joint_embedding = self.caption_joint_embedding_layer(hidden)
        else:
            x_caption_joint_embedding = torch.zeros((x_img.shape[0], self.joint_embedding_size), device=x_img.device)

        # Fusion
        x_fusion = x_img_joint_embedding + x_caption_joint_embedding
        
        # Prediction
        pred_intent = self.intent_prediction_layer(x_fusion)
        pred_semiotic = self.semiotic_prediction_layer(x_fusion)
        pred_contextual = self.contextual_prediction_layer(x_fusion)

        output = {
            'intent': pred_intent,
            'semiotic': pred_semiotic,
            'contextual': pred_contextual,
        }
        return output

if __name__ == '__main__':
    model = Model(vocab_size=100)
