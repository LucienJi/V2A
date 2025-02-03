import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..common import MLP, CNN
from .transformer import PositionalEncoding, TransformerEncoder
from collections import OrderedDict
from robomimic.models.obs_core import VisualCore, Randomizer
from copy import deepcopy
from einops import rearrange, repeat, reduce


class Prototypes(nn.Module):
    def __init__(self, embedding_dim, num_prototypes):
        """
        A linear layer that maps an embedding to the space of prototypes.
        
        Args:
            embedding_dim (int): Dimension of the video embedding.
            num_prototypes (int): Number of prototypes (clusters).
        """
        super().__init__()
        self.prototypes = nn.Linear(embedding_dim, num_prototypes, bias=False)
    
    def forward(self, x):
        # x: (batch, embedding_dim)
        # Returns: (batch, num_prototypes)
        return self.prototypes(x)

class VisualMotionEncoder_v1(nn.Module):
    def __init__(self,
                 img_encoder:CNN,
                 state_encoder:MLP,
                 video_encoder:TransformerEncoder,
                 num_prototypes:int,
                 ):
        """
        A visual encoder that processes a sequence of video frames.
        1. input sequence of frames (batch, seq_len, C, H, W) 
        2. input sequence of robot state (batch, seq_len, robot_state_dim)

        3. Apply CNN to each frame to get a sequence of frame embeddings (batch, seq_len, embedding_dim)
        4. Apply MLP to robot state to get a sequence of robot state embeddings (batch, seq_len, embedding_dim)
        5. Concatenate frame embeddings and robot state embeddings along the feature dimension (batch, seq_len, 2*embedding_dim)
        6. Two different method to get the final embedding for the clip of video in form of (batch, embedding_dim) 
            1. Mean pooling over the sequence length
            2. Use the cls token from the transformer encoder
        """
        super(VisualMotionEncoder_v1, self).__init__()
        
        self.img_encoder = img_encoder
        self.state_encoder = state_encoder 
        self.video_encoder = video_encoder 
    
    def forward(self,image, state):
        """
        Args:
            image (torch.Tensor): Input sequence of video frames. Shape: (batch, seq_len, C, H, W)
            state (torch.Tensor): Input sequence of robot state. Shape: (batch, seq_len, state_dim)
        Returns:
            torch.Tensor: Final embedding for the clip of video. Shape: (batch, embedding_dim)
        """
        img_encoding = self.img_representation(image)
        state_encoding = self.state_representation(state)
        x = torch.cat([img_encoding, state_encoding], dim=-1)
        x = self.video_encoder(x)
        return x
    

    def img_representation(self,image):
        """
        Args:
            image (torch.Tensor): Input sequence of video frames. Shape: (batch, seq_len, C, H, W)
        Returns:
            torch.Tensor: Final embedding for the clip of video. Shape: (batch, embedding_dim)
        """
        batch, seq_len, C, H, W = image.shape
        image = rearrange(image, 'b t c h w  -> (b t) c h w')
        image_encoding = self.img_encoder(image)
        image_encoding = rearrange(image_encoding, '(b t) c h w -> b t c h w', b=batch)
        return image_encoding

    def state_representation(self,state):
        """
        Args:
            state (torch.Tensor): Input sequence of robot state. Shape: (batch, seq_len, state_dim)
        Returns:
            torch.Tensor: Final embedding for the clip of video. Shape: (batch, embedding_dim)
        """
        state_encoding = self.state_encoder(state)
        return state_encoding
    
        