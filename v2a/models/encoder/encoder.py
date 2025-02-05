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
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor
from v2a.models import utils



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
        self.num_prototypes = num_prototypes
        if self.num_prototypes > 0:
            self.prototypes = SwaVPrototypes(input_dim=video_encoder.rep_dim, n_prototypes=num_prototypes)
        else:
            self.prototypes = None
    
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
    

class SwaVPrototypes(nn.Module):
    """Multihead Prototypes used for SwaV.

    Each output feature is assigned to a prototype, SwaV solves the swapped
    prediction problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.

    Attributes:
        input_dim:
            The input dimension of the head.
        n_prototypes:
            Number of prototypes.
        n_steps_frozen_prototypes:
            Number of steps during which we keep the prototypes fixed.

    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_prototypes: Union[List[int], int] = 512,
        n_steps_frozen_prototypes: int = 0,
    ):
        """Intializes the SwaVPrototypes module with the specified parameters"""
        super(SwaVPrototypes, self).__init__()

        # Default to a list of 1 if n_prototypes is an int.
        self.n_prototypes = (
            n_prototypes if isinstance(n_prototypes, list) else [n_prototypes]
        )
        self._is_single_prototype = True if isinstance(n_prototypes, int) else False
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, prototypes) for prototypes in self.n_prototypes]
        )
        self.n_steps_frozen_prototypes = n_steps_frozen_prototypes

    def forward(
        self, x: Tensor, step: Optional[int] = None
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass of the SwaVPrototypes module.

        Args:
            x:
                Input tensor.
            step:
                Current training step.

        Returns:
            The logits after passing through the prototype heads. Returns a single tensor
            if there's one prototype head, otherwise returns a list of tensors.
        """
        self._freeze_prototypes_if_required(step)
        out = []
        for layer in self.heads:
            out.append(layer(x))
        return out[0] if self._is_single_prototype else out

    def normalize(self) -> None:
        """Normalizes the prototypes so that they are on the unit sphere."""
        for layer in self.heads:
            utils.normalize_weight(layer.weight)

    def _freeze_prototypes_if_required(self, step: Optional[int] = None) -> None:
        """Freezes the prototypes if the specified number of steps has been reached."""
        if self.n_steps_frozen_prototypes > 0:
            if step is None:
                raise ValueError(
                    "`n_steps_frozen_prototypes` is greater than 0, please"
                    " provide the `step` argument to the `forward()` method."
                )
            self.requires_grad_(step >= self.n_steps_frozen_prototypes)