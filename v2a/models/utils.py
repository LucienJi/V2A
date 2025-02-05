import math
import random
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import Identity, Module, Sequential, functional, init
from torch.nn.modules import CrossMapLRN2d, GroupNorm, LayerNorm, LocalResponseNorm
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.parameter import Parameter
from torchvision.ops import StochasticDepth

@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension."""
    weight.div_(torch.norm(weight, dim=dim, keepdim=keepdim))