from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn
import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
# from .algo import register_algo_factory_func, Algo
from mimicplay.algo import register_algo_factory_func, PolicyAlgo
from ..models.policy.gmm_policy import GMMActorNetwork

class GMMPoicyAlgo(PolicyAlgo):
    def __init__(self,algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device):
        
        super().__init__(algo_config,obs_config,global_config,obs_key_shapes,ac_dim,device)


    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self.nets["policy"] = GMMActorNetwork(
            image_output_dim = 256,
            state_input_dim = 18,
            state_hidden_dims = [256, 256],
            state_output_dim = 64,
            skill_input_dim = 0,
            skill_output_dim = 128,
            combined_hidden_dims = [256, 256],
            action_dim = 6,
            num_modes = 5,
            min_std = 0.0001,
            std_activation = "softplus",
            use_tanh = False,
            low_noise_eval = True
        )
        self.nets = self.nets.float().to(self.device)


    def process_batch_for_training(self, batch):
        return batch 
    
    def train_on_batch(self, batch, epoch, validate=False):
        pass