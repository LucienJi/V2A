from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn
import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from ..models.encoder.encoder import VisualMotionEncoder_v1
from ..models.encoder.transformer import TransformerEncoder, PositionalEncoding
from ..models.common import MLP, CNN
from .algo import register_algo_factory_func, Algo
from .cluster import SwaVLoss

class EncoderAlgo(Algo):
    def __init__(self,algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device):
        
        super().__init__(algo_config,obs_config,global_config,obs_key_shapes,ac_dim,device)

        self.swav_loss = SwaVLoss()

    def _create_networks(self):
        
        ## MLP
        robot_state_input_dim = 0
        for obs_name in self.obs_config.encoder.robot_state:
            robot_state_input_dim += self.obs_key_shapes.get(obs_name,[0])[0]
        robot_mlp = MLP(
            input_dim = robot_state_input_dim, 
            out_size= self.algo_config.encoder.embedding_dim
        )

        ## CNN
        assert len(self.obs_config.encoder.rgb) == 1 , print(len(self.obs_config.encoder.rgb))
        input_shape = self.obs_key_shapes[self.obs_config.encoder.rgb[0]]
        img_encoder = CNN(
            out_size= self.algo_config.encoder.embedding_dim,
            input_shape = input_shape,
        )
        ## Transformer
        transformer = TransformerEncoder(
            input_dim = self.algo_config.encoder.embedding_dim,
            query_dim = self.algo_config.encoder.embedding_dim,
            heads = self.algo_config.encoder.heads,
            n_layer = self.algo_config.encoder.n_layer,
            rep_dim = self.algo_config.encoder.embedding_dim,
            pos_encoder = PositionalEncoding(size = self.algo_config.encoder.embedding_dim, 
                                             max_len = 100),

        )

        self.nets['encoder'] = VisualMotionEncoder_v1(
            img_encoder = img_encoder,
            state_encoder = robot_mlp,
            video_encoder = transformer,
            num_prototypes = self.algo_config.encoder.num_prototypes,
        )


    def process_batch_for_training(self, batch):
        return batch 
    
    def train_on_batch(self, batch, epoch, validate=False):
        pass

    def cluster_loss(self, batch):
        """
        Implement the SwAV loss function.
        From Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
        Args:
            batch (dict): batch of data
        """
        obs = batch['obs'] ## rgb video # (batch, seq_len, C, H, W)
        state = batch['robot_state'] ## robot state

        ## apply augmentation to the obs 
        obs_1 = self.transform_obs(obs)
        obs_2 = self.transform_obs(obs)

        feature1 = self.nets['encoder'].forward(obs_1, state)
        feature2 = self.nets['encoder'].forward(obs_2, state)

        #! TODO whether we need projection head ? 
        ## compute the swav loss
        normed_feature1 = nn.functional.normalize(feature1, dim=1, p=2)
        normed_feature2 = nn.functional.normalize(feature2, dim=1, p=2)
        
        p_f1 = self.nets['encoder'].prototypes(normed_feature1)
        p_f2 = self.nets['encoder'].prototypes(normed_feature2)
        #! TODO we could use memory ? 
        loss = self.swav_loss.forward(
            p_f1,
            p_f1,
        )

        return loss 
    
    def temporal_contrastive_loss(self,batch):
        obs = batch['obs'] 
        state = batch['robot_state']

        pos_obs = batch['pos_obs']
        pos_state = batch['pos_robot_state']
        


        


       