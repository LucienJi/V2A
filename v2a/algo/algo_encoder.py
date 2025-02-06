from collections import OrderedDict
import numpy as np 
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
from .algo_utils import build_transform

class EncoderAlgo(Algo):
    def __init__(self,algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device):
        
        super().__init__(algo_config,obs_config,global_config,obs_key_shapes,ac_dim,device)

        self.obs_transform = build_transform(self.global_config.algo.encoder.obs_augmentation)

        # cluster loss 
        self.swav_loss = SwaVLoss()
        self.cluster_loss_coef = self.global_config.algo.encoder.cluster.cluster_coef
        

        # temporal contrastive loss 
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.positive_window = self.global_config.algo.encoder.tcl.positive_window
        self.negative_window = self.global_config.algo.encoder.tcl.negative_window
        self.num_negative_samples = self.global_config.algo.encoder.tcl.num_negatives
        self.num_positive_samples = 1
        self.tcl_loss_coef = self.global_config.algo.encoder.tcl.tcl_coef

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

        robot_state = []
        for name in self.obs_config.encoder.robot_state:
            robot_state.append(batch['obs'][name])
        robot_state = torch.cat(robot_state, dim=-1)

        robot_view = []
        for name in self.obs_config.encoder.rgb:
            print("name", name)
            robot_view.append(batch['obs'][name].permute(0, 1, 4, 2, 3))
        robot_view = torch.cat(robot_view, dim=2)
        print("robot_view", robot_view.shape)

        robot_state = robot_state.to(self.device)
        robot_view = robot_view.to(self.device) 

        batch['obs'] = robot_view
        batch['robot_state'] = robot_state
        B,SEQ,C,H,W = robot_view.shape
        augmented = self.obs_transform(batch['obs'].reshape(-1,C,H,W))
        batch['aug_obs'] = augmented.reshape(B,SEQ,C,H,W)

        return batch 
    
    def train_on_batch(self, batch, epoch, validate=False):
        """
        We do need some data agumentation methods before train on the batch

        Train on batch should return the loss and the metrics
        """
        obs = batch['obs'] # shape (batch, seq_len, C, H, W)    
        robot_state = batch['robot_state'] # shape (batch, seq_len, robot_state_dim)
        aug_obs = batch['aug_obs'] # shape (batch, seq_len, C, H, W)

        embedding1 = self.nets['encoder'](obs, robot_state)
        embedding2 = self.nets['encoder'](aug_obs, robot_state)

        cluster_loss = self.cluster_loss(embedding1, embedding2)
        temporal_contrastive_loss = self.temporal_contrastive_loss(embedding1) 

        loss = cluster_loss + self.tcl_loss_coef * temporal_contrastive_loss
        info = {
            'losses': loss,
            'cluster_loss': cluster_loss.item(),
            'temporal_contrastive_loss': temporal_contrastive_loss.item(),
        }
        return info


        

    def cluster_loss(self, feature1, feature2):
        
        #! TODO whether we need projection head ? 
        ## compute the swav loss
        self.nets['encoder'].prototypes.normlize()

        normed_feature1 = nn.functional.normalize(feature1, dim=1, p=2)
        normed_feature2 = nn.functional.normalize(feature2, dim=1, p=2)
        
        p_f1 = self.nets['encoder'].prototypes(normed_feature1)
        p_f2 = self.nets['encoder'].prototypes(normed_feature2)
        #! TODO we could use memory ? 
        loss = self.swav_loss.forward(
            p_f1,
            p_f2,
        )

        return loss 
    
    def temporal_contrastive_loss(self,embeddings):
    
        # Batch version 
        total_segments = embeddings.shape[0]
        # Precompute candidate indices for each segment:
        pos_candidates = []
        neg_candidates = []

        for idx in range(total_segments):
            # Create list of candidate indices for positives:
            pos_range = list(range(max(idx - self.positive_window, 0), idx)) + \
                        list(range(idx + 1, min(idx + self.positive_window + 1, total_segments)))
            # Randomly choose one candidate per segment:
            pos_idx = np.random.choice(pos_range, 1)[0]
            pos_candidates.append(pos_idx)
            
            # Create list of candidate indices for negatives:
            neg_range = list(range(0, max(idx - self.negative_window, 0))) + \
                        list(range(min(idx + self.negative_window, total_segments), total_segments))
            neg_idxs = np.random.choice(neg_range, self.num_negative_samples, replace=True)
            neg_candidates.append(neg_idxs)

        # Convert to tensors
        pos_indices = torch.tensor(pos_candidates)             # shape: (total_segments,)
        neg_indices = torch.tensor(neg_candidates)               # shape: (total_segments, num_negatives)

        # Stack positive and negative indices for each segment:
        # This gives a tensor of shape (total_segments, 1 + num_negatives)
        all_indices = torch.cat([pos_indices.unsqueeze(1), neg_indices], dim=1)
        # Gather features in one go:
        # all_features: shape (total_segments, 1 + num_negatives, F_DIM)
        all_features = embeddings[all_indices]

        anchors = embeddings 
        others = all_features 
        logits = torch.bmm(anchors, others.transpose(1, 2)).squeeze(1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        return loss 
    


        


       