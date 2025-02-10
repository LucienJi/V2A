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
from ..models.common import MLP, CNN, StateProjector
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

        # state alignment loss
        self.dynamic_contrastive_loss_coef = self.global_config.algo.encoder.mcr.mcr_coef

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

        self.nets['state_projector'] = StateProjector(
            input_dim = robot_state_input_dim,
            n_seq = self.algo_config.encoder.n_seq,
            out_size = self.algo_config.encoder.embedding_dim,
        )


        self.nets = self.nets.float().to(self.device)



    def process_batch_for_training(self, batch):
        robot_state = []
        for name in self.obs_config.encoder.robot_state:
            robot_state.append(batch['obs'][name])
        robot_state = torch.cat(robot_state, dim=-1)

        robot_view = []
        for name in self.obs_config.encoder.rgb:
            robot_view.append(batch['obs'][name].permute(0, 1, 4, 2, 3))
        robot_view = torch.cat(robot_view, dim=2)


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

        s_embedding = self.nets['state_projector'](robot_state)

        cluster_loss = self.cluster_loss(embedding1, embedding2)
        temporal_contrastive_loss, pos_indices, neg_indices = self.temporal_contrastive_loss(embedding1) 
        s_loss, v_s_loss, s_v_loss,pos_indices, neg_indices = self.dynamic_contrastive_loss(embedding1, s_embedding, pos_indices, neg_indices)

        loss = self.cluster_loss_coef * cluster_loss \
                + self.tcl_loss_coef * temporal_contrastive_loss \
                + self.dynamic_contrastive_loss_coef * (s_loss + v_s_loss + s_v_loss)
                
        info = {
            'losses': loss,
            'cluster_loss': cluster_loss.item(),
            'temporal_contrastive_loss': temporal_contrastive_loss.item(),
            "state_alignment_loss": s_loss.item(),
            "visual_state_alignment_loss": v_s_loss.item(),
            "state_visual_alignment_loss": s_v_loss.item(),
        }
        return info


        

    def cluster_loss(self, feature1, feature2):
        
        #! TODO whether we need projection head ? 
        ## compute the swav loss
        self.nets['encoder'].prototypes.normalize()  ## normalize the prototypes

        normed_feature1 = nn.functional.normalize(feature1, dim=1, p=2) ## normalize the features
        normed_feature2 = nn.functional.normalize(feature2, dim=1, p=2)
        
        p_f1 = self.nets['encoder'].prototypes(normed_feature1) ## similarity matrix. shape (batch, num_prototypes)
        p_f2 = self.nets['encoder'].prototypes(normed_feature2)
        #! TODO we could use memory ? 
        loss = self.swav_loss.forward(
            p_f1,
            p_f2,
        )

        return loss 
    
    def temporal_contrastive_loss(self,embeddings,pos_indices = None, neg_indices = None):
    
        # Batch version 
        total_segments = embeddings.shape[0]
        # Precompute candidate indices for each segment:
        if pos_indices is None or neg_indices is None:
            pos_candidates = []
            neg_candidates = []
            positive_window = self.positive_window if self.positive_window > total_segments//2 else total_segments//2
            negative_window = self.negative_window if self.negative_window < total_segments//2 else total_segments//2

            for idx in range(total_segments):
                # Create list of candidate indices for positives:
                pos_range = list(range(max(idx - positive_window, 0), idx)) + \
                            list(range(idx + 1, min(idx + positive_window + 1, total_segments)))
                # Randomly choose one candidate per segment:
                pos_idx = np.random.choice(pos_range, 1)[0]
                pos_candidates.append(pos_idx)
                
                # Create list of candidate indices for negatives:
                neg_range = list(range(0, max(idx - negative_window, 0))) + \
                            list(range(min(idx + negative_window, total_segments), total_segments))
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

        anchors = embeddings.unsqueeze(1)
        others = all_features 
        logits = torch.bmm(anchors, others.transpose(1, 2)).squeeze(1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        return loss, pos_indices, neg_indices
    
    def dynamic_contrastive_loss(self, v_embedding, s_embedding,pos_indices = None, neg_indices = None):
        """
        v_embedding: visual embedding
        s_embedding: state embedding 

        TWo lossess: 
        1. contrastive for state
        2. contrastive between visual and state
        """
        total_segments = v_embedding.shape[0]
        if pos_indices is None or neg_indices is None:
            pos_candidates = []
            neg_candidates = []
            positive_window = self.positive_window if self.positive_window > total_segments//2 else total_segments//2
            negative_window = self.negative_window if self.negative_window < total_segments//2 else total_segments//2

            for idx in range(total_segments):
                # Create list of candidate indices for positives:
                pos_range = list(range(max(idx - positive_window, 0), idx)) + \
                            list(range(idx + 1, min(idx + positive_window + 1, total_segments)))
                # Randomly choose one candidate per segment:
                pos_idx = np.random.choice(pos_range, 1)[0]
                pos_candidates.append(pos_idx)
                
                # Create list of candidate indices for negatives:
                neg_range = list(range(0, max(idx - negative_window, 0))) + \
                            list(range(min(idx + negative_window, total_segments), total_segments))
                neg_idxs = np.random.choice(neg_range, self.num_negative_samples, replace=True)
                neg_candidates.append(neg_idxs)

            # Convert to tensors
            pos_indices = torch.tensor(pos_candidates)             # shape: (total_segments,)
            neg_indices = torch.tensor(neg_candidates)               # shape: (total_segments, num_negatives)

        # Stack positive and negative indices for each segment:
        # This gives a tensor of shape (total_segments, 1 + num_negatives)
        all_indices = torch.cat([pos_indices.unsqueeze(1), neg_indices], dim=1)

        # For contrastive between states 
        s_anchors = s_embedding.unsqueeze(1)
        s_others = s_embedding[all_indices]
        s_logits = torch.bmm(s_anchors, s_others.transpose(1, 2)).squeeze(1)
        s_labels = torch.zeros(s_logits.size(0), dtype=torch.long, device=s_logits.device)
        s_loss = self.cross_entropy_loss(s_logits, s_labels)

        # For contrastive  visual -> state
        v_anchors = v_embedding.unsqueeze(1)
        v_s_logits = torch.bmm(v_anchors, s_others.transpose(1, 2)).squeeze(1)
        v_s_labels = torch.zeros(v_s_logits.size(0), dtype=torch.long, device=v_s_logits.device)
        v_s_loss = self.cross_entropy_loss(v_s_logits, v_s_labels)
 
        # For contrastive state -> visual
        v_others = v_embedding[all_indices]
        s_v_logits = torch.bmm(s_anchors, v_others.transpose(1, 2)).squeeze(1)
        s_v_labels = torch.zeros(s_v_logits.size(0), dtype=torch.long, device=s_v_logits.device)
        s_v_loss = self.cross_entropy_loss(s_v_logits, s_v_labels)

        return s_loss, v_s_loss, s_v_loss,pos_indices, neg_indices



        


       