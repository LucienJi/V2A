#!/usr/bin/env python
"""
Clean ACT Policy Implementation with Explicit Arguments

This implementation produces a fixed-length chunk of actions from a single image and robot state.
It uses:
  - A pretrained ResNet-18 for image features.
  - An MLP for the robot state.
  - A VAE branch (active during training) to capture multimodality.
  - A DETR-like transformer (encoder and decoder) with sinusoidal positional embeddings.
  - A temporal ensembler to deterministically select one action from the chunk.
"""

import math
import numpy as np
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn as nn
from torch import Tensor 
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

# ---------------- Utility Functions ----------------

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> torch.Tensor:
    """Generates a sinusoidal positional embedding table of shape (num_positions, dimension)."""
    def get_angle_vec(pos):
        return [pos / np.power(10000, 2 * (i // 2) / dimension) for i in range(dimension)]
    sinusoid_table = np.array([get_angle_vec(pos) for pos in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()

def get_activation_fn(activation: str) -> callable:
    """Returns an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation: {activation}")

# ---------------- 2D Sinusoidal Positional Embedding ----------------

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """
    2D sinusoidal positional embeddings for image feature maps.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.two_pi = 2 * math.pi
        self.eps = 1e-6
        self.temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        y_range = torch.linspace(0, self.two_pi, steps=H, device=x.device).unsqueeze(1).repeat(1, W)
        x_range = torch.linspace(0, self.two_pi, steps=W, device=x.device).unsqueeze(0).repeat(H, 1)
        dim_t = torch.arange(self.dimension, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dimension)
        pos_x = x_range.unsqueeze(-1) / dim_t  # (H, W, dimension)
        pos_y = y_range.unsqueeze(-1) / dim_t  # (H, W, dimension)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return pos

# ---------------- Transformer Components ----------------

class ACTEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dim_feedforward: int, dropout: float,
                 pre_norm: bool, feedforward_activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.activation = get_activation_fn(feedforward_activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        attn_out = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(attn_out)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(ff_out)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class ACTEncoder(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, n_layers: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool, feedforward_activation: str):
        super().__init__()
        self.layers = nn.ModuleList([
            ACTEncoderLayer(dim_model, n_heads, dim_feedforward, dropout, pre_norm, feedforward_activation)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)

class ACTDecoderLayer(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dim_feedforward: int, dropout: float,
                 pre_norm: bool, feedforward_activation: str):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.activation = get_activation_fn(feedforward_activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                decoder_pos_embed: torch.Tensor = None, encoder_pos_embed: torch.Tensor = None) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if decoder_pos_embed is None else x + decoder_pos_embed
        x = skip + self.dropout1(self.self_attn(q, k, value=x)[0])
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        q = x if decoder_pos_embed is None else x + decoder_pos_embed
        k = encoder_out if encoder_pos_embed is None else encoder_out + encoder_pos_embed
        x = skip + self.dropout2(self.multihead_attn(q, k, value=encoder_out)[0])
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(ff_out)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

class ACTDecoder(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, n_layers: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool, feedforward_activation: str):
        super().__init__()
        self.layers = nn.ModuleList([
            ACTDecoderLayer(dim_model, n_heads, dim_feedforward, dropout, pre_norm, feedforward_activation)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                decoder_pos_embed: torch.Tensor = None, encoder_pos_embed: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        return self.norm(x)

# ---------------- Temporal Ensembler ----------------

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

# ---------------- ACT Network ----------------

class ACT(nn.Module):
    """
    The ACT network that produces a chunk of actions given inputs.

    It includes a VAE encoder branch (if use_vae is True) for multimodality,
    an image backbone (pretrained ResNet-18) for image features, and a DETR-like
    transformer (with encoder and decoder) that outputs a fixed-length action chunk.
    """
    def __init__(self,
                 use_vae: bool,
                 robot_state_feature: int,
                 image_features: bool,
                 action_feature: int,
                 skill_feature: int,
                 dim_model: int,
                 n_heads: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 n_vae_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 pre_norm: bool,
                 feedforward_activation: str,
                 chunk_size: int,
                 latent_dim: int):
        super().__init__()
        self.use_vae = use_vae
        self.robot_state_feature = robot_state_feature
        self.image_features = image_features
        self.action_feature = action_feature
        self.skill_feature = skill_feature
        self.dim_model = dim_model
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        # VAE branch for multimodality.
        if self.use_vae:
            self.vae_encoder = ACTEncoder(dim_model, n_heads, n_vae_encoder_layers, dim_feedforward, dropout, pre_norm, feedforward_activation)
            self.vae_cls_embed = nn.Embedding(1, dim_model)
            if self.robot_state_feature is not None:
                self.vae_robot_proj = nn.Linear(robot_state_feature, dim_model)
            self.vae_action_proj = nn.Linear(action_feature, dim_model)
            self.vae_latent_proj = nn.Linear(dim_model, latent_dim * 2)
            self.register_buffer("vae_pos_enc", create_sinusoidal_pos_embedding(num_tokens, dim_model).unsqueeze(0))
            num_tokens = 1 + chunk_size + (1 if self.robot_state_feature is not None else 0)
        # Image backbone.
        if self.image_features:
            backbone = torchvision.models.resnet18(pretrained=True)
            self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
            self.img_proj = nn.Conv2d(backbone.fc.in_features, dim_model, kernel_size=1)
        
         # Input projections for encoder tokens.
        if self.robot_state_feature is not None:
            self.state_proj = nn.Linear(robot_state_feature, dim_model)
        if self.skill_feature is not None:
            self.skill_proj = nn.Linear(skill_feature, dim_model)
        self.latent_proj = nn.Linear(latent_dim, dim_model)

        # Transformer encoder and decoder.
        self.encoder = ACTEncoder(dim_model, n_heads, n_encoder_layers, dim_feedforward, dropout, pre_norm, feedforward_activation)
        self.decoder = ACTDecoder(dim_model, n_heads, n_decoder_layers, dim_feedforward, dropout, pre_norm, feedforward_activation)
       
        # 1D positional embedding for encoder tokens.
        num_1d_tokens = 1 + (1 if self.robot_state_feature is not None else 0) + (1 if self.skill_feature is not None else 0)
        self.enc_1d_pos = nn.Embedding(num_1d_tokens, dim_model)
        # 2D positional embedding for image features.
        if self.image_features:
            self.cam_pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)
    
        # Decoder query embedding.
        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.dec_query_embed = nn.Embedding(chunk_size, dim_model)
        # Final action head.
        self.action_head = nn.Linear(dim_model, action_feature)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        B = batch["robot_state"].shape[0]
        # VAE branch.
        if self.use_vae and "action" in batch:
            cls_embed = einops.repeat(
                self.vae_cls_embed.weight, "1 d -> b 1 d", b=B
            )  # (B, 1, D)
            state_embed = self.vae_robot_proj(batch["robot_state"]).unsqueeze(1) if self.robot_state_feature is not None else None
            action_embed = self.vae_action_proj(batch["action"])  # (B, chunk_size, D)
            if state_embed is not None:
                vae_input = torch.cat([cls_embed, state_embed, action_embed], dim=1)  # shape (B, num_tokens, D)
            else:
                vae_input = torch.cat([cls_embed, action_embed], dim=1)
            pos_enc = self.vae_pos_enc.clone().detach().permute(1, 0, 2)  # (token_count, 1, D)
            cls_token_out = self.vae_encoder(vae_input.permute(1, 0, 2), pos_embed=pos_enc)[0]  # (B, D)
            latent_params = self.vae_latent_proj(cls_token_out)
            mu = latent_params[:, :self.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma = latent_params[:, self.latent_dim:]
            latent_sample = mu + (log_sigma.div(2).exp()) * torch.randn_like(mu)
        else:
            mu = log_sigma = None
            latent_sample = torch.zeros(B, self.latent_dim, device=batch["robot_state"].device)
        # Build transformer encoder tokens.
        tokens = [self.latent_proj(latent_sample)]  # (B, D)
        pos_tokens = [self.enc_1d_pos.weight]        # (num_1d_tokens, D)

        if self.robot_state_feature is not None:
            tokens.append(self.state_proj(batch["robot_state"]))  # (B, D)
        
        if self.skill_feature is not None:
            tokens.append(self.skill_proj(batch["skill"]))
            
        tokens = torch.stack(tokens, dim=0)       # (num_tokens, B, D)
        pos_tokens = torch.stack(pos_tokens, dim=0).unsqueeze(1).expand(-1, B, -1) # (num_tokens, B, D)

        # Process image features.
        if self.image_features:
            img_feats = self.backbone(batch["images"])["feature_map"]  # (B, C, h, w)
            img_feats = self.img_proj(img_feats)  # (B, D, h, w)
            cam_pos = self.cam_pos_embed(img_feats)  # (1, D, h, w)
            img_tokens = einops.rearrange(img_feats, "b c h w -> (h w) b c")  # (S, B, D)
            cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")         # (S, B, D)
            tokens = torch.cat([tokens, img_tokens], dim=0)
            pos_tokens = torch.cat([pos_tokens, cam_pos], dim=0)


        encoder_out = self.encoder(tokens, pos_embed=pos_tokens)
        # Prepare decoder queries.
        dec_queries = self.dec_query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        dec_in = torch.zeros_like(dec_queries)
        dec_out = self.decoder.forward(
            x = dec_in,
            encoder_out = encoder_out,
            decoder_pos_embed = dec_queries,
            encoder_pos_embed = pos_tokens,
        )
        dec_out = dec_out.transpose(0, 1)  # (B, chunk_size, D)
        actions = self.action_head(dec_out)
        return actions, (mu, log_sigma)

# ---------------- ACT Policy ----------------

class ACTPolicy(nn.Module):
    """
    ACT Policy: wraps the ACT network and a temporal ensembler.
    It normalizes inputs/outputs (here, identity functions), runs the ACT model,
    and uses the temporal ensembler to output one deterministic action.
    """
    def __init__(self,
                 use_vae: bool,
                 robot_state_feature: int,
                 image_features: bool,
                 action_feature: int,
                 skill_feature: int,
                 dim_model: int,
                 n_heads: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 n_vae_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 pre_norm: bool,
                 feedforward_activation: str,
                 chunk_size: int,
                 latent_dim: int,
                 temporal_ensemble_coeff: float,
                 kl_weight: float,
                 n_action_steps: int):
        super().__init__()
        self.use_vae = use_vae
        self.robot_state_feature = robot_state_feature
        self.image_features = image_features
        self.action_feature = action_feature
        self.skill_feature = skill_feature
        self.dim_model = dim_model
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.n_action_steps = n_action_steps
        self.kl_weight = kl_weight

        self.model = ACT(
            use_vae,
            robot_state_feature,
            image_features,
            action_feature,
            skill_feature,
            dim_model,
            n_heads,
            n_encoder_layers,
            n_decoder_layers,
            n_vae_encoder_layers,
            dim_feedforward,
            dropout,
            pre_norm,
            feedforward_activation,
            chunk_size,
            latent_dim,
        )
        self.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, chunk_size)

        self.reset()

    def reset(self):
        """Call this when the environment resets."""
        self.temporal_ensembler.reset()
        self._action_queue = deque([], maxlen=self.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
        action = self.temporal_ensembler.update(actions)
        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss_dict["loss"] = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss_dict["loss"] = l1_loss

        return loss_dict

# ---------------- Testing ----------------

if __name__ == '__main__':
    # Specify explicit arguments.
    use_vae = True
    robot_state_feature = 10      # e.g. 10-dimensional robot state vector
    image_features = True
    action_feature = 4            # e.g. 4-dimensional action vector
    dim_model = 256
    n_heads = 8
    n_encoder_layers = 4
    n_decoder_layers = 4
    n_vae_encoder_layers = 2
    dim_feedforward = 2048
    dropout = 0.1
    pre_norm = True
    feedforward_activation = "relu"
    chunk_size = 10
    latent_dim = 32
    temporal_ensemble_coeff = 0.01
    kl_weight = 0.1
    n_action_steps = 5

    # Create a dummy batch.
    B = 2
    dummy_batch = {
        "observation.state": torch.randn(B, robot_state_feature),
        "observation.images": torch.randn(B, 3, 224, 224),
        "action": torch.randn(B, chunk_size, action_feature),
    }

    # Instantiate the policy.
    policy = ACTPolicy(
        use_vae,
        robot_state_feature,
        image_features,
        action_feature,
        dim_model,
        n_heads,
        n_encoder_layers,
        n_decoder_layers,
        n_vae_encoder_layers,
        dim_feedforward,
        dropout,
        pre_norm,
        feedforward_activation,
        chunk_size,
        latent_dim,
        temporal_ensemble_coeff,
        kl_weight,
        n_action_steps,
    )

    # Test forward pass (training mode).
    output = policy.forward(dummy_batch)
    print("Training loss dict:", output)
    # Test action selection.
    action = policy.select_action(dummy_batch)
    print("Selected action shape:", action.shape)