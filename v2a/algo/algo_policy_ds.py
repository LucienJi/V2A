# v2a/algo/algo_policy.py
from collections import OrderedDict
import torch
import torch.nn as nn
from robomimic.algo import PolicyAlgo
# from robomimic.utils.obs_utils import ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from ..models.policy.gmm_policy import GMMActorNetwork

class GMMPolicyAlgo(PolicyAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device):
        super().__init__(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        
        # Get params from config
        gmm_config = self.algo_config.gmm
        self.nets["policy"] = GMMActorNetwork(
            image_output_dim=gmm_config.image_output_dim,
            state_input_dim=gmm_config.state_input_dim,
            state_hidden_dims=gmm_config.state_hidden_dims,
            state_output_dim=gmm_config.state_output_dim,
            skill_input_dim=gmm_config.skill_input_dim,
            skill_output_dim=gmm_config.skill_output_dim,
            combined_hidden_dims=gmm_config.combined_hidden_dims,
            action_dim=self.ac_dim,
            num_modes=gmm_config.num_modes,
            min_std=gmm_config.min_std,
            std_activation=gmm_config.std_activation,
            use_tanh=gmm_config.use_tanh,
            low_noise_eval=gmm_config.low_noise_eval
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = dict()
        
        # Process image observations
        input_batch["obs"] = {
            "image": batch["obs"]["eye_in_hand_rgb"][:, 0, :],  # [B, C, H, W]
            "robot_state": batch["obs"]["robot_states"][:, 0, :]
        }
        
        # Process actions
        input_batch["actions"] = batch["actions"][:, 0, :]
        
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _forward_training(self, batch):
        # Get action distribution from policy
        dists = self.nets["policy"].forward_train(
            obs={
                "image": batch["obs"]["image"],
                "robot_state": batch["obs"]["robot_state"]
            }
        )
        return dists

    def _compute_losses(self, predictions, batch):
        # Calculate negative log likelihood loss
        log_probs = predictions.log_prob(batch["actions"])
        action_loss = -log_probs.mean()
        
        return OrderedDict(
            log_probs=log_probs.mean(),
            action_loss=action_loss
        )

    def _train_step(self, losses):
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def get_action(self, obs_dict, goal_dict=None):
        self.set_eval()
        with torch.no_grad():
            action = self.nets["policy"](obs_dict)
        return action

    def log_info(self, info):
        log = super().log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        return log