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
            # image_output_dim=gmm_config.image_output_dim,
            # state_input_dim=gmm_config.state_input_dim,
            # state_hidden_dims=gmm_config.state_hidden_dims,
            # state_output_dim=gmm_config.state_output_dim,
            # skill_input_dim=gmm_config.skill_input_dim,
            # skill_output_dim=gmm_config.skill_output_dim,
            # combined_hidden_dims=gmm_config.combined_hidden_dims,
            # action_dim=self.ac_dim,
            # num_modes=gmm_config.num_modes,
            # min_std=gmm_config.min_std,
            # std_activation=gmm_config.std_activation,
            # use_tanh=gmm_config.use_tanh,
            # low_noise_eval=gmm_config.low_noise_eval
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = dict()
        
        # Process image observations
        input_batch["obs"] = {
            "image": batch["obs"]["image"][:, 0, :],  # [B, C, H, W]
            "robot_state": batch["obs"]["robot_state"][:, 0, :]
        }
        
        # Process actions
        input_batch["actions"] = batch["actions"][:, 0, :]
        
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))
    
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(GMMPolicyAlgo, self).train_on_batch(batch, epoch, validate=validate)
            
            # forward, get action distribution
            action_dist = self._forward_training(batch)
            
            # compute loss and log info
            losses = self._compute_losses(action_dist, batch)
            info["losses"] = TensorUtils.detach(losses)

            # log info of action distribution
            with torch.no_grad():
                info["predictions"] = {
                    "actions": action_dist.sample(),
                    # "means": action_dist.component_distribution.loc.mean(dim=1),
                    # "logits": action_dist.mixture_distribution.logits
                }

                info["predictions"] = TensorUtils.detach(info["predictions"])
                
            # info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        # Get action distribution from policy
        dists = self.nets["policy"].forward(
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
        # with torch.no_grad():
        #     action = self.nets["policy"](obs_dict)
            
        action = self.nets["policy"].sample_action(obs_dict)
        return action

    # def log_info(self, info):
    #     log = super().log_info(info)
    #     log["Loss"] = info["losses"]["action_loss"].item()
    #     log["Log_Likelihood"] = info["losses"]["log_probs"].item()
    #     return log
    
    def log_info(self, info):
        log = super().log_info(info)
        log.update({
            "Loss": info["losses"]["action_loss"].item(),
            "Log_Likelihood": info["losses"]["log_probs"].item(),
            "Grad_Norm": sum(info.get("policy_grad_norms", [0.0])).item()
        })
        return log