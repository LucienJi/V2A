import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision import models

##############################################
# Helper: Modified ResNet-18 for Image Encoding
##############################################

def create_resnet18_encoder(pretrained=True, output_dim=512):
    """
    Loads a pretrained ResNet-18 and removes its classification head.
    Optionally, you can add a linear projection so that the output has dimension output_dim.
    """
    resnet = models.resnet18(pretrained=pretrained)
    # Remove the last fc layer; resnet.avgpool outputs a 512-d feature map (after flattening)
    modules = list(resnet.children())[:-1]  # remove the fc layer
    encoder = nn.Sequential(*modules)  # now encoder outputs shape: (B, 512, 1, 1)
    
    # We'll add a projection layer to flatten and (optionally) change the feature dimension.
    projection = nn.Linear(512, output_dim)
    
    # Wrap them in a single module
    class ResNet18Encoder(nn.Module):
        def __init__(self, encoder, projection):
            super(ResNet18Encoder, self).__init__()
            self.encoder = encoder
            self.projection = projection

        def forward(self, x):
            # x shape: (B, 3, H, W)
            feat = self.encoder(x)  # (B, 512, 1, 1)
            feat = feat.view(feat.size(0), -1)  # (B, 512)
            out = self.projection(feat)         # (B, output_dim)
            return out

    return ResNet18Encoder(encoder, projection)

##############################################
# Helper: MLP for Robot State Processing
##############################################

def create_state_mlp(input_dim, hidden_dims, output_dim):
    """
    Creates an MLP that processes the robot state.
    hidden_dims is a list of hidden layer sizes.
    """
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

##############################################
# GMM Actor Network for Robot Manipulation
##############################################

class GMMActorNetwork(nn.Module):
    """
    A policy network that takes as input:
        - A single frame image.
        - A robot state vector.
    It uses a pretrained ResNet-18 to encode the image and an MLP to encode the state.
    The features are concatenated and passed through an MLP to predict the parameters
    of a Gaussian Mixture Model (GMM) distribution over actions.
    """

    def __init__(
        self,
        image_output_dim=256,      # Dimension of image features after projection.
        state_input_dim=18,        # Dimension of the raw robot state vector.
        state_hidden_dims=[256, 256],# Hidden sizes for state MLP.
        state_output_dim=64,       # Output dimension for state features.
        skill_input_dim = 0,
        skill_output_dim = 128,
        combined_hidden_dims=[256, 256], # Hidden sizes for the combined MLP.
        action_dim=6,              # Dimension of the action vector.
        num_modes=5,               # Number of GMM modes.
        min_std=0.0001,              # Minimum standard deviation.
        std_activation="softplus", # Activation function for std (can be "softplus" or "exp").
        use_tanh=False,            # If True, wrap the final distribution in a tanh transform.
        low_noise_eval=True,       # If True, use low noise at eval time (for deterministic mode).
    ):
        super(GMMActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh
        self.skill_output_dim = skill_output_dim if skill_input_dim > 0 else 0 

        self.skill_proj = nn.Linear(skill_input_dim, skill_output_dim) if skill_input_dim > 0 and self.skill_output_dim > 0 else None

        # Select activation for scale.
        if std_activation == "softplus":
            self.std_activation_fn = F.softplus
        elif std_activation == "exp":
            self.std_activation_fn = torch.exp
        else:
            raise ValueError("Unsupported std_activation. Choose 'softplus' or 'exp'.")

        # Image encoder: a pretrained ResNet-18 (we freeze the backbone for testing if desired)
        self.image_encoder = create_resnet18_encoder(pretrained=True, output_dim=image_output_dim)
        # Optionally freeze image encoder parameters for testing:
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        # State encoder: simple MLP
        self.state_encoder = create_state_mlp(state_input_dim, state_hidden_dims, state_output_dim)

        # Combined feature dimension:
        combined_dim = image_output_dim + state_output_dim + self.skill_output_dim 

        # Combined MLP: this network outputs the parameters for the GMM.
        # The total output dimension is: num_modes * (2*action_dim + 1)
        #   - For each mode, we need:
        #       * mean (action_dim)
        #       * scale (action_dim)
        #       * logit (1 scalar) for the mixture component weight.
        self.gmm_param_dim = num_modes * (2 * action_dim + 1)
        mlp_layers = []
        prev_dim = combined_dim
        for h in combined_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h))
            mlp_layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        mlp_layers.append(nn.Linear(prev_dim, self.gmm_param_dim))
        self.combined_mlp = nn.Sequential(*mlp_layers)

    def forward(self, obs, skill=None):
        """
        Args:
            obs (dict): Dictionary with keys:
                - "image": Tensor of shape (B, 3, H, W)
                - "robot_state": Tensor of shape (B, state_input_dim)
        Returns:
            A sampled action from the policy distribution (Tensor of shape (B, action_dim)).
            (Optionally, you could also return the distribution for loss computations.)
        """
        image = obs["image"]
        robot_state = obs["robot_state"]

        # Process image and state
        image_feat = self.image_encoder(image)         # (B, image_output_dim)
        state_feat = self.state_encoder(robot_state)     # (B, state_output_dim)
        if self.skill_proj is not None and skill is not None:
            skill_feat = self.skill_proj(skill)
            combined_feat = torch.cat([image_feat, state_feat, skill_feat], dim=1)
        else:
            combined_feat = torch.cat([image_feat, state_feat], dim=1)
        # Get GMM parameters from the combined MLP.
        gmm_params = self.combined_mlp(combined_feat)    # (B, gmm_param_dim)
        B = gmm_params.size(0)
        M = self.num_modes
        D_a = self.action_dim

        # Split parameters.
        # First M * D_a values: means
        means = gmm_params[:, :M * D_a].view(B, M, D_a)
        # Next M * D_a values: scales (pre-activation)
        scales = gmm_params[:, M * D_a:2 * M * D_a].view(B, M, D_a)
        # Last M values: logits for mixture weights.
        logits = gmm_params[:, 2 * M * D_a:].view(B, M)

        # Apply tanh to means (if not using a tanh-wrapped distribution)
        if not self.use_tanh:
            means = torch.tanh(means)

        # Process scales: apply activation and add a minimum std.
        if self.low_noise_eval and (not self.training):
            scales = torch.ones_like(scales) * 1e-4
        else:
            scales = self.std_activation_fn(scales) + self.min_std

        # Create the component (Gaussian) distribution.
        # Here each component is a Normal distribution with independent dimensions.
        component_dist = D.Normal(loc=means, scale=scales)
        # Make it an independent distribution over the action dimensions.
        component_dist = D.Independent(component_dist, 1)

        # Create the mixture distribution.
        mixture_dist = D.Categorical(logits=logits)

        # Create the full mixture distribution.
        gmm = D.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

        # Optionally wrap with a tanh distribution to squash the output to [-1, 1].
        if self.use_tanh:
            # We define a simple Tanh wrapper below.
            gmm = TanhWrappedDistribution(gmm)

        # For training we often need to return the distribution (to compute log-likelihood)
        # Here we sample an action.
        action = gmm.sample()  # (B, action_dim)
        return action

##############################################
# Tanh Wrapped Distribution (optional)
##############################################

class TanhWrappedDistribution(D.TransformedDistribution):
    """
    Wraps a distribution in a Tanh transformation.
    Useful for ensuring actions lie in [-1, 1].
    """
    def __init__(self, base_dist):
        transforms = [D.transforms.TanhTransform()]
        super().__init__(base_dist, transforms)
        self.base_dist = base_dist

    @property
    def mean(self):
        # Return the mean, transformed by tanh.
        m = self.base_dist.mean
        return torch.tanh(m)

##############################################
# Example Usage
##############################################

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy observation:
    batch_size = 4
    # Assume image size is 3 x 224 x 224
    dummy_image = torch.randn(batch_size, 3, 224, 224, device=device)
    # Assume robot state is a vector of size 10
    dummy_state = torch.randn(batch_size, 10, device=device)

    obs = {
        "image": dummy_image,
        "robot_state": dummy_state,
    }

    # Create the policy network.
    policy_net = GMMActorNetwork(
        image_output_dim=256,
        state_input_dim=10,
        state_hidden_dims=[64, 64],
        state_output_dim=64,
        skill_input_dim=0,
        skill_output_dim=128,
        combined_hidden_dims=[256, 256],
        action_dim=6,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        use_tanh=True,           # Set to True if you want the final actions in [-1, 1]
        low_noise_eval=True,
    ).to(device)

    # Set in evaluation mode for testing (optional)
    policy_net.eval()

    # Forward pass: get a sampled action.
    with torch.no_grad():
        action = policy_net.forward(obs)
    print("Sampled action shape:", action.shape)  # Expected: (batch_size, action_dim)
    print("Sampled action:", action)