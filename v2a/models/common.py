import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, out_size, input_shape=(3, 84, 84),) -> None:
        """
        Args:
            out_size (int): the output size of the final linear layer.
            input_shape (tuple): shape of the input image (channels, height, width).
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute the flattened feature dimension dynamically
        dummy = torch.zeros(1, *input_shape)  # batch size 1
        dummy_out = self.cnn(dummy)
        n_features = dummy_out.shape[1]
        self.linear = nn.Linear(n_features, out_size)

    def forward(self, images):
        features = self.cnn(images)
        return self.linear(features)

class MLP(nn.Module):
    def __init__(self,input_dim, out_size,) -> None:
        """
        Args:
            input_dim (int): the input dimension of the input features.
            out_size (int): the output size of the final linear layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.linear = nn.Linear(256, out_size)

    def forward(self, images):
        features = self.mlp(images)
        return self.linear(features)