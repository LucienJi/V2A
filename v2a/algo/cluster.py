from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def sinkhorn(
    out: Tensor,
    iterations: int = 3,
    epsilon: float = 0.05,
    gather_distributed: bool = False,
) -> Tensor:
    """Distributed sinkhorn algorithm.

    Returns:
        Soft codes Q assigning each feature to a prototype.
    """
    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    # Get the exponential matrix and make it sum to 1
    Q = torch.exp(out / epsilon).t()
    sum_Q = torch.sum(Q)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    B = Q.shape[1] * world_size

    for _ in range(iterations):
        # Normalize rows
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        # Normalize columns
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class SwaVLoss(nn.Module):
    """Implementation of the SwaV loss.

    Attributes:
        temperature:
            Temperature parameter used for cross entropy calculations.
        sinkhorn_iterations:
            Number of iterations of the sinkhorn algorithm.
        sinkhorn_epsilon:
            Temperature parameter used in the sinkhorn algorithm.
        sinkhorn_gather_distributed:
            If True, features from all GPUs are gathered to calculate the
            soft codes in the sinkhorn algorithm.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_gather_distributed: bool = False,
    ):
        """Initializes the SwaVLoss module with the specified parameters.

        Args:
            temperature:
                Temperature parameter used for cross-entropy calculations.
            sinkhorn_iterations:
                Number of iterations of the sinkhorn algorithm.
            sinkhorn_epsilon:
                Temperature parameter used in the sinkhorn algorithm.
            sinkhorn_gather_distributed:
                If True, features from all GPUs are gathered to calculate the
                soft codes in the sinkhorn algorithm.

        Raises:
            ValueError: If sinkhorn_gather_distributed is True but torch.distributed
                is not available.
        """
        super(SwaVLoss, self).__init__()
        if sinkhorn_gather_distributed and not dist.is_available():
            raise ValueError(
                "sinkhorn_gather_distributed is True but torch.distributed is not "
                "available. Please set gather_distributed=False or install a torch "
                "version with distributed support."
            )

        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_gather_distributed = sinkhorn_gather_distributed

    def subloss(self, z: Tensor, q: Tensor) -> Tensor:
        """Calculates the cross entropy for the SwaV prediction problem.

        Args:
            z:
                Similarity of the features and the SwaV prototypes.
            q:
                Codes obtained from Sinkhorn iterations.

        Returns:
            Cross entropy between predictions z and codes q.
        """
        return -torch.mean(
            torch.sum(q * F.log_softmax(z / self.temperature, dim=1), dim=1)
        )

    def forward(
        self,
        features1: List[Tensor],
        features2: List[Tensor],
        queue_outputs: Union[List[Tensor], None] = None,
    ) -> Tensor:
        n_features = len(features1)
        with torch.no_grad():
            # Append queue outputs
            if queue_outputs is not None:
                features1 = torch.cat((features1, queue_outputs.detach())) # (B+N, C)
                features2 = torch.cat((features2, queue_outputs.detach()))
                # Compute the codes
            q1 = sinkhorn(
                    features1,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )
            q2 = sinkhorn(
                    features2,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )

            # Drop queue similarities
            if queue_outputs is not None:
                q1 = q1[: n_features]
                q2 = q2[: n_features]
            print(q1.shape, q2.shape)
        # Compute subloss for each pair of crops
        loss1 = self.subloss(z = features1, q = q2)
        loss2 = self.subloss(z = features2, q = q1)
        loss = loss1 + loss2        

        return loss / 2.0