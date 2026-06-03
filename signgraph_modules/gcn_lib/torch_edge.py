import torch
import torch.nn as nn


class DenseDilatedKnnGraph(nn.Module):
    """Dense KNN graph builder used by local SignGraph convolutions."""

    def __init__(self, kernel_size, dilation=1, stochastic=False, epsilon=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon

    def forward(self, x, y=None, relative_pos=None):
        source = x if y is None else y
        x_nodes = x.squeeze(-1).transpose(1, 2).contiguous()
        y_nodes = source.squeeze(-1).transpose(1, 2).contiguous()
        dist = torch.cdist(x_nodes, y_nodes)
        if relative_pos is not None and relative_pos.shape[-2:] == dist.shape[-2:]:
            dist = dist + relative_pos.to(device=dist.device, dtype=dist.dtype)
        k = min(self.kernel_size * self.dilation, dist.shape[-1])
        idx = dist.topk(k=k, dim=-1, largest=False).indices
        if self.dilation > 1:
            idx = idx[..., :: self.dilation]
        return idx[..., : min(self.kernel_size, idx.shape[-1])]
