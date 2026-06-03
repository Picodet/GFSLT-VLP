import torch
import torch.nn as nn


def act_layer(act):
    if act is None or act == "identity":
        return nn.Identity()
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise NotImplementedError(f"Unsupported activation: {act}")


def norm_layer(norm, channels):
    if norm is None or norm == "identity":
        return nn.Identity()
    norm = norm.lower()
    if norm in {"batch", "bn"}:
        return nn.BatchNorm2d(channels)
    if norm in {"instance", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    raise NotImplementedError(f"Unsupported norm: {norm}")


class BasicConv(nn.Module):
    def __init__(self, channels, act="relu", norm=None, bias=True):
        super().__init__()
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            layers.append(norm_layer(norm, channels[i]))
            layers.append(act_layer(act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def batched_index_select(x, idx):
    """Select dense graph neighbors from x shaped [B, C, N, 1]."""
    if idx.dim() != 3:
        raise ValueError(f"Expected edge index with shape [B, N, K], got {tuple(idx.shape)}")
    idx = idx.to(device=x.device, dtype=torch.long)
    batch, channels, nodes, _ = x.shape
    _, target_nodes, neighbors = idx.shape
    flat = x.squeeze(-1).transpose(1, 2).contiguous()
    gather_idx = idx.unsqueeze(-1).expand(batch, target_nodes, neighbors, channels)
    selected = torch.gather(flat.unsqueeze(1).expand(batch, target_nodes, nodes, channels), 2, gather_idx)
    return selected.permute(0, 3, 1, 2).contiguous()
