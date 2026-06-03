import torch
import torch.nn as nn
from einops import rearrange

from timm.models.layers import DropPath


class TemporalGraph(nn.Module):
    """Temporal SignGraph module with pure-PyTorch top-k temporal aggregation."""

    def __init__(self, in_channels, k=4, drop_path=0.0):
        super().__init__()
        self.k = k
        self.reduction_channel = in_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0)),
            nn.BatchNorm3d(in_channels),
        )
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0)),
            nn.BatchNorm3d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, batch):
        total_frames, channels, height, width = x.shape
        if batch <= 0 or total_frames % batch != 0:
            raise ValueError(f"Invalid batch={batch} for temporal graph input with {total_frames} frames")
        time = total_frames // batch
        video = rearrange(x.view(batch, time, channels, height, width), "b t c h w -> b c t h w")
        reduced = self.down_conv(video)
        if time <= 1:
            return self.drop_path(self.up_conv(reduced)).permute(0, 2, 1, 3, 4).contiguous().view(total_frames, channels, height, width)

        nodes = rearrange(reduced, "b c t h w -> b t (h w) c")
        out = nodes.clone()
        k = min(self.k, height * width)
        for t in range(time - 1):
            sim = -torch.cdist(nodes[:, t], nodes[:, t + 1])
            idx = sim.topk(k=k, dim=-1).indices
            neighbors = torch.gather(nodes[:, t + 1].unsqueeze(1).expand(-1, height * width, -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, channels))
            out[:, t] = out[:, t] + neighbors.mean(dim=2)
            back_idx = sim.transpose(1, 2).topk(k=k, dim=-1).indices
            back_neighbors = torch.gather(nodes[:, t].unsqueeze(1).expand(-1, height * width, -1, -1), 2, back_idx.unsqueeze(-1).expand(-1, -1, -1, channels))
            out[:, t + 1] = out[:, t + 1] + back_neighbors.mean(dim=2)
        out = rearrange(out, "b t (h w) c -> b c t h w", h=height, w=width)
        out = self.up_conv(out)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(total_frames, channels, height, width)
        return self.drop_path(out)
