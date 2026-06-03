import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .pos_embed import get_2d_relative_pos_embed
from .torch_edge import DenseDilatedKnnGraph
from .torch_nn import BasicConv, act_layer, batched_index_select

from timm.models.layers import DropPath


class EdgeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super().__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index)
        x_j = batched_index_select(x if y is None else y, edge_index)
        value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return value


class DyGraphConv2d(EdgeConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv="edge", act="relu", norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        if conv != "edge":
            raise NotImplementedError("This GFSLT integration currently supports SignGraph edge convolution only.")
        super().__init__(in_channels, out_channels, act, norm, bias)
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        batch, channels, height, width = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
        x_nodes = x.reshape(batch, channels, -1, 1).contiguous()
        y_nodes = None if y is None else y.reshape(batch, channels, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x_nodes, y_nodes, relative_pos)
        x = super().forward(x_nodes, edge_index, y_nodes)
        return x.reshape(batch, -1, height, width).contiguous()


class Grapher(nn.Module):
    """Local SignGraph module operating on per-frame spatial feature maps."""

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv="edge", act="relu", norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super().__init__()
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias), nn.BatchNorm2d(in_channels))
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=bias), nn.BatchNorm2d(in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels, int(n ** 0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(relative_pos_tensor, size=(n, n // (r * r)), mode="bicubic", align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1).mean(0), requires_grad=False)

    def _get_relative_pos(self, height, width):
        if self.relative_pos is None or height * width == self.n:
            return self.relative_pos
        reduced = max(height * width // (self.r * self.r), 1)
        return F.interpolate(self.relative_pos.unsqueeze(0).unsqueeze(0), size=(height * width, reduced), mode="bicubic", align_corners=False).squeeze(0).squeeze(0)

    def forward(self, x):
        x = self.fc1(x)
        _, _, height, width = x.shape
        x = self.graph_conv(x, self._get_relative_pos(height, width))
        x = self.fc2(x)
        return self.drop_path(x)
