import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange

from .gcn_lib.torch_vertex import Grapher
from .gcn_lib.temgraph import TemporalGraph

__all__ = ["ResNet", "resnet18", "resnet34"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, use_local_graph=True, use_temporal_graph=True):
        super().__init__()
        self.inplanes = 64
        self.use_local_graph = use_local_graph
        self.use_temporal_graph = use_temporal_graph
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.localG = Grapher(in_channels=256, kernel_size=3, dilation=1, conv="edge", act="relu", norm="batch", bias=True, stochastic=False, epsilon=0.0, r=1, n=14 * 14, drop_path=0.0, relative_pos=True)
        self.temporalG = TemporalGraph(k=14 * 14 // 4, in_channels=256, drop_path=0.0)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.localG2 = Grapher(in_channels=512, kernel_size=4, dilation=1, conv="edge", act="relu", norm="batch", bias=True, stochastic=False, epsilon=0.0, r=1, n=7 * 7, drop_path=0.0, relative_pos=True)
        self.temporalG2 = TemporalGraph(k=7 * 7, in_channels=512, drop_path=0.0)
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _apply_graphs(self, x, batch, local_graph, temporal_graph, alpha_local, alpha_temporal):
        _, channels, time, height, width = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        if self.use_local_graph:
            x = x + local_graph(x) * alpha_local
        if self.use_temporal_graph:
            x = x + temporal_graph(x, batch) * alpha_temporal
        return x.view(batch, time, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x):
        batch = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self._apply_graphs(x, batch, self.localG, self.temporalG, self.alpha[0], self.alpha[1])
        x = self.layer4(x)
        x = self._apply_graphs(x, batch, self.localG2, self.temporalG2, self.alpha[2], self.alpha[3])
        _, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * time, channels, height, width)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _inflate_2d_state_dict(state_dict):
    inflated = {}
    for name, value in state_dict.items():
        if value.ndim == 4 and ("conv" in name or "downsample.0.weight" in name):
            value = value.unsqueeze(2)
        inflated[name] = value
    return inflated


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(_inflate_2d_state_dict(model_zoo.load_url(model_urls["resnet18"])), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(_inflate_2d_state_dict(model_zoo.load_url(model_urls["resnet34"])), strict=False)
    return model
