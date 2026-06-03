import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from definition import PAD_IDX
from fusion_logger import log_event, register_selected_forward_hooks, summarize_obj, tensor_summary
import signgraph_modules.resnet as signgraph_resnet


class SignGraphResNetForGFSLT(nn.Module):
    """
    Adapter that preserves the GFSLT visual feature protocol while running the
    SignGraph graph-enhanced ResNet internally.

    GFSLT input:
        x       [sumT, 3, H, W]
        lengths [B]
    SignGraph input:
        video   [B, 3, maxT, H, W]
    Output:
        feat    [B, maxT, 512]
    """

    def __init__(
        self,
        backbone_name="resnet18",
        adaptive_pool=True,
        pretrained=False,
        use_local_graph=True,
        use_temporal_graph=True,
        log_shapes=False,
        log_every=50,
        hook_backbone=False,
    ):
        super().__init__()
        self.backbone = getattr(signgraph_resnet, backbone_name)(
            pretrained=pretrained,
            use_local_graph=use_local_graph,
            use_temporal_graph=use_temporal_graph,
        )
        self.backbone.fc = nn.Identity()
        if adaptive_pool:
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.log_shapes = log_shapes
        self.log_every = log_every
        self.forward_step = 0
        self.hook_handles = []
        if hook_backbone:
            target_names = [
                "conv1",
                "maxpool",
                "layer1",
                "layer2",
                "layer3",
                "localG",
                "temporalG",
                "layer4",
                "localG2",
                "temporalG2",
                "avgpool",
                "fc",
            ]
            self.hook_handles = register_selected_forward_hooks(self.backbone, target_names=target_names, log_every=log_every)

    def _should_log(self):
        return self.log_shapes and self.forward_step % self.log_every == 0

    def _length_sum(self, lengths):
        if torch.is_tensor(lengths):
            return int(lengths.sum().item())
        return int(sum(int(length) for length in lengths))

    def forward(self, x, lengths):
        self.forward_step += 1
        length_sum = self._length_sum(lengths)
        if self._should_log():
            log_event(
                "adapter.input",
                step=self.forward_step,
                x=tensor_summary(x, "flattened_frames"),
                lengths=summarize_obj(lengths, "src_length_batch"),
                length_sum=length_sum,
                x_first_dim=int(x.shape[0]),
            )
        if length_sum != int(x.shape[0]):
            log_event("adapter.length_mismatch", step=self.forward_step, length_sum=length_sum, x_first_dim=int(x.shape[0]))
            raise ValueError(f"Length mismatch: sum(lengths)={length_sum}, x.shape[0]={x.shape[0]}")
        x_batch = []
        start = 0
        for length in lengths:
            length = int(length)
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x_batch = pad_sequence(x_batch, batch_first=True, padding_value=PAD_IDX)
        if self._should_log():
            log_event("adapter.after_pad", step=self.forward_step, x_batch=tensor_summary(x_batch, "x_batch_[B,T,C,H,W]"))
        x_video = x_batch.permute(0, 2, 1, 3, 4).contiguous()
        if self._should_log():
            log_event("adapter.before_signgraph", step=self.forward_step, x_video=tensor_summary(x_video, "x_video_[B,C,T,H,W]"))
        batch, _, time, _, _ = x_video.shape
        feat = self.backbone(x_video)
        if self._should_log():
            log_event("adapter.after_signgraph", step=self.forward_step, feat=tensor_summary(feat, "feat_[B*T,512]"))
        feat = feat.view(batch, time, -1)
        if self._should_log():
            log_event("adapter.output", step=self.forward_step, feat=tensor_summary(feat, "feat_[B,T,512]"))
        return feat
