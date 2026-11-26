"""Full author HyperDS modules, with explicit hour queries and field feedback.

Upstream is pinned in modal_app.py. No satellite branch or loss is replaced by a
station-bias surrogate. Canonical field order is u10, v10, t2m, sp, tp1h.
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_REVISION = "1fe6d0b08e759548655009e1747428f9136a7839"


class HyperDS(nn.Module):
    def __init__(self, mode="paper", satellite_shape=(128, 256)):
        super().__init__()
        if mode not in ("paper", "direct", "autoregressive"):
            raise ValueError(mode)
        upstream = Path(os.environ.get("HYPERDS_UPSTREAM_ROOT", "/opt/hyperds"))
        if not (upstream / "models/dual_transnet.py").is_file():
            raise FileNotFoundError(f"Clone the pinned author repository into {upstream}")
        sys.path.insert(0, str(upstream))
        from models.dual_transnet import DualTransNet
        from models.variable_net import VariableNet
        from utils.positional_encoding import SineCosPE

        self.mode = mode
        self.fusion = DualTransNet(
            h8_backbone_cfg={"name":"resnet18", "in_channels":4, "pretrained":False,
                             "num_classes":None, "out_keys":["block4"]},
            pred_sfc_backbone_cfg={"name":"resnet18", "in_channels":5, "pretrained":False,
                                   "num_classes":None, "out_keys":["block4"]},
            trans_cfg=SimpleNamespace(d_model=64),
            head_cfg=SimpleNamespace(feat_channels=[256,128,64,64,32], img_size=[32,64], feat_size=[16,32]),
        )
        self.feature_shape = ((satellite_shape[0]+7)//8, (satellite_shape[1]+7)//8)
        token_count = self.feature_shape[0] * self.feature_shape[1]
        # Upstream hardcodes 19*29 tokens for China; only spatial size changes.
        self.fusion.pe_parameters3 = nn.Parameter(torch.rand(1, token_count, 64))
        self.decoders = nn.ModuleList([VariableNet(token_count, 64, 128) for _ in range(5)])
        self.spatial_encoding = SineCosPE(2, N_freqs=16, include_input=False)
        self.hour_encoding = SineCosPE(1, N_freqs=32, include_input=False)

    def encode_satellite(self, satellite):
        if satellite.ndim != 5 or satellite.shape[1:3] != (2,4):
            raise ValueError("satellite must be [batch, 2 frames, 4 bands, row, col]")
        features = [self.fusion.h8_backbone(satellite[:,frame])[1]["block4"] for frame in (0,1)]
        features = self.fusion.h8_conv3(torch.cat(features, dim=1))
        tokens = features.flatten(2).transpose(1,2).contiguous() + self.fusion.pe_parameters3
        # This is the same self-attention encoder used by DualTransNet.forward.
        return self.fusion.transformer3.encoder(tokens)[0]

    def encode_field(self, field, satellite_context):
        features = self.fusion.pred_sfc_backbone(field)[1]["block4"]
        features = self.fusion.pred_conv3(features)
        features = F.interpolate(features, size=self.feature_shape, mode="bilinear", align_corners=True)
        query = features.flatten(2).transpose(1,2).contiguous() + self.fusion.pe_parameters3
        fused = self.fusion.transformer3.decoder(satellite_context, query)[0]
        fused = self.fusion.transformer3.projection(fused)
        return fused.transpose(1,2).reshape(field.shape[0],64,*self.feature_shape)

    def decode(self, context, field, coordinates, hour=0):
        # grid_sample aligns coordinates to cell boundaries (align_corners=False).
        values = F.grid_sample(field, coordinates[:,None], mode="bilinear",
                               padding_mode="border", align_corners=False).squeeze(2).transpose(1,2)
        embedding = self.spatial_encoding((coordinates+1)/2)
        if self.mode != "paper":
            h = coordinates.new_full((*coordinates.shape[:-1],1), float(hour)/48)
            embedding = embedding + self.hour_encoding(h)-self.hour_encoding(torch.zeros_like(h))
        # Use each author VariableNet directly. INF_Net.forward_single upstream
        # concatenates P and T in reverse order; explicit canonical ordering fixes it.
        residual = torch.cat([decoder(context,embedding,values[:,:,index])
                              for index,decoder in enumerate(self.decoders)], dim=-1)
        return values + residual

    def forward(self, field, satellite, coordinates, hour=0):
        return self.decode(self.encode_field(field,self.encode_satellite(satellite)), field, coordinates, hour)


def subgrid_queries(batch, height, width, device, stations=None, random=True):
    """Ten interior samples per cell, including station coordinates in their cells."""
    y, x = torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing="ij")
    cells = torch.stack((x,y),dim=-1).reshape(1,height*width,1,2)
    if random:
        offsets = torch.rand(batch,height*width,10,2,device=device)
    else:
        # Deterministic stratified samples for repeatable validation.
        offsets = torch.tensor([[.1,.2],[.3,.2],[.5,.2],[.7,.2],[.9,.2],
                                [.1,.8],[.3,.8],[.5,.8],[.7,.8],[.9,.8]],device=device)
        offsets = offsets.expand(batch,height*width,10,2)
    samples = 2*(cells+offsets)/torch.tensor([width,height],device=device)-1
    if stations is not None:
        for station in range(stations.shape[1]):
            xy = stations[:,station]
            indices = ((xy+1)/2*torch.tensor([width,height],device=device)).long()
            indices[:,0].clamp_(0,width-1)
            indices[:,1].clamp_(0,height-1)
            flat = indices[:,1]*width+indices[:,0]
            # Distinct slots preserve multiple stations in the same pixel.
            samples[torch.arange(batch,device=device),flat,station%10] = xy
    return samples.reshape(batch,height*width*10,2)


def grid_average(prediction, height, width):
    return prediction.reshape(prediction.shape[0],height,width,10,5).mean(dim=3).permute(0,3,1,2)
