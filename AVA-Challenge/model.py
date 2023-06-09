from detectron2.modeling.roi_heads import *
from detectron2.modeling import ShapeSpec
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, get_norm



class ChannelAttention(nn.Module):
    def __init__(self, in_feat, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_feat, in_feat // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_feat // ratio, in_feat, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        average_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        maximum_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = average_out + maximum_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super(SpatialAttention, self).__init__()

        assert kernel in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        average_out = torch.mean(x, dim=1, keepdim=True)
        maximum_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([average_out, maximum_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelGate(nn.Module):
    def __init__(self, gate_chnl, reduc_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_chnl
        self.mlp = nn.Sequential(
            nn.Linear(gate_chnl, gate_chnl // reduc_ratio),
            nn.ReLU(),
            nn.Linear(gate_chnl // reduc_ratio, gate_chnl)
            )
        self.pool_types = pool_types
                    
    def forward(self, x):
        chnl_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                chnl_att_raw = self.mlp( avg_pool.view(avg_pool.size(0), -1))
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                chnl_att_raw = self.mlp( max_pool.view(max_pool.size(0), -1))
            if chnl_att_sum is None:
                chnl_att_sum = chnl_att_raw
            else:
                chnl_att_sum = chnl_att_sum + chnl_att_raw

        scale = torch.sigmoid( chnl_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self, in_feat):
        super(SpatialGate, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_feat, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.layers(x)

class BAM(nn.Module):
    def __init__(self, gate_chnl):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_chnl)
        self.spatial_att = SpatialGate(gate_chnl)

    def forward(self,x):
        att = 1 + self.channel_att(x) * self.spatial_att(x)
        return att * x

'''Following functions as taken from Detectron2 official page  - 

https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/mask_head.py

'''

@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead_(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"
        print('LOADING RESIDUAL BOTTLENECK ATTENTION MaskRCNN')
        
        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            
            self.add_module("mask_fcn{}".format(k + 1), conv)
            if k == 0:    # Add BAM only after the first conv layer
                bam = BAM(conv_dim)    
                self.add_module("bam{}".format(k + 1), bam)   
                self.conv_norm_relus.append(bam)

            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim
            
        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            if layer=='Conv2D':
                
                weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret
    

    def layers(self, x):
        residual = x
        for layer in self:
            if x.numel() == 0:
                print(f"Empty tensor found with shape {x.shape}")
                return x
            x = layer(x)
        conv1x1 = nn.Conv2d(x.size(1), residual.size(1), kernel_size=1).to(x.device)
        x = conv1x1(x)
        # Apply upsampling to the residual before adding it to x
        x_upsampled = F.interpolate(x, size=residual.shape[2:], mode='bilinear', align_corners=False)

        x = x_upsampled + residual
        x = self.deconv(x)
        return self.predictor(x)



@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead_multi(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"
        print('LOADING MULTIHEAD ATTENTION MaskRCNN')

        self.conv_norm_relus = []
        self.attention_layers = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

            # Add multihead attention after the first convolution layer
            if k == 0:
                bam = BAM(conv_dim)
                ca = ChannelAttention(conv_dim)
                sa = SpatialAttention()

                self.add_module("bam", bam)
                self.add_module("ca", ca)
                self.add_module("sa", sa)

                self.attention_layers.append((bam, ca, sa))

            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret


    def layers(self, x):
        residual = x

        for i, layer in enumerate(self.conv_norm_relus):
            if x.numel() == 0:
                print(f"Empty tensor found with shape {x.shape}")
                return x
            x = layer(x)

            # Apply multihead attention after the first convolution layer
            if i == 0:
                bam, ca, sa = self.attention_layers[0]
                x_bam = bam(x)
                x_ca = ca(x)
                x_sa = sa(x)
                x = x_bam * x_ca * x_sa

        x = self.deconv(x)
        return self.predictor(x)

