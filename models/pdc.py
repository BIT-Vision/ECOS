import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d, DeformConv2d
import math


class DeformConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 offset_groups: int = 2,
                 with_mask: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.offset_groups = offset_groups
        self.with_mask = with_mask

        scale = 3 if self.with_mask else 2

        self.conv_offset = nn.Conv2d(self.in_channels,
                                     scale * self.offset_groups * self.kernel_size * self.kernel_size,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=self.bias)
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.conv_offset(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.conv_offset(x)
            mask = None
        return deform_conv2d(x,
                             offset=offset,
                             weight=self.weight,
                             bias=self.bias,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             mask=mask)


class DeformLayer(DeformConv):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 offset_groups: int = 8,
                 with_mask: bool = True):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         offset_groups=offset_groups)

    def forward(self, x, feat):
        if self.with_mask:
            oh, ow, mask = self.conv_offset(feat).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.conv_offset(feat)
            mask = None
        return deform_conv2d(x,
                             offset=offset,
                             weight=self.weight,
                             bias=self.bias,
                             stride=self.stride,
                             padding=self.padding,
                             dilation=self.dilation,
                             mask=mask)