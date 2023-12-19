import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class BaseUNet(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_output_channels=1,
                 skip_type='sum',
                 activation='sigmoid',
                 num_encoders=4,
                 base_num_channels=32,
                 num_residual_blocks=2,
                 norm=None,
                 use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert (self.num_input_channels > 0)
        assert (self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(
                self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                   input_size // 2,
                                   kernel_size=5,
                                   padding=2,
                                   norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels,
                              1,
                              activation=None,
                              norm=self.norm)


class RecurrentEncoder(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self,
                 num_input_channels,
                 num_output_channels=1,
                 skip_type='sum',
                 recurrent_block_type='convlstm',
                 activation='sigmoid',
                 num_encoders=4,
                 base_num_channels=64,
                 num_residual_blocks=2,
                 norm=None,
                 use_upsample_conv=True):
        super(RecurrentEncoder,
              self).__init__(num_input_channels, num_output_channels, skip_type, activation, num_encoders,
                             base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.input_dim = num_input_channels

        self.head = ConvLayer(num_input_channels,
                              self.base_num_channels,
                              kernel_size=5,
                              stride=1,
                              padding=2,
                              norm=self.norm)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(
                RecurrentConvLayer(input_size,
                                   output_size,
                                   kernel_size=5,
                                   stride=2,
                                   padding=2,
                                   recurrent_block_type=recurrent_block_type,
                                   norm=self.norm))

        self.out_conv = nn.ModuleList()
        self.out_conv.append(ConvLayer(64, 256, kernel_size=3, stride=2, padding=1, norm=self.norm))
        self.out_conv.append(ConvLayer(128, 512, kernel_size=3, stride=2, padding=1, norm=self.norm))
        self.out_conv.append(ConvLayer(256, 1024, kernel_size=3, stride=2, padding=1, norm=self.norm))

    def forward(self, x, prev_states=None):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        x = self.head(x)

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(self.out_conv[i](x))
            states.append(state)

        return blocks, states


class RecurrentRefEncoder(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self,
                 num_input_channels,
                 num_output_channels=1,
                 skip_type='sum',
                 recurrent_block_type='convlstm',
                 activation='sigmoid',
                 num_encoders=4,
                 base_num_channels=64,
                 num_residual_blocks=2,
                 norm=None,
                 use_upsample_conv=True):
        super(RecurrentRefEncoder,
              self).__init__(num_input_channels, num_output_channels, skip_type, activation, num_encoders,
                             base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.input_dim = num_input_channels
        # frame conv
        self.head = ConvLayer(num_input_channels + 2,
                              self.base_num_channels,
                              kernel_size=5,
                              stride=2,
                              padding=2,
                              norm=self.norm)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(
                RecurrentConvLayer(input_size,
                                   output_size,
                                   kernel_size=5,
                                   stride=2,
                                   padding=2,
                                   recurrent_block_type=recurrent_block_type,
                                   norm=self.norm))

    def forward(self, x, masks, prev_states=None):
        """
        :param x: B C H W
        :param masks: B no H W
        :return: B x no x C x H x W
        """
        batchsize, num_objects = masks.shape[:2]

        others = torch.cat([
            torch.sum(masks[:, [j for j in range(num_objects) if i != j]], dim=1, keepdim=True)
            for i in range(num_objects)
        ], 1)

        # b no 2 h w
        g = torch.stack([masks, others], dim=2)

        x = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        g = torch.cat([x, g], dim=2)

        x = g.flatten(0, 1)
        x = self.head(x)
        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            states.append(state)
            blocks.append(x.view(batchsize, num_objects, *x.shape[1:]))
        out = x.view(batchsize, num_objects, *x.shape[1:])

        return out, states


def build_recurrent_backbone(cfg, mode='seg'):
    if mode == 'seg':
        backbone = RecurrentEncoder(num_input_channels=cfg.MODEL.INPUT_DIM,
                                    base_num_channels=32,
                                    num_encoders=3,
                                    norm=cfg.MODEL.NORM)
    else:
        backbone = RecurrentRefEncoder(num_input_channels=cfg.MODEL.INPUT_DIM,
                                       base_num_channels=32,
                                       num_encoders=3,
                                       norm=cfg.MODEL.NORM)
    return backbone