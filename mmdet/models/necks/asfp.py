import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmcv.cnn import ConvModule, xavier_init

from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module()
class ASFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(ASFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        self.asfp_convs = nn.ModuleList()
        for i in range(self.num_levels):
            asfp_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.asfp_convs.append(asfp_conv)

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.ws = []
        self.bs = []

        for i in range(self.num_levels):
            self.ws.append(Parameter(torch.ones(1, self.in_channels, 1, 1).cuda()))
            self.bs.append(Parameter(torch.zeros(1, self.in_channels, 1, 1).cuda()))

    def init_weights(self):
        """Initialize the weights of FPN module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    # def slice_as(self, src, dst):
    #     """Slice ``src`` as ``dst``
    #     Note:
    #         ``src`` should have the same or larger size than ``dst``.
    #
    #     Args:
    #         src (torch.Tensor): Tensors to be sliced.
    #         dst (torch.Tensor): ``src`` will be sliced to have the same
    #             size as ``dst``.
    #
    #     Returns:
    #         torch.Tensor: Sliced tensor.
    #     """
    #     assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
    #     if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
    #         return src
    #     else:
    #         return src[:, :, :dst.size(2), :dst.size(3)]

    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        gates = []
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            gathered = self.ws[i] * inputs[i] + self.bs[i]
            gates.append(nn.Sigmoid()(gathered))
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(gathered, output_size=gather_size)
            else:
                gathered = F.interpolate(gathered, size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            residual = self.asfp_convs[i](residual)
            outs.append(gates[i] * inputs[i] + (1 - gates[i]) * residual)

        return tuple(outs)