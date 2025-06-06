"""
Creates a MobileNetV4 Model as defined in:
Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard. (2024).
MobileNetV4 - Universal Models for the Mobile Ecosystem
arXiv preprint arXiv:2404.10518.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from collections import OrderedDict

__all__ = ['mobilenetv4_conv_small', 'mobilenetv4_conv_medium', 'mobilenetv4_conv_large']

from mmdet.models import BACKBONES
from mmdet3d.utils import get_root_logger
from mmcv.runner import _load_checkpoint


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"

    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3',
                                  conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio,
                 use_layer_scale=False
                 ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        self.use_res_connect = (inp == oup and stride == 1)

        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

        self._use_layer_scale = use_layer_scale
        if self._use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(oup, self.layer_scale_init_value)

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        if self._use_layer_scale:
            x = self.layer_scale(x)
        if self.use_res_connect:
            x = x + shortcut
        return x


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
                 dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py &
                             https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention2d.py
        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj = conv_2d(inp, num_heads * key_dim, 1, 1, norm=False, act=False)

        self.key = nn.Sequential()
        self.value = nn.Sequential()
        if self.kv_strides > 1:
            self.key.add_module('_key_dw_conv',
                                conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False))
            self.value.add_module('_value_dw_conv',
                                  conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False))
        self.key.add_module('_key_proj', conv_2d(inp, key_dim, 1, 1, norm=False, act=False))
        self.value.add_module('_value_proj', conv_2d(inp, key_dim, 1, 1, norm=False, act=False))

        self._output_proj = conv_2d(num_heads * key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def _reshape_projected_query(self, t: torch.Tensor, num_heads: int, key_dim: int):
        """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
        s = t.shape
        t = t.reshape(s[0], num_heads, key_dim, -1)
        return t.transpose(-1, -2).contiguous()

    def _reshape_input(self, t: torch.Tensor):
        """Reshapes a tensor to three dimensions, keeping the batch and channels."""
        s = t.shape
        t = t.reshape(s[0], s[1], -1).transpose(1, 2)
        return t.unsqueeze(1).contiguous()

    def _reshape_output(self, t: torch.Tensor, num_heads: int, h_px: int, w_px: int):
        """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
        s = t.shape
        feat_dim = s[-1] * num_heads
        t = t.transpose(1, 2)
        return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        batch_size, seq_length, H, W = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        # desired q shape: [b, h, k, n x n] - [b, l, h, k]
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)
        k = self.key(x)
        # output shape of k: [b, k, p], p = m x m
        k = self._reshape_input(k)
        v = self.value(x)
        # output shape of v: [ b, p, k], p = m x m
        v = self._reshape_input(v)

        # calculate attn score
        q = q * (self.key_dim ** -0.5)
        attn_score = q @ k.transpose(-1, -2)
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.dropout(attn_score)
        output = attn_score @ v

        # reshape o into [b, hk, n, n,]
        output = self._reshape_output(output, self.num_heads, H // self.query_h_strides, W // self.query_w_strides)
        output = self._output_proj(output)
        return output


class MNV4LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            inp,
            num_heads,
            key_dim,
            value_dim,
            query_h_strides,
            query_w_strides,
            kv_strides,
            use_layer_scale,
            use_multi_query,
            use_residual=True
    ):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)

        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(inp, self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            specs_ = layer_spec['block_specs'][i]
            if isinstance(specs_, list):
                args = dict(zip(schema_, layer_spec['block_specs'][i]))
                layers.add_module(f"convbn_{i}", conv_2d(**args))
            elif specs_ == "AdaptiveAvgPool2d":
                layers.add_module(f"AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(1))
            else:
                raise NotImplementedError
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio', 'use_layer_scale', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            mhsa = args.pop("mhsa") if "mhsa" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(mhsa_schema_, [args['oup']] + (mhsa)))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, block_specs, num_classes=1000, detection=False, pretrained=None, init_cfg=None):
        super().__init__()

        if not detection:
            self.num_classes = num_classes
        self.spec = block_specs
        self.detection = detection
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5
        self.layer5 = build_blocks(self.spec['layer5'])

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x1, x2, x3, x4, x5]

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for key, value in _state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]  # 移除'backbone.'前缀 (长度为9)
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value  # 保留非backbone前缀的键

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
            if missing_keys:
                logger.warning(f"MobileNet v4_init_weight\nMissing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"MobileNet v4_init_weight\nUnexpected keys: {unexpected_keys}")


@BACKBONES.register_module()
class mobilenetv4_conv_small(MobileNetV4):
    def __init__(self, **kwargs):
        """
        Constructs a MobileNetV4-Conv-Small model
        """
        block_specs = {
            "conv0": {
                "block_name": "convbn",
                "num_blocks": 1,
                "block_specs": [
                    [3, 32, 3, 2]
                ]
            },
            "layer1": {
                "block_name": "convbn",
                "num_blocks": 2,
                "block_specs": [
                    [32, 32, 3, 2],
                    [32, 32, 1, 1]
                ]
            },
            "layer2": {
                "block_name": "convbn",
                "num_blocks": 2,
                "block_specs": [
                    [32, 96, 3, 2],
                    [96, 64, 1, 1]
                ]
            },
            "layer3": {
                "block_name": "uib",
                "num_blocks": 6,
                "block_specs": [
                    [64, 96, 5, 5, True, 2, 3, False],
                    [96, 96, 0, 3, True, 1, 2, False],
                    [96, 96, 0, 3, True, 1, 2, False],
                    [96, 96, 0, 3, True, 1, 2, False],
                    [96, 96, 0, 3, True, 1, 2, False],
                    [96, 96, 3, 0, True, 1, 4, False],
                ]
            },
            "layer4": {
                "block_name": "uib",
                "num_blocks": 6,
                "block_specs": [
                    [96, 128, 3, 3, True, 2, 6, False],
                    [128, 128, 5, 5, True, 1, 4, False],
                    [128, 128, 0, 5, True, 1, 4, False],
                    [128, 128, 0, 5, True, 1, 3, False],
                    [128, 128, 0, 3, True, 1, 4, False],
                    [128, 128, 0, 3, True, 1, 4, False],
                ]
            },
            "layer5": {
                "block_name": "convbn",
                "num_blocks": 3,
                "block_specs": [
                    [128, 960, 1, 1],
                    "AdaptiveAvgPool2d",
                    [960, 1280, 1, 1]
                ]
            }
        }
        super().__init__(block_specs, **kwargs)


@BACKBONES.register_module()
class mobilenetv4_conv_medium(MobileNetV4):
    def __init__(self, **kwargs):
        """
        Constructs a MobileNetV4-Conv-Medium model
        """
        block_specs = {
            "conv0": {
                "block_name": "convbn",
                "num_blocks": 1,
                "block_specs": [
                    [3, 32, 3, 2]
                ]
            },
            "layer1": {
                "block_name": "fused_ib",
                "num_blocks": 1,
                "block_specs": [
                    [32, 48, 2, 4.0, False]
                ]
            },
            "layer2": {
                "block_name": "uib",
                "num_blocks": 2,
                "block_specs": [
                    [48, 80, 3, 5, True, 2, 4, False],
                    [80, 80, 3, 3, True, 1, 2, False]
                ]
            },
            "layer3": {
                "block_name": "uib",
                "num_blocks": 8,
                "block_specs": [
                    [80, 160, 3, 5, True, 2, 6, False],
                    [160, 160, 3, 3, True, 1, 4, False],
                    [160, 160, 3, 3, True, 1, 4, False],
                    [160, 160, 3, 5, True, 1, 4, False],
                    [160, 160, 3, 3, True, 1, 4, False],
                    [160, 160, 3, 0, True, 1, 4, False],
                    [160, 160, 0, 0, True, 1, 2, False],
                    [160, 160, 3, 0, True, 1, 4, False]
                ]
            },
            "layer4": {
                "block_name": "uib",
                "num_blocks": 11,
                "block_specs": [
                    [160, 256, 5, 5, True, 2, 6, False],
                    [256, 256, 5, 5, True, 1, 4, False],
                    [256, 256, 3, 5, True, 1, 4, False],
                    [256, 256, 3, 5, True, 1, 4, False],
                    [256, 256, 0, 0, True, 1, 4, False],
                    [256, 256, 3, 0, True, 1, 4, False],
                    [256, 256, 3, 5, True, 1, 2, False],
                    [256, 256, 5, 5, True, 1, 4, False],
                    [256, 256, 0, 0, True, 1, 4, False],
                    [256, 256, 0, 0, True, 1, 4, False],
                    [256, 256, 5, 0, True, 1, 2, False]
                ]
            },
            "layer5": {
                "block_name": "convbn",
                "num_blocks": 3,
                "block_specs": [
                    [256, 960, 1, 1],
                    "AdaptiveAvgPool2d",
                    [960, 1280, 1, 1]
                ]
            }
        }
        super().__init__(block_specs, **kwargs)


@BACKBONES.register_module()
class mobilenetv4_conv_large(MobileNetV4):
    def __init__(self, **kwargs):
        """
        Constructs a MobileNetV4-Conv-Large model
        """
        block_specs = {
            "conv0": {
                "block_name": "convbn",
                "num_blocks": 1,
                "block_specs": [
                    [3, 24, 3, 2]
                ]
            },
            "layer1": {
                "block_name": "fused_ib",
                "num_blocks": 1,
                "block_specs": [
                    [24, 48, 2, 4.0, False]
                ]
            },
            "layer2": {
                "block_name": "uib",
                "num_blocks": 2,
                "block_specs": [
                    [48, 96, 3, 5, True, 2, 4],
                    [96, 96, 3, 3, True, 1, 4]
                ]
            },
            "layer3": {
                "block_name": "uib",
                "num_blocks": 11,
                "block_specs": [
                    [96, 192, 3, 5, True, 2, 4, False],
                    [192, 192, 3, 3, True, 1, 4, False],
                    [192, 192, 3, 3, True, 1, 4, False],
                    [192, 192, 3, 3, True, 1, 4, False],
                    [192, 192, 3, 5, True, 1, 4, False],
                    [192, 192, 5, 3, True, 1, 4, False],
                    [192, 192, 5, 3, True, 1, 4, False],
                    [192, 192, 5, 3, True, 1, 4, False],
                    [192, 192, 5, 3, True, 1, 4, False],
                    [192, 192, 5, 3, True, 1, 4, False],
                    [192, 192, 3, 0, True, 1, 4, False]
                ]
            },
            "layer4": {
                "block_name": "uib",
                "num_blocks": 13,
                "block_specs": [
                    [192, 512, 5, 5, True, 2, 4, False],
                    [512, 512, 5, 5, True, 1, 4, False],
                    [512, 512, 5, 5, True, 1, 4, False],
                    [512, 512, 5, 5, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False],
                    [512, 512, 5, 3, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False],
                    [512, 512, 5, 3, True, 1, 4, False],
                    [512, 512, 5, 5, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False],
                    [512, 512, 5, 0, True, 1, 4, False]
                ]
            },
            "layer5": {
                "block_name": "convbn",
                "num_blocks": 3,
                "block_specs": [
                    [512, 960, 1, 1],
                    "AdaptiveAvgPool2d",
                    [960, 1280, 1, 1]
                ]
            }
        }
        super().__init__(block_specs, **kwargs)
