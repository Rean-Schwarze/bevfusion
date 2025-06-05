from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            image_size: Tuple[int, int],
            feature_size: Tuple[int, int],
            xbound: Tuple[float, float, float],
            ybound: Tuple[float, float, float],
            zbound: Tuple[float, float, float],
            dbound: Tuple[float, float, float],
            downsample: int = 1,
            is_xtransform: bool = False,
            height_expand: bool = False,
            add_depth_features: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            height_expand=height_expand,
            add_depth_features=add_depth_features
        )
        self.is_xtransform = is_xtransform
        if self.add_depth_features and self.height_expand:
            dtransform_in_channels = 6
        else:
            dtransform_in_channels = 1
        self.dtransform = nn.Sequential(
            nn.Conv2d(dtransform_in_channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # self.xtransform = nn.Sequential(  # change for SwiftFormer, input[6, 256, 64, 176]
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        # )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d, **kwargs):
        """
        x：图像特征
        d：激光点云在图像物理坐标系下的深度信息。
        """
        # x为neck部分输出的图像特征 [1, 6, 256, 32, 88])
        # d [1, 6, 6, 256, 704]
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:])  # [6, 6, 256, 704]
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        if self.is_xtransform:
            x = self.xtransform(x)  # change for SwiftFormer
        # print(f"\n\nShape of x: {x.shape}\n\n")
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        if self.is_xtransform:
            x = x.view(B, N, self.C, self.D, fH // 2, fW // 2)  # change for SwiftFormer
        else:
            x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def print_kwargs(self, **kwargs):
        print("-----------**kwargs-----------")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print("-----------**kwargs-----------")

    def forward(self, *args, **kwargs):
        # self.print_kwargs(**kwargs)
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x
