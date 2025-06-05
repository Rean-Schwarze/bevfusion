# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import sys

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import wrap_fp16_model, force_fp32
from mmengine.registry import init_default_scope
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d
from pytorch_quantization.nn import TensorQuantizer
from thop import profile, clever_format
from torch.nn import functional as F
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.utils.flops_counter import count_sparseconv, count_mha, count_window_msa
from mmdet.models.backbones.swin import WindowMSA, ShiftWindowMSA
from mmdet3d.models.utils.transformer import MultiheadAttention
from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d
from mmdet3d.utils import recursive_eval
from qat.lean.quantize import SparseConvolutionQunat, QuantAdd

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--type", help="bevfusion | kd")
    parser.add_argument('--ptq', action='store_true')
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
             "the inference speed",
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


@torch.no_grad()
@force_fp32()
def voxelize(encoders, points):
    feats, coords, sizes = [], [], []
    for k, res in enumerate(points):
        ret = encoders.lidar.voxelize(res)
        if len(ret) == 3:
            # hard voxelize
            f, c, n = ret
        else:
            assert len(ret) == 2
            f, c = ret
            n = None
        feats.append(f)
        coords.append(F.pad(c, (1, 0), mode="constant", value=k))
        if n is not None:
            sizes.append(n)

    feats = torch.cat(feats, dim=0)
    coords = torch.cat(coords, dim=0)
    if len(sizes) > 0:
        sizes = torch.cat(sizes, dim=0)

    return feats, coords, sizes


def count_quant_conv2d(m: QuantConv2d, x, y):
    x = x[0]
    batch_size, in_channels, h, w = x.size()
    out_channels, _, kh, kw = m.weight.size()

    # 卷积操作的MACs
    kernel_ops = kh * kw * (in_channels // m.groups)
    bias_ops = 1 if m.bias is not None else 0
    output_elements = batch_size * out_channels * y.size(2) * y.size(3)
    conv_macs = output_elements * (kernel_ops + bias_ops)

    # 输入量化的操作（缩放和舍入）
    input_quant_macs = output_elements * 2  # 一次乘法和一次加法

    # 权重量化的操作（缩放和舍入，按通道计算）
    weight_elements = out_channels * in_channels * kh * kw
    weight_quant_macs = weight_elements * 2

    total_macs = conv_macs + input_quant_macs + weight_quant_macs
    m.total_ops = torch.tensor([int(total_macs)])


def count_sparse_convolution_quant(m: SparseConvolutionQunat, x, y):
    indice_dict = y.indice_dict[m.indice_key]
    kmap_size = indice_dict[-2].sum().item()
    m.total_ops += kmap_size * x[0].features.shape[1] * y.features.shape[1]


def count_quant_add(m: QuantAdd, x, y):
    # 获取输入张量
    input1, input2 = x

    # 假设两个输入维度相同，取第一个输入的元素数
    elements = input1.numel()

    # 量化加法操作：
    # 1. 两个输入的量化（各2次操作：缩放+舍入）
    # 2. 一次加法操作
    # 3. 输出的量化（2次操作：缩放+舍入）
    quant_ops = elements * 2 * 2  # 两个输入的量化
    add_ops = elements  # 加法操作
    output_quant_ops = elements * 2  # 输出量化

    # 总操作数
    total_ops = quant_ops + add_ops + output_quant_ops

    # 设置计算量和参数数量（QuantAdd通常没有可训练参数）
    m.total_ops = torch.tensor([int(total_ops)])
    m.total_params = torch.tensor([0])


def main():
    args = parse_args()

    if args.type in ['fastbev']:
        cfg = Config.fromfile(args.config)
    else:
        configs.load(args.config, recursive=True)
        cfg = Config(recursive_eval(configs), filename=args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    if args.ptq:
        model = torch.load(args.checkpoint).module
    else:
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    split_line = '=' * 45

    print(f'\n{split_line} Start Calculating MACs and Params of type {args.type} {split_line}')

    encoders = None
    fuser = None
    decoder = None
    heads = None
    if args.type == 'bevfusion':
        encoders = model.encoders
        fuser = model.fuser
        decoder = model.decoder
        heads = model.heads
    elif args.type == 'kd':
        encoders = model.encoders_student
        fuser = model.fuser_student
        decoder = model.decoder_student
        heads = model.heads_student

    count = 0
    macs_items = []
    params_items = []

    for data in data_loader:
        count += 1
        if count > 1:
            break
        if args.type == 'fastbev':
            img = torch.randn(4, 3, 256, 704).cuda()

            macs, params = profile(model.backbone, inputs=(img,))
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Backbone macs: {macs}\tparams: {params}\n')

            neck_input = model.backbone(img)
            macs, params = profile(model.neck, inputs=(neck_input,))
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Neck macs: {macs}\tparams: {params}\n')

            neck_3d_input = torch.randn(4, 256, 200, 200, 4).cuda()
            macs, params = profile(model.neck_3d, inputs=(neck_3d_input,))
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Neck 3d macs: {macs}\tparams: {params}\n')

            feature_bev = model.neck_3d(neck_3d_input)
            macs, params = profile(model.bbox_head, inputs=(feature_bev,))
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Bbox head macs: {macs}\tparams: {params}\n')

        else:
            points = [i.cuda() for i in data["points"].data[0]]
            radar = [i.cuda() for i in data["radar"].data[0]]
            camera2ego = [i.cuda() for i in data["camera2ego"].data[0]][0]
            lidar2ego = [i.cuda() for i in data["lidar2ego"].data[0]][0]
            lidar2camera = [i.cuda() for i in data["lidar2camera"].data[0]][0]
            lidar2image = [i.cuda() for i in data["lidar2image"].data[0]]
            camera_intrinsics = [i.cuda() for i in data["camera_intrinsics"].data[0]][0]
            camera2lidar = [i.cuda() for i in data["camera2lidar"].data[0]][0]
            camera2lidar = camera2lidar.unsqueeze(0)
            img_aug_matrix = [i.cuda() for i in data["img_aug_matrix"].data[0]][0]
            img_aug_matrix = img_aug_matrix.unsqueeze(0)
            lidar_aug_matrix = [i.cuda() for i in data["lidar_aug_matrix"].data[0]][0]
            lidar_aug_matrix = lidar_aug_matrix.unsqueeze(0)
            metas = data["metas"].data

            img = torch.randn(4, 3, 256, 704).cuda()
            macs, params = profile(encoders.camera.backbone, inputs=(img,), custom_ops={
                QuantConv2d: count_quant_conv2d,
                # TensorQuantizer: count_tensor_quantizer,
                WindowMSA: count_window_msa,
            })
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Camera backbone macs: {macs}\tparams: {params}\n')

            quant_custom_op = {}
            if args.ptq:
                quant_custom_op = {
                    # TensorQuantizer: count_tensor_quantizer,
                    QuantConv2d: count_quant_conv2d,
                }
            neck_input = encoders.camera.backbone(img)
            macs, params = profile(encoders.camera.neck, inputs=(neck_input,), custom_ops=quant_custom_op)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Camera neck macs: {macs}\tparams: {params}\n')
            neck_output = encoders.camera.neck(neck_input)

            features = []
            x = torch.randn(1, 6, 256, 32, 88).cuda()
            macs, params = profile(encoders.camera.vtransform, inputs=(x, points, radar, camera2ego, lidar2ego,
                                                                       lidar2camera,
                                                                       lidar2image, camera_intrinsics, camera2lidar,
                                                                       img_aug_matrix,
                                                                       lidar_aug_matrix, metas,),
                                   custom_ops=quant_custom_op)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Camera vtransform macs: {macs}\tparams: {params}\n')
            camera_features = encoders.camera.vtransform(x, points, radar, camera2ego, lidar2ego,
                                                         lidar2camera,
                                                         lidar2image, camera_intrinsics, camera2lidar,
                                                         img_aug_matrix,
                                                         lidar_aug_matrix, metas)
            features.append(camera_features)

            del radar, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            points = points[0]
            points = points.unsqueeze(0)
            b = 100
            num_batches = points.shape[1] // b
            macs_list = []
            params_list = []
            for i in range(num_batches):
                batch_points = points[:, i * b: (i + 1) * b, :]
                batch_macs, batch_params = profile(encoders.lidar.voxelize, inputs=(batch_points,),
                                                   custom_ops=quant_custom_op)
                macs_list.append(batch_macs)
                params_list.append(batch_params)
            macs = sum(macs_list)
            params = sum(params_list)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Lidar voxelize macs: {macs}\tparams: {params}\n')
            feats, coords, sizes = voxelize(encoders, points)

            feats = torch.sum(feats, dim=1)
            batch_size = coords[-1, 0] + 1
            macs, params = profile(encoders.lidar.backbone, inputs=(feats, coords, batch_size, sizes),
                                   custom_ops={SparseConv3d: count_sparseconv,
                                               SubMConv3d: count_sparseconv,
                                               QuantConv2d: count_quant_conv2d,
                                               SparseConvolutionQunat: count_sparse_convolution_quant,
                                               QuantAdd: count_quant_add,
                                               })
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Lidar backbone macs: {macs}\tparams: {params}\n')
            lidar_features = encoders.lidar.backbone(feats, coords, batch_size, sizes=sizes)
            features.append(lidar_features)

            features = features[::-1]

            macs, params = profile(fuser, inputs=(features,), custom_ops=quant_custom_op)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Fuser macs: {macs}\tparams: {params}\n')
            fused_features = fuser(features)

            macs, params = profile(decoder.backbone, inputs=(fused_features,), custom_ops=quant_custom_op)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Decoder backbone macs: {macs}\tparams: {params}\n')
            decoder_backbone_output = decoder.backbone(fused_features)

            macs, params = profile(decoder.neck, inputs=(decoder_backbone_output,), custom_ops=quant_custom_op)
            macs_items.append(macs)
            params_items.append(params)
            macs, params = clever_format([macs, params], "%.3f")
            print(f'Decoder neck macs: {macs}\tparams: {params}\n')
            decoder_neck_output = decoder.neck(decoder_backbone_output)

            for type, head in heads.items():
                if type == "object":
                    head.shared_conv.half()
                    head.heatmap_head.half()
                    macs, params = profile(head, inputs=(decoder_neck_output, metas),
                                           custom_ops={
                                               MultiheadAttention: count_mha,
                                               QuantConv2d: count_quant_conv2d,
                                           })
                    macs_items.append(macs)
                    params_items.append(params)
                    macs, params = clever_format([macs, params], "%.3f")
                    print(f'Heads macs: {macs}\tparams: {params}\n')

        print(f'{split_line}')

        macs, params = clever_format([sum(macs_items), sum(params_items)], "%.3f")
        print(f'Total macs: {macs}\tTotal params: {params}')

        print(f'{split_line}')

    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
