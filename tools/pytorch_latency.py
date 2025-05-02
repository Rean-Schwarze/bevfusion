import argparse
import copy
import os
import time
import warnings

import mmcv
import torch
from torch.nn import functional as F
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model, force_fp32
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--type", help="bevfusion | kd")

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


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    samples_per_gpu = cfg.data.samples_per_gpu
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    dataset = data_loader.dataset

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model.cuda().eval()

    device = 'cuda:0'

    camera_backbone_times = []
    camera_neck_times = []
    camera_vtransform_times = []
    lidar_voxelize_times = []
    lidar_backbone_times = []
    fuser_times = []
    decoder_backbone_times = []
    decoder_neck_times = []
    heads_times = []

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

    num_runs = 0
    print("---------------------- Start ----------------------")
    for data in data_loader:
        num_runs += 1

        img = torch.randn(24, 3, 3, 3).cuda()

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

        features = []
        start_time = time.time()
        with torch.no_grad():
            x = encoders.camera.backbone(img)
            time2 = time.time()

            x = encoders.camera.neck(x)
            time3 = time.time()

            x = torch.randn(1, 6, 256, 32, 88).cuda()
            camera_features = encoders.camera.vtransform(x, points, radar, camera2ego, lidar2ego,
                                                         lidar2camera,
                                                         lidar2image, camera_intrinsics, camera2lidar,
                                                         img_aug_matrix,
                                                         lidar_aug_matrix, metas)
            features.append(camera_features)
            time4 = time.time()

            points = points[0]
            points = points.unsqueeze(0)
            feats, coords, sizes = voxelize(encoders, points)
            time5 = time.time()

            feats = torch.sum(feats, dim=1)
            batch_size = coords[-1, 0] + 1
            lidar_features = encoders.lidar.backbone(feats, coords, batch_size, sizes=sizes)
            features.append(lidar_features)
            time6 = time.time()

            features = features[::-1]

            fused_features = fuser(features)
            time7 = time.time()

            x = decoder.backbone(fused_features)
            time8 = time.time()

            x = decoder.neck(x)
            time9 = time.time()

            for type, head in heads.items():
                if type == "object":
                    head.shared_conv.half()
                    head.heatmap_head.half()
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas[0])

            time10 = time.time()

        camera_backbone_times.append(time2 - start_time)
        camera_neck_times.append(time3 - time2)
        camera_vtransform_times.append(time4 - time3)
        lidar_voxelize_times.append(time5 - time4)
        lidar_backbone_times.append(time6 - time5)
        fuser_times.append(time7 - time6)
        decoder_backbone_times.append(time8 - time7)
        decoder_neck_times.append(time9 - time8)
        heads_times.append(time10 - time9)

        print(f"Run {num_runs} completed, time: {time10 - start_time:.2f} seconds")

    camera_backbone_avg_time = sum(camera_backbone_times) / num_runs
    camera_neck_avg_time = sum(camera_neck_times) / num_runs
    camera_vtransform_avg_time = sum(camera_vtransform_times) / num_runs
    lidar_voxelize_avg_time = sum(lidar_voxelize_times) / num_runs
    lidar_backbone_avg_time = sum(lidar_backbone_times) / num_runs
    fuser_avg_time = sum(fuser_times) / num_runs
    decoder_backbone_avg_time = sum(decoder_backbone_times) / num_runs
    decoder_neck_avg_time = sum(decoder_neck_times) / num_runs
    heads_avg_time = sum(heads_times) / num_runs

    overall_avg_time = (camera_backbone_avg_time + camera_neck_avg_time + camera_vtransform_avg_time +
                        lidar_voxelize_avg_time + lidar_backbone_avg_time + fuser_avg_time +
                        decoder_backbone_avg_time + decoder_neck_avg_time + heads_avg_time)

    print(f'Camera Backbone average latency: {camera_backbone_avg_time * 1000:.2f} ms')
    print(f'Camera Neck average latency: {camera_neck_avg_time * 1000:.2f} ms')
    print(f'Camera Vtransform average latency: {camera_vtransform_avg_time * 1000:.2f} ms')
    print(f'Lidar Voxelize average latency: {lidar_voxelize_avg_time * 1000:.2f} ms')
    print(f'Lidar Backbone average latency: {lidar_backbone_avg_time * 1000:.2f} ms')
    print(f'Fuser average latency: {fuser_avg_time * 1000:.2f} ms')
    print(f'Decoder Backbone average latency: {decoder_backbone_avg_time * 1000:.2f} ms')
    print(f'Decoder Neck average latency: {decoder_neck_avg_time * 1000:.2f} ms')
    print(f'Heads average latency: {heads_avg_time * 1000:.2f} ms')
    print(f'Overall average latency of {num_runs} runs : {overall_avg_time * 1000:.2f} ms')


if __name__ == "__main__":
    main()
