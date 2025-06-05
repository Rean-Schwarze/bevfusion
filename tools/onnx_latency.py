import argparse

import torch
import onnxruntime
import numpy as np
import time

from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model, load_checkpoint
from torchpack.utils.config import configs
from mmcv import Config, DictAction

from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="onnx test (and eval) a model")
    parser.add_argument("config", help="config file path")
    parser.add_argument("path", help="onnx file path (folder) e.g: /root/autodl-tmp/model/swint/")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("Device: ", onnxruntime.get_device())
    print("Available providers: ", onnxruntime.get_available_providers())

    opts = onnxruntime.SessionOptions()
    opts.enable_profiling = True

    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT",
                                            "cudnn_conv_use_max_workspace": '1'}),
                 "CPUExecutionProvider"]

    # 加载 ONNX 模型
    session_camera_backbone = onnxruntime.InferenceSession(args.path + "camera.backbone.onnx", opts=opts,
                                                           providers=providers)
    session_camera_vtransform = onnxruntime.InferenceSession(args.path + "camera.vtransform.onnx", opts=opts,
                                                             providers=providers)
    session_lidar_backbone = onnxruntime.InferenceSession(args.path + "lidar.backbone.xyz.onnx", opts=opts,
                                                          providers=providers)
    session_fuser = onnxruntime.InferenceSession(args.path + "fuser.onnx", opts=opts,
                                                 providers=providers)
    session_head_bbox = onnxruntime.InferenceSession(args.path + "head.bbox.onnx", opts=opts,
                                                     providers=providers)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # if "CLASSES" in checkpoint.get("meta", {}):
    #     model.CLASSES = checkpoint["meta"]["CLASSES"]
    # else:
    #     model.CLASSES = dataset.CLASSES
    # model.eval()

    num_runs = 0
    camera_backbone_times = []
    camera_vtransform_times = []
    lidar_backbone_times = []
    fuser_times = []
    head_bbox_times = []
    overall_times = []

    for data in data_loader:
        num_runs += 1
        if num_runs > 5:
            break

        img = data["img"].data[0]
        depth = data["depths"].data  # [1,6,256,704]
        depth = depth.unsqueeze(2)
        depth = depth.repeat(1, 1, 6, 1, 1)  # 匹配shape（ [1,6,6,256,704]

        feat_in = torch.randn(1, 80, 360, 360).cuda()
        x = torch.randn(1, 5).cuda()

        start_time = time.time()

        outputs = session_camera_backbone.run(None, {"img": img.numpy(), "depth": depth.numpy()})
        time2 = time.time()

        camera_feature = session_camera_vtransform.run(None, {"feat_in": feat_in.numpy()})
        time3 = time.time()

        # SparseConvolution 算子未实现，无法run
        lidar_feature = session_lidar_backbone.run(None, {"0": x.numpy()})
        time4 = time.time()

        fused_feature = session_fuser.run(None, {"camera": camera_feature.numpy(),
                                                 "lidar": lidar_feature.numpy()})
        time5 = time.time()

        heads_outputs = session_head_bbox.run(None, {"middle": fused_feature.numpy()})
        time6 = time.time()

        camera_backbone_times.append(time2 - start_time)
        camera_vtransform_times.append(time3 - time2)
        lidar_backbone_times.append(time4 - time3)
        fuser_times.append(time5 - time4)
        head_bbox_times.append(time6 - time5)
        overall_times.append(time6 - start_time)

        print(f"Run {num_runs}: {overall_times[-1]} seconds")

    print(f"Camera Backbone average inference time in {num_runs} runs: {sum(camera_backbone_times)/num_runs * 1000:.2f} ms")
    print(f"Camera Vtransform average inference time in {num_runs} runs: {sum(camera_vtransform_times)/num_runs * 1000:.2f} ms")
    print(f"Lidar Backbone average inference time in {num_runs} runs: {sum(lidar_backbone_times)/num_runs * 1000:.2f} ms")
    print(f"Fuser average inference time in {num_runs} runs: {sum(fuser_times)/num_runs * 1000:.2f} ms")
    print(f"Head Bbox average inference time in {num_runs} runs: {sum(head_bbox_times)/num_runs * 1000:.2f} ms")
    print(f"Overall average inference time in {num_runs} runs: {sum(overall_times)/num_runs * 1000:.2f} ms")


if __name__ == "__main__":
    main()
