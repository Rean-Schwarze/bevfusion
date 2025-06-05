import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info, wrap_fp16_model, _load_checkpoint
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from ultralytics import YOLO


def main():
    #     dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int)
    args, opts = parser.parse_known_args()

    distributed = False

    cfg = load_config(args, opts, False)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(
        f"GPU {torch.cuda.current_device()} of {torch.cuda.device_count()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"device_capability: {torch.cuda.get_device_capability()}")
    # logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    # if "yolo_v11" in cfg.model.encoders.camera.backbone.type:
    #     ckpt_path = "./pretrained/yolo11m.pt"
    #     yolo_model = YOLO(ckpt_path)
    #     model.encoders.camera.backbone = yolo_model
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model.init_weights()
    model.cuda()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed,
        validate=True,
        timestamp=timestamp,
    )
#     x = torch.randn(4, 3, 256, 704).cuda()
#     y = model.encoders.camera.backbone(x)
#     print(f"type(y): {type(y)}")
#     print(f"len(y): {len(y)}")
#     for i in y:
#         print(f"type(i): {type(i)}")
#         print(i.shape)


def load_config(args, opts, is_teacher):
    if is_teacher:
        dump_name = "configs_teacher.yaml"
        configs.load(args.teacher_config, recursive=True)
    else:
        dump_name = "configs.yaml"
        configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    cfg.dist_params = dict(backend='nccl')
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, dump_name))
    return cfg


if __name__ == "__main__":
    main()
