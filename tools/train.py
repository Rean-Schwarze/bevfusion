import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import init_dist
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    # dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    # parser.add_argument("--use-distillation", action='store_true', help="Enable knowledge distillation during training")
    # parser.add_argument('--distillation-alpha',
    #                     default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    # parser.add_argument("--teacher-config", metavar="FILE", help="teacher model's config file")
    # parser.add_argument("--teacher-ckpt-path", metavar="FILE", help="teacher model's ckpt")
    args, opts = parser.parse_known_args()

    # distillation = args.use_distillation

    distributed = False
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)

    cfg = load_config(args, opts, False)
    # if args.use_distillation:
    #     cfg_teacher = load_config(args, opts, True)

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
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
    model.init_weights()
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
    # torch.cuda.set_device(dist.local_rank())
    cfg.dist_params = dict(backend='nccl')
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, dump_name))
    return cfg


def load_teacher_model(args, opts):
    cfg = load_config(args=args, opts=opts, is_teacher=True)
    teacher_path = args.teacher_ckpt_path

    teacher_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    teacher_ckpt = torch.load(teacher_path)
    teacher_model.load_state_dict(teacher_ckpt['state_dict'])
    teacher_model.eval()
    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()

    return teacher_model


if __name__ == "__main__":
    main()
