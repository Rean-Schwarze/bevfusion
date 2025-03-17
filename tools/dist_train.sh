#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
LOADFROM=$4
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG --model.encoders.camera.backbone.init_cfg.checkpoint $CHECKPOINT --load_from $LOADFROM --launcher pytorch ${@:4}