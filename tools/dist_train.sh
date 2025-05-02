#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
LOADFROM=$4
RUNDIR=$5
RESUMEFROM=$6
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --run-dir $RUNDIR --launcher pytorch ${@:7} --model.encoders.camera.backbone.init_cfg.checkpoint $CHECKPOINT --load_from $LOADFROM --resume_from $RESUMEFROM