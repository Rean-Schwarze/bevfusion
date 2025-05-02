#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
RUNDIR=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG -run-dir $RUNDIR --launcher pytorch ${@:4}