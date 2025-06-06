#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUTPKL=$4
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox --out $OUTPKL --launcher pytorch ${@:5}