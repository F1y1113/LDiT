#!/bin/bash

MASTER_PORT=$(shuf -i 30000-39999 -n 1)
echo "Using MASTER_PORT: $MASTER_PORT"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_noaction.py \
    --config config/nwm_cdit_ha.yaml \
    --ckpt-every 4000 \
    --eval-every 20000 \
    --bfloat16 1 \
    --epochs 200 \
    --torch-compile 0