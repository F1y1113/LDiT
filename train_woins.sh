#!/bin/bash

MASTER_PORT=37124

NUM_GPUS=4

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --config config/nwm_cdit_ha_woins.yaml \
    --ckpt-every 2000 \
    --eval-every 15000 \
    --bfloat16 1 \
    --epochs 300 \
    --torch-compile 0