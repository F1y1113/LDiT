#!/bin/bash

MASTER_PORT=37124

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --config config/nwm_cdit_ha_context_1.yaml \
    --ckpt-every 4000 \
    --eval-every 20000 \
    --bfloat16 1 \
    --epochs 200 \
    --torch-compile 0