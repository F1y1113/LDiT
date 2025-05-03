#!/bin/bash

MASTER_PORT=37124

NUM_GPUS=4

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_wohuman.py \
    --config config/nwm_cdit_wohuman.yaml \
    --ckpt-every 4000 \
    --eval-every 100 \
    --bfloat16 1 \
    --epochs 300 \
    --torch-compile 0