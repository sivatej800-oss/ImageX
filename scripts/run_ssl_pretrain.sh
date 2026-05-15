#!/bin/bash
# Self-supervised pre-training of SwinUNETR on plain head MRIs (multi-GPU DDP).
# Run from the repository root, inside a MONAI Docker container.
# Volumes: /train holds train/, test/, log/; /cache holds the MONAI persistent cache.

python -W ignore -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    src/segmentation/ssl_pretrain_ddp.py \
    --traindir=/train/train \
    --testdir=/train/test \
    --logdir=/train/log \
    --cachedir=/cache \
    --epochs=500
