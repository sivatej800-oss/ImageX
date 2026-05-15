#!/bin/bash
# Launch a MONAI Docker container for SSL pre-training.
# Edit the two host paths below, then run from the repository root.
#   /PATH/TO/IMAGEX  -> this repository (mounted as /train inside the container)
#   /PATH/TO/CACHE   -> a host folder for the MONAI persistent cache

docker run --gpus all --rm -ti \
    -v /PATH/TO/IMAGEX:/train \
    -v /PATH/TO/CACHE:/cache \
    --ipc=host \
    projectmonai/monai:latest
