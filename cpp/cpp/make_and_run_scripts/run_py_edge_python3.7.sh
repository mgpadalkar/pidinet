#!/bin/bash

# params
# ENV_NAME="tf-1.14"
ENV_NAME="IRMID"
model="pidinet_tiny_converted"
image="pyscripts/test_image.bmp"
config="carv4"
checkpoint="trained_models/table5_pidinet-tiny-l.pth"

source ${HOME}/localInstalls/set_python3.7.sh
conda activate ${ENV_NAME}

python pyscripts/edge_detector.py --model $model --image $image --config $config --checkpoint $checkpoint --resize_factor 0.3
