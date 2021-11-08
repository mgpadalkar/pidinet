#!/bin/bash

# params
# ENV_NAME="tf-1.14"
ENV_NAME="IRMID"
# model="pidinet_tiny_converted"
# image="pyscripts/test_image.bmp"
# config="carv4"
# checkpoint="trained_models/table5_pidinet-tiny-l.pth"

source ${HOME}/localInstalls/set_python3.7.sh
conda activate ${ENV_NAME}

if [ "$5" -ge 1 ]; then
  sa="--sa"
else
  sa=""
fi

if [ "$6" -ge 1 ]; then
  dil="--dil"
else
  dil=""
fi


cmd="python pyscripts/edge_detector.py --image $1 --model $2 --config $3 --checkpoint $4 $sa $dil --resize_factor $7"
echo $cmd
$cmd
