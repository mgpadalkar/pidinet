ENV_NAME=IRMID

# set path for python having tensorflow
source ~/localInstalls/set_opencv-3.4.4.sh
source ~/localInstalls/set_python3.7.sh
conda activate ${ENV_NAME}
export LD_LIBRARY_PATH=~/localInstalls/anaconda3/envs/tf-1.14/lib:$LD_LIBRARY_PATH


# command
./bin/test_edge_detector $1 $2 $3
