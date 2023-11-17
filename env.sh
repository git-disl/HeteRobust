#export CUDA_HOME=/usr/local/cuda-11.0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/compat/
conda activate $1
export PYTHONPATH=$PYTHONPATH:`pwd`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/git/mxnet

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_USE_FUSION=0
